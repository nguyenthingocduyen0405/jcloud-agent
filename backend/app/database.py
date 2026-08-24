from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


INSTANCE_TABLE_SQL = """
CREATE TABLE instances (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL COLLATE NOCASE,
    image TEXT NOT NULL,
    vcpus INTEGER NOT NULL,
    ram_gb INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, name)
)
"""


class Repository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL DEFAULT 'mock-session',
                    user_id TEXT NOT NULL DEFAULT 'mock-user',
                    project_id TEXT NOT NULL DEFAULT 'mock-project',
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_requests (
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'ko',
                    parameters TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, user_id, project_id)
                )
                """
            )
            pending_request_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(pending_requests)").fetchall()
            }
            if "language" not in pending_request_columns:
                connection.execute(
                    "ALTER TABLE pending_requests ADD COLUMN language TEXT NOT NULL DEFAULT 'ko'"
                )
            operation_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(operations)").fetchall()
            }
            for column, default in (
                ("session_id", "mock-session"),
                ("user_id", "mock-user"),
                ("project_id", "mock-project"),
            ):
                if column not in operation_columns:
                    connection.execute(
                        f"ALTER TABLE operations ADD COLUMN {column} TEXT NOT NULL DEFAULT '{default}'"
                    )

            instance_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'instances'"
            ).fetchone()
            if not instance_exists:
                connection.execute(INSTANCE_TABLE_SQL)
            elif self._instances_need_migration(connection):
                self._migrate_instances(connection)

    @staticmethod
    def _instances_need_migration(connection: sqlite3.Connection) -> bool:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(instances)")}
        if "session_id" not in columns:
            return True
        for index in connection.execute("PRAGMA index_list(instances)").fetchall():
            if not index["unique"]:
                continue
            indexed_columns = [
                row["name"]
                for row in connection.execute(f"PRAGMA index_info('{index['name']}')").fetchall()
            ]
            if indexed_columns == ["name"]:
                return True
        return False

    @staticmethod
    def _migrate_instances(connection: sqlite3.Connection) -> None:
        old_columns = {row["name"] for row in connection.execute("PRAGMA table_info(instances)")}
        connection.execute("ALTER TABLE instances RENAME TO instances_legacy")
        connection.execute(INSTANCE_TABLE_SQL)
        session_expression = "session_id" if "session_id" in old_columns else "'mock-session'"
        connection.execute(
            f"""
            INSERT INTO instances (id, session_id, name, image, vcpus, ram_gb, status, created_at)
            SELECT id, {session_expression}, name, image, vcpus, ram_gb, status, created_at
            FROM instances_legacy
            """
        )
        connection.execute("DROP TABLE instances_legacy")

    @staticmethod
    def _seed_session(connection: sqlite3.Connection, session_id: str) -> None:
        now = utc_now()
        connection.executemany(
            """
            INSERT INTO instances
            (id, session_id, name, image, vcpus, ram_gb, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (f"vm-{uuid4().hex[:12]}", session_id, "web-demo", "Ubuntu 22.04", 2, 4, "ACTIVE", now),
                (f"vm-{uuid4().hex[:12]}", session_id, "test-01", "Ubuntu 22.04", 1, 2, "SHUTOFF", now),
            ],
        )

    def ensure_session(self, session_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                "SELECT COUNT(*) FROM instances WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            if count == 0:
                self._seed_session(connection, session_id)

    def list_instances(self, session_id: str) -> list[dict[str, Any]]:
        self.ensure_session(session_id)
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, image, vcpus, ram_gb, status, created_at
                FROM instances WHERE session_id = ? ORDER BY created_at, name
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_instance(self, session_id: str, name: str) -> dict[str, Any] | None:
        self.ensure_session(session_id)
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, name, image, vcpus, ram_gb, status, created_at
                FROM instances WHERE session_id = ? AND name = ? COLLATE NOCASE
                """,
                (session_id, name),
            ).fetchone()
        return dict(row) if row else None

    def create_instance(self, session_id: str, instance: dict[str, Any]) -> dict[str, Any]:
        self.ensure_session(session_id)
        record = {**instance, "session_id": session_id}
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO instances
                (id, session_id, name, image, vcpus, ram_gb, status, created_at)
                VALUES (:id, :session_id, :name, :image, :vcpus, :ram_gb, :status, :created_at)
                """,
                record,
            )
        return instance

    def set_instance_status(
        self, session_id: str, name: str, status: str
    ) -> dict[str, Any] | None:
        self.ensure_session(session_id)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE instances SET status = ?
                WHERE session_id = ? AND name = ? COLLATE NOCASE
                """,
                (status, session_id, name),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                """
                SELECT id, name, image, vcpus, ram_gb, status, created_at
                FROM instances WHERE session_id = ? AND name = ? COLLATE NOCASE
                """,
                (session_id, name),
            ).fetchone()
        return dict(row) if row else None

    def reset_session(self, session_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM operations WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM pending_requests WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM instances WHERE session_id = ?", (session_id,))
            self._seed_session(connection, session_id)
        return self.list_instances(session_id)

    def get_pending_request(
        self, *, session_id: str, user_id: str, project_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT action, language, parameters, created_at, updated_at
                FROM pending_requests
                WHERE session_id = ? AND user_id = ? AND project_id = ?
                """,
                (session_id, user_id, project_id),
            ).fetchone()
        if not row:
            return None
        pending = dict(row)
        pending["parameters"] = json.loads(pending["parameters"])
        return pending

    def upsert_pending_request(
        self,
        *,
        session_id: str,
        user_id: str,
        project_id: str,
        action: str,
        language: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_requests
                    (session_id, user_id, project_id, action, language, parameters,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, user_id, project_id) DO UPDATE SET
                    action = excluded.action,
                    language = excluded.language,
                    parameters = excluded.parameters,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    user_id,
                    project_id,
                    action,
                    language,
                    json.dumps(parameters),
                    now,
                    now,
                ),
            )
        return {
            "action": action,
            "language": language,
            "parameters": parameters,
            "created_at": now,
            "updated_at": now,
        }

    def clear_pending_request(
        self, *, session_id: str, user_id: str, project_id: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                DELETE FROM pending_requests
                WHERE session_id = ? AND user_id = ? AND project_id = ?
                """,
                (session_id, user_id, project_id),
            )

    def create_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        record = {**operation, "payload": json.dumps(operation["payload"])}
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operations
                (id, session_id, user_id, project_id, action, status, summary, payload,
                 result, error, created_at, updated_at)
                VALUES (:id, :session_id, :user_id, :project_id, :action, :status, :summary,
                        :payload, NULL, NULL, :created_at, :updated_at)
                """,
                record,
            )
        return operation

    def get_operation(
        self, operation_id: str, *, session_id: str, user_id: str, project_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM operations
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                """,
                (operation_id, session_id, user_id, project_id),
            ).fetchone()
        return self._decode_operation(row)

    def claim_operation(
        self, operation_id: str, *, session_id: str, user_id: str, project_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations SET status = 'running', updated_at = ?
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                  AND status = 'waiting_for_confirmation'
                """,
                (utc_now(), operation_id, session_id, user_id, project_id),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                """
                SELECT * FROM operations
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                """,
                (operation_id, session_id, user_id, project_id),
            ).fetchone()
            return self._decode_operation(row)

    def cancel_operation(
        self, operation_id: str, *, session_id: str, user_id: str, project_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE operations
                SET status = 'cancelled', result = NULL, error = NULL, updated_at = ?
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                  AND status = 'waiting_for_confirmation'
                """,
                (utc_now(), operation_id, session_id, user_id, project_id),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute(
                """
                SELECT * FROM operations
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                """,
                (operation_id, session_id, user_id, project_id),
            ).fetchone()
            return self._decode_operation(row)

    def update_operation(
        self,
        operation_id: str,
        status: str,
        *,
        session_id: str,
        user_id: str,
        project_id: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE operations SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE id = ? AND session_id = ? AND user_id = ? AND project_id = ?
                """,
                (
                    status,
                    json.dumps(result) if result else None,
                    error,
                    utc_now(),
                    operation_id,
                    session_id,
                    user_id,
                    project_id,
                ),
            )
        return self.get_operation(
            operation_id, session_id=session_id, user_id=user_id, project_id=project_id
        )

    @staticmethod
    def _decode_operation(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        operation = dict(row)
        operation["payload"] = json.loads(operation["payload"])
        operation["result"] = json.loads(operation["result"]) if operation["result"] else None
        return operation
