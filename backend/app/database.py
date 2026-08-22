from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS instances (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    image TEXT NOT NULL,
                    vcpus INTEGER NOT NULL,
                    ram_gb INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            count = connection.execute("SELECT COUNT(*) FROM instances").fetchone()[0]
            if count == 0:
                connection.executemany(
                    """
                    INSERT INTO instances
                    (id, name, image, vcpus, ram_gb, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        ("vm-demo-01", "web-demo", "Ubuntu 22.04", 2, 4, "ACTIVE", utc_now()),
                        ("vm-test-01", "test-01", "Ubuntu 22.04", 1, 2, "SHUTOFF", utc_now()),
                    ],
                )

    def list_instances(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM instances ORDER BY created_at, name"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_instance(self, name: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM instances WHERE lower(name) = lower(?)", (name,)
            ).fetchone()
        return dict(row) if row else None

    def create_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO instances
                (id, name, image, vcpus, ram_gb, status, created_at)
                VALUES (:id, :name, :image, :vcpus, :ram_gb, :status, :created_at)
                """,
                instance,
            )
        return instance

    def set_instance_status(self, name: str, status: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE instances SET status = ? WHERE lower(name) = lower(?)",
                (status, name),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                "SELECT * FROM instances WHERE lower(name) = lower(?)", (name,)
            ).fetchone()
        return dict(row) if row else None

    def create_operation(self, operation: dict[str, Any]) -> dict[str, Any]:
        record = {**operation, "payload": json.dumps(operation["payload"])}
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operations
                (id, action, status, summary, payload, result, error, created_at, updated_at)
                VALUES (:id, :action, :status, :summary, :payload, NULL, NULL, :created_at, :updated_at)
                """,
                record,
            )
        return operation

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE id = ?", (operation_id,)
            ).fetchone()
        if not row:
            return None
        operation = dict(row)
        operation["payload"] = json.loads(operation["payload"])
        operation["result"] = json.loads(operation["result"]) if operation["result"] else None
        return operation

    def update_operation(
        self,
        operation_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE operations
                SET status = ?, result = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, json.dumps(result) if result else None, error, utc_now(), operation_id),
            )
        return self.get_operation(operation_id)

