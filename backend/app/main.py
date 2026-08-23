from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .cloud import CloudClient, MockCloudClient
from .database import Repository, utc_now
from .llm import LLMClient, LLMClientError, create_llm_client
from .policy import ALLOWED_ACTIONS, MUTATING_ACTIONS, contains_sensitive_value, is_prohibited_request
from .schemas import (
    ActionParameters,
    ChatRequest,
    ChatResponse,
    LLMDecision,
    Operation,
    RequestIdentity,
    SandboxResetResponse,
)


def get_request_identity(
    session_id: Annotated[str, Header(alias="X-Session-ID")],
    user_id: Annotated[str, Header(alias="X-User-ID")] = "mock-user",
    project_id: Annotated[str, Header(alias="X-Project-ID")] = "mock-project",
) -> RequestIdentity:
    return RequestIdentity(session_id=session_id, user_id=user_id, project_id=project_id)


IdentityDependency = Annotated[RequestIdentity, Depends(get_request_identity)]


def create_app(
    database_path: str | None = None,
    *,
    llm_client: LLMClient | None = None,
    cloud_client: CloudClient | None = None,
) -> FastAPI:
    resolved_path = database_path or os.getenv(
        "DATABASE_PATH", str(Path(__file__).resolve().parents[1] / "data" / "jcloud_agent.db")
    )
    repository = Repository(resolved_path)
    cloud = cloud_client or MockCloudClient(repository)
    llm = llm_client or create_llm_client()
    llm_provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        repository.initialize()
        yield

    application = FastAPI(title="JCloud Agent MVP", version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Session-ID", "X-User-ID", "X-Project-ID"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "cloud": "mock", "llm_provider": llm_provider}

    @application.get("/api/instances")
    def list_instances(identity: IdentityDependency) -> list[dict]:
        return cloud.list_instances(identity.session_id)

    @application.get("/api/quota")
    def get_quota(identity: IdentityDependency) -> dict[str, int]:
        return cloud.get_quota(identity.session_id)

    @application.post("/api/sandbox/reset", response_model=SandboxResetResponse)
    def reset_sandbox(identity: IdentityDependency) -> SandboxResetResponse:
        instances = cloud.reset_sandbox(identity.session_id)
        return SandboxResetResponse(status="reset", instances=instances)

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, identity: IdentityDependency) -> ChatResponse:
        all_user_text = "\n".join(
            [request.message, *(item.content for item in request.conversation_context)]
        )
        if contains_sensitive_value(all_user_text):
            return ChatResponse(
                message="대화에 API key, token, 비밀번호 또는 private key를 입력하지 마세요. 이 요청은 LLM에 전달되지 않았습니다."
            )
        if is_prohibited_request(request.message):
            return ChatResponse(
                message="이 요청은 JCloud Agent의 안전 범위를 벗어나므로 지원되지 않습니다."
            )

        cloud_context = {
            "quota": cloud.get_quota(identity.session_id),
            "images": [
                {
                    "name": image["name"],
                    "operating_system": image["operating_system"],
                    "version": image["version"],
                }
                for image in cloud.list_images()
            ],
            "flavors": [
                {"name": flavor["name"], "vcpus": flavor["vcpus"], "ram_gb": flavor["ram_gb"]}
                for flavor in cloud.list_flavors()
            ],
            "instance_names": [
                instance["name"] for instance in cloud.list_instances(identity.session_id)
            ],
        }
        try:
            raw_decision = llm.parse_message(
                request.message,
                [item.model_dump() for item in request.conversation_context],
                cloud_context,
            )
            decision = LLMDecision.model_validate(raw_decision)
        except (LLMClientError, ValidationError, ValueError, TypeError, RuntimeError):
            return ChatResponse(
                message="현재 요청을 안전하게 해석할 수 없습니다. 어떠한 작업도 실행되지 않았습니다."
            )

        if decision.decision_type != "action":
            return ChatResponse(message=decision.message)
        if decision.action not in ALLOWED_ACTIONS:
            return ChatResponse(message="이 작업은 아직 지원되지 않습니다.")

        if decision.action == "list_instances":
            instances = cloud.list_instances(identity.session_id)
            return ChatResponse(message=decision.message, data=instances)
        if decision.action == "get_quota":
            quota = cloud.get_quota(identity.session_id)
            message = f"사용 가능한 자원은 {quota['available_vcpus']} vCPU와 RAM {quota['available_ram_gb']} GB입니다."
            return ChatResponse(message=message, data=quota)
        if decision.action == "list_images":
            return ChatResponse(message=decision.message, data=cloud.list_images())
        if decision.action == "list_flavors":
            return ChatResponse(message=decision.message, data=cloud.list_flavors())
        if decision.action not in MUTATING_ACTIONS:
            return ChatResponse(message="이 작업은 아직 지원되지 않습니다.")

        response_message = decision.message
        try:
            if decision.action == "plan_create_instance":
                uses_default_ubuntu = (
                    decision.parameters.operating_system is not None
                    and decision.parameters.operating_system.strip().lower() == "ubuntu"
                    and decision.parameters.operating_system_version is None
                )
                payload = resolve_instance_plan(decision.parameters, cloud)
                cloud.plan_create_instance(identity.session_id, payload)
                if uses_default_ubuntu and "24.04" not in response_message:
                    response_message += " Ubuntu 버전을 지정하지 않아 Ubuntu 24.04를 기본으로 선택합니다."
                summary = (
                    f"{payload['name']} 생성: {payload['image']}, "
                    f"{payload['vcpus']} vCPU, RAM {payload['ram_gb']} GB"
                )
                operation_action = "create_instance"
            else:
                name = decision.parameters.name
                if not name:
                    return ChatResponse(message="어떤 머신을 대상으로 작업할까요?")
                if not repository.get_instance(identity.session_id, name):
                    raise ValueError(f"머신 '{name}'을(를) 찾을 수 없습니다.")
                operation_action = decision.action
                payload = {"name": name}
                verbs = {
                    "start_instance": "시작",
                    "stop_instance": "중지",
                    "reboot_instance": "재부팅",
                }
                summary = f"머신 {name} {verbs[operation_action]}"
        except ValueError as exc:
            return ChatResponse(message=str(exc))

        now = utc_now()
        operation = repository.create_operation(
            {
                "id": uuid4().hex,
                "session_id": identity.session_id,
                "user_id": identity.user_id,
                "project_id": identity.project_id,
                "action": operation_action,
                "status": "waiting_for_confirmation",
                "summary": summary,
                "payload": payload,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        return ChatResponse(
            message=f"{response_message} 계획이 준비되었습니다. 실행 전에 확인해 주세요.",
            operation=Operation.model_validate(operation),
        )

    @application.get("/api/operations/{operation_id}", response_model=Operation)
    def get_operation(operation_id: str, identity: IdentityDependency) -> Operation:
        operation = repository.get_operation(
            operation_id,
            session_id=identity.session_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        return Operation.model_validate(operation)

    @application.post("/api/operations/{operation_id}/confirm", response_model=Operation)
    def confirm_operation(operation_id: str, identity: IdentityDependency) -> Operation:
        owned_operation = repository.get_operation(
            operation_id,
            session_id=identity.session_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not owned_operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        operation = repository.claim_operation(
            operation_id,
            session_id=identity.session_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not operation:
            raise HTTPException(status_code=409, detail="Operation is no longer awaiting confirmation")
        try:
            if operation["action"] == "create_instance":
                result = cloud.create_instance(identity.session_id, operation["payload"])
            elif operation["action"] == "start_instance":
                result = cloud.start_instance(identity.session_id, operation["payload"]["name"])
            elif operation["action"] == "stop_instance":
                result = cloud.stop_instance(identity.session_id, operation["payload"]["name"])
            elif operation["action"] == "reboot_instance":
                result = cloud.reboot_instance(identity.session_id, operation["payload"]["name"])
            else:
                raise ValueError("Operation action is not allowed")
            updated = repository.update_operation(
                operation_id,
                "completed",
                session_id=identity.session_id,
                user_id=identity.user_id,
                project_id=identity.project_id,
                result=result,
            )
        except ValueError as exc:
            updated = repository.update_operation(
                operation_id,
                "failed",
                session_id=identity.session_id,
                user_id=identity.user_id,
                project_id=identity.project_id,
                error=str(exc),
            )
        return Operation.model_validate(updated)

    @application.post("/api/operations/{operation_id}/cancel", response_model=Operation)
    def cancel_operation(operation_id: str, identity: IdentityDependency) -> Operation:
        owned_operation = repository.get_operation(
            operation_id,
            session_id=identity.session_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not owned_operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        updated = repository.cancel_operation(
            operation_id,
            session_id=identity.session_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not updated:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Operation cannot be cancelled")
        return Operation.model_validate(updated)

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.is_dir():
        application.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return application


def resolve_instance_plan(parameters: ActionParameters, cloud: CloudClient) -> dict[str, Any]:
    if not parameters.operating_system or parameters.vcpus is None or parameters.ram_gb is None:
        raise ValueError("운영체제, vCPU 수와 RAM 용량을 알려 주세요.")
    if parameters.requires_gpu:
        raise ValueError("이 MVP에서는 GPU를 아직 지원하지 않습니다.")

    os_name = parameters.operating_system.strip().lower()
    requested_version = parameters.operating_system_version
    if os_name == "ubuntu" and requested_version is None:
        requested_version = "24.04"
    image = next(
        (
            item
            for item in cloud.list_images()
            if item["operating_system"].lower() == os_name
            and (requested_version is None or item.get("version") == requested_version)
        ),
        None,
    )
    if not image:
        raise ValueError("요청한 운영체제에 허용된 이미지를 찾을 수 없습니다.")
    flavor = next(
        (
            item
            for item in cloud.list_flavors()
            if item["vcpus"] == parameters.vcpus and item["ram_gb"] == parameters.ram_gb
        ),
        None,
    )
    if not flavor:
        raise ValueError("요청한 CPU와 RAM에 정확히 일치하는 허용된 flavor가 없습니다.")

    return {
        "name": parameters.name or f"{os_name}-demo",
        "image_id": image["id"],
        "image": image["name"],
        "flavor_id": flavor["id"],
        "flavor": flavor["name"],
        "vcpus": parameters.vcpus,
        "ram_gb": parameters.ram_gb,
        "requires_gpu": bool(parameters.requires_gpu),
    }


app = create_app()
