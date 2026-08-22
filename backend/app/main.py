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
)


def get_request_identity(
    session_id: Annotated[str, Header(alias="X-Session-ID")] = "mock-session",
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
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "cloud": "mock", "llm_provider": llm_provider}

    @application.get("/api/instances")
    def list_instances() -> list[dict]:
        return cloud.list_instances()

    @application.get("/api/quota")
    def get_quota() -> dict[str, int]:
        return cloud.get_quota()

    @application.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, identity: IdentityDependency) -> ChatResponse:
        all_user_text = "\n".join(
            [request.message, *(item.content for item in request.conversation_context)]
        )
        if contains_sensitive_value(all_user_text):
            return ChatResponse(
                message="Không gửi API key, token, mật khẩu hoặc private key vào cuộc trò chuyện. Yêu cầu chưa được chuyển tới LLM."
            )
        if is_prohibited_request(request.message):
            return ChatResponse(
                message="Yêu cầu này chưa được hỗ trợ vì nằm ngoài phạm vi an toàn của JCloud Agent."
            )

        cloud_context = {
            "quota": cloud.get_quota(),
            "images": [
                {"name": image["name"], "operating_system": image["operating_system"]}
                for image in cloud.list_images()
            ],
            "flavors": [
                {"name": flavor["name"], "vcpus": flavor["vcpus"], "ram_gb": flavor["ram_gb"]}
                for flavor in cloud.list_flavors()
            ],
            "instance_names": [instance["name"] for instance in cloud.list_instances()],
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
                message="Không thể hiểu yêu cầu một cách an toàn lúc này. Không có thao tác nào được thực hiện."
            )

        if decision.decision_type != "action":
            return ChatResponse(message=decision.message)
        if decision.action not in ALLOWED_ACTIONS:
            return ChatResponse(message="Thao tác này chưa được hỗ trợ.")

        if decision.action == "list_instances":
            instances = cloud.list_instances()
            return ChatResponse(message=decision.message, data=instances)
        if decision.action == "get_quota":
            quota = cloud.get_quota()
            message = f"Còn {quota['available_vcpus']} vCPU và {quota['available_ram_gb']} GB RAM."
            return ChatResponse(message=message, data=quota)
        if decision.action == "list_images":
            return ChatResponse(message=decision.message, data=cloud.list_images())
        if decision.action == "list_flavors":
            return ChatResponse(message=decision.message, data=cloud.list_flavors())
        if decision.action not in MUTATING_ACTIONS:
            return ChatResponse(message="Thao tác này chưa được hỗ trợ.")

        try:
            if decision.action == "plan_create_instance":
                payload = resolve_instance_plan(decision.parameters, cloud)
                cloud.plan_create_instance(payload)
                summary = (
                    f"Tạo {payload['name']} với {payload['image']}, "
                    f"{payload['vcpus']} vCPU và {payload['ram_gb']} GB RAM"
                )
                operation_action = "create_instance"
            else:
                name = decision.parameters.name
                if not name:
                    return ChatResponse(message="Bạn muốn thao tác với máy nào?")
                if not repository.get_instance(name):
                    raise ValueError(f"Không tìm thấy máy '{name}'")
                operation_action = decision.action
                payload = {"name": name}
                verbs = {
                    "start_instance": "Khởi động",
                    "stop_instance": "Tắt",
                    "reboot_instance": "Khởi động lại",
                }
                summary = f"{verbs[operation_action]} máy {name}"
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
            message=f"{decision.message} Kế hoạch đã sẵn sàng; vui lòng xác nhận trước khi thực hiện.",
            operation=Operation.model_validate(operation),
        )

    @application.get("/api/operations/{operation_id}", response_model=Operation)
    def get_operation(operation_id: str, identity: IdentityDependency) -> Operation:
        operation = repository.get_operation(
            operation_id,
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
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not owned_operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        operation = repository.claim_operation(
            operation_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not operation:
            raise HTTPException(status_code=409, detail="Operation is no longer awaiting confirmation")
        try:
            if operation["action"] == "create_instance":
                result = cloud.create_instance(operation["payload"])
            elif operation["action"] == "start_instance":
                result = cloud.start_instance(operation["payload"]["name"])
            elif operation["action"] == "stop_instance":
                result = cloud.stop_instance(operation["payload"]["name"])
            elif operation["action"] == "reboot_instance":
                result = cloud.reboot_instance(operation["payload"]["name"])
            else:
                raise ValueError("Operation action is not allowed")
            updated = repository.update_operation(operation_id, "completed", result=result)
        except ValueError as exc:
            updated = repository.update_operation(operation_id, "failed", error=str(exc))
        return Operation.model_validate(updated)

    @application.post("/api/operations/{operation_id}/cancel", response_model=Operation)
    def cancel_operation(operation_id: str, identity: IdentityDependency) -> Operation:
        owned_operation = repository.get_operation(
            operation_id,
            user_id=identity.user_id,
            project_id=identity.project_id,
        )
        if not owned_operation:
            raise HTTPException(status_code=404, detail="Operation not found")
        updated = repository.cancel_operation(
            operation_id,
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
        raise ValueError("Vui lòng cho biết hệ điều hành, số vCPU và dung lượng RAM.")
    if parameters.requires_gpu:
        raise ValueError("GPU chưa được hỗ trợ trong MVP này.")

    os_name = parameters.operating_system.strip().lower()
    image = next(
        (item for item in cloud.list_images() if item["operating_system"].lower() == os_name),
        None,
    )
    if not image:
        raise ValueError("Không tìm thấy image được phép cho hệ điều hành đã yêu cầu.")
    flavor = next(
        (
            item
            for item in cloud.list_flavors()
            if item["vcpus"] == parameters.vcpus and item["ram_gb"] == parameters.ram_gb
        ),
        None,
    )
    if not flavor:
        raise ValueError("Không có flavor được phép khớp chính xác với CPU và RAM đã yêu cầu.")

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
