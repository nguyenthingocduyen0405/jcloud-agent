from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from .schemas import ActionParameters, LLMDecision


class LLMClientError(RuntimeError):
    """Raised when a provider cannot return a validated decision."""


class LLMClient(ABC):
    provider_name = "unknown"

    @abstractmethod
    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        """Convert user language to a validated decision without executing any action."""


def _plain(text: str) -> str:
    return " ".join(text.lower().strip().split())


CREATE_PHRASES = (
    "생성",
    "만들",
    "create instance",
    "create vm",
    "new instance",
    "new vm",
    "tạo máy",
    "tạo vm",
    "tạo instance",
    "khởi tạo máy",
    "máy ảo mới",
)


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def detect_language(
    text: str,
    conversation_context: list[dict[str, str]],
    fallback: str = "ko",
) -> str:
    candidates = [text, *(
        item.get("content", "")
        for item in reversed(conversation_context)
        if item.get("role") == "user"
    )]
    for candidate in candidates:
        value = _plain(candidate)
        if re.search(r"[가-힣]", value):
            return "ko"
        if re.search(r"[ăâđêôơưà-ỹ]", value) or _has_any(
            value,
            ("cho tôi", "giúp tôi", "máy ảo", "danh sách", "khởi động", "còn lại"),
        ):
            return "vi"
        if _has_any(
            value,
            ("please", "show", "list", "create", "start", "stop", "reboot", "how much"),
        ):
            return "en"
    return fallback


MESSAGES = {
    "unsupported": {
        "ko": "이 요청은 JCloud Agent의 안전 범위를 벗어나므로 지원되지 않습니다.",
        "vi": "Yêu cầu này nằm ngoài phạm vi an toàn của JCloud Agent nên không được hỗ trợ.",
        "en": "This request is outside JCloud Agent's safe scope and is not supported.",
    },
    "list_instances": {
        "ko": "현재 가상 머신 목록입니다.",
        "vi": "Đây là danh sách máy ảo hiện tại.",
        "en": "Here is the current virtual machine list.",
    },
    "list_images": {
        "ko": "사용 가능한 이미지 목록입니다.",
        "vi": "Đây là danh sách image hiện có.",
        "en": "Here is the list of available images.",
    },
    "list_flavors": {
        "ko": "사용 가능한 flavor 목록입니다.",
        "vi": "Đây là danh sách flavor hiện có.",
        "en": "Here is the list of available flavors.",
    },
    "target": {
        "ko": "어떤 머신을 대상으로 작업할까요?",
        "vi": "Bạn muốn thao tác với máy nào?",
        "en": "Which machine should I operate on?",
    },
    "gpu_unsupported": {
        "ko": "이 MVP에서는 GPU 머신을 아직 지원하지 않습니다.",
        "vi": "Bản MVP này chưa hỗ trợ máy sử dụng GPU.",
        "en": "This MVP does not support GPU machines yet.",
    },
    "capabilities": {
        "ko": "가상 머신, 할당량, 이미지, flavor를 조회하거나 머신 생성, 시작, 중지, 재부팅 계획을 도와드릴 수 있습니다.",
        "vi": "Tôi có thể xem máy ảo, quota, image và flavor, hoặc lập kế hoạch tạo, khởi động, dừng và khởi động lại máy.",
        "en": "I can inspect virtual machines, quota, images, and flavors, or plan machine creation, start, stop, and reboot operations.",
    },
}


def _message(key: str, language: str) -> str:
    return MESSAGES[key][language]


def _extract_vcpus(text: str) -> int | None:
    for pattern in (
        r"(\d+)\s*(?:v?cpu)(?:\s*개)?",
        r"(?:v?cpu)\s*[:=]?\s*(\d+)(?:\s*개)?",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_ram_gb(text: str) -> int | None:
    for pattern in (
        r"(?:ram)\s*[:=]?\s*(\d+)\s*gb",
        r"(\d+)\s*gb(?:\s*(?:of\s*)?ram)?",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _missing_create_fields(assistant_text: str) -> frozenset[str]:
    fields = set()
    if "vcpu" in assistant_text:
        fields.add("vcpus")
    if "ram" in assistant_text:
        fields.add("ram_gb")
    if _has_any(assistant_text, ("운영체제", "hệ điều hành", "operating system")):
        fields.add("operating_system")
    return frozenset(fields)


def _pending_create_context(
    conversation_context: list[dict[str, str]],
) -> tuple[str, frozenset[str]] | None:
    context = conversation_context[-10:]
    if len(context) < 2:
        return None
    prior_user, prior_assistant = context[-2], context[-1]
    assistant_text = _plain(prior_assistant.get("content", ""))
    user_text = _plain(prior_user.get("content", ""))
    missing_fields = _missing_create_fields(assistant_text)
    if (
        prior_user.get("role") == "user"
        and prior_assistant.get("role") == "assistant"
        and missing_fields
        and _has_any(user_text, CREATE_PHRASES)
    ):
        return user_text, missing_fields
    return None


def _normalize_create_followup(text: str, missing_fields: frozenset[str]) -> str | None:
    if (
        _extract_vcpus(text) is not None
        or _extract_ram_gb(text) is not None
        or ("operating_system" in missing_fields and "ubuntu" in text)
    ):
        return text

    number_match = re.fullmatch(r"(\d+)(?:\s*개)?", text)
    if not number_match or len(missing_fields) != 1:
        return None

    value = number_match.group(1)
    missing_field = next(iter(missing_fields))
    if missing_field == "vcpus":
        return f"{value} vcpu"
    if missing_field == "ram_gb":
        return f"ram {value} gb"
    return None


def _required_create_fields(parameters: dict[str, Any]) -> frozenset[str]:
    missing = set()
    if not parameters.get("operating_system"):
        missing.add("operating_system")
    if parameters.get("vcpus") is None:
        missing.add("vcpus")
    if parameters.get("ram_gb") is None:
        missing.add("ram_gb")
    return frozenset(missing)


def _pending_create_prompt(parameters: dict[str, Any], followup: str) -> str:
    parts = ["create instance"]
    operating_system = parameters.get("operating_system")
    if operating_system:
        parts.append(str(operating_system))
    version = parameters.get("operating_system_version")
    if version:
        parts.append(str(version))
    if parameters.get("vcpus") is not None:
        parts.append(f"{parameters['vcpus']} vcpu")
    if parameters.get("ram_gb") is not None:
        parts.append(f"ram {parameters['ram_gb']} gb")
    if parameters.get("requires_gpu") is False:
        parts.append("no gpu")
    name = parameters.get("name")
    if name:
        parts.append(f"name {name}")
    parts.append(followup)
    return " ".join(parts)


def _instance_name(text: str) -> str | None:
    value = _plain(text)
    for pattern in (
        r"(?:이름(?:은|이|을)?|name|tên)\s*[:=：]?\s*([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})",
        r"(?:인스턴스|가상\s*머신|머신|서버|instance|vm|máy(?:\s*ảo)?)\s+(?:이름(?:은|이|을)?\s*)?([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})",
        r"([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})\s*(?:인스턴스|가상\s*머신|머신|서버|instance|vm|máy(?:\s*ảo)?)",
    ):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


class MockLLMClient(LLMClient):
    """Deterministic local implementation used by default and in automated tests."""

    provider_name = "mock"

    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        text = _plain(message.strip())
        stored_pending = cloud_context.get("pending_request")
        fallback_language = (
            stored_pending.get("language", "ko") if isinstance(stored_pending, dict) else "ko"
        )
        language = detect_language(message, conversation_context, fallback_language)
        analysis_text = text
        pending_create = None
        if not _has_any(text, CREATE_PHRASES):
            if stored_pending and stored_pending.get("action") == "plan_create_instance":
                stored_parameters = stored_pending.get("parameters", {})
                missing_fields = _required_create_fields(stored_parameters)
                followup = _normalize_create_followup(text, missing_fields)
                if followup:
                    analysis_text = _pending_create_prompt(stored_parameters, followup)
                elif re.fullmatch(r"\d+(?:\s*개)?", text) and len(missing_fields) > 1:
                    ambiguity = {
                        "ko": "이 숫자가 vCPU인지 RAM인지 함께 알려 주세요.",
                        "vi": "Hãy cho biết con số này là vCPU hay RAM.",
                        "en": "Please specify whether this number is vCPU or RAM.",
                    }
                    return LLMDecision(
                        decision_type="clarification",
                        pending_action="plan_create_instance",
                        parameters=ActionParameters.model_validate(stored_parameters),
                        message=ambiguity[language],
                    )
            else:
                pending_create = _pending_create_context(conversation_context)
            if analysis_text == text and not stored_pending and pending_create:
                prior_create, missing_fields = pending_create
                followup = _normalize_create_followup(text, missing_fields)
                if followup:
                    analysis_text = f"{prior_create} {followup}"

        if any(term in analysis_text for term in ("삭제", "delete", "shell", "powershell", "cmd.exe", "firewall", "방화벽", "controller", "compute node")):
            return LLMDecision(
                decision_type="answer",
                message=_message("unsupported", language),
            )
        if _has_any(text, ("이미지 목록", "이미지 보여", "list image", "show image", "danh sách image", "các image")):
            return self._action("list_images", _message("list_images", language))
        if _has_any(text, ("flavor 목록", "플레이버 목록", "list flavor", "show flavor", "danh sách flavor", "các flavor")):
            return self._action("list_flavors", _message("list_flavors", language))
        if "quota" in text or "할당량" in text or (
            "cpu" in text and _has_any(text, ("남", "얼마", "available", "còn", "khả dụng", "bao nhiêu"))
        ):
            quota = cloud_context.get("quota", {})
            available_vcpus = quota.get("available_vcpus", "?")
            available_ram = quota.get("available_ram_gb", "?")
            quota_messages = {
                "ko": f"사용 가능한 자원은 {available_vcpus} vCPU와 RAM {available_ram} GB입니다.",
                "vi": f"Tài nguyên khả dụng là {available_vcpus} vCPU và {available_ram} GB RAM.",
                "en": f"The available capacity is {available_vcpus} vCPU and {available_ram} GB of RAM.",
            }
            return self._action("get_quota", quota_messages[language])
        if _has_any(
            text,
            ("목록", "리스트", "보여", "조회", "list instance", "list vm", "show instance", "danh sách máy", "liệt kê máy", "xem máy ảo"),
        ):
            return self._action("list_instances", _message("list_instances", language))
        if _has_any(analysis_text, CREATE_PHRASES):
            if _has_any(analysis_text, ("강력한", "고성능", "powerful", "mạnh", "hiệu năng cao")):
                clarification = {
                    "ko": "용도와 필요한 vCPU 수, RAM 용량을 알려 주세요.",
                    "vi": "Hãy cho tôi biết mục đích sử dụng, số vCPU và dung lượng RAM cần thiết.",
                    "en": "Please provide the workload, required vCPU count, and RAM capacity.",
                }
                return LLMDecision(
                    decision_type="clarification",
                    pending_action="plan_create_instance",
                    message=clarification[language],
                )
            vcpus = _extract_vcpus(analysis_text)
            ram_gb = _extract_ram_gb(analysis_text)
            operating_system = "ubuntu" if "ubuntu" in analysis_text else None
            name_match = re.search(
                r"(?:이름(?:은|을)?|name|tên(?:\s+là)?)\s*[:=：]?\s*([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})",
                analysis_text,
            )
            version_match = re.search(r"\b(22\.04|24\.04)\b", analysis_text)
            version = version_match.group(1) if version_match else None
            if _has_any(analysis_text, ("gpu 필요 없", "gpu는 필요 없", "gpu 없이", "gpu 불필요", "no gpu", "không cần gpu", "không gpu")):
                requires_gpu = False
            elif "gpu" in analysis_text:
                requires_gpu = True
            else:
                requires_gpu = False
            parameters = ActionParameters(
                operating_system=operating_system,
                operating_system_version=version,
                vcpus=vcpus,
                ram_gb=ram_gb,
                requires_gpu=requires_gpu,
                name=name_match.group(1) if name_match else None,
            )
            if requires_gpu:
                return LLMDecision(
                    decision_type="answer",
                    message=_message("gpu_unsupported", language),
                )
            missing = []
            if not operating_system:
                missing.append({"ko": "운영체제", "vi": "hệ điều hành", "en": "operating system"}[language])
            if vcpus is None:
                missing.append("vCPU")
            if ram_gb is None:
                missing.append("RAM")
            if missing:
                joined = ", ".join(missing)
                clarification = {
                    "ko": f"다음 정보를 알려 주세요: {joined}.",
                    "vi": f"Vui lòng cho biết thêm: {joined}.",
                    "en": f"Please provide the following: {joined}.",
                }
                return LLMDecision(
                    decision_type="clarification",
                    pending_action="plan_create_instance",
                    parameters=parameters,
                    message=clarification[language],
                )
            plan_messages = {
                "ko": "적합한 구성을 확인하겠습니다.",
                "vi": "Tôi sẽ kiểm tra cấu hình phù hợp.",
                "en": "I will validate the requested configuration.",
            }
            default_messages = {
                "ko": " Ubuntu 버전을 지정하지 않아 Ubuntu 24.04를 기본으로 선택합니다.",
                "vi": " Bạn chưa chỉ định phiên bản Ubuntu nên Ubuntu 24.04 sẽ được chọn mặc định.",
                "en": " No Ubuntu version was specified, so Ubuntu 24.04 will be selected by default.",
            }
            default_message = default_messages[language] if version is None else ""
            return self._action(
                "plan_create_instance",
                f"{plan_messages[language]}{default_message}",
                parameters,
            )
        for action, phrases, reply in (
            ("reboot_instance", ("재부팅", "다시 시작", "reboot", "restart instance", "khởi động lại"), {
                "ko": "머신 재부팅 계획을 준비하겠습니다.", "vi": "Tôi sẽ chuẩn bị kế hoạch khởi động lại máy.", "en": "I will prepare a machine reboot plan.",
            }),
            ("start_instance", ("시작", "켜 줘", "부팅", "start instance", "start vm", "khởi động", "bật máy"), {
                "ko": "머신 시작 계획을 준비하겠습니다.", "vi": "Tôi sẽ chuẩn bị kế hoạch khởi động máy.", "en": "I will prepare a machine start plan.",
            }),
            ("stop_instance", ("중지", "정지", "꺼 줘", "stop instance", "stop vm", "dừng máy", "tắt máy"), {
                "ko": "머신 중지 계획을 준비하겠습니다.", "vi": "Tôi sẽ chuẩn bị kế hoạch dừng máy.", "en": "I will prepare a machine stop plan.",
            }),
        ):
            if _has_any(text, phrases):
                name = _instance_name(message)
                if not name:
                    return LLMDecision(decision_type="clarification", message=_message("target", language))
                return self._action(action, reply[language], ActionParameters(name=name))
        return LLMDecision(
            decision_type="answer",
            message=_message("capabilities", language),
        )

    @staticmethod
    def _action(action: str, message: str, parameters: ActionParameters | None = None) -> LLMDecision:
        return LLMDecision(
            decision_type="action",
            action=action,
            parameters=parameters or ActionParameters(),
            message=message,
            requires_confirmation=False,
        )


class FastPathLLMClient(LLMClient):
    """Resolve high-confidence VM intents locally before using a remote provider."""

    def __init__(self, fallback: LLMClient) -> None:
        self.fallback = fallback
        self.local = MockLLMClient()
        self.provider_name = f"{fallback.provider_name}+fast-path"

    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        local_decision = self.local.parse_message(
            message,
            conversation_context,
            cloud_context,
        )
        stored_pending = cloud_context.get("pending_request")
        fallback_language = (
            stored_pending.get("language", "ko") if isinstance(stored_pending, dict) else "ko"
        )
        language = detect_language(message, conversation_context, fallback_language)
        is_generic_fallback = (
            local_decision.decision_type == "answer"
            and local_decision.message == _message("capabilities", language)
        )
        if not is_generic_fallback:
            return local_decision
        return self.fallback.parse_message(message, conversation_context, cloud_context)


class OpenAILLMClient(LLMClient):
    """OpenAI Responses API adapter. It receives no tools and cannot execute cloud actions."""

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        max_output_tokens: int = 500,
        reasoning_effort: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM_MODEL is required when LLM_PROVIDER=openai")
        if not api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=timeout_seconds)
        self.client = client

    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        prompt_payload = {
            "conversation_context": conversation_context[-10:],
            "cloud_context": cloud_context,
            "message": message,
        }
        try:
            request_options: dict[str, Any] = {
                "model": self.model,
                "store": False,
                "max_output_tokens": self.max_output_tokens,
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": json.dumps(prompt_payload, ensure_ascii=False),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "jcloud_decision",
                        "schema": strict_llm_decision_schema(),
                        "strict": True,
                    }
                },
                "timeout": self.timeout_seconds,
            }
            if self.reasoning_effort:
                request_options["reasoning"] = {"effort": self.reasoning_effort}
            response = self.client.responses.create(
                **request_options,
            )
            return LLMDecision.model_validate_json(response.output_text)
        except Exception as exc:
            raise LLMClientError("The LLM provider did not return a valid decision") from exc


SYSTEM_INSTRUCTIONS = """Act as the natural-language planner for a safe JCloud assistant.
Return only a decision that matches the supplied JSON schema. You have no tools and must never claim
that an action has already run. Allowed actions are list_instances, get_quota, list_images,
list_flavors, plan_create_instance, start_instance, stop_instance, and reboot_instance.

Use cloud_context as the source of truth for quota, images, flavors, and instance names. For read-only
requests, provide a concise, useful message using those values. For a create request, operating system,
vCPU count, and RAM are required; GPU is optional and defaults to false. Ask only for required fields
that are actually missing. Understand equivalent parameter orders such as "4 CPU", "CPU 4", and
"CPU 4개". Extract Ubuntu 22.04 or 24.04 into operating_system_version. If the user only says Ubuntu,
leave operating_system_version null and explicitly say Ubuntu 24.04 will be selected by default.

Use conversation_context only when the immediately preceding assistant message asked for missing
details. If cloud_context.pending_request exists, treat its parameters as collected state, merge new
details into it, and set pending_action=plan_create_instance on clarifications. Do not revive an older
request after the user has changed topics. Reply in the language of the current user message, including
Vietnamese, Korean, or English. Refuse deletion, shell commands,
controller or compute changes, shared-network changes, and opening all firewall access. Never output
credentials, tokens, passwords, or private keys. Set requires_confirmation=false; only the backend
decides confirmation policy, validates metadata, and performs verified work."""


def create_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-5-nano").strip() or "gpt-5-nano"
    if provider == "auto":
        provider = "openai" if api_key else "mock"
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        openai_client = OpenAILLMClient(
            model=model,
            api_key=api_key,
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500")),
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "minimal").strip() or None,
        )
        fast_path_enabled = os.getenv("LLM_FAST_PATH", "true").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        return FastPathLLMClient(openai_client) if fast_path_enabled else openai_client
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def strict_llm_decision_schema() -> dict[str, Any]:
    """Return the Pydantic schema normalized for OpenAI strict Structured Outputs."""
    schema = deepcopy(LLMDecision.model_json_schema())

    def normalize(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                normalize(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("default") is None:
            node.pop("default", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
        for value in node.values():
            normalize(value)

    normalize(schema)
    return schema
