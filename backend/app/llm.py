from __future__ import annotations

import json
import os
import re
import unicodedata
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from .schemas import ActionParameters, LLMDecision


class LLMClientError(RuntimeError):
    """Raised when a provider cannot return a validated decision."""


class LLMClient(ABC):
    @abstractmethod
    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        """Convert user language to a validated decision without executing any action."""


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(
        character for character in normalized if unicodedata.category(character) != "Mn"
    )
    return without_marks.replace("đ", "d")


def _instance_name(text: str) -> str | None:
    match = re.search(r"(?:may|instance)\s+([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})", _plain(text))
    return match.group(1) if match else None


class MockLLMClient(LLMClient):
    """Deterministic local implementation used by default and in automated tests."""

    def parse_message(
        self,
        message: str,
        conversation_context: list[dict[str, str]],
        cloud_context: dict[str, Any],
    ) -> LLMDecision:
        del cloud_context
        text = _plain(message.strip())
        create_phrases = ("tao ", "tao may", "tao instance", "create instance", "create vm")
        analysis_text = text
        if not any(phrase in text for phrase in create_phrases) and (
            re.search(r"\d+\s*(?:cpu|vcpu)", text) or re.search(r"\d+\s*gb", text)
        ):
            prior_create = next(
                (
                    _plain(item["content"])
                    for item in reversed(conversation_context[-10:])
                    if item.get("role") == "user"
                    and any(phrase in _plain(item.get("content", "")) for phrase in create_phrases)
                ),
                None,
            )
            if prior_create:
                analysis_text = f"{prior_create} {text}"

        if any(term in analysis_text for term in ("xoa", "delete", "shell", "powershell", "cmd.exe", "firewall", "controller", "compute node")):
            return LLMDecision(
                decision_type="answer",
                message="Yêu cầu này chưa được hỗ trợ vì nằm ngoài phạm vi an toàn của JCloud Agent.",
            )
        if any(phrase in text for phrase in ("liet ke", "danh sach", "list instance", "list may")):
            return self._action("list_instances", "Tôi sẽ liệt kê các máy ảo hiện có.")
        if "quota" in text or ("cpu" in text and any(word in text for word in ("con", "bao nhieu", "available"))):
            return self._action("get_quota", "Tôi sẽ kiểm tra quota hiện tại.")
        if any(phrase in text for phrase in ("image nao", "liet ke image", "list image")):
            return self._action("list_images", "Tôi sẽ liệt kê các image được phép.")
        if any(phrase in text for phrase in ("flavor nao", "liet ke flavor", "list flavor")):
            return self._action("list_flavors", "Tôi sẽ liệt kê các flavor được phép.")
        if analysis_text.startswith("tao ") or any(
            phrase in analysis_text for phrase in create_phrases
        ):
            if "may manh" in analysis_text or "powerful" in analysis_text:
                return LLMDecision(
                    decision_type="clarification",
                    message="Bạn sử dụng máy cho mục đích gì và có cần GPU không?",
                )
            cpu_match = re.search(r"(\d+)\s*(?:cpu|vcpu)", analysis_text)
            ram_match = re.search(r"(?:ram\s*)?(\d+)\s*gb(?:\s*ram)?", analysis_text)
            if not cpu_match or not ram_match:
                return LLMDecision(
                    decision_type="clarification",
                    message="Vui lòng cho biết số vCPU, dung lượng RAM và có cần GPU hay không.",
                )
            requires_gpu = None
            if any(phrase in analysis_text for phrase in ("khong can gpu", "khong gpu", "no gpu")):
                requires_gpu = False
            elif "gpu" in analysis_text:
                requires_gpu = True
            else:
                requires_gpu = False
            name_match = re.search(r"(?:ten|name)\s+([a-zA-Z0-9][a-zA-Z0-9_-]{0,62})", analysis_text)
            version_match = re.search(r"\b(22\.04|24\.04)\b", analysis_text)
            version = version_match.group(1) if version_match else None
            default_message = (
                " Bạn không nêu phiên bản Ubuntu nên tôi mặc định chọn Ubuntu 24.04."
                if "ubuntu" in analysis_text and version is None
                else ""
            )
            return self._action(
                "plan_create_instance",
                f"Tôi sẽ kiểm tra cấu hình phù hợp.{default_message}",
                ActionParameters(
                    operating_system="ubuntu" if "ubuntu" in analysis_text else None,
                    operating_system_version=version,
                    vcpus=int(cpu_match.group(1)),
                    ram_gb=int(ram_match.group(1)),
                    requires_gpu=requires_gpu,
                    name=name_match.group(1) if name_match else None,
                ),
            )
        for action, phrases, reply in (
            ("reboot_instance", ("khoi dong lai", "reboot", "restart instance"), "Tôi sẽ lập kế hoạch khởi động lại máy."),
            ("start_instance", ("khoi dong", "start instance", "start may"), "Tôi sẽ lập kế hoạch khởi động máy."),
            ("stop_instance", ("tat may", "stop instance", "stop may"), "Tôi sẽ lập kế hoạch tắt máy."),
        ):
            if any(phrase in text for phrase in phrases):
                name = _instance_name(message)
                if not name:
                    return LLMDecision(decision_type="clarification", message="Bạn muốn thao tác với máy nào?")
                return self._action(action, reply, ActionParameters(name=name))
        return LLMDecision(
            decision_type="answer",
            message="Tôi có thể giúp xem máy, quota, image, flavor hoặc lập kế hoạch tạo, khởi động, tắt và reboot máy.",
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


class OpenAILLMClient(LLMClient):
    """OpenAI Responses API adapter. It receives no tools and cannot execute cloud actions."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        timeout_seconds: float = 15.0,
        max_output_tokens: int = 500,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("LLM_MODEL is required when LLM_PROVIDER=openai")
        if not api_key:
            raise ValueError("LLM_API_KEY is required when LLM_PROVIDER=openai")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
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
            response = self.client.responses.create(
                model=self.model,
                store=False,
                max_output_tokens=self.max_output_tokens,
                instructions=SYSTEM_INSTRUCTIONS,
                input=json.dumps(prompt_payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "jcloud_decision",
                        "schema": strict_llm_decision_schema(),
                        "strict": True,
                    }
                },
                timeout=self.timeout_seconds,
            )
            return LLMDecision.model_validate_json(response.output_text)
        except Exception as exc:
            raise LLMClientError("The LLM provider did not return a valid decision") from exc


SYSTEM_INSTRUCTIONS = """You only classify a user's JCloud request into the supplied JSON schema.
You have no tools and must never claim to execute an action. Allowed actions are list_instances,
get_quota, list_images, list_flavors, plan_create_instance, start_instance, stop_instance, and
reboot_instance. Refuse delete, shell commands, controller/compute changes, shared-network changes,
and opening all firewall access. Ask a clarification instead of guessing missing CPU, RAM, OS, GPU,
or target instance details. Extract Ubuntu 22.04 or 24.04 into operating_system_version. If the user
only says Ubuntu, leave operating_system_version null and explicitly say that Ubuntu 24.04 will be
selected by default. Never output credentials, tokens, passwords, or private keys. Set
requires_confirmation=false; the backend alone decides confirmation policy and performs verified work.
Reply to users in Vietnamese unless their message uses another language."""


def create_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        return OpenAILLMClient(
            model=os.getenv("LLM_MODEL", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500")),
        )
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
