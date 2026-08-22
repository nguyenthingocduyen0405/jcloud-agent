from __future__ import annotations

import re
import unicodedata


ALLOWED_ACTIONS = frozenset({
    "list_instances",
    "get_quota",
    "list_images",
    "list_flavors",
    "plan_create_instance",
    "start_instance",
    "stop_instance",
    "reboot_instance",
})

MUTATING_ACTIONS = frozenset({
    "plan_create_instance",
    "start_instance",
    "stop_instance",
    "reboot_instance",
})


def plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    value = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
    return value.replace("đ", "d")


def is_prohibited_request(text: str) -> bool:
    value = plain(text)
    prohibited = (
        "xoa instance",
        "xoa may",
        "xoa tat ca",
        "delete instance",
        "delete all",
        "shell",
        "powershell",
        "cmd.exe",
        "controller node",
        "compute node",
        "network dung chung",
        "shared network",
        "mo toan bo firewall",
        "open all firewall",
        "0.0.0.0/0",
    )
    return any(term in value for term in prohibited)


def contains_sensitive_value(text: str) -> bool:
    value = plain(text)
    patterns = (
        r"(?:api[_ -]?key|token|password|mat khau|private key)\s*[:=]\s*\S+",
        r"-----begin (?:rsa |ec |openssh )?private key-----",
    )
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)

