from __future__ import annotations

import re


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
    return text.lower().strip()


def is_prohibited_request(text: str) -> bool:
    value = plain(text)
    prohibited = (
        "인스턴스 삭제",
        "머신 삭제",
        "모두 삭제",
        "delete instance",
        "delete all",
        "shell",
        "powershell",
        "cmd.exe",
        "controller node",
        "compute node",
        "공유 네트워크",
        "shared network",
        "방화벽 전체 개방",
        "open all firewall",
        "0.0.0.0/0",
    )
    return any(term in value for term in prohibited)


def contains_sensitive_value(text: str) -> bool:
    value = plain(text)
    patterns = (
        r"(?:api[_ -]?key|token|password|비밀번호|암호|private key)\s*[:=]\s*\S+",
        r"-----begin (?:rsa |ec |openssh )?private key-----",
    )
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)
