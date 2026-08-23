from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .database import Repository


class CloudClient(ABC):
    @abstractmethod
    def list_instances(self, session_id: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_quota(self, session_id: str) -> dict[str, int]: ...

    @abstractmethod
    def list_images(self) -> list[dict[str, str]]: ...

    @abstractmethod
    def list_flavors(self) -> list[dict[str, int | str]]: ...

    @abstractmethod
    def plan_create_instance(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def create_instance(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def start_instance(self, session_id: str, name: str) -> dict[str, Any]: ...

    @abstractmethod
    def stop_instance(self, session_id: str, name: str) -> dict[str, Any]: ...

    @abstractmethod
    def reboot_instance(self, session_id: str, name: str) -> dict[str, Any]: ...

    @abstractmethod
    def reset_sandbox(self, session_id: str) -> list[dict[str, Any]]: ...


class MockCloudClient(CloudClient):
    TOTAL_VCPUS = 16
    TOTAL_RAM_GB = 64

    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def list_instances(self, session_id: str) -> list[dict[str, Any]]:
        return self.repository.list_instances(session_id)

    def get_quota(self, session_id: str) -> dict[str, int]:
        instances = self.list_instances(session_id)
        used_vcpus = sum(int(item["vcpus"]) for item in instances)
        used_ram_gb = sum(int(item["ram_gb"]) for item in instances)
        return {
            "total_vcpus": self.TOTAL_VCPUS,
            "used_vcpus": used_vcpus,
            "available_vcpus": self.TOTAL_VCPUS - used_vcpus,
            "total_ram_gb": self.TOTAL_RAM_GB,
            "used_ram_gb": used_ram_gb,
            "available_ram_gb": self.TOTAL_RAM_GB - used_ram_gb,
        }

    def list_images(self) -> list[dict[str, str]]:
        return [
            {"id": "img-ubuntu-2204", "name": "Ubuntu 22.04", "operating_system": "ubuntu", "version": "22.04"},
            {"id": "img-ubuntu-2404", "name": "Ubuntu 24.04", "operating_system": "ubuntu", "version": "24.04"},
        ]

    def list_flavors(self) -> list[dict[str, int | str]]:
        return [
            {"id": "flavor-small", "name": "small", "vcpus": 1, "ram_gb": 2},
            {"id": "flavor-medium", "name": "medium", "vcpus": 2, "ram_gb": 4},
            {"id": "flavor-large", "name": "large", "vcpus": 4, "ram_gb": 16},
        ]

    def plan_create_instance(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        image = next((item for item in self.list_images() if item["id"] == payload["image_id"]), None)
        if not image or image["name"] != payload["image"]:
            raise ValueError("Image is not allowed")
        flavor = next((item for item in self.list_flavors() if item["id"] == payload["flavor_id"]), None)
        if not flavor or flavor["vcpus"] != payload["vcpus"] or flavor["ram_gb"] != payload["ram_gb"]:
            raise ValueError("Flavor is not allowed")
        if payload.get("requires_gpu"):
            raise ValueError("GPU instances are not supported in this MVP")
        quota = self.get_quota(session_id)
        if payload["vcpus"] > quota["available_vcpus"]:
            raise ValueError("Not enough available CPU quota")
        if payload["ram_gb"] > quota["available_ram_gb"]:
            raise ValueError("Not enough available RAM quota")
        if self.repository.get_instance(session_id, payload["name"]):
            raise ValueError("An instance with this name already exists")
        return payload

    def create_instance(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        plan = self.plan_create_instance(session_id, payload)
        return self.repository.create_instance(session_id, {
            "id": f"vm-{uuid4().hex[:10]}",
            "name": plan["name"],
            "image": plan["image"],
            "vcpus": plan["vcpus"],
            "ram_gb": plan["ram_gb"],
            "status": "ACTIVE",
            "created_at": datetime.now(UTC).isoformat(),
        })

    def start_instance(self, session_id: str, name: str) -> dict[str, Any]:
        instance = self.repository.set_instance_status(session_id, name, "ACTIVE")
        if not instance:
            raise ValueError(f"Instance '{name}' was not found")
        return instance

    def stop_instance(self, session_id: str, name: str) -> dict[str, Any]:
        instance = self.repository.set_instance_status(session_id, name, "SHUTOFF")
        if not instance:
            raise ValueError(f"Instance '{name}' was not found")
        return instance

    def reboot_instance(self, session_id: str, name: str) -> dict[str, Any]:
        instance = self.repository.set_instance_status(session_id, name, "ACTIVE")
        if not instance:
            raise ValueError(f"Instance '{name}' was not found")
        return instance

    def reset_sandbox(self, session_id: str) -> list[dict[str, Any]]:
        return self.repository.reset_session(session_id)
