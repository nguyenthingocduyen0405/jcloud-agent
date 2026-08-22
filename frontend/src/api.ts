import type { ChatResponse, Operation } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Yêu cầu thất bại (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function sendMessage(message: string): Promise<ChatResponse> {
  return request("/api/chat", { method: "POST", body: JSON.stringify({ message }) });
}

export function decideOperation(id: string, decision: "confirm" | "cancel"): Promise<Operation> {
  return request(`/api/operations/${id}/${decision}`, { method: "POST" });
}

