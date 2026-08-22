import type { ChatResponse, ConversationContextMessage, Operation } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
const MOCK_IDENTITY_HEADERS = {
  "X-Session-ID": "mock-session",
  "X-User-ID": "mock-user",
  "X-Project-ID": "mock-project",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...MOCK_IDENTITY_HEADERS, ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Yêu cầu thất bại (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export type HealthResponse = {
  status: "ok";
  cloud: string;
  llm_provider: string;
};

export function checkBackend(signal?: AbortSignal): Promise<HealthResponse> {
  return request("/api/health", { signal });
}

export function sendMessage(
  message: string,
  conversationContext: ConversationContextMessage[],
): Promise<ChatResponse> {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_context: conversationContext.slice(-10) }),
  });
}

export function decideOperation(id: string, decision: "confirm" | "cancel"): Promise<Operation> {
  return request(`/api/operations/${id}/${decision}`, { method: "POST" });
}
