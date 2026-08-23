import type { ChatResponse, ConversationContextMessage, Operation } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
export const SESSION_STORAGE_KEY = "jcloud_agent_session_id";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

export function getOrCreateSessionId(
  storage: StorageLike = window.localStorage,
  createUuid: () => string = () => crypto.randomUUID(),
): string {
  const existing = storage.getItem(SESSION_STORAGE_KEY);
  if (existing) return existing;
  const sessionId = createUuid();
  storage.setItem(SESSION_STORAGE_KEY, sessionId);
  return sessionId;
}

function identityHeaders() {
  return {
    "X-Session-ID": getOrCreateSessionId(),
    "X-User-ID": "mock-user",
    "X-Project-ID": "mock-project",
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...identityHeaders(), ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `요청에 실패했습니다 (${response.status})`);
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

export type SandboxResetResponse = {
  status: "reset";
  instances: Array<Record<string, unknown>>;
};

export function resetSandbox(): Promise<SandboxResetResponse> {
  return request("/api/sandbox/reset", { method: "POST" });
}
