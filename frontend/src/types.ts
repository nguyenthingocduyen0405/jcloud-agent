export type OperationStatus =
  | "waiting_for_confirmation"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface Operation {
  id: string;
  action: "create_instance" | "start_instance" | "stop_instance" | "reboot_instance";
  status: OperationStatus;
  summary: string;
  payload: Record<string, unknown>;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ChatResponse {
  message: string;
  operation?: Operation | null;
  data?: unknown;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  operation?: Operation | null;
  data?: unknown;
}
