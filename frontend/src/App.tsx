import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { checkBackend, decideOperation, resetSandbox, sendMessage } from "./api";
import type { ChatMessage, ConversationContextMessage, Operation } from "./types";
import "./styles.css";

const suggestions = [
  "Liệt kê máy của tôi",
  "Tôi còn bao nhiêu CPU và RAM?",
  "Tạo máy Ubuntu 4 CPU, RAM 16 GB",
  "Khởi động máy test-01",
];

export const initialMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Xin chào. Tôi là JCloud Agent chạy với dữ liệu giả lập. Tôi có thể liệt kê máy, kiểm tra quota và lập kế hoạch tạo, khởi động hoặc tắt máy.",
};

const sensitiveValuePattern = /(?:api[_ -]?key|token|password|mật khẩu|private key)\s*[:=]\s*\S+/i;
type ConnectionStatus = "connecting" | "ready" | "failed";

export function buildConversationContext(messages: ChatMessage[]): ConversationContextMessage[] {
  return messages
    .filter((message) => message.text.trim() && !sensitiveValuePattern.test(message.text))
    .map((message) => ({ role: message.role, content: message.text.slice(0, 500) }))
    .slice(-10);
}

function Status({ value }: { value: Operation["status"] }) {
  const labels: Record<Operation["status"], string> = {
    waiting_for_confirmation: "Chờ xác nhận",
    running: "Đang chạy",
    completed: "Hoàn thành",
    failed: "Thất bại",
    cancelled: "Đã hủy",
  };
  return <span className={`status status--${value}`}>{labels[value]}</span>;
}

function DataPreview({ data }: { data: unknown }) {
  if (!data) return null;
  if (Array.isArray(data)) {
    return (
      <div className="instance-grid">
        {data.map((item) => {
          const vm = item as Record<string, unknown>;
          return (
            <div className="instance" key={String(vm.id)}>
              <strong>{String(vm.name)}</strong>
              <span>{String(vm.image)}</span>
              <span>{String(vm.vcpus)} vCPU · {String(vm.ram_gb)} GB</span>
              <span className="instance__state">{String(vm.status)}</span>
            </div>
          );
        })}
      </div>
    );
  }
  const quota = data as Record<string, unknown>;
  if ("available_vcpus" in quota) {
    return (
      <div className="quota">
        <div><strong>{String(quota.available_vcpus)}</strong><span>vCPU còn lại</span></div>
        <div><strong>{String(quota.available_ram_gb)} GB</strong><span>RAM còn lại</span></div>
      </div>
    );
  }
  return null;
}

export function OperationDetails({ payload }: { payload: Operation["payload"] }) {
  const details = [
    ["Tên máy", payload.name],
    ["Hệ điều hành", payload.image ?? payload.operating_system],
    ["CPU", payload.vcpus === undefined ? undefined : `${String(payload.vcpus)} vCPU`],
    ["RAM", payload.ram_gb === undefined ? undefined : `${String(payload.ram_gb)} GB`],
    ["GPU", payload.requires_gpu === undefined ? undefined : payload.requires_gpu ? "Có" : "Không"],
    ["Flavor", payload.flavor],
  ].filter((detail): detail is [string, unknown] => detail[1] !== undefined && detail[1] !== null);

  return (
    <dl>
      {details.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{String(value)}</dd></div>)}
    </dl>
  );
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("connecting");
  const [sandboxNotice, setSandboxNotice] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const connectionAttemptRef = useRef(0);

  const connect = useCallback(async () => {
    const attempt = ++connectionAttemptRef.current;
    setConnectionStatus("connecting");
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 60_000);

    try {
      const health = await checkBackend(controller.signal);
      if (health.status !== "ok") throw new Error("Backend chưa sẵn sàng");
      if (attempt === connectionAttemptRef.current) setConnectionStatus("ready");
    } catch {
      if (attempt === connectionAttemptRef.current) setConnectionStatus("failed");
    } finally {
      window.clearTimeout(timeout);
    }
  }, []);

  useEffect(() => {
    void connect();
  }, [connect]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy || connectionStatus !== "ready") return;
    setInput("");
    setBusy(true);
    const conversationContext = buildConversationContext(messages);
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", text: trimmed }]);
    try {
      const response = await sendMessage(trimmed, conversationContext);
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", text: response.message, operation: response.operation, data: response.data },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", text: error instanceof Error ? error.message : "Không thể kết nối backend." },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function handleDecision(messageId: string, operation: Operation, decision: "confirm" | "cancel") {
    setBusy(true);
    setMessages((current) => current.map((message) =>
      message.id === messageId ? { ...message, operation: { ...operation, status: "running" } } : message
    ));
    try {
      const updated = await decideOperation(operation.id, decision);
      setMessages((current) => current.map((message) =>
        message.id === messageId ? { ...message, operation: updated } : message
      ));
      setMessages((current) => [...current, {
        id: crypto.randomUUID(),
        role: "assistant",
        text: updated.status === "completed" ? "Thao tác giả lập đã hoàn thành." : "Kế hoạch đã được hủy.",
        data: updated.result,
      }]);
    } catch (error) {
      setMessages((current) => current.map((message) =>
        message.id === messageId ? { ...message, operation: { ...operation, status: "failed", error: String(error) } } : message
      ));
    } finally {
      setBusy(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void submit(input);
  }

  function startNewConversation() {
    setMessages([initialMessage]);
    setInput("");
    setSandboxNotice("Đã bắt đầu cuộc trò chuyện mới. Dữ liệu sandbox không thay đổi.");
  }

  async function handleResetSandbox() {
    const confirmed = window.confirm(
      "Reset sandbox chỉ xóa máy ảo và kế hoạch mô phỏng của phiên trình duyệt hiện tại. Bạn có muốn tiếp tục?",
    );
    if (!confirmed) return;

    setBusy(true);
    setSandboxNotice(null);
    try {
      await resetSandbox();
      setMessages([initialMessage]);
      setInput("");
      setSandboxNotice("Sandbox đã được reset về hai máy mặc định: web-demo và test-01.");
    } catch (error) {
      setSandboxNotice(error instanceof Error ? error.message : "Không thể reset sandbox.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand__mark">JC</span><div><strong>JCloud Agent</strong><small>Mock environment</small></div></div>
        <div className={`environment environment--${connectionStatus}`} aria-live="polite">
          <span className="pulse" />
          {connectionStatus === "connecting" && "Đang kết nối..."}
          {connectionStatus === "ready" && "Sandbox sẵn sàng"}
          {connectionStatus === "failed" && "Mất kết nối"}
        </div>
        <div className="sidebar__copy">
          <h2>An toàn từ thiết kế</h2>
          <p>AI chỉ hiểu yêu cầu và lập kế hoạch. Backend kiểm tra rồi chỉ thực thi sau khi bạn xác nhận.</p>
        </div>
        <div className="guardrails">
          <span>Không có credential</span><span>Không gọi OpenStack thật</span><span>Không chạy shell</span>
        </div>
      </aside>

      <section className="chat">
        <header>
          <div><span className="eyebrow">CONTROL PLANE ASSISTANT</span><h1>Quản lý cloud bằng hội thoại</h1></div>
          <div className="header-actions">
            <button type="button" onClick={startNewConversation} disabled={busy}>Cuộc trò chuyện mới</button>
            <button type="button" className="reset-button" onClick={() => void handleResetSandbox()} disabled={busy || connectionStatus !== "ready"}>Reset sandbox</button>
            <span className="mode">MOCK MODE</span>
          </div>
        </header>
        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`message message--${message.role}`}>
              <div className="message__label">{message.role === "user" ? "Bạn" : "Agent"}</div>
              <div className="bubble">
                <p>{message.text}</p>
                <DataPreview data={message.data} />
                {message.operation && (
                  <div className="operation">
                    <div className="operation__top"><span>KẾ HOẠCH THAO TÁC</span><Status value={message.operation.status} /></div>
                    <h3>{message.operation.summary}</h3>
                    <OperationDetails payload={message.operation.payload} />
                    {message.operation.error && <p className="error">{message.operation.error}</p>}
                    {message.operation.status === "waiting_for_confirmation" && (
                      <div className="actions">
                        <button className="button button--primary" onClick={() => void handleDecision(message.id, message.operation!, "confirm")}>Xác nhận</button>
                        <button className="button button--ghost" onClick={() => void handleDecision(message.id, message.operation!, "cancel")}>Hủy</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </article>
          ))}
          {busy && <div className="typing"><i /><i /><i /></div>}
          <div ref={endRef} />
        </div>
        <div className="composer-area">
          {sandboxNotice && <div className="sandbox-notice" role="status">{sandboxNotice}</div>}
          <div className={`connection connection--${connectionStatus}`} role="status" aria-live="polite">
            <div className="connection__message">
              <span className="connection__indicator" aria-hidden="true" />
              <div>
                <strong>
                  {connectionStatus === "connecting" && "Đang kết nối tới sandbox..."}
                  {connectionStatus === "ready" && "Sandbox sẵn sàng"}
                  {connectionStatus === "failed" && "Không thể kết nối tới sandbox"}
                </strong>
                {connectionStatus === "connecting" && <small>Lần khởi động đầu tiên có thể mất khoảng một phút.</small>}
                {connectionStatus === "failed" && <small>Backend chưa phản hồi. Vui lòng thử kết nối lại.</small>}
              </div>
            </div>
            {connectionStatus === "failed" && <button type="button" onClick={() => void connect()}>Thử lại</button>}
          </div>
          <div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => void submit(suggestion)} disabled={busy || connectionStatus !== "ready"}>{suggestion}</button>)}</div>
          <form className="composer" onSubmit={onSubmit}>
            <input
              aria-label="Yêu cầu quản lý máy ảo"
              maxLength={500}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={connectionStatus === "ready" ? "Nhập yêu cầu, ví dụ: Tạo máy Ubuntu 4 CPU, RAM 16 GB" : "Đang chờ kết nối tới sandbox..."}
              disabled={busy || connectionStatus !== "ready"}
            />
            <button type="submit" disabled={busy || connectionStatus !== "ready" || !input.trim()} aria-label="Gửi">→</button>
          </form>
          <small className="disclaimer">Mọi thay đổi đều yêu cầu xác nhận · Dữ liệu chỉ là giả lập</small>
        </div>
      </section>
    </main>
  );
}
