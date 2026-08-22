import { FormEvent, useEffect, useRef, useState } from "react";
import { decideOperation, sendMessage } from "./api";
import type { ChatMessage, ConversationContextMessage, Operation } from "./types";
import "./styles.css";

const suggestions = [
  "Liệt kê máy của tôi",
  "Tôi còn bao nhiêu CPU và RAM?",
  "Tạo máy Ubuntu 4 CPU, RAM 16 GB",
  "Khởi động máy test-01",
];

const initialMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Xin chào. Tôi là JCloud Agent chạy với dữ liệu giả lập. Tôi có thể liệt kê máy, kiểm tra quota và lập kế hoạch tạo, khởi động hoặc tắt máy.",
};

const sensitiveValuePattern = /(?:api[_ -]?key|token|password|mật khẩu|private key)\s*[:=]\s*\S+/i;

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

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([initialMessage]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
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

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand__mark">JC</span><div><strong>JCloud Agent</strong><small>Mock environment</small></div></div>
        <div className="environment"><span className="pulse" />Local sandbox</div>
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
          <span className="mode">MOCK MODE</span>
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
                    <dl>{Object.entries(message.operation.payload).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl>
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
          <div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => void submit(suggestion)} disabled={busy}>{suggestion}</button>)}</div>
          <form className="composer" onSubmit={onSubmit}>
            <input aria-label="Yêu cầu quản lý máy ảo" maxLength={500} value={input} onChange={(event) => setInput(event.target.value)} placeholder="Nhập yêu cầu, ví dụ: Tạo máy Ubuntu 4 CPU, RAM 16 GB" />
            <button type="submit" disabled={busy || !input.trim()} aria-label="Gửi">→</button>
          </form>
          <small className="disclaimer">Mọi thay đổi đều yêu cầu xác nhận · Dữ liệu chỉ là giả lập</small>
        </div>
      </section>
    </main>
  );
}
