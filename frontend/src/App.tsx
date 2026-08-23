import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { checkBackend, decideOperation, resetSandbox, sendMessage } from "./api";
import type { ChatMessage, ConversationContextMessage, Operation } from "./types";
import "./styles.css";

const suggestions = [
  "내 가상 머신 목록 보여 줘",
  "CPU와 RAM이 얼마나 남았어?",
  "Ubuntu 24.04, CPU 4개, RAM 16 GB 머신 생성",
  "test-01 머신 시작",
];

export const initialMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "안녕하세요. 모의 데이터로 실행되는 JCloud Agent입니다. 가상 머신 목록과 할당량을 확인하고 머신 생성, 시작 또는 중지 계획을 준비할 수 있습니다.",
};

const sensitiveValuePattern = /(?:api[_ -]?key|token|password|비밀번호|암호|private key)\s*[:=]\s*\S+/i;
type ConnectionStatus = "connecting" | "ready" | "failed";

export function buildConversationContext(messages: ChatMessage[]): ConversationContextMessage[] {
  return messages
    .filter((message) => message.text.trim() && !sensitiveValuePattern.test(message.text))
    .map((message) => ({ role: message.role, content: message.text.slice(0, 500) }))
    .slice(-10);
}

function Status({ value }: { value: Operation["status"] }) {
  const labels: Record<Operation["status"], string> = {
    waiting_for_confirmation: "확인 대기",
    running: "실행 중",
    completed: "완료",
    failed: "실패",
    cancelled: "취소됨",
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
        <div><strong>{String(quota.available_vcpus)}</strong><span>사용 가능한 vCPU</span></div>
        <div><strong>{String(quota.available_ram_gb)} GB</strong><span>사용 가능한 RAM</span></div>
      </div>
    );
  }
  return null;
}

export function OperationDetails({ payload }: { payload: Operation["payload"] }) {
  const details = [
    ["머신 이름", payload.name],
    ["운영체제", payload.image ?? payload.operating_system],
    ["CPU", payload.vcpus === undefined ? undefined : `${String(payload.vcpus)} vCPU`],
    ["RAM", payload.ram_gb === undefined ? undefined : `${String(payload.ram_gb)} GB`],
    ["GPU", payload.requires_gpu === undefined ? undefined : payload.requires_gpu ? "필요" : "불필요"],
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
      if (health.status !== "ok") throw new Error("백엔드가 아직 준비되지 않았습니다.");
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
        { id: crypto.randomUUID(), role: "assistant", text: error instanceof Error ? error.message : "백엔드에 연결할 수 없습니다." },
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
        text: updated.status === "completed" ? "모의 작업이 완료되었습니다." : "계획이 취소되었습니다.",
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
    setSandboxNotice("새 대화를 시작했습니다. Sandbox 데이터는 변경되지 않았습니다.");
  }

  async function handleResetSandbox() {
    const confirmed = window.confirm(
      "Sandbox 초기화는 현재 브라우저 세션의 모의 가상 머신과 작업 계획만 삭제합니다. 계속할까요?",
    );
    if (!confirmed) return;

    setBusy(true);
    setSandboxNotice(null);
    try {
      await resetSandbox();
      setMessages([initialMessage]);
      setInput("");
      setSandboxNotice("Sandbox가 기본 머신 두 대(web-demo, test-01)로 초기화되었습니다.");
    } catch (error) {
      setSandboxNotice(error instanceof Error ? error.message : "Sandbox를 초기화할 수 없습니다.");
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
          {connectionStatus === "connecting" && "연결 중..."}
          {connectionStatus === "ready" && "Sandbox 준비 완료"}
          {connectionStatus === "failed" && "연결 끊김"}
        </div>
        <div className="sidebar__copy">
          <h2>안전을 고려한 설계</h2>
          <p>AI는 요청을 이해하고 계획만 작성합니다. 백엔드가 검증하며 사용자가 확인한 후에만 실행합니다.</p>
        </div>
        <div className="guardrails">
          <span>Credential 미사용</span><span>실제 OpenStack 미연결</span><span>Shell 실행 금지</span>
        </div>
      </aside>

      <section className="chat">
        <header>
          <div><span className="eyebrow">CONTROL PLANE ASSISTANT</span><h1>대화형 클라우드 관리</h1></div>
          <div className="header-actions">
            <button type="button" onClick={startNewConversation} disabled={busy}>새 대화</button>
            <button type="button" className="reset-button" onClick={() => void handleResetSandbox()} disabled={busy || connectionStatus !== "ready"}>Sandbox 초기화</button>
            <span className="mode">MOCK MODE</span>
          </div>
        </header>
        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article key={message.id} className={`message message--${message.role}`}>
              <div className="message__label">{message.role === "user" ? "사용자" : "Agent"}</div>
              <div className="bubble">
                <p>{message.text}</p>
                <DataPreview data={message.data} />
                {message.operation && (
                  <div className="operation">
                    <div className="operation__top"><span>작업 계획</span><Status value={message.operation.status} /></div>
                    <h3>{message.operation.summary}</h3>
                    <OperationDetails payload={message.operation.payload} />
                    {message.operation.error && <p className="error">{message.operation.error}</p>}
                    {message.operation.status === "waiting_for_confirmation" && (
                      <div className="actions">
                        <button className="button button--primary" onClick={() => void handleDecision(message.id, message.operation!, "confirm")}>확인</button>
                        <button className="button button--ghost" onClick={() => void handleDecision(message.id, message.operation!, "cancel")}>취소</button>
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
                  {connectionStatus === "connecting" && "Sandbox에 연결 중..."}
                  {connectionStatus === "ready" && "Sandbox 준비 완료"}
                  {connectionStatus === "failed" && "Sandbox에 연결할 수 없습니다"}
                </strong>
                {connectionStatus === "connecting" && <small>첫 시작에는 약 1분이 걸릴 수 있습니다.</small>}
                {connectionStatus === "failed" && <small>백엔드가 응답하지 않습니다. 다시 연결해 주세요.</small>}
              </div>
            </div>
            {connectionStatus === "failed" && <button type="button" onClick={() => void connect()}>다시 시도</button>}
          </div>
          <div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => void submit(suggestion)} disabled={busy || connectionStatus !== "ready"}>{suggestion}</button>)}</div>
          <form className="composer" onSubmit={onSubmit}>
            <input
              aria-label="가상 머신 관리 요청"
              maxLength={500}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={connectionStatus === "ready" ? "요청 입력 (예: Ubuntu 24.04, CPU 4개, RAM 16 GB 머신 생성)" : "Sandbox 연결을 기다리는 중..."}
              disabled={busy || connectionStatus !== "ready"}
            />
            <button type="submit" disabled={busy || connectionStatus !== "ready" || !input.trim()} aria-label="전송">→</button>
          </form>
          <small className="disclaimer">모든 변경 작업은 확인 필요 · 모의 데이터만 사용</small>
        </div>
      </section>
    </main>
  );
}
