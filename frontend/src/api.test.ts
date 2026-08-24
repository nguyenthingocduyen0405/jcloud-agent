import { beforeEach, describe, expect, it, vi } from "vitest";
import { checkBackend, getOrCreateSessionId, resetSandbox, sendMessage, SESSION_STORAGE_KEY } from "./api";

describe("browser session", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("creates a UUID for a browser that has no session yet", () => {
    const sessionId = getOrCreateSessionId();
    expect(sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it("creates a session UUID once and reuses it after reload", () => {
    const first = getOrCreateSessionId(localStorage, () => "first-browser-uuid");
    const second = getOrCreateSessionId(localStorage, () => "must-not-be-used");

    expect(first).toBe("first-browser-uuid");
    expect(second).toBe(first);
    expect(localStorage.getItem(SESSION_STORAGE_KEY)).toBe(first);
  });

  it("sends the stored session ID with every backend request", async () => {
    localStorage.setItem(SESSION_STORAGE_KEY, "persisted-browser-uuid");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "ok", cloud: "mock", llm_provider: "mock" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await checkBackend();

    const headers = fetchMock.mock.calls[0][1]?.headers as Record<string, string>;
    expect(headers["X-Session-ID"]).toBe("persisted-browser-uuid");
    expect(headers["X-Session-ID"]).not.toBe("mock-session");
  });

  it("sends conversation context and reuses the same session for follow-up answers", async () => {
    localStorage.setItem(SESSION_STORAGE_KEY, "conversation-session-uuid");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ message: "ok" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const context = [
      { role: "user" as const, content: "Tạo máy Ubuntu 24.04 RAM 16 GB" },
      { role: "assistant" as const, content: "Vui lòng cho biết thêm: vCPU." },
    ];

    await sendMessage("4", context);

    const init = fetchMock.mock.calls[0][1];
    const headers = init?.headers as Record<string, string>;
    expect(headers["X-Session-ID"]).toBe("conversation-session-uuid");
    expect(JSON.parse(String(init?.body))).toEqual({
      message: "4",
      conversation_context: context,
    });
  });

  it("calls the session-scoped sandbox reset endpoint", async () => {
    localStorage.setItem(SESSION_STORAGE_KEY, "reset-browser-uuid");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ status: "reset", instances: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await resetSandbox();

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/sandbox\/reset$/),
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "X-Session-ID": "reset-browser-uuid" }),
      }),
    );
  });
});
