import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App, { OperationDetails } from "./App";

const apiMocks = vi.hoisted(() => ({
  checkBackend: vi.fn(),
  sendMessage: vi.fn(),
  decideOperation: vi.fn(),
  resetSandbox: vi.fn(),
}));

vi.mock("./api", () => apiMocks);

describe("App sandbox controls", () => {
  beforeEach(() => {
    apiMocks.checkBackend.mockReset().mockResolvedValue({
      status: "ok",
      cloud: "mock",
      llm_provider: "mock",
    });
    apiMocks.sendMessage.mockReset().mockResolvedValue({ message: "가상 머신 목록", data: [] });
    apiMocks.decideOperation.mockReset();
    apiMocks.resetSandbox.mockReset().mockResolvedValue({
      status: "reset",
      instances: [{ name: "web-demo" }, { name: "test-01" }],
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("keeps chat disabled until the backend is ready", async () => {
    let resolveHealth!: (value: { status: "ok"; cloud: string; llm_provider: string }) => void;
    apiMocks.checkBackend.mockReturnValueOnce(new Promise((resolve) => { resolveHealth = resolve; }));
    render(<App />);

    const input = screen.getByLabelText("가상 머신 관리 요청");
    expect(input).toBeDisabled();
    expect(screen.getByText("Sandbox에 연결 중...")).toBeInTheDocument();

    resolveHealth({ status: "ok", cloud: "mock", llm_provider: "mock" });
    await waitFor(() => expect(input).toBeEnabled());
    expect(screen.getAllByText("Sandbox 준비 완료")).toHaveLength(2);
  });

  it("offers retry after a failed connection", async () => {
    apiMocks.checkBackend.mockRejectedValueOnce(new Error("offline"));
    render(<App />);

    const retry = await screen.findByRole("button", { name: "다시 시도" });
    apiMocks.checkBackend.mockResolvedValueOnce({ status: "ok", cloud: "mock", llm_provider: "mock" });
    fireEvent.click(retry);

    await waitFor(() => expect(screen.getByLabelText("가상 머신 관리 요청")).toBeEnabled());
    expect(apiMocks.checkBackend).toHaveBeenCalledTimes(2);
  });

  it("starts a new conversation without resetting sandbox", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("가상 머신 관리 요청")).toBeEnabled());
    const input = screen.getByLabelText("가상 머신 관리 요청");
    fireEvent.change(input, { target: { value: "가상 머신 목록" } });
    fireEvent.submit(input.closest("form")!);
    expect(await screen.findByText("가상 머신 목록")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "새 대화" }));

    expect(screen.queryByText("가상 머신 목록")).not.toBeInTheDocument();
    expect(screen.getByText(/안녕하세요\. 모의 데이터로 실행되는 JCloud Agent/)).toBeInTheDocument();
    expect(apiMocks.resetSandbox).not.toHaveBeenCalled();
  });

  it("confirms reset, calls the reset endpoint and clears the conversation", async () => {
    render(<App />);
    await waitFor(() => expect(screen.getByLabelText("가상 머신 관리 요청")).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: "Sandbox 초기화" }));

    await waitFor(() => expect(apiMocks.resetSandbox).toHaveBeenCalledTimes(1));
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining("현재 브라우저 세션"));
    expect(await screen.findByText(/Sandbox가 기본 머신 두 대/)).toBeInTheDocument();
  });
});

describe("OperationDetails", () => {
  it("shows friendly fields and hides technical IDs", () => {
    render(<OperationDetails payload={{
      name: "ubuntu-demo",
      image: "Ubuntu 24.04",
      image_id: "img-ubuntu-2404",
      flavor: "large",
      flavor_id: "flavor-large",
      vcpus: 4,
      ram_gb: 16,
      requires_gpu: false,
      session_id: "secret-session",
    }} />);

    expect(screen.getByText("머신 이름")).toBeInTheDocument();
    expect(screen.getByText("Ubuntu 24.04")).toBeInTheDocument();
    expect(screen.getByText("large")).toBeInTheDocument();
    expect(screen.queryByText("img-ubuntu-2404")).not.toBeInTheDocument();
    expect(screen.queryByText("flavor-large")).not.toBeInTheDocument();
    expect(screen.queryByText("secret-session")).not.toBeInTheDocument();
  });
});
