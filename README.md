# JCloud Agent MVP

Ứng dụng web chạy cục bộ để thử nghiệm quản lý máy ảo bằng hội thoại. Cloud hiện dùng dữ liệu giả lập; không kết nối JCloud/OpenStack thật và không chạy lệnh shell do người dùng hoặc AI tạo ra.

## Kiến trúc an toàn

```text
Người dùng
    │ ngôn ngữ tự nhiên
    ▼
React chat UI ──HTTP──► FastAPI
                           │
                           ▼
                LLMClient.parse_message()
                  │ chỉ trả LLMDecision đã
                  │ được Pydantic kiểm tra
                  ▼
            allowlist + policy + xác minh metadata
                  │
                  ├─ thao tác đọc: CloudClient
                  │
                  └─ thao tác thay đổi: tạo operation
                         │ waiting_for_confirmation
                         ▼
                    người dùng xác nhận
                         ▼
                 MockCloudClient ──► SQLite
```

LLM chỉ hiểu ý định và trả JSON có cấu trúc. LLM không nhận tool, không gọi `CloudClient`, không chạy shell và không thực hiện thay đổi. Chỉ backend được phép ánh xạ action qua allowlist, tìm image/flavor thật từ `CloudClient`, kiểm tra quota/chính sách và thực hiện operation đã được xác nhận.

`CloudClient` vẫn là ranh giới tích hợp. Có thể thêm `OpenStackCloudClient` sau này mà không thay đổi giao diện chat hoặc lớp LLM.

## Yêu cầu

- Windows 10/11
- Python 3.11 trở lên
- Node.js 20 trở lên

## Cấu hình LLM

Sao chép file cấu hình mẫu tại thư mục gốc:

```powershell
Copy-Item .env.example .env
```

### Chạy bằng MockLLMClient

Không cần API key và không gọi Internet:

```dotenv
LLM_PROVIDER=mock
LLM_MODEL=
LLM_API_KEY=
```

Đây là chế độ mặc định và cũng là chế độ dùng trong test tự động.

### Chạy bằng OpenAI

Đặt API key **chỉ trong file `.env` ở thư mục gốc**, không đặt trong frontend, source code, log hoặc `.env.example`:

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-mini
LLM_API_KEY=đặt-api-key-thật-tại-đây
```

Chọn một model có Structured Outputs và Responses API mà tài khoản của bạn được phép sử dụng. Adapter OpenAI gửi request với `store=false`, không cấu hình tool, và kiểm tra lại toàn bộ output bằng `LLMDecision` của Pydantic. Nếu provider lỗi hoặc output không hợp lệ, backend trả thông báo an toàn và không tạo operation.

Sau khi đổi provider, cần khởi động lại backend.

## Khởi động backend trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
cd "C:\Users\dn160\OneDrive\Desktop\jcloud_agent\backend"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --env-file ..\.env
```

Nếu chưa tạo `.env`, bỏ phần `--env-file ..\.env`; backend sẽ tự dùng `LLM_PROVIDER=mock`.

- API: <http://127.0.0.1:8000>
- API docs: <http://127.0.0.1:8000/docs>
- Health/provider hiện tại: <http://127.0.0.1:8000/api/health>

## Khởi động frontend trên Windows

Trong cửa sổ PowerShell khác:

```powershell
cd "C:\Users\dn160\OneDrive\Desktop\jcloud_agent\frontend"
npm.cmd install
npm.cmd run dev
```

Mở <http://127.0.0.1:5173>. Frontend mặc định gọi backend tại `http://127.0.0.1:8000`.

## Structured output

Mọi provider phải triển khai:

```python
parse_message(message, conversation_context, cloud_context)
```

Và phải trả `LLMDecision` hợp lệ với một trong ba loại:

- `action`: yêu cầu một action nằm trong allowlist.
- `clarification`: thiếu dữ liệu quan trọng, cần hỏi lại.
- `answer`: chỉ trả lời hoặc từ chối, không thao tác.

Allowlist gồm `list_instances`, `get_quota`, `list_images`, `list_flavors`, `plan_create_instance`, `start_instance`, `stop_instance` và `reboot_instance`.

Không hỗ trợ xóa instance, shell command, thay đổi controller/compute node, thay đổi network dùng chung hoặc mở toàn bộ firewall.

## Kiểm thử

Test backend luôn inject `MockLLMClient`; không gọi API thật và không đọc API key:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Kiểm tra frontend:

```powershell
cd frontend
npm.cmd run build
```

## Yêu cầu mẫu

- `Liệt kê máy của tôi.`
- `Tôi còn bao nhiêu CPU?`
- `Tạo Ubuntu 4 CPU và 16 GB RAM.`
- `Tạo cho tôi một máy mạnh.`
- `Khởi động máy test-01.`
- `Tắt máy test-01.`
- `Khởi động lại máy test-01.`

Các thao tác tạo, khởi động, tắt và reboot chỉ tạo operation ở trạng thái `waiting_for_confirmation`. Dữ liệu chỉ thay đổi sau khi người dùng chọn **Xác nhận**.

## Dữ liệu

- SQLite mặc định: `backend/data/jcloud_agent.db`.
- Hai máy mẫu: `web-demo` và `test-01`.
- Backend không lưu nội dung hội thoại hoặc API key vào SQLite.
- `.env` đã nằm trong `.gitignore`; không commit file này.

