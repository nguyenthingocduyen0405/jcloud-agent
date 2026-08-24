# JCloud Agent MVP

자연어 대화를 통해 가상 머신 관리 흐름을 시험할 수 있는 로컬 웹 애플리케이션입니다. 현재 클라우드 계층은 모의 데이터를 사용하며, 실제 JCloud/OpenStack에 연결하지 않습니다. 또한 사용자나 AI가 생성한 셸 명령을 실행하지 않습니다.

## 안전한 아키텍처

```text
사용자
  │ 자연어 요청
  ▼
React 채팅 UI ──HTTP──► FastAPI
                            │
                            ▼
                 LLMClient.parse_message()
                   │ Pydantic 검증을 통과한
                   │ LLMDecision만 반환
                   ▼
             허용 목록 + 정책 + 메타데이터 검증
                   │
                   ├─ 조회 작업: CloudClient
                   │
                   └─ 변경 작업: operation 생성
                          │ waiting_for_confirmation
                          ▼
                       사용자 확인
                          ▼
                  MockCloudClient ──► SQLite
```

LLM은 사용자의 의도를 이해하고 구조화된 JSON을 반환하는 역할만 담당합니다. LLM에는 도구가 제공되지 않으며, `CloudClient` 호출, 셸 실행 또는 리소스 변경을 직접 수행할 수 없습니다. 백엔드만 작업 허용 목록을 적용하고, `CloudClient`에서 실제 image/flavor 정보를 조회하여 검증하고, quota와 정책을 확인한 후 사용자가 승인한 operation을 실행할 수 있습니다.

`CloudClient`는 클라우드 연동 경계로 유지됩니다. 향후 채팅 UI나 LLM 계층을 변경하지 않고 `OpenStackCloudClient`를 추가할 수 있습니다.

## 요구 사항

- Windows 10/11
- Python 3.11 이상
- Node.js 20 이상

## LLM 설정

프로젝트 루트에서 예제 환경 설정 파일을 복사합니다.

```powershell
Copy-Item .env.example .env
```

### 자동 선택 모드로 실행

기본 `auto` 모드는 `OPENROUTER_API_KEY`가 있으면 OpenRouter를 우선 사용하고, 그렇지 않으면
`LLM_API_KEY`가 있을 때 OpenAI를 사용합니다. 두 키가 모두 없으면 인터넷 연결이 필요 없는
`MockLLMClient`를 안전한 fallback으로 사용합니다.

```dotenv
LLM_PROVIDER=auto
LLM_MODEL=gpt-5-nano
LLM_API_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=google/gemma-4-31b-it:free
```

항상 mock만 사용하려면 `LLM_PROVIDER=mock`으로 지정하세요. 자동화 테스트는 외부 API를 호출하지
않도록 항상 `MockLLMClient`를 주입합니다.

### OpenAI로 실행

API 키는 반드시 프로젝트 루트의 `.env` 파일에만 저장하세요. 프런트엔드, 소스 코드, 로그 또는 `.env.example`에는 API 키를 입력하지 마세요.

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-nano
LLM_API_KEY=여기에-실제-API-키-입력
```

계정에서 사용할 수 있고 Responses API와 Structured Outputs를 지원하는 모델을 선택해야 합니다. OpenAI 어댑터는 `store=false`로 요청하고 도구를 구성하지 않으며, 모든 출력을 Pydantic의 `LLMDecision`으로 다시 검증합니다. Provider 오류가 발생하거나 출력이 유효하지 않으면 백엔드는 안전한 오류 메시지만 반환하고 operation을 생성하지 않습니다.

### 무료 OpenRouter 모델로 실행

한국어 자연어 요청을 위한 기본 무료 모델은 `google/gemma-4-31b-it:free`입니다. OpenRouter
Dashboard에서 API 키를 만든 뒤 `.env` 또는 Render secret에만 저장하세요.

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=여기에-실제-OpenRouter-API-키-입력
OPENROUTER_MODEL=google/gemma-4-31b-it:free
OPENROUTER_TIMEOUT_SECONDS=20
OPENROUTER_MAX_OUTPUT_TOKENS=500
OPENROUTER_ATTEMPTS=2
```

무료 Gemma endpoint는 JSON object 출력을 요청하고 모든 응답을 `LLMDecision`으로 다시
검증합니다. 잘못된 JSON은 한 번만 재시도하며, 두 번 모두 실패하면 operation을 만들지 않습니다.

Provider 설정을 변경한 후에는 백엔드를 다시 시작해야 합니다.

## Windows에서 백엔드 실행

프로젝트 디렉터리에서 PowerShell을 엽니다.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --env-file ..\.env
```

`.env` 파일을 만들지 않았다면 `--env-file ..\.env` 부분을 제외하세요. 이 경우 백엔드는 자동으로 `LLM_PROVIDER=mock`을 사용합니다.

- API: <http://127.0.0.1:8000>
- API 문서: <http://127.0.0.1:8000/docs>
- 현재 상태 및 provider: <http://127.0.0.1:8000/api/health>

## Windows에서 프런트엔드 실행

별도의 PowerShell 창에서 다음 명령을 실행합니다.

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

브라우저에서 <http://127.0.0.1:5173>을 엽니다. 프런트엔드는 기본적으로 `http://127.0.0.1:8000`의 백엔드를 호출합니다.

## Render에 공개 배포

저장소 루트의 `render.yaml`은 FastAPI와 React 정적 빌드를 하나의 공개 Web Service로 배포합니다. Render에서 이 GitHub 저장소를 Blueprint로 연결하면 자동으로 빌드하고 `onrender.com` URL을 발급합니다.

- 배포 모드는 `LLM_PROVIDER=auto`입니다. Render의 `OPENROUTER_API_KEY` secret을 설정하면 무료
  Gemma 모델을 우선 사용합니다. OpenRouter 키가 없고 `LLM_API_KEY`만 있으면 OpenAI를 사용하며,
  두 키가 모두 없으면 mock fallback을 사용합니다.
- API 키는 Render Dashboard의 secret 환경 변수로만 설정하며 저장소에 커밋하지 않습니다.
- 프로덕션 React 빌드는 FastAPI와 같은 origin에서 제공됩니다.
- `main` 브랜치에 push하면 Render가 자동으로 다시 배포합니다.
- free Web Service의 로컬 파일 시스템은 영구 저장소가 아닙니다. 따라서 SQLite 데이터는 service restart, redeploy 또는 spin-down 후 초기 mock 데이터로 돌아갈 수 있습니다.

이 구성은 공개 MVP 데모용입니다. 실제 사용자 데이터나 실제 클라우드 리소스 관리에는 사용하지 마세요.

## 구조화된 출력

모든 provider는 다음 메서드를 구현해야 합니다.

```python
parse_message(message, conversation_context, cloud_context)
```

그리고 다음 세 가지 유형 중 하나에 해당하는 유효한 `LLMDecision`을 반환해야 합니다.

- `action`: 허용 목록에 있는 작업을 요청합니다.
- `clarification`: 중요한 정보가 부족하여 사용자에게 추가 질문이 필요합니다.
- `answer`: 설명하거나 요청을 거절하며 어떠한 작업도 수행하지 않습니다.

작업 허용 목록은 `list_instances`, `get_quota`, `list_images`, `list_flavors`, `plan_create_instance`, `start_instance`, `stop_instance`, `reboot_instance`입니다.

인스턴스 삭제, 셸 명령 실행, controller/compute node 변경, 공유 네트워크 변경, 전체 방화벽 개방은 지원하지 않습니다.

OpenAI provider는 strict Structured Outputs를 사용하며 요청마다 기본 15초 timeout과 500 output-token 제한을 적용합니다. 현재 실제로 선택된 provider는 `/api/health`의 `llm_provider`에서 확인할 수 있습니다. 다음 환경 변수로 값을 조정할 수 있습니다.

```dotenv
LLM_TIMEOUT_SECONDS=15
LLM_MAX_OUTPUT_TOKENS=500
LLM_REASONING_EFFORT=minimal
LLM_FAST_PATH=true
```

`LLM_FAST_PATH=true`이면 목록, quota, 생성, 시작, 중지, 재부팅처럼 로컬 규칙이 확실하게
해석할 수 있는 요청은 외부 LLM 호출 없이 즉시 처리합니다. 로컬 규칙이 확신할 수 없는 요청만
선택된 provider로 전달됩니다. `/api/health`에 `openrouter+fast-path` 또는 `openai+fast-path`가
표시되면 이 최적화가 활성화된 것입니다. `LLM_REASONING_EFFORT=minimal`은 `gpt-5-nano`의 분류·라우팅 작업에서
불필요한 추론 시간을 줄입니다. 다른 모델로 변경할 때는 해당 모델이 지원하는 reasoning effort를
확인하거나 빈 값으로 설정하세요.

프런트엔드는 현재 메시지를 제외한 최근 대화 텍스트를 최대 10개까지 `conversation_context`로 전송합니다. operation payload와 민감한 값은 대화 context에 포함하지 않습니다. Ubuntu 버전을 생략하면 백엔드는 검증된 `Ubuntu 24.04` image를 기본으로 선택하고, 사용자 응답에도 기본 선택임을 명시합니다. `Ubuntu 22.04` 또는 `Ubuntu 24.04`를 지정하면 해당 버전과 정확히 일치하는 image만 사용합니다.

## 브라우저 세션과 모의 격리

첫 방문 시 프런트엔드는 무작위 UUID를 생성하여 브라우저의 `localStorage`에 `jcloud_agent_session_id`라는 이름으로 저장합니다. 새로고침해도 같은 UUID를 재사용하며 모든 백엔드 요청에 `X-Session-ID`로 전달합니다. 이 값은 화면이나 일반 로그에 표시하지 않습니다.

- `X-User-ID: mock-user`
- `X-Project-ID: mock-project`

Mock 모드의 instance, 사용 quota, operation은 session별로 분리됩니다. 각 새 session에는 `web-demo`와 `test-01`이 별도로 생성됩니다. 다른 session의 instance와 operation은 조회하거나 변경할 수 없습니다. 단, 이 UUID 격리는 데모 sandbox의 데이터 분리 기능일 뿐이며 인증이나 권한 부여가 아닙니다. 실제 서비스에서는 Keystone 또는 별도의 신뢰할 수 있는 인증 계층이 필요합니다.

화면의 **새 대화**는 프런트엔드 대화 기록만 초기화하며 instance, quota, operation은 변경하지 않습니다. **Reset sandbox**는 확인 창을 거쳐 현재 브라우저 session의 모의 instance와 operation만 삭제하고 기본 VM 두 개를 다시 생성합니다.

## SQLite 스키마 업그레이드

백엔드 시작 시 기존 `instances` 테이블에 전역 `UNIQUE(name)` 제약이 있으면 자동 migration을 수행합니다. 기존 행은 손실 없이 `mock-session`에 귀속시키고 테이블을 `UNIQUE(session_id, name)` 구조로 교체합니다. 이후 브라우저 UUID session은 각각 독립된 기본 데이터를 받습니다.

업그레이드 전에 중요한 로컬 mock 데이터가 있다면 `backend/data/jcloud_agent.db`를 복사해 두는 것이 좋습니다. migration이 불가능한 손상된 개발용 DB라면 백엔드를 중지한 뒤 해당 DB 파일을 삭제하여 깨끗한 mock DB를 다시 만들 수 있습니다. Render 무료 파일 시스템은 임시이므로 재배포나 재시작 때 SQLite 데이터가 초기화될 수 있습니다.

## 테스트

백엔드 테스트는 항상 `MockLLMClient`를 주입하므로 실제 API를 호출하거나 API 키를 읽지 않습니다.

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

프런트엔드는 다음 명령으로 확인합니다.

```powershell
cd frontend
npm.cmd test
npm.cmd run build
```

## 요청 예시

- `내 가상 머신 목록을 보여 줘.`
- `사용 가능한 CPU가 얼마나 남았어?`
- `Ubuntu, CPU 4개, RAM 16GB인 머신을 만들어 줘.`
- `강력한 머신을 하나 만들어 줘.`
- `test-01 머신을 시작해 줘.`
- `test-01 머신을 중지해 줘.`
- `test-01 머신을 재부팅해 줘.`

생성, 시작, 중지, 재부팅 요청은 먼저 `waiting_for_confirmation` 상태의 operation만 생성합니다. 사용자가 **확인**을 선택한 후에만 모의 데이터가 변경됩니다.

## 데이터

- 기본 SQLite 경로: `backend/data/jcloud_agent.db`
- session별 기본 가상 머신: `web-demo`, `test-01`
- 백엔드는 대화 내용이나 API 키를 SQLite에 저장하지 않습니다.
- `.env`는 `.gitignore`에 포함되어 있으므로 커밋하지 않습니다.
