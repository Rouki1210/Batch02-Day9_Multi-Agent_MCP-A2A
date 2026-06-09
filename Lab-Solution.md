# Lab Solution: Multi-Agent MCP/A2A

Tài liệu này tổng hợp lời giải cho codelab trong `CODELAB.md`, dựa trên code hiện có trong repo. Mục tiêu là hiểu tiến trình từ gọi LLM trực tiếp đến hệ thống multi-agent phân tán dùng A2A protocol.

## 0. Chuẩn bị môi trường

Yêu cầu:

- Python 3.11+
- `uv`
- OpenRouter API key

Các bước chạy:

```bash
uv sync
cp .env.example .env
# Điền OPENROUTER_API_KEY vào .env
```

File cấu hình LLM nằm ở `common/llm.py`. Hàm `get_llm()` tạo `ChatOpenAI` trỏ tới OpenRouter:

```python
return ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
)
```

`temperature=0.3` giúp câu trả lời ổn định hơn so với temperature cao.

## 1. Stage 1: Direct LLM Calling

Chạy:

```bash
uv run python stages/stage_1_direct_llm/main.py
```

File chính: `stages/stage_1_direct_llm/main.py`.

### Trả lời câu hỏi

1. LLM được khởi tạo bằng `get_llm()` từ `common/llm.py`. Hàm này trả về một client `ChatOpenAI` dùng OpenRouter làm OpenAI-compatible endpoint.

2. Message gửi vào LLM là danh sách gồm:

```python
messages = [
    SystemMessage(content="You are a legal expert..."),
    HumanMessage(content=QUESTION),
]
```

3. `SystemMessage` định nghĩa vai trò, phong cách, và ràng buộc của model. `HumanMessage` là câu hỏi thực tế của người dùng. Tách hai loại message này giúp prompt rõ hơn và dễ mở rộng sang agent/tool flow ở các stage sau.

### Bài tập 1.1

Thay đổi biến `QUESTION`, ví dụ:

```python
QUESTION = "Một startup cần lưu ý gì khi ký NDA với đối tác nước ngoài?"
```

Sau đó chạy lại file Stage 1.

### Bài tập 1.2

Đáp án đã có trong `common/llm.py`: thêm `temperature=0.3` vào `ChatOpenAI(...)`.

## 2. Stage 2: LLM + RAG & Tools

Chạy:

```bash
uv run python stages/stage_2_rag_tools/main.py
```

File chính: `stages/stage_2_rag_tools/main.py`.

Stage này thêm:

- Knowledge base giả lập: `LEGAL_KNOWLEDGE`
- Tool search: `search_legal_database`
- Tool tính toán: `calculate_damages`
- Tool thời hiệu: `check_statute_of_limitations`
- Manual tool loop: LLM chọn tool, code gọi tool, rồi đưa kết quả tool về lại LLM

### Trả lời câu hỏi

1. Decorator `@tool` được dùng trước các hàm tool như `search_legal_database`, `calculate_damages`, `check_statute_of_limitations`.

2. `LEGAL_KNOWLEDGE` là list các dict. Mỗi entry có:

```python
{
    "id": "...",
    "keywords": [...],
    "text": "...",
}
```

3. LLM bind với tool bằng:

```python
llm_with_tools = llm.bind_tools(TOOLS)
```

Sau đó `response.tool_calls` cho biết LLM muốn gọi tool nào và với tham số gì.

### Bài tập 2.1: Thêm luật lao động

Thêm entry này vào `LEGAL_KNOWLEDGE`:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

Trong stage demo chính, entry này đã được thêm sẵn.

### Bài tập 2.2: Tạo tool thời hiệu

Thêm tool:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.

    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm vào danh sách tool:

```python
TOOLS = [search_legal_database, calculate_damages, check_statute_of_limitations]
```

Nếu làm trong `exercises/exercise_2_tools.py`, cần xử lý thêm khi tool được gọi:

```python
elif tool_call["name"] == "check_statute_of_limitations":
    tool_result = check_statute_of_limitations.invoke(tool_call["args"])
```

## 3. Stage 3: Single Agent với ReAct

Chạy:

```bash
uv run python stages/stage_3_single_agent/main.py
```

File chính: `stages/stage_3_single_agent/main.py`.

Stage 3 thay manual tool loop bằng `create_react_agent`. Agent tự lặp:

1. Think: quyết định cần làm gì
2. Act: gọi tool
3. Observe: đọc kết quả tool
4. Lặp lại đến khi đủ thông tin

### Trả lời câu hỏi

1. `create_react_agent()` nằm trong hàm `main()`:

```python
graph = create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)
```

2. Khác Stage 2: không còn vòng lặp tool call tự viết. LangGraph tự quản lý việc gọi tool nhiều bước.

3. Code chỉ cần gọi agent graph một lần, hoặc stream các update:

```python
async for chunk in graph.astream(inputs, stream_mode="updates"):
    ...
```

### Bài tập 3.1: Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.

    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào `TOOLS`:

```python
TOOLS = [
    search_legal_database,
    calculate_penalty,
    check_compliance_requirements,
    search_case_law,
]
```

Trong stage demo chính, tool này đã được thêm.

### Bài tập 3.2: Debug agent reasoning

Với version LangGraph hiện tại trong repo, code đang debug bằng `graph.astream(..., stream_mode="updates")`, in ra tool calls, tool observations, và final answer. Đây là cách phù hợp hơn so với chỉ thêm `verbose=True`.

## 4. Stage 4: Multi-Agent In-Process

Chạy:

```bash
uv run python stages/stage_4_milti_agent/main.py
```

Lưu ý: tên thư mục trong repo là `stage_4_milti_agent`.

File chính: `stages/stage_4_milti_agent/main.py`.

Luồng graph:

```text
analyze_law -> check_routing -> [tax + compliance + privacy] -> aggregate -> END
```

### Trả lời câu hỏi

1. Shared state là `LegalState(TypedDict)`, gồm câu hỏi, phân tích luật, cờ routing, kết quả từng specialist, và final answer.

2. Các agent/node chính:

- `analyze_law`
- `check_routing`
- `call_tax_specialist`
- `call_compliance_specialist`
- `call_privacy_specialist`
- `aggregate`

3. `Send()` API nằm trong `route_to_specialists`. Hàm này trả về list `Send(...)`; LangGraph sẽ chạy các nhánh specialist song song khi có nhiều Send.

4. Graph được build trong `create_graph()` bằng `StateGraph`, `add_node`, `add_edge`, và `add_conditional_edges`.

### Bài tập 4.1: Thêm Privacy Agent

Trong `exercises/exercise_4_multiagent.py`, implement:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    llm = get_llm()
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.

Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy, GDPR, data protection, quyền riêng tư,
nghĩa vụ thông báo khi rò rỉ dữ liệu, và rủi ro xử phạt nếu có.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node:

```python
graph.add_node("privacy_agent", privacy_agent)
```

Thêm edge:

```python
graph.add_edge("privacy_agent", "aggregate_results")
```

### Bài tập 4.2: Conditional routing

Thêm vào `check_routing`:

```python
if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
    tasks.append(Send("privacy_agent", state))
```

Thêm privacy vào phần tổng hợp:

```python
if state.get("privacy_analysis"):
    sections.append(f"🔒 PHÂN TÍCH PRIVACY/GDPR:\n{state['privacy_analysis']}")
```

Trong stage demo chính, privacy agent đã được tích hợp sẵn bằng `needs_privacy`, `call_privacy_specialist`, và `privacy_result`.

## 5. Stage 5: Distributed A2A System

Chạy toàn bộ hệ thống:

```bash
./start_all.sh
```

Terminal khác:

```bash
uv run python test_client.py
```

Nếu dùng Windows PowerShell và không chạy được `.sh`, có thể chạy từng service:

```bash
uv run python -m registry
uv run python -m tax_agent
uv run python -m compliance_agent
uv run python -m law_agent
uv run python -m customer_agent
```

Các port:

| Service | Port | Vai trò |
|---|---:|---|
| Registry | 10000 | Lưu agent registry và discovery |
| Customer Agent | 10100 | Entry point nhận câu hỏi |
| Law Agent | 10101 | Orchestrator pháp lý |
| Tax Agent | 10102 | Specialist về thuế |
| Compliance Agent | 10103 | Specialist về compliance |

### Luồng request

```text
User/test_client
  -> Customer Agent
  -> Registry discover("legal_question")
  -> Law Agent
  -> analyze_law
  -> check_routing
  -> Registry discover("tax_question")        -> Tax Agent
  -> Registry discover("compliance_question") -> Compliance Agent
  -> aggregate
  -> Customer Agent
  -> User
```

### Dynamic discovery

Mỗi agent tự đăng ký với Registry khi startup bằng `common.registry_client.register(...)`.

Registry có endpoint:

- `POST /register`
- `GET /discover/{task}`
- `GET /agents`
- `GET /health`

Ví dụ Law Agent đăng ký task `legal_question`, Tax Agent đăng ký `tax_question`, Compliance Agent đăng ký `compliance_question`.

Khi cần gọi agent khác, code không hardcode URL mà dùng:

```python
endpoint = await discover("tax_question")
```

Sau đó gọi qua A2A:

```python
result = await delegate(
    endpoint=endpoint,
    question=state["question"],
    context_id=state["context_id"],
    trace_id=state["trace_id"],
    depth=state.get("delegation_depth", 0) + 1,
)
```

### A2A message metadata

`common/a2a_client.py` gắn metadata vào message:

```python
metadata={
    "trace_id": trace_id,
    "context_id": context_id,
    "delegation_depth": depth,
}
```

Ý nghĩa:

- `trace_id`: theo dõi một request đi qua nhiều service
- `context_id`: giữ ngữ cảnh A2A của request/conversation
- `delegation_depth`: chống vòng lặp delegate vô hạn

Trong `law_agent/graph.py`, `MAX_DELEGATION_DEPTH = 3`. Nếu depth đạt giới hạn, Law Agent bỏ qua sub-agent delegation.

### Nếu dừng Tax Agent

Khi Tax Agent không chạy:

1. Registry có thể không tìm thấy `tax_question`, hoặc endpoint không phản hồi.
2. `call_tax()` bắt exception.
3. `tax_result` được gán dạng:

```text
[Tax analysis unavailable: ...]
```

Hệ thống vẫn có thể aggregate phần law/compliance còn lại thay vì crash toàn bộ request.

## 6. So sánh 5 stages

| Stage | Pattern | Điểm chính | Hạn chế |
|---|---|---|---|
| 1 | Direct LLM | Đơn giản, gọi LLM trực tiếp | Không tool, không retrieval |
| 2 | LLM + Tools | Có RAG/tools, grounded hơn | Tool loop tự viết, thường một vòng |
| 3 | ReAct Agent | Agent tự gọi tool nhiều bước | Một agent xử lý nhiều domain |
| 4 | Multi-Agent in-process | Specialist agents, chạy song song | Vẫn là một process, coupling cao |
| 5 | Distributed A2A | Mỗi agent là service riêng, discovery động | Khó debug hơn, latency cao hơn |

## 7. Câu hỏi ôn tập

### Khi nào nên dùng single agent thay vì multi-agent?

Dùng single agent khi bài toán nhỏ, ít domain, ít yêu cầu scale độc lập, và logic orchestration chưa phức tạp. Multi-agent phù hợp hơn khi cần chuyên môn hóa, phân quyền, scale riêng từng năng lực, hoặc chạy song song nhiều nhánh phân tích.

### A2A hơn REST/gRPC thông thường ở điểm nào?

REST/gRPC chỉ định nghĩa cách service gọi nhau. A2A bổ sung abstraction dành cho agent: Agent Card, task/message format, artifact, context, và cách agent tự mô tả capability. Nhờ đó agent có thể discover và tương tác linh hoạt hơn.

### Làm sao prevent infinite delegation loops?

Repo này dùng `delegation_depth` trong metadata và `MAX_DELEGATION_DEPTH = 3`. Mỗi lần delegate tăng depth lên 1. Khi đạt giới hạn, agent không delegate tiếp.

### Tại sao cần Registry?

Registry giúp agent discovery động. Nếu hardcode URL, mỗi lần đổi port, scale service, hoặc thay agent implementation đều phải sửa code. Registry tách logic "cần task gì" khỏi "agent nào đang phục vụ task đó".

## 8. Bài cộng điểm: Latency

### Cách đo latency

Có thể đo quanh lệnh gọi client:

```bash
time uv run python test_client.py
```

Hoặc sửa `test_client.py`:

```python
from time import perf_counter

start = perf_counter()
response = await client.send_message(request)
elapsed = perf_counter() - start
print(f"Latency: {elapsed:.2f}s")
```

### Vì sao latency cao?

Stage 5 gọi nhiều LLM/service:

1. Customer Agent dùng LLM để quyết định delegate
2. Law Agent gọi LLM để phân tích luật
3. Law Agent gọi LLM để routing
4. Tax Agent gọi LLM
5. Compliance Agent gọi LLM
6. Law Agent gọi LLM để aggregate

Dù Tax và Compliance chạy song song, tổng latency vẫn gồm nhiều hop HTTP và nhiều lần LLM inference.

### Phương án giảm latency

Các hướng khả thi:

- Thay LLM routing bằng keyword routing cho câu hỏi rõ ràng.
- Cho Customer Agent delegate trực tiếp mọi câu hỏi pháp lý, giảm một bước reasoning không cần thiết.
- Dùng model nhanh hơn cho routing/aggregation, model mạnh hơn cho phân tích chính.
- Cache `discover(task)` để giảm call Registry lặp lại.
- Giữ parallel delegation ở Law Agent bằng `Send`.
- Giới hạn prompt và `max_tokens` để response ngắn hơn.

Một thay đổi dễ demo:

```python
question_lower = state["question"].lower()
needs_tax = any(kw in question_lower for kw in ["tax", "irs", "thuế", "offshore"])
needs_compliance = any(kw in question_lower for kw in ["compliance", "sec", "regulation", "sox", "gdpr"])
```

Thay routing LLM trong `law_agent/graph.py` bằng keyword routing sẽ bỏ được một LLM call. Sau đó đo lại bằng `perf_counter()` trong `test_client.py` và so sánh trước/sau.

## 9. Checklist hoàn thành lab

- Stage 1 chạy được direct LLM.
- Stage 2 hiểu `LEGAL_KNOWLEDGE`, `@tool`, `bind_tools`, và tool-call loop.
- Stage 3 hiểu `create_react_agent` và ReAct loop.
- Stage 4 hiểu `StateGraph`, `Send`, specialist agents, và aggregate.
- Stage 5 chạy được Registry + 4 agents qua A2A.
- Theo dõi được `trace_id` trong logs.
- Hiểu vai trò của `context_id`, `delegation_depth`, Registry, Agent Card, và A2A message.

