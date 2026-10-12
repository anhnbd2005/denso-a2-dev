# DENSO A2 — Master Know-How Accumulator

Ứng dụng mẫu cho challenge **DENSO A2 — Accumulate Master Know-How for AI to Learn**. Dự án số hóa ghi chú chuyên gia thành tri thức có cấu trúc, đưa vào kho của phiên làm việc, tra cứu bằng Hybrid retrieval, tạo câu trả lời có căn cứ bằng LLM và đánh giá retrieval trên benchmark do đội xây dựng.

> Toàn bộ dữ liệu đi kèm là synthetic/mock. Đây không phải dữ liệu vận hành, benchmark chính thức hay kết quả xác nhận tại nhà máy DENSO.

## 1. Bài toán và phạm vi

Kinh nghiệm hiện trường thường nằm trong ghi chú và trao đổi miệng, khó tìm lại và dễ mất khi chuyển giao. DENSO A2 triển khai vòng lặp hiện tại:

```text
EXPERT NOTES → LLM STRUCTURED EXTRACTION → KNOWLEDGE BASE
                                              ↓
                 ANSWER + SOURCES ← GROUNDED LLM GENERATION
                                              ↑
                         HYBRID RETRIEVAL → OOS GUARDRAIL
```

Nguyên mẫu không bao gồm Interview, Governance, hàng đợi phê duyệt hoặc workflow review.

## 2. Kiến trúc hiện tại

- **Capture:** LLM OpenAI-compatible bóc tách ghi chú tự do thành bản nháp `KnowledgeItem`; người dùng kiểm tra/sửa rồi mới lưu. CSV/XLSX dùng bộ nhập có validation.
- **Knowledge Base:** 30 thẻ synthetic frozen (5 thẻ nền + 25 thẻ mở rộng) được nạp đầu phiên; bản ghi capture/import bổ sung trong `st.session_state`.
- **Hybrid retrieval:** BM25 và semantic vector retrieval, hợp nhất bằng Reciprocal Rank Fusion (RRF), sau đó áp dụng cổng out-of-scope (OOS) trước generation.
- **Grounded generation:** câu hỏi gốc và `KnowledgeItem` được retrieve được gửi tới cùng OpenAI-compatible client. LLM chỉ được phép dùng evidence đã cung cấp; source ID được kiểm tra và gắn bằng chương trình.
- **Evaluation:** evaluator chạy riêng benchmark phát triển 10 câu và held-out nội bộ 38 câu trên corpus 30 thẻ, không trộn hai loại bằng chứng.

LLM thật được dùng ở hai điểm: trích xuất có cấu trúc trong **Thu thập** và tạo câu trả lời có căn cứ trong **Tra cứu**. Embedding model của semantic retrieval là thành phần riêng, không phải generation LLM.

## 3. Bốn trang chính

1. **Tra cứu** — tìm evidence bằng BM25 + semantic + RRF, từ chối OOS trước LLM, rồi tạo câu trả lời tiếng Việt kèm nguồn đã xác thực.
2. **Thu thập** — trích xuất từ ghi chú bằng LLM hoặc nhập CSV/XLSX; chỉnh sửa trước khi lưu.
3. **Kho tri thức** — duyệt/lọc bản ghi, xem metadata và Knowledge Map của bản ghi được chọn.
4. **Báo cáo** — phân biệt snapshot lịch sử với lần đánh giá/đo retrieval latency mới.

Mục **Cài đặt nâng cao** chỉ hiển thị cấu hình kết nối và health check; API key không được hiển thị.

## 4. Tính năng cốt lõi

- Schema Pydantic và `extra_metadata` mở rộng cho dữ liệu nhà máy.
- Import CSV/XLSX, alias cột, phát hiện ID trùng và validation theo dòng.
- Dynamic reindex cho BM25 và Chroma vector index trong phiên.
- Grounded LLM generation với context giới hạn, nguồn do chương trình xác thực và không fallback im lặng.
- Knowledge Map HTML/CSS có escape dữ liệu người dùng và hiển thị đầy đủ nội dung.
- OOS trả về thông báo không đủ tri thức, không tạo hướng dẫn kỹ thuật thay thế.
- Report chạy evaluator thật và đo retrieval bằng `time.perf_counter()`.

## 5. Cấu trúc dự án

```text
app.py                  # Giao diện Streamlit bốn trang
schema.py               # KnowledgeItem và schema đánh giá
workflow_capture.py     # Trích xuất ghi chú/transcript
llm_client.py           # Client OpenAI-compatible
batch_importer.py       # Import CSV/XLSX
hybrid_retriever.py     # BM25 + vector + RRF + OOS
retriever.py            # Lexical baseline
rag_engine.py           # Engine RAG thử nghiệm riêng
assistant.py            # Hybrid RAG orchestration, grounded prompt và source validation
knowledge_graph.py      # Knowledge Map HTML/CSS
reporting.py            # Benchmark phát triển, held-out nội bộ và retrieval latency
operational_metrics.py  # Tính toán scenario có nhãn bằng chứng
ui_components.py        # Sidebar, hero và style dùng chung
run_local.py            # Nạp .env cục bộ rồi chạy Streamlit
data/                   # Fixture synthetic và benchmark của đội
tests/                  # Offline regression và validation scripts
tests/real/             # Test API thật, chỉ chạy chủ động khi có credential
```

## 6. Yêu cầu môi trường

Môi trường hiện tại đã chạy regression với **Python 3.13.5 trên Windows**. Dùng `.venv` riêng; nếu dùng phiên bản Python khác, cần chạy lại toàn bộ test. Các package chính nằm trong `requirements.txt`: Streamlit, Pydantic, OpenAI SDK, ChromaDB, sentence-transformers, pandas, NumPy và openpyxl.

Semantic model có thể cần được tải ở lần chạy đầu nếu chưa có trong cache. Vì vậy không nên mô tả ứng dụng là hoàn toàn offline; LLM capture luôn phụ thuộc endpoint do thành viên cấu hình.

## 7. Cài đặt

```powershell
git clone https://github.com/<OWNER>/<PRIVATE_REPOSITORY>.git
cd <PRIVATE_REPOSITORY>
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Thành viên phải được mời và chấp nhận quyền truy cập repository Private trước khi clone.

## 8. Cấu hình `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

```dotenv
OPENAI_COMPAT_ENABLED=true
OPENAI_COMPAT_BASE_URL=https://YOUR_REACHABLE_PROVIDER/v1
OPENAI_COMPAT_MODEL=deepseek-v4-flash
OPENAI_COMPAT_API_KEY=YOUR_PRIVATE_API_KEY
```

`run_local.py` ánh xạ `OPENAI_COMPAT_BASE_URL` → `OPENAI_BASE_URL` và `OPENAI_COMPAT_API_KEY` → `OPENAI_API_KEY` cho tiến trình con. Giá trị trong `.env` ghi đè biến môi trường kế thừa; không có giá trị mặc định là endpoint riêng của chủ dự án.

Để mở UI mà không dùng LLM, đặt `OPENAI_COMPAT_ENABLED=false` hoặc chưa tạo `.env`. Khi tắt, launcher loại bỏ `OPENAI_BASE_URL` và `OPENAI_API_KEY` kế thừa khỏi tiến trình con. Kho tri thức và Report vẫn xem được; thao tác extraction hoặc grounded generation sẽ báo lỗi cấu hình rõ ràng và không giả làm kết quả LLM thành công.

## 9. Chạy ứng dụng

```powershell
.\.venv\Scripts\python.exe run_local.py
```

Có thể truyền tùy chọn Streamlit, ví dụ khi cổng 8501 đang được dùng:

```powershell
.\.venv\Scripts\python.exe run_local.py --server.port 8502
```

Launcher dùng chính `sys.executable`, đặt working directory về thư mục dự án, không cài package, không sửa `.env` và không in credential.

## 10. Kiểm thử

Test được tổ chức thành các module standalone. Chạy lệnh từ project root để imports và đường dẫn fixture được resolve nhất quán. `pytest` có thể chưa được cài trong `.venv`.

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"
Get-ChildItem tests\test_*.py | ForEach-Object {
    .\.venv\Scripts\python.exe -m "tests.$($_.BaseName)"
}
.\.venv\Scripts\python.exe -m tests.validate_data
.\.venv\Scripts\python.exe -m tests.validate_test_data
.\.venv\Scripts\python.exe evaluate_usefulness.py
.\.venv\Scripts\python.exe evaluate_rag.py
.\.venv\Scripts\python.exe evaluate_hybrid.py
.\.venv\Scripts\python.exe evaluate_heldout.py
.\.venv\Scripts\python.exe -m pip check
```

Hai test mạng thật được tách khỏi suite offline và chỉ chạy chủ động sau khi nạp endpoint/model/key hợp lệ:

```powershell
.\.venv\Scripts\python.exe -m tests.real.test_openai_compatible
.\.venv\Scripts\python.exe -m tests.real.test_real_grounded_rag
```

Không dùng dummy credential để tuyên bố REAL PASS. `tests/test_grounded_assistant.py` là mocked contract test, không phải bằng chứng inference thật.

## 11. Phương pháp benchmark

`data/test_benchmark.json` gồm 10 câu hỏi do đội xây dựng. Evaluator chấm bốn thành phần: retrieval, root cause, action và safety; một câu đạt khi điểm trung bình ≥ 0,75. Kết quả lịch sử đã xác thực trên benchmark phát triển:

| Chế độ | Kết quả lịch sử |
| --- | --- |
| Lexical | 9/10 (90%) |
| Semantic | 8/10 (80%) |
| Hybrid | 10/10 (100%) — đã hiệu chỉnh trên chính bộ phát triển |

Các số này đo hợp đồng retrieval/evaluator đã thiết lập, không phải accuracy của câu trả lời LLM. Chúng cũng không phải benchmark chính thức của DENSO, production accuracy hay bằng chứng tổng quát hóa.

Đánh giá held-out nội bộ dùng `data/mock_knowledge.json` cùng `data/extended_knowledge.json`, tổng cộng 30 thẻ synthetic. `data/heldout_benchmark.json` gồm 30 câu trong phạm vi và 8 câu ngoài phạm vi; ngưỡng OOS 0,55 được giữ nguyên, không hiệu chỉnh lại trên bộ này. Chạy bằng:

```powershell
.\.venv\Scripts\python.exe evaluate_heldout.py
```

Kết quả đã tái lập: **26/30 top-1 trong phạm vi, 8/8 OOS, tổng 34/38**. Bốn trường hợp chưa đạt là `HO-009`, `HO-014`, `HO-017` và `HO-021`. Đây là held-out do đội xây dựng trên dữ liệu synthetic, không phải đánh giá độc lập hoặc dữ liệu sản xuất của DENSO.

## 12. Báo cáo và giả định vận hành

`reporting.py` giữ benchmark phát triển 10 câu trên 5 thẻ nền để so sánh với snapshot lịch sử, đồng thời cung cấp benchmark held-out 38 câu trên đầy đủ 30 thẻ synthetic. Item capture/import trong session không tham gia các benchmark. Snapshot lịch sử trong `data/historical_benchmark_snapshot.json` được hiển thị tách biệt với kết quả chạy mới.

Retrieval latency được đo lại bằng `time.perf_counter()` và không bao gồm LLM latency, thời gian con người chẩn đoán hoặc hiệu quả vận hành nhà máy. Ví dụ **45 phút / 1,5 phút = 30×** được ghi nhãn **GIẢ ĐỊNH**; đó không phải baseline hay cải thiện thực tế do DENSO xác nhận.

## 13. Cộng tác nhóm

- Mỗi thành viên dùng `.env` riêng và endpoint/model/key mà máy mình truy cập được.
- Không chia sẻ key qua Git, README, issue hoặc pull request.
- Không commit `.venv`, cache model, Chroma storage, log hoặc browser profile.
- Trước khi merge: chạy regression scripts, frozen hash check, `pip check` và UI smoke test.

## 14. Bảo mật và quyền riêng tư

`.env`, `.env.*` và `.streamlit/secrets.toml` bị ignore; chỉ `.env.example` placeholder được phép track. Dự án không yêu cầu đường dẫn `F:`/`G:` hoặc proxy riêng của chủ dự án để vận hành thông thường. Không đưa dữ liệu nhà máy, credential, token hoặc log chứa bí mật vào repository.

## 15. Giới hạn đã biết

- Bản ghi thêm mới chỉ sống trong Streamlit session, chưa có persistence production.
- Semantic embedding model/cache và generation LLM endpoint phụ thuộc môi trường từng máy.
- Held-out hiện là benchmark nội bộ synthetic; OOS threshold và kết quả vẫn cần được xác thực độc lập trên dữ liệu mới hoặc dữ liệu đã được DENSO phê duyệt.
- Không cam kết tương thích với mọi OpenAI-compatible provider.
- Grounded prompt và numerical guard giảm rủi ro nhưng không bảo đảm tuyệt đối không hallucination.
- Chưa xác minh triển khai on-premise hoặc hiệu quả vận hành thực tế.

## 16. Công bố dữ liệu synthetic/mock

Các KnowledgeItem, log nhập mẫu, câu hỏi benchmark và scenario vận hành đi kèm đều phục vụ phát triển/demo. Không được diễn giải chúng như hồ sơ chuyên gia, số liệu nhà máy hoặc kết quả chính thức của DENSO.
