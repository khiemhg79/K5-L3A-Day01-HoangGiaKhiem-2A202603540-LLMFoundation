# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.920   0.947
2. BoW + MLP (deep learning)     0.952   0.974
3a. Encoder (MiniLM) + kNN       0.917   0.974
3b. Encoder (MiniLM) + LogReg    0.935   1.000
3c. Encoder (MiniLM) + MLP head  0.941   1.000
4a. LLM nhỏ zero-shot            0.976   0.976

## 3. Dự đoán và thực tế

1. **`llm_models`**
   - Mình đoán: gpt-6.1-terra nhỉnh hơn một chút (F1 cao hơn ít), đắt gấp ~10 lần, ngang với 3b (Encoder + LogReg).
   - Kết quả: Chưa có vì bị limit

2. **`ml_lr` (lr = 0.001 và lr = 50)**
   - predict("ml_lr", "lr=0.001 thì quá nhỏ học chậm, lr=50 thì quá lớn học k sâu ")
   - Kết quả: loss cuối lr = 0.001 là 0.625, lr = 50 là 0.007.
   - Vì sao lệch: lr = 50 vẫn ra loss thấp vì dữ liệu BoW gần như tách được tuyến tính, sigmoid có clip nên không nổ; nhưng đường loss lởm chởm.

3. **`dl_relu` (bỏ ReLU)**
   - predict("dl_relu", "2 Linear gộp thành 1, F1 thay đổi ít ")
   - Không ReLU — TEST F1: 0.947
   - Vì sao lệch: 

## 4. Phân tích 50 case khó


## 5. Thử thách cá nhân

Đề #2: giảm vocab xuống 100 từ và tăng lên 5000 từ, train lại BoW + LogReg, báo F1 trên tập test.
Cách làm: mỗi kích thước vocab trong {100, 2000, 5000} mình chạy build_vocab(texts_tr, max_size). Vocab chỉ xây từ tập train để không rò rỉ dữ liệu test. Sau đó vectorize cả train và test bằng cùng vocab đó, train train_logreg, đánh giá trên test. Mình đo thêm % tin test bị thành vector rỗng.
vocab	F1	    Precision	Recall	% tin rỗng
100	    0.852	0.973	    0.758	2.19%
2000	0.920	1.000	    0.853	0.13%
5000	0.914	1.000	    0.842	0.00%

## 6. Phương án cho sản phẩm thật

BoW + LogReg hoặc Encoder MiniLM + LogReg
F1 trên test là 0.92
chi phí rẻ hơn nhưng chất lượng sẽ k bằng những con đắt tiền

## 7. AI log và điều đã học

Phần	Bạn hỏi AI gì (prompt)	AI trả lời/viết gì	Bạn đã sửa / kiểm chứng gì
4 – Cấu hình	Cách dùng OpenAI, cách lấy API key; provider nào miễn phí	Hướng dẫn lấy key OpenAI (cần nạp ≥ $5); gợi ý Gemini (AI Studio) và Groq miễn phí	Chọn Gemini, tạo key trên aistudio.google.com, thêm vào Colab Secrets, chạy ô cấu hình thấy in đúng provider/model
4.1 – call_openai	Viết TODO 4.1 để chạy được với Gemini	Code gọi chat.completions.create với **provider_kwargs(max_tokens), đo latency bằng time.perf_counter()	Ô test giả lập báo lỗi vì AI dùng .strip() → bỏ .strip(); chạy lại test pass
4.2 – Prompt	Viết build_classification_prompt (zero-/few-shot)	Prompt mô tả nhiệm vụ, yêu cầu trả lời 1 từ, chèn ví dụ few-shot, kết thúc bằng Label:	In thử prompt few-shot, kiểm tra đúng định dạng Message: """...""" / Label:
4.3 – parse_label	Viết hàm đọc nhãn từ câu trả lời	Regex \b, xử lý 'not spam'/'non-spam' trước, trả None khi không rõ	AI test 15 câu trả lời mẫu (Spam., **spam**, hamburger…); tôi kiểm tra lại kết quả
4.4 – llm_classify_batch	Viết hàm phân loại cả loạt + đo chi phí	Vòng lặp gọi call_llm qua retry_with_backoff, token ≈ số từ / 0.75, nghỉ 4s cho gói free	Chạy ô kiểm tra giả lập 4.1 + 4.4 → pass
4.A / 4.B / Pareto / Hard eval	Gợi ý dự đoán cho các ô predict(...)	Dự đoán F1, tỉ lệ chi phí (terra đắt 10× luna; few-shot ~2.5×), cách nằm trên Pareto	Tự tính lại tỉ lệ giá từ PRICE_PER_1M; so dự đoán với kết quả thật
5 – AI log	Tạo bảng AI log gồm 4 cột: Phần · Bạn hỏi gì · AI trả lời gì · Bạn đã sửa/kiểm chứng gì; sau đó yêu cầu làm bảng đẹp hơn	Tổng hợp lại toàn bộ các lần hỏi AI trong buổi lab thành bảng; viết code pandas Styler (màu tiêu đề, dòng xen kẽ, tự xuống dòng)	Đối chiếu từng dòng với những gì mình đã làm thật; tự điền các ô còn để trống; sửa nội dung cho đúng

- Mỗi mô hình sẽ có 1 điểm yếu khác nhau. BoW chỉ đếm nên sẽ bị lừa bởi các từ biến dạng như "C0ngr4tul4t10ns" và tin ham có từ giống spam.