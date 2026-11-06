# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV** 2A202603436
- **Họ và tên** Phạm Nguyên Vũ
- **SEED** 2423

## 1. Bằng chứng

| Tệp | Đã có trong thư mục nộp? |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [lab1_report.json](lab1_report.json) — kết quả từ ô báo cáo cuối | ☐ |
| [hand_calc.jpg](hand_calc.jpg) — ảnh viết tay có các bước tính | ☐ |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin (ước tính) |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,894 | 0,919 | 0,077 | 0 |
| BoW + MLP | 0,931 | 0,947 | 0,306 | 0 |
| MiniLM + kNN | 0,948 | 0,974 | 60,140 | 0 |
| MiniLM + LogReg | 0,962 | 0,974 | 39,615 | 0 |
| MiniLM + MLP head | 0,952 | 0,974 | 38,345 | 0 |
| LLM nhỏ zero-shot | — | 0,000 | 1.088,440 | 0,014 |
| LLM lớn zero-shot | — | 0,000 | 1.083,606 | 0,144 |
| LLM nhỏ few-shot | — | 0,000 | 1.887,320 | 0,040 |

Ở cả ba lượt LLM trên SUB 40, notebook ghi nhận 40/40 phản hồi không hợp lệ và gán ham cho các phản hồi này. F1 bằng 0 phản ánh lỗi nhận nhãn trong lần chạy.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán | Kết quả và giải thích |
| --- | --- | --- |
| 1.B — Learning rate | Với lr = 50, loss có thể dao động do bước cập nhật lớn. | Loss tăng vọt lên khoảng 2,6 ở đầu rồi ổn định, kết thúc ở 0,005. Bước lớn có thể vượt qua vùng loss thấp lúc đầu, nhưng những lần cập nhật sau vẫn làm train loss giảm. |
| 2.B — Bỏ ReLU | F1 sẽ giảm vì mạng chỉ còn phép biến đổi tuyến tính. | Cả hai bản đều đạt F1 0,931 ở độ chính xác hiển thị. Với BoW và tập test này, lớp phi tuyến chưa cho thấy lợi ích qua F1. |
| 3.B — Tin ngụy trang | Encoder + LogReg sẽ đúng nhiều hơn BoW + LogReg. | Cả hai đúng 5/6. BoW vẫn nhận ra các từ gợi ý như “your” và “mobile” trong tin mời nhận kỳ nghỉ. Cả hai bỏ lọt tin giả thông báo điểm thưởng sắp hết hạn. |

## 4. Phân tích 50 tin khó

BoW + MLP giảm F1 nhiều nhất, từ 0,947 trên SUB 40 xuống 0,368, và sai cả 6 tin né từ khóa. MiniLM + kNN đạt F1 cao nhất trên bộ khó (0,679). Spam giọng tự nhiên vẫn khó với cả ba mô hình MiniLM, mỗi mô hình chỉ đúng 3/10 tin. Tập test thường có thể chứa các tin gần trùng với tập train, trong khi bộ 50 tin chủ động tập trung vào các tình huống dễ gây nhầm.

- Tin #29 dùng “C0ngr4tulat10ns” để né từ khóa. BoW xem cách viết này là một token khác với “congratulations”, nên có thể bỏ mất tín hiệu spam.
- Tin #16 mở đầu như lời hỏi thăm của bạn cũ rồi mới mời mua mỹ phẩm. Cả năm mô hình BoW và MiniLM đều bỏ lọt tin này.

LLM zero-shot và LLM dùng prompt phòng thủ đều đúng 1/6 tin ở nhóm prompt injection. Kết quả của hai lượt tương đương việc gán ham cho mọi tin, nên chưa đánh giá được hiệu quả của prompt phòng thủ.

## 5. Thử thách cá nhân

Đề #6 yêu cầu so sánh bản trả một từ với bản JSON có nhãn và lý do. Tôi dùng cùng model nhỏ, cùng 10 tin từ SUB và giới hạn 128 token cho cả hai bản. Mỗi bản dùng hàm đọc nhãn tương ứng.

| Định dạng | Hợp lệ | Đúng | Token vào (ước tính) | Token ra (ước tính) | Chi phí (USD, ước tính) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Một từ | 10/10 | 8/10 | 541,3 | 13,3 | 0,000061 |
| JSON | 10/10 | 9/10 | 754,7 | 160,0 | 0,000155 |

JSON tốn khoảng 2,5 lần chi phí và đoán đúng thêm một tin. Mẫu 10 tin còn nhỏ để kết luận JSON phân loại tốt hơn. Tôi sẽ dùng bản một từ khi chỉ cần nhãn, và dùng JSON khi người kiểm tra cần xem lý do gắn cờ.

## 6. Phương án cho sản phẩm thật

BoW + LogReg, BoW + MLP, MiniLM + LogReg và MiniLM + MLP head đạt p95 dưới 50 ms/tin. Tôi chọn BoW + MLP vì có F1 test 0,931, p95 0,306 ms, thời gian huấn luyện 12,074 giây và thông lượng cả lô đo được khoảng 27.633 tin/s, so với nhu cầu trung bình khoảng 116 tin/s. MiniLM + LogReg đạt F1 cao hơn (0,962) nhưng trong lần đo này xử lý khoảng 39 tin/s.

Có thể dùng BoW + MLP xử lý tin có điểm spam dưới 0,3 hoặc trên 0,7, rồi chuyển các tin ở giữa sang MiniLM + LogReg để xét thêm ngữ nghĩa. Cần đo p95 và thông lượng của cả hệ thống sau khi kết hợp.

Nếu dùng LLM cho 300 triệu tin trong tháng 30 ngày, đơn giá trong bảng cho mức ước tính lần lượt là 4.200 USD cho model nhỏ zero-shot, 43.200 USD cho model lớn zero-shot và 12.000 USD cho model nhỏ few-shot. Các mức này dựa trên lần gọi có phản hồi lỗi và có thể thay đổi khi nhận được nhãn hợp lệ.

## 7. AI log và điều đã học

| Phần | Tôi hỏi AI gì | AI hỗ trợ gì | Tôi đã sửa hoặc kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0 và tính tay | Tin trùng ảnh hưởng điểm test thế nào? Dùng công thức gì cho năm câu tính tay? | Giải thích rò rỉ dữ liệu và trình bày công thức. | Đối chiếu với 4.516 tin ham trên tổng 5.169 tin sau khi bỏ trùng. |
| Phần 1 — ML | Cách viết BoW, sigmoid, LogReg và metrics? Đọc biểu đồ loss thế nào? | Giải thích khái niệm và cung cấp đoạn mã cho các TODO. | Sửa lỗi gõ vocal thành vocab, chạy kiểm tra và đối chiếu nhận xét với biểu đồ learning rate. |
| Phần 2 — DL | Cách xây MLP, huấn luyện và nhận biết overfitting? | Giải thích ReLU, vòng lặp huấn luyện và cung cấp đoạn mã cho các TODO. | Chạy kiểm tra 2.2, đối chiếu validation loss thấp nhất ở epoch 8 và F1 khi bỏ ReLU. |
| Phần 3 — Embedding | Cách tính cosine, lấy top-k và bỏ phiếu kNN? | Giải thích các bước, cung cấp đoạn mã và hướng dẫn sửa lỗi thụt lề. | Chạy kiểm tra cosine và top-k, sau đó đối chiếu kết quả tìm kiếm và thí nghiệm tin ngụy trang. |
| Phần 4 — LLM | các hàm và yêu cầu của notebook. Vì sao gọi thật lại báo invalid = 40? | hướng dẫn và hướng dẫn xem phản hồi thô. | Rà soát định dạng prompt và cách đọc “Label: spam”, “Not spam” theo kiểm tra của notebook. Xác nhận phản hồi đầu tiên của cả hai model ở 4.A là chuỗi rỗng. |
| Phần 5 — Tổng hợp | Đọc biểu đồ Pareto, phân tích lỗi và chọn mô hình như thế nào? | Giúp diễn giải số liệu, soạn nhận xét và đề xuất router. | Đối chiếu với bảng F1, p95 và kết quả 50 tin khó của lần chạy. |


Tôi hiểu vì sao một bộ lọc có F1 cao trên tập test thường vẫn có thể bỏ lọt nhiều tin cố tình né từ khóa. Tôi cũng hiểu vì sao cần kiểm tra số phản hồi hợp lệ trước khi dùng F1 để so sánh các lượt gọi LLM.
