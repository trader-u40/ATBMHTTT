# P3/P4 chạy lại sau khi sửa pad_id

Sửa: lúc train detector, `pad_id` là id của `None` (OOV) trong khi val/test và `XSSDetectEnv` dùng `<pad>`. Giờ cả ba đều dùng `<pad>`.

Lệnh:
```
python src/run_p3.py --phase all --runs-dir runs/p3_padfix --ppo-seeds 42,43,44,46,47 --timesteps 20000
python src/run_p4.py --phase all --algo ppo --seeds 42,43,44,46,47 --timesteps 20000 --p3-dir runs/p3_padfix --runs-dir runs/p4_padfix
python src/wilcoxon_p3p4.py --p3 runs/p3_padfix/p3_eval.json --p4 runs/p4_padfix/p4_eval.json --out data/wilcoxon_p3p4_padfix.json
```
Dataset: vẫn là `mereani_exec_oracle.csv` cũ (chưa chuyển sang bản lọc bằng oracle mới).

## Detector (test n=2168)
| | Acc | P | R | F1 |
|---|---:|---:|---:|---:|
| cũ | 0.991 | 0.990 | 0.992 | 0.991 |
| padfix | 0.993 | 0.992 | 0.994 | 0.993 |

## Agent (400 payload test, 20k bước)
| Seed | P3 escape | P3 ER | P3 escape còn chạy JS | P4 escape (= TASR, RR(E)=0) | P4 ER |
|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 3.00% | 3 | 19 | 4.75% |
| 43 | 2 | 0.50% | 0 | 4 | 1.00% |
| 44 | 19 | 4.75% | 12 | 4 | 1.00% |
| 46 | 2 | 0.50% | 2 | 2 | 0.50% |
| 47 | 2 | 0.50% | 2 | 3 | 0.75% |
| **mean** | | **1.85%** | **0.95%** | | **1.60%** |

Số cũ (3 seed, pad_id lỗi): P3 ER 0.75%, P4 ER/TASR 0.25% (= sàn FN detector).

## Đọc kết quả
- Vẫn **không** tái hiện được ER ~99% của Pasini.
- P4 không còn dính sàn FN 0.25%: oracle trong reward giúp agent tìm được biến thể vừa né vừa chạy JS (TASR 1.6% > 0.95% của P3). Kết luận cũ "P4 ≤ P3 trên mọi seed" không còn đúng.
- Wilcoxon ER P3 vs P4: n_nonzero=4, p=0.875 (p nhỏ nhất đạt được với n=4 là 0.125). Chỉ mang tính mô tả; cần ≥6 seed khác 0 mới có thể đạt p<0.05.
- Phương sai giữa seed lớn (0.5%–4.75%); nên báo theo từng seed + Wilson CI (`data/wilcoxon_p3p4_padfix.json`).
