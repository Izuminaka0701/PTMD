# PTMD — Dynamic API Resolution Detection via MCAFF

Đồ án môn **Phân tích Mã độc**: phát hiện và phân loại kỹ thuật **Dynamic API Resolution** (G1/G2/G3) kết hợp **API Call Behavior** bằng mô hình **MCAFF (Multi-Head Cross-Attention Feature Fusion)**.

> **Cảnh báo:** Dự án này dùng malware thật cho mục đích nghiên cứu. Chỉ chạy trên **máy ảo cô lập**, không kết nối mạng production, không commit sample vào git.

---

## Kiến trúc MCAFF — Multi-Head Cross-Attention Feature Fusion

### Ý tưởng chính (Novelty)

Thay vì dùng concat/gated fusion thông thường hay GNN phức tạp, MCAFF xem **mỗi nhóm feature** (Static PE, Behavior, Graph-derived) như **một "token"**, rồi áp dụng **Multi-Head Cross-Attention** để các nhóm tự động học được **mối quan hệ tương tác** lẫn nhau:

- Static features (có LoadLibrary trong IAT) **cross-attend** với Behavior (có chain LoadLib→GetProcAddr) → model tự học 2 signal này reinforce nhau → **G1**
- Behavior (NtQuery patterns cao) **cross-attend** với Graph (ít resolve edges) → model nhận ra thiếu standard resolver → **G2**
- Graph (nhiều hash nodes) **cross-attend** với Behavior (hash constants trong args) → **G3**

### Sơ đồ kiến trúc

```
                        ┌─── Linear Proj ─── token₁ [d_model] ───┐
Static PE [64]  ────────┘                                         │
                                                                  ├── Multi-Head Cross-Attention
Behavior [15]  ─── Linear Proj ─── token₂ [d_model] ─────────────┤    (num_heads=4, layers=2)
                                                                  │
Graph-derived [5] ─ Linear Proj ─── token₃ [d_model] ────────────┘
                                         │
                                         ├── + Group Type Embeddings
                                         │
                                         ▼
                                Residual + LayerNorm
                                         │
                                         ▼
                               FFN (Feed-Forward Network)
                                         │
                                         ▼
                              Mean Pooling over tokens
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                    G1/G2/G3 Group Head    Family Head (auxiliary)
```

### 3 nguồn features (1 luồng dữ liệu)

| Nhóm feature | Nguồn | Dims | Mô tả |
|--------------|-------|------|-------|
| **Static PE** | PE file (IAT) | 64 | Import features, entropy, section info, G1/G2/G3 API indicators |
| **Behavior** | Sandbox API log | 15 | g1_chain_ratio, g2_peb_ratio, g3_hash_arg_ratio, pattern flags, derived |
| **Graph-derived** | Static + Dynamic | 5 | num_modules, num_apis, num_hash_nodes, num_resolves, num_hash_edges |

> **So với 3-stream approach (GNN/LSTM):** MCAFF gộp thành 1 luồng duy nhất nhưng vẫn mô hình hóa tương tác liên nhóm qua cross-attention, chạy hoàn toàn trên CPU.

### Phân loại kỹ thuật (MITRE)

| Nhóm | Kỹ thuật | Ví dụ Family |
|------|----------|--------------|
| **G1** | LoadLibrary + GetProcAddress | CHIMNEYSWEEP, Raccoon Stealer, WannaCry |
| **G2** | PEB Traversal, không dùng Win32 resolve | PlugX, Cobalt Strike, Lazarus shellcode |
| **G3** | Hash/XOR encrypted API names | TONESHELL (DJB2), CLAIMLOADER (XOR), SplatDropper (seed 131313) |

---

## Cấu trúc thư mục

```
PTMD/
├── config/default.yaml          # Hyperparameters (MCAFF)
├── data/
│   ├── metadata/
│   │   ├── mitre_families.json  # 17 MITRE families + G1/G2/G3 mapping
│   │   └── sample_manifest.json # Danh sách sample (tạo sau bước 1)
│   ├── samples/                 # PE binaries (KHÔNG commit)
│   ├── sandbox_logs/            # API call logs từ dynamic analysis
│   └── processed/               # Features đã trích xuất (unified)
├── src/
│   ├── features/
│   │   ├── static_pe.py         # Static PE/IAT extractor
│   │   ├── api_behavior.py      # Behavior statistics extractor
│   │   ├── api_graph.py         # Graph topology extractor
│   │   └── unified_features.py  # Unified Feature Extractor (gộp 3 nguồn)
│   ├── models/mcaff_model.py    # MCAFF model (Cross-Attention)
│   ├── datasets/                # PyTorch dataset
│   └── training/                # Trainer + metrics
└── scripts/
    ├── check_api.py              # Kiểm tra MalwareBazaar API key
    ├── 00_download_samples.py    # Tải sample từ manifest
    ├── 01_fetch_sample_metadata.py
    ├── 02_extract_features.py    # Trích xuất unified features
    ├── 03_train_model.py         # Train MCAFF
    ├── 04_evaluate.py            # Evaluate + attention visualization
    ├── 05_predict.py             # Predict với attention interpretability
    ├── run_pipeline.py           # Chạy full pipeline (CPU)
    └── import_malware_api_dataset.py
```

---

## Hướng dẫn chạy trên Máy ảo (VM)

### 1. Yêu cầu VM

- Windows 10/11 hoặc Linux (Ubuntu 22.04+)
- Python 3.10+
- **Không cần GPU** — MCAFF chạy hoàn toàn trên CPU
- **Tắt shared folders** hoặc chỉ mount one-way
- Snapshot VM trước khi phân tích malware

### 2. Cài đặt

```bash
cd PTMD
python -m venv .venv
.venv\Scripts\activate        # Windows

pip install -r requirements.txt

# Cấu hình API key MalwareBazaar
copy .env.example .env
# Sửa .env: ABUSECH_API_KEY=<key từ https://auth.abuse.ch/>

python scripts/check_api.py   # Kiểm tra kết nối API
```

> **Lưu ý:** Không cần cài `torch-geometric` hay CUDA toolkit — MCAFF chỉ dùng `torch` (CPU).

### 3. Pipeline đầy đủ

```bash
# Chạy tất cả (fetch → extract → train → eval)
python scripts/run_pipeline.py --step all --limit 5 --download

# Hoặc từng bước:

# Bước 0: Kiểm tra API
python scripts/check_api.py

# Bước 1: Lấy metadata + tải sample
python scripts/01_fetch_sample_metadata.py --limit 5 --download
# Password zip: infected

# Bước 1b (tuỳ chọn): Tạo demo logs để test pipeline
python scripts/generate_demo_logs.py

# Bước 2: Dynamic analysis trên VM (Cuckoo / API Monitor / Procmon)
# Export log JSON → data/sandbox_logs/<sha256>.json
# Format mẫu: data/sandbox_logs/example_log_format.json

# Bước 3: Trích xuất features (unified)
python scripts/02_extract_features.py

# Bước 4: Train MCAFF model (CPU)
python scripts/03_train_model.py --epochs 50

# Bước 5: Evaluate
python scripts/04_evaluate.py

# Bước 6: Predict sample mới
python scripts/05_predict.py --sample data/samples/xxx.exe --log data/sandbox_logs/xxx.json
```

---

## Nguồn sample công khai

### Repo dataset riêng (khuyến nghị)

Repo [malware-api-dataset](https://github.com/nguyentrancaotri40-cmd/malware-api-dataset) **tương thích trực tiếp** với PTMD:

| File trong repo | Dùng cho PTMD |
|-----------------|---------------|
| `samples.csv` | SHA256 + family → `sample_manifest.json` |
| `verified_samples.csv` | Lọc sample DYN-API (5/331) |
| `samples/*.vir.zip` | Static PE analysis |

```bash
git clone https://github.com/nguyentrancaotri40-cmd/malware-api-dataset.git
python scripts/import_malware_api_dataset.py --dataset-dir ../malware-api-dataset --pe-only
```

> **Lưu ý:** Repo cung cấp metadata + PE static, **chưa có API call logs**. Cần chạy thêm dynamic analysis trên VM cho Behavior features.

| Nguồn | URL | Ghi chú |
|-------|-----|---------|
| MalwareBazaar | https://bazaar.abuse.ch/ | API miễn phí, tag theo family |
| VirusTotal | https://www.virustotal.com/ | Cần API key |
| Malware Traffic Analysis | https://malware-traffic-analysis.net/ | PCAP + sample kèm theo |
| theZoo | https://github.com/ytisf/theZoo | Research only, git submodule |

Script `01_fetch_sample_metadata.py` query MalwareBazaar theo tag mapping trong `FAMILY_TAGS`.

---

## Mô hình MCAFF — Chi tiết

### Cross-Attention Mechanism

```
Input: 3 feature groups → project lên d_model = 64
       ↓
Token sequence: [token_static, token_behavior, token_graph]  shape: [batch, 3, 64]
       ↓
+ Group Type Embeddings (learnable, giống segment embeddings trong BERT)
       ↓
Multi-Head Self-Attention (heads=4) × 2 layers
  → Mỗi token attend đến 2 tokens còn lại
  → Capture: Static↔Behavior, Static↔Graph, Behavior↔Graph interactions
       ↓
Mean Pooling → [batch, 64]
       ↓
├── Group Head: Linear(64→64)→GELU→Dropout→Linear(64→3) → G1/G2/G3
└── Family Head: Linear(64→64)→GELU→Dropout→Linear(64→20) → Family classification
```

- **Interpretability:** Attention weights cho thấy feature group nào đóng góp nhiều nhất cho từng prediction → xuất biểu đồ cho báo cáo
- **Metrics:** Accuracy, F1 macro, confusion matrix per G1/G2/G3
- **Output:** `checkpoints/best_model.pt`, `confusion_matrix.png`, `attention_weights.png`

---

## Mapping MITRE Families

Chi tiết đầy đủ trong `data/metadata/mitre_families.json`:

- **G1:** S1149 CHIMNEYSWEEP, S1148 Raccoon Stealer, S0366 WannaCry
- **G2:** S0013 PlugX, S0154 Cobalt Strike
- **G3:** S1053 AvosLocker, S0534 Bazar, S1236 CLAIMLOADER, S1239 TONESHELL, S1232 SplatDropper, S1099 Samurai, ...

APT Groups: G0094 Kimsuky, G0032 Lazarus, G0129 Mustang Panda

---

## Sandbox log format

```json
{
  "api_calls": [
    {"timestamp": 0.01, "api": "LoadLibraryA", "module": "kernel32.dll", "args": ["user32.dll"]},
    {"timestamp": 0.02, "api": "GetProcAddress", "module": "kernel32.dll", "args": ["MessageBoxA"]}
  ]
}
```

Export từ:
- **API Monitor** → filter kernel32/ntdll → export CSV → convert JSON
- **Procmon** → Export XML → script convert
- **Cuckoo Sandbox** → `behavior.processes[].calls`

---

## Lưu ý đồ án

1. **Không chạy malware trên máy host** — chỉ trên VM snapshot
2. **Không push sample lên GitHub** — `.gitignore` đã exclude `data/samples/`
3. Số lượng sample tối thiểu khuyến nghị: **≥5 sample/family**, tổng **≥50** để model hội tụ
4. Nếu thiếu dynamic log, model vẫn chạy với static features (behavior + graph = zeros)
5. Ghi rõ trong báo cáo: nguồn sample, hash SHA256, ngày phân tích, công cụ sandbox
6. **Attention weights** có thể dùng để giải thích kết quả trong báo cáo (interpretability)

---

## Tham khảo

- MITRE ATT&CK — Dynamic API Resolution: https://attack.mitre.org/techniques/T1027/007/
- MalwareBazaar API: https://bazaar.abuse.ch/api/
- Vaswani et al. "Attention Is All You Need" (2017) — Multi-Head Attention mechanism
- PyTorch: https://pytorch.org/
