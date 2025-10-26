# Hướng dẫn Setup Conda Environment cho APML Project

### Bước 1: Tải và cài đặt Anaconda/Miniconda

- Truy cập: https://www.anaconda.com/products/distribution
- Tải phiên bản Python 3.12 cho Windows
- Chạy file installer và làm theo hướng dẫn

### Bước 2: Xác minh cài đặt
Mở Command Prompt hoặc PowerShell và chạy:
```bash
conda --version
```

## 🔧 Tạo môi trường từ file environment.yml

### Bước 1: Di chuyển đến thư mục project
```bash
cd /path/to/APML
```

### Bước 2: Tạo môi trường từ file environment.yml
```bash
conda env create -f environment.yml
```

### Bước 3: Kích hoạt môi trường
```bash
conda activate MLAI
```

## ✅ Kiểm tra cài đặt

### Kiểm tra Python version:
```bash
python --version
# Kết quả mong đợi: Python 3.12.7
```

### Kiểm tra PyTorch và CUDA:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Kiểm tra các thư viện chính:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import cv2
import transformers
print("Tất cả thư viện đã được cài đặt thành công!")
```

## 🛠️ Sử dụng môi trường

### Kích hoạt môi trường mỗi khi làm việc:
```bash
conda activate MLAI
```

### Tắt môi trường:
```bash
conda deactivate
```

### Xem danh sách môi trường:
```bash
conda env list
```

## 📚 Tham khảo thêm >>>

- **Conda Cheatsheet**: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- **Conda User Guide**: https://docs.conda.io/projects/conda/en/latest/user-guide/

## 🐛 Xử lý sự cố

### Lỗi CUDA không hoạt động:
1. Kiểm tra driver NVIDIA: `nvidia-smi`
2. Cài đặt CUDA Toolkit từ NVIDIA
3. Cài đặt lại PyTorch với CUDA support

### Lỗi package conflicts:
```bash
conda clean --all
conda env create -f environment.yml --force
```
