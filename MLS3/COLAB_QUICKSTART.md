# MLS3 Colab 快速启动指南

## 🎯 最简单的方法（3步完成）

### 方法 A：从 GitHub 直接运行（推荐）⭐

**仓库地址**: https://github.com/ishikisiko/MSL.git

在 Colab 新笔记本中依次运行：

```python
# 步骤 1: 克隆项目
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3

# 步骤 2: 安装依赖
!pip install -q tensorflow keras numpy pandas matplotlib seaborn psutil \
              memory-profiler tensorflow-model-optimization onnx onnxruntime \
              scikit-learn tqdm pyyaml

# 步骤 3: 运行（如果已有基线模型）
!python run_optimizations.py

# 或完整运行（包括训练基线，需要1-2小时）
# !python part1_baseline_model.py  # 30-60分钟
# !python run_optimizations.py     # 20-40分钟
```

---

### 方法 B：使用专用 Notebook（新手友好）⭐

1. 上传 `colab_setup.ipynb` 到 Colab
2. 按顺序运行所有单元格（自动克隆项目）
3. 等待完成并下载结果

---

### 方法 C：使用 Google Drive（最灵活）

**本地准备：**
无需准备，直接在 Colab 运行

**在 Colab 中运行：**
```python
# 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 在 Drive 中工作
%cd /content/drive/MyDrive
!mkdir -p MLS3_Project
%cd MLS3_Project

# 克隆项目
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3

# 安装和运行
!pip install -q -r requirements.txt
!python run_optimizations.py

# 结果自动保存到 Drive，下次可直接继续
```

---

## ⚡ 超快速测试（仅验证环境）

```python
# 单行测试
!pip install -q tensorflow && python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

## 🔥 最小化代码（仅测试模型创建）

```python
# 不训练，仅创建和查看模型
!pip install -q tensorflow keras

from tensorflow import keras
from tensorflow.keras import layers

def create_test_model():
    base = keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        alpha=0.5,
        include_top=False,
        weights='imagenet'
    )
    model = keras.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = create_test_model()
model.summary()
print(f"\n参数量: {model.count_params():,}")
```

---

## 📥 下载结果

```python
# 打包所有结果
!zip -r results.zip optimized_models/ results/ logs/ *.keras

# 下载
from google.colab import files
files.download('results.zip')
```

---

## 💾 使用 Google Drive（推荐长时间运行）

```python
# 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 在 Drive 中工作
%cd /content/drive/MyDrive
!mkdir -p MLS3
%cd MLS3

# 上传项目文件（首次）
from google.colab import files
uploaded = files.upload()

# 解压和运行
!unzip -q MLS3_colab_*.zip -d .
%cd MLS3
!pip install -q -r requirements.txt
!python run_optimizations.py

# 结果自动保存到 Drive，下次可直接继续
```

---

## 🎓 教学模式（分步理解）

### 第1步：只加载数据
```python
from part1_baseline_model import load_and_preprocess_data
train_ds, val_ds, test_ds = load_and_preprocess_data(batch_size=64)
print("✓ 数据加载完成")
```

### 第2步：创建模型
```python
from part2_optimizations import create_latency_optimized_model
model = create_latency_optimized_model(
    input_shape=(128, 128, 3),
    num_classes=10,
    alpha=0.5
)
model.summary()
```

### 第3步：量化
```python
from part2_optimizations import dynamic_range_quantization
from tensorflow import keras

model = keras.models.load_model('baseline_mobilenetv2.keras')
dynamic_range_quantization(model, 'quantized.tflite')
```

### 第4步：性能分析
```python
from performance_profiler import profile_model_comprehensive

config = {"power_budget_w": 5.0, "memory_budget_mb": 1024, "tdp_watts": 10.0}
results = profile_model_comprehensive(model, test_ds, config)
```

---

## 🐛 常见问题速查

### GPU 未启用
```
运行时 → 更改运行时类型 → GPU → 保存
```

### 模块未找到
```python
# 确认工作目录
!pwd
%cd /content/MLS3  # 或你的项目路径
```

### 内存不足
```python
# 减小批量大小
train_ds, val_ds, test_ds = load_and_preprocess_data(batch_size=32)
```

### 基线模型缺失
```python
# 检查
import os
print(os.path.exists('baseline_mobilenetv2.keras'))

# 需要先训练
!python part1_baseline_model.py
```

---

## ⏱️ 时间规划

| 任务 | 时间 | 可跳过 |
|------|------|--------|
| 环境设置 | 2-5分钟 | ✗ |
| 基线训练 | 30-60分钟 | ✓ (如有模型) |
| 优化流程 | 20-40分钟 | ✗ |
| 结果分析 | 5-10分钟 | ✗ |
| **总计** | **~1-2小时** | |

---

## ✅ 一键复制代码块

### 完整运行（有基线模型）
```python
!git clone https://github.com/ishikisiko/MSL.git && \
%cd MSL/MLS3 && \
pip install -q -r requirements.txt && \
python run_optimizations.py
```

### 完整运行（无基线模型）
```python
!git clone https://github.com/ishikisiko/MSL.git && \
%cd MSL/MLS3 && \
pip install -q -r requirements.txt && \
python part1_baseline_model.py && \
python run_optimizations.py
```

### 仅测试导入
```python
!git clone https://github.com/ishikisiko/MSL.git && \
%cd MSL/MLS3 && \
pip install -q tensorflow keras && \
python quick_test.py
```

---

## 🎓 选择合适的方法

| 场景 | 推荐方法 | 说明 |
|------|----------|------|
| 首次运行，了解流程 | 方法 B (专用 Notebook) | 有完整说明 |
| 快速运行，已熟悉 | 方法 A (GitHub) | 最快 |
| 多次运行，长时间 | 方法 C (Google Drive) | 结果持久化 |
| 仅测试特定功能 | 教学模式 | 分步理解 |

---

## 🎉 快速开始

**最快的方式：**

1. 打开 https://colab.research.google.com
2. 新建笔记本
3. 复制粘贴"一键复制代码块"中的命令
4. 运行
5. 等待完成
6. 下载结果

就这么简单！🚀

---

需要帮助？查看完整文档：`COLAB_SETUP.md`
