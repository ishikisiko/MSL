# 🚀 在 Google Colab 运行 MLS3 项目

本指南提供在 Google Colab 上运行 MLS3 多文件项目的便捷方法。

## 📋 准备工作

在开始前，请确保：
- 有 Google 账号（用于访问 Colab）
- 建议启用 GPU 加速（Colab 菜单：运行时 → 更改运行时类型 → GPU）
- 项目仓库：https://github.com/ishikisiko/MSL.git

---

## 🎯 方法一：使用专用 Colab Notebook（推荐⭐）

### 步骤：

1. **打开准备好的 Notebook**
   - 直接上传 `colab_setup.ipynb` 到 Colab
   - 或访问：https://colab.research.google.com
   - 点击 "文件" → "上传笔记本" → 选择 `colab_setup.ipynb`

2. **自动从 GitHub 克隆**
   
   Notebook 会自动执行：
   ```python
   !git clone https://github.com/ishikisiko/MSL.git
   %cd MSL/MLS3
   ```

3. **按顺序运行单元格**
   - 环境设置 → 依赖安装 → 训练/优化 → 查看结果 → 下载

4. **下载结果**
   - 运行最后的下载单元格
   - 获得包含所有模型和结果的 ZIP 文件

---

## 🔥 方法二：直接运行（最快⚡）

### 在 Colab 新建笔记本，复制运行以下代码：

```python
# 单元格 1: 完整自动化设置
# 从 GitHub 克隆
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3

# 安装依赖
!pip install -q tensorflow keras numpy pandas matplotlib seaborn
!pip install -q psutil memory-profiler tensorflow-model-optimization
!pip install -q onnx onnxruntime scikit-learn tqdm pyyaml

# 检查 GPU
import tensorflow as tf
print("GPU 可用:", tf.config.list_physical_devices('GPU'))

# 快速测试
!python -c "import part1_baseline_model; import part2_optimizations; print('✓ 模块导入成功')"
```

```python
# 单元格 2: 运行完整流程（如果已有基线模型）
!python run_optimizations.py
```

```python
# 单元格 3: 或分步运行
# 步骤 1: 训练基线（如需要，30-60分钟）
!python part1_baseline_model.py

# 步骤 2: 运行优化（20-40分钟）
!python run_optimizations.py
```

```python
# 单元格 4: 打包下载
!zip -r MLS3_results.zip optimized_models/ results/ logs/ baseline_mobilenetv2.keras

from google.colab import files
files.download('MLS3_results.zip')
```

---
## 📦 方法三：使用 Google Drive 同步（最灵活）

### 适合需要多次运行和保存中间结果的情况

```python
# 单元格 1: 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 在 Drive 中创建项目目录
!mkdir -p /content/drive/MyDrive/MLS3
%cd /content/drive/MyDrive/MLS3
```

## 📦 方法三：使用 Google Drive（最灵活⭐）

### 适合需要多次运行和保存中间结果的情况

```python
# 单元格 1: 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 在 Drive 中创建项目目录
!mkdir -p /content/drive/MyDrive/MLS3_Project
%cd /content/drive/MyDrive/MLS3_Project
```

```python
# 单元格 2: 克隆项目（首次运行）
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
```

```python
# 单元格 3: 安装依赖和运行
!pip install -q -r requirements.txt

# 运行流程
!python run_optimizations.py
```

**优势：**
- 所有文件自动保存到 Google Drive
- 断线后可以继续运行
- 结果永久保存，不会因会话结束而丢失
- 下次打开可直接使用，无需重新克隆

**后续运行：**
```python
# 直接使用已克隆的项目
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/MLS3_Project/MSL/MLS3
!git pull  # 获取最新更新
!python run_optimizations.py
```

---

## 🛠️ 快速问题排查

### 问题 1: 模块导入失败

```python
# 检查当前目录
!pwd
!ls -la

# 确保在正确的目录
%cd MSL/MLS3  # 或你的项目路径
```

### 问题 2: GPU 未启用

1. Colab 菜单：运行时 → 更改运行时类型
2. 硬件加速器：选择 "GPU"
3. 保存 → 会话将重启

### 问题 3: 内存不足

```python
# 减小批量大小
# 在 run_optimizations.py 中修改或直接运行：
from part1_baseline_model import load_and_preprocess_data
train_ds, val_ds, test_ds = load_and_preprocess_data(batch_size=32)  # 默认64
```

### 问题 4: 会话超时

- Colab 免费版：12小时会话限制
- 建议使用 Google Drive 方法保存中间结果
- 或分段运行：先训练基线 → 保存 → 第二天运行优化

### 问题 5: 基线模型未找到

```python
# 检查文件是否存在
import os
print("基线模型存在:", os.path.exists('baseline_mobilenetv2.keras'))

# 如果不存在，先训练
!python part1_baseline_model.py
```

---

## 📊 推荐工作流程

### 完整流程（首次运行）

```
Day 1: 训练基线
├─ 上传项目文件到 Colab/Drive
├─ 安装依赖
├─ 运行 part1_baseline_model.py (30-60分钟)
└─ 下载 baseline_mobilenetv2.keras 备份

Day 2: 优化和分析
├─ 上传基线模型（如果使用新会话）
├─ 运行 run_optimizations.py (20-40分钟)
├─ 查看性能对比
└─ 下载所有结果文件
```

### 快速测试流程

```python
# 1. 只测试模型创建（不训练）
from part2_optimizations import create_latency_optimized_model
model = create_latency_optimized_model(input_shape=(128,128,3), num_classes=10)
model.summary()

# 2. 只测试量化
from part2_optimizations import dynamic_range_quantization
from tensorflow import keras
model = keras.models.load_model('baseline_mobilenetv2.keras')
dynamic_range_quantization(model, 'test.tflite')

# 3. 只测试性能分析
from performance_profiler import profile_model_comprehensive
# ... (见 colab_setup.ipynb 示例)
```

---

## 💡 高级技巧

### 1. 并行运行多个实验

```python
# 创建多个模型变体进行对比
alphas = [0.35, 0.5, 0.75, 1.0]
models = {}

for alpha in alphas:
    model = create_latency_optimized_model(
        input_shape=(128, 128, 3),
        num_classes=10,
        alpha=alpha
    )
    models[f'alpha_{alpha}'] = model
```

### 2. 自定义性能分析

```python
# 针对特定平台配置
platform_configs = {
    "mobile_low_power": {
        "power_budget_w": 2.0,
        "memory_budget_mb": 512,
        "tdp_watts": 3.0,
    },
    "embedded_device": {
        "power_budget_w": 0.5,
        "memory_budget_mb": 256,
        "tdp_watts": 1.0,
    },
}

for name, config in platform_configs.items():
    results = profile_model_comprehensive(model, test_ds, config)
    print(f"\n{name} results:")
    print_profiling_results(results, name)
```

### 3. 可视化结果

```python
# 在 Colab 中直接绘图
import matplotlib.pyplot as plt

# 性能对比雷达图
import numpy as np

categories = ['Latency', 'Memory', 'Energy', 'Accuracy', 'Throughput']
baseline_scores = [50, 100, 100, 100, 50]
optimized_scores = [90, 60, 70, 95, 85]

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
baseline_scores += baseline_scores[:1]
optimized_scores += optimized_scores[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='Baseline')
ax.plot(angles, optimized_scores, 'o-', linewidth=2, label='Optimized')
ax.fill(angles, baseline_scores, alpha=0.25)
ax.fill(angles, optimized_scores, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.legend()
plt.title('Model Performance Comparison')
plt.show()
```

---

## 📚 资源链接

- **Google Colab 文档**: https://colab.research.google.com/notebooks/intro.ipynb
- **TensorFlow GPU 支持**: https://www.tensorflow.org/install/gpu
- **本项目 GitHub**: https://github.com/ishikisiko/MSL
- **Colab Pro**: 考虑升级以获得更长会话和更好 GPU

---

## ✅ 检查清单

运行前确认：
- [ ] 已上传所有必要的 Python 文件
- [ ] 已安装所有依赖（运行 requirements.txt）
- [ ] GPU 已启用（如果需要加速）
- [ ] 有足够的 Google Drive 空间（约 2-3 GB）
- [ ] 了解预计运行时间（完整流程约 1-2 小时）

运行后确认：
- [ ] `optimized_models/` 目录包含所有模型文件
- [ ] `results/` 目录包含性能报告
- [ ] 已下载所有结果到本地
- [ ] 性能指标符合预期

---

## 🎉 完成！

按照以上任一方法，你应该能够在 Colab 上顺利运行完整的 MLS3 项目。

如有问题，请检查：
1. 文件路径是否正确
2. 依赖是否完全安装
3. 运行时是否使用 GPU
4. 查看 `logs/` 目录中的错误日志

祝顺利完成作业！🚀
