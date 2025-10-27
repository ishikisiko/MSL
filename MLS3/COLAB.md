# 🚀 MLS3 在 Google Colab 上运行

> **一键运行** | **零配置** | **GPU 加速**

---

## ⚡ 快速开始（30 秒）

在 [Google Colab](https://colab.research.google.com) 新笔记本中运行：

```python
# 一键完整运行
!git clone https://github.com/ishikisiko/MSL.git && cd MSL/MLS3 && \
pip install -q "numpy>=2.0,<2.3" tensorflow-model-optimization line-profiler && \
python run_optimizations.py
```

**就这么简单！** 🎉

---

## 📋 三种运行方式

### 方法 1: 一键命令（最快）⭐

```python
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
!pip install -q "numpy>=2.0,<2.3" tensorflow-model-optimization line-profiler
!python run_optimizations.py
```

**优势**: 最快速，适合单次运行  
**耗时**: 1-2 小时（含训练）

### 方法 2: 交互式 Notebook（推荐新手）

1. 上传 `colab_setup.ipynb` 到 Colab
2. 按顺序运行单元格
3. 查看中间结果

**优势**: 分步执行，易于调试  
**耗时**: 1-2 小时

### 方法 3: Google Drive（持久化存储）

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3

!pip install -q "numpy>=2.0,<2.3" tensorflow-model-optimization line-profiler
!python run_optimizations.py
```

**优势**: 结果保存在 Drive，断线可恢复  
**耗时**: 1-2 小时（首次）

---

## ⏱️ 时间预估

| 步骤 | GPU | CPU | 可跳过 |
|------|-----|-----|--------|
| 克隆项目 | 30秒 | 30秒 | ✗ |
| 安装依赖 | 20秒 | 20秒 | ✗ |
| 训练基线 | 30-45分钟 | 60-120分钟 | ✓ |
| 运行优化 | 20-30分钟 | 40-60分钟 | ✗ |
| **总计** | **~1小时** | **~2小时** | |

💡 **建议**: 启用 GPU（运行时 → 更改运行时类型 → T4 GPU）

---

## 🔧 常见问题

### 1. 依赖冲突错误

**症状**: `ERROR: pip's dependency resolver...`

**解决**: 使用最小化安装（已包含在上述命令中）

```python
# 只安装必需的包，避免冲突
!pip install -q "numpy>=2.0,<2.3"  # 修复 NumPy 版本
!pip install -q tensorflow-model-optimization line-profiler
```

**原因**: Colab 已预装 200+ 包，重装会引发冲突

### 2. GPU 未检测到

**解决**: 
1. 菜单: **运行时** → **更改运行时类型**
2. 硬件加速器: 选择 **T4 GPU**
3. 保存并重新连接

**验证**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### 3. 文件未找到

**解决**: 确保在正确的目录
```python
import os
print(os.getcwd())  # 应显示 /content/MSL/MLS3
%cd /content/MSL/MLS3  # 如不是，切换到此目录
```

### 4. 内存不足 (OOM)

**解决**: 
```python
# 方法 1: 启用 GPU 内存增长
import tensorflow as tf
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# 方法 2: 减小批量大小（修改代码）
# batch_size = 16  # 从 32 降到 16
```

### 5. 会话断开

**解决**: 使用 Google Drive 方法（方法 3），结果自动保存

**更多问题**: 查看 [COLAB_TROUBLESHOOTING.md](COLAB_TROUBLESHOOTING.md)

---

## 📊 生成的文件

运行完成后将生成：

```
MLS3/
├── optimized_models/          # 优化后的模型
│   ├── latency_optimized.keras
│   ├── memory_optimized.keras
│   ├── energy_optimized.keras
│   └── *.tflite              # 量化模型
├── results/                   # 性能结果
│   ├── performance_comparison.png
│   ├── project_summary.json
│   └── *.csv
└── baseline_mobilenetv2.keras # 基线模型
```

**下载**: 左侧文件浏览器 → 右键 → 下载

---

## 🧪 快速测试

只测试环境，不运行完整流程：

```python
# 测试导入
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
!pip install -q "numpy>=2.0,<2.3" tensorflow-model-optimization line-profiler

# 验证环境
import numpy as np
import tensorflow as tf
print(f"✓ NumPy: {np.__version__}")
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPU: {tf.config.list_physical_devices('GPU')}")

# 测试模块导入
import part1_baseline_model
print("✓ 模块导入成功")
```

---

## 💡 最佳实践

### ✅ 推荐做法
- 使用最小化安装命令（避免依赖冲突）
- 启用 GPU（速度提升 2-3 倍）
- 定期下载结果到本地
- 使用 Drive 方法进行长时间运行

### ❌ 避免做法
- 不要运行 `pip install -r requirements.txt`（会触发大量冲突）
- 不要忽视 NumPy 版本限制
- 不要在 CPU 模式下训练（太慢）

---

## 📚 相关文档

- **[COLAB_TROUBLESHOOTING.md](COLAB_TROUBLESHOOTING.md)** - 完整的故障排除指南
- **[COLAB_SETUP.md](COLAB_SETUP.md)** - 详细的设置说明和高级选项
- **[README.md](README.md)** - 项目主文档
- **[HOWTO_RUN.md](HOWTO_RUN.md)** - 本地运行指南

---

## 🎯 下一步

1. ✅ 成功运行项目
2. 📥 下载生成的文件
3. 📊 分析性能对比结果
4. 🚀 尝试自定义参数（见 COLAB_SETUP.md）

---

**版本**: v5.0  
**更新**: 2025-10-28  
**仓库**: https://github.com/ishikisiko/MSL
