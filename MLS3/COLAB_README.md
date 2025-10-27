# 🚀 MLS3 Google Colab 运行指南

> 快速在 Google Colab 上运行整个 MLS3 多文件项目

---

## ⚡ 超快速开始（3 步完成）

### 方法 1: 从 GitHub 直接运行（最快）⭐

**仓库地址**: https://github.com/ishikisiko/MSL.git

在 [Google Colab](https://colab.research.google.com) 新笔记本中运行：

```python
# 一键完整运行
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
!pip install -q -r requirements.txt
!python run_optimizations.py
```

### 方法 2: 使用专用 Notebook（推荐新手）⭐

1. 上传 `colab_setup.ipynb` 到 Colab
2. 按顺序运行所有单元格（会自动从 GitHub 克隆项目）
3. 下载结果

### 方法 3: Google Drive 持久化（多次运行）

```python
# 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 切换到 Drive 目录
%cd /content/drive/MyDrive
!mkdir -p MLS3_Project
%cd MLS3_Project

# 克隆项目（首次）
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3

# 安装和运行
!pip install -q -r requirements.txt
!python run_optimizations.py
```

---

## 📚 完整文档

| 文档 | 说明 | 适合 |
|------|------|------|
| **[COLAB_INDEX.md](COLAB_INDEX.md)** | 📑 所有资源索引 | 所有人 |
| **[COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)** | ⚡ 快速开始 | 赶时间 |
| **[COLAB_SETUP.md](COLAB_SETUP.md)** | 📖 详细指南 | 深入学习 |
| **[COLAB_SUMMARY.md](COLAB_SUMMARY.md)** | 📋 方案总结 | 需要概览 |
| **[colab_setup.ipynb](colab_setup.ipynb)** | 📔 交互式 Notebook | 喜欢图形化 |

---

## 🎯 选择你的路径

### 🏃 快速路径（15 分钟设置）
```
COLAB_QUICKSTART.md → 复制代码 → 运行 → 完成
```

### 📚 学习路径（1 小时）
```
COLAB_SETUP.md → colab_setup.ipynb → 实践 → 掌握
```

### 🎓 完整路径（深入理解）
```
COLAB_INDEX.md → COLAB_SUMMARY.md → COLAB_SETUP.md → 高级应用
```

---

## ⏱️ 时间规划

| 步骤 | GPU | CPU | 可跳过 |
|------|-----|-----|--------|
| 环境设置 | 5 分钟 | 5 分钟 | ✗ |
| 基线训练 | 30-45 分钟 | 60-120 分钟 | ✓ (如有模型) |
| 优化流程 | 20-30 分钟 | 40-60 分钟 | ✗ |
| 结果分析 | 5 分钟 | 5 分钟 | ✗ |
| **总计** | **~1 小时** | **~2 小时** | |

**💡 建议：** 启用 GPU 加速（Runtime → Change runtime type → GPU）

---

## 📦 工具脚本

### prepare_colab.py - 打包项目
```powershell
python prepare_colab.py
```
生成适合上传的 ZIP 文件

### generate_colab_diagrams.py - 生成图表
```powershell
python generate_colab_diagrams.py
```
创建工作流程可视化图表

### quick_test.py - 快速测试
```powershell
python quick_test.py
```
验证模块导入是否正常

---

## 🎉 生成的文件

运行完成后你将获得：

```
📁 优化模型
  ├─ latency_optimized.keras
  ├─ memory_optimized.keras
  ├─ energy_optimized.keras
  └─ *.tflite (量化版本)

📁 性能结果
  ├─ performance_comparison.png
  ├─ project_summary.json
  └─ *.csv (详细指标)

📁 运行日志
  └─ *.log
```

---

## 💡 常见问题速查

| 问题 | 解决方案 | 文档 |
|------|----------|------|
| GPU 未启用 | Runtime → Change runtime type → GPU | COLAB_QUICKSTART.md |
| 模块未找到 | 确认工作目录：`%cd /content/MLS3` | COLAB_SETUP.md |
| 内存不足 | 减小批量大小：`batch_size=32` | COLAB_SETUP.md |
| 基线模型缺失 | 先运行：`!python part1_baseline_model.py` | COLAB_QUICKSTART.md |
| 会话超时 | 使用 Google Drive 方法 | COLAB_SETUP.md |

---

## 🌟 推荐使用

### 第一次使用？
1. 阅读 **COLAB_QUICKSTART.md**（5 分钟）
2. 上传 **colab_setup.ipynb**
3. 运行并观察

### 经验用户？
1. 复制一键命令
2. 粘贴到 Colab
3. 直接运行

### 需要多次运行？
1. 查看 **COLAB_SETUP.md** → Google Drive 方法
2. 结果持久化
3. 随时继续

---

## 📞 需要帮助？

1. **📑 查看索引** → [COLAB_INDEX.md](COLAB_INDEX.md)
2. **⚡ 快速指南** → [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)  
3. **📖 详细文档** → [COLAB_SETUP.md](COLAB_SETUP.md)
4. **🐛 运行测试** → `!python quick_test.py`

---

## ✅ 快速检查

运行前确认：
- [ ] 选择了运行方法
- [ ] 了解预计时间
- [ ] 准备好文件
- [ ] 启用 GPU（推荐）

---

## 🚀 立即开始

```python
# 复制这段代码到 Colab 新笔记本
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
!pip install -q -r requirements.txt

# 检查 GPU
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))

# 运行完整流程
!python run_optimizations.py
```

**就这么简单！** 🎉

---

**文档版本：** 1.0  
**最后更新：** 2025-10-28  
**项目主页：** [README.md](README.md)
