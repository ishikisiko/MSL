# MLS3 Colab 运行方案总结

## 📦 已创建的文件

为了便于在 Google Colab 运行这个多文件项目，已创建以下文件：

### 1. **colab_setup.ipynb** ⭐ 推荐
- 完整的 Jupyter Notebook
- 包含所有设置步骤和说明
- 适合首次使用或需要详细指导的用户
- 支持三种上传方式：GitHub / ZIP / 手动上传
- 包含结果可视化和下载功能

### 2. **COLAB_SETUP.md**
- 详细的使用文档
- 三种运行方法的完整说明
- 问题排查和高级技巧
- 适合深入了解 Colab 运行机制

### 3. **COLAB_QUICKSTART.md**
- 快速启动指南
- 一键复制代码块
- 时间规划和方法选择
- 适合快速上手

### 4. **prepare_colab.py**
- 自动打包脚本
- 创建完整包或轻量级包
- 生成快速启动代码
- 适合本地准备后上传

---

## 🎯 使用流程

### 方案 A：使用专用 Notebook（最简单）

1. **上传 Notebook**
   ```
   打开 https://colab.research.google.com
   上传 colab_setup.ipynb
   ```

2. **选择上传方式**
   - GitHub 克隆（如果项目在 GitHub）
   - ZIP 上传（使用 prepare_colab.py 打包）
   - 手动上传文件

3. **运行所有单元格**
   ```
   Runtime → Run all
   ```

4. **下载结果**
   ```
   运行最后的下载单元格
   获得包含所有模型的 ZIP
   ```

---

### 方案 B：直接运行（最快）

**如果项目在 GitHub：**

在 Colab 新笔记本中运行：
```python
# 一键完整运行
!git clone https://github.com/ishikisiko/MSL.git && \
cd MSL/MLS3 && \
pip install -q -r requirements.txt && \
python run_optimizations.py
```

**如果需要训练基线：**
```python
!git clone https://github.com/ishikisiko/MSL.git && \
cd MSL/MLS3 && \
pip install -q -r requirements.txt && \
python part1_baseline_model.py && \
python run_optimizations.py
```

---

### 方案 C：使用 Google Drive（最灵活）

适合多次运行和保存中间结果：

```python
# 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 工作目录
%cd /content/drive/MyDrive
!mkdir -p MLS3
%cd MLS3

# 上传项目（首次）
from google.colab import files
uploaded = files.upload()  # 上传 ZIP

# 解压
!unzip -q MLS3_colab_*.zip

# 运行
!pip install -q -r requirements.txt
!python run_optimizations.py
```

**优势：**
- 结果自动保存到 Drive
- 断线可继续
- 下次直接使用

---

## 📋 准备清单

### 本地准备（推荐做法）

1. **打包项目**
   ```powershell
   python prepare_colab.py
   ```
   - 选择 "1" 创建完整包
   - 或选择 "2" 创建轻量级包（需在 Colab 训练）

2. **检查生成的 ZIP**
   - `MLS3_colab_YYYYMMDD_HHMMSS.zip` - 完整包
   - `MLS3_colab_lite_YYYYMMDD_HHMMSS.zip` - 轻量级包

3. **准备上传**
   - 文件大小应该在可接受范围（通常 < 100MB）
   - 如果包含基线模型会更大

---

## ⏱️ 时间估算

| 步骤 | 使用 GPU | 使用 CPU | 可跳过 |
|------|----------|----------|--------|
| 环境设置 | 2-5 分钟 | 2-5 分钟 | ✗ |
| 基线训练 | 30-45 分钟 | 60-120 分钟 | ✓ |
| 优化流程 | 20-30 分钟 | 40-60 分钟 | ✗ |
| 结果分析 | 5 分钟 | 5 分钟 | ✗ |
| **总计** | **~1 小时** | **~2 小时** | |

**建议：**
- 启用 GPU 加速（Runtime → Change runtime type → GPU）
- 如果已有基线模型，可跳过训练步骤

---

## 🔧 快速命令参考

### 检查 GPU
```python
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))
```

### 检查文件
```bash
!ls -la
!pwd
```

### 安装依赖
```bash
!pip install -q -r requirements.txt
```

### 运行项目
```bash
# 完整流程
!python run_optimizations.py

# 仅训练基线
!python part1_baseline_model.py

# 快速测试
!python quick_test.py
```

### 打包下载
```bash
!zip -r results.zip optimized_models/ results/ logs/
```
```python
from google.colab import files
files.download('results.zip')
```

---

## 📊 生成的文件

运行完成后会生成：

```
optimized_models/
├── latency_optimized.keras          # 延迟优化模型
├── memory_optimized.keras           # 内存优化模型
├── energy_optimized.keras           # 能耗优化模型
├── latency_optimized_dynamic.tflite # 动态量化
├── latency_optimized_ptq_int8.tflite# INT8 量化
├── baseline.tflite                  # TFLite 基线
└── baseline_data.cc                 # TFLite Micro

results/
├── performance_comparison.png       # 性能对比图
├── project_summary.json            # 项目摘要
└── *.csv                           # 性能数据

logs/
└── *.log                           # 运行日志
```

---

## 🎓 学习路径

### 初学者路径
1. 阅读 `COLAB_QUICKSTART.md`
2. 使用 `colab_setup.ipynb`
3. 按步骤运行并观察输出
4. 查看生成的文件和报告

### 中级路径
1. 阅读 `COLAB_SETUP.md`
2. 选择合适的运行方法
3. 自定义参数和配置
4. 尝试不同的优化策略

### 高级路径
1. 直接使用命令行运行
2. 修改源代码进行实验
3. 使用 Drive 进行持久化
4. 批量运行多个实验

---

## ❓ 常见问题

### Q1: 是否必须使用 Colab？
不是，可以在本地运行。Colab 提供：
- 免费 GPU
- 无需本地环境配置
- 易于分享

### Q2: 需要 Colab Pro 吗？
不需要。免费版足够，但 Pro 提供：
- 更长会话时间
- 更好的 GPU（A100/V100）
- 更多内存

### Q3: 数据会丢失吗？
会话结束后临时文件会丢失。建议：
- 使用 Google Drive 保存
- 及时下载重要结果
- 或使用 GitHub 同步

### Q4: 能否在手机上运行？
可以！Colab 支持手机浏览器，但：
- 建议使用桌面版获得更好体验
- 上传文件可能不便
- 查看结果可能较小

### Q5: 如何中断后继续？
使用 Google Drive 方法：
- 文件保存在 Drive
- 重新连接后继续运行
- 或使用检查点恢复

---

## 🚀 最佳实践

1. **使用 GPU**
   - 始终启用 GPU 加速
   - 检查 GPU 是否被识别

2. **分步运行**
   - 首次运行建议分步
   - 观察每步输出
   - 确认无误后继续

3. **保存结果**
   - 及时下载重要文件
   - 使用 Drive 持久化
   - 记录关键指标

4. **监控资源**
   - 观察内存使用
   - 避免 OOM 错误
   - 必要时减小批量大小

5. **版本管理**
   - 记录运行配置
   - 保存关键模型
   - 文档化实验结果

---

## 📚 相关文档

- `README.md` - 项目总览
- `HOWTO_RUN.md` - 本地运行指南
- `COLAB_SETUP.md` - Colab 详细文档
- `COLAB_QUICKSTART.md` - Colab 快速指南
- `ASS.md` - 作业要求
- `AGENTS.md` - 开发规范

---

## 🎉 开始使用

选择最适合你的方法：

| 你的情况 | 推荐方案 | 文档 |
|---------|---------|------|
| 首次使用，需要指导 | colab_setup.ipynb | COLAB_SETUP.md |
| 快速运行，已熟悉 | 直接运行 | COLAB_QUICKSTART.md |
| 项目在 GitHub | GitHub 克隆 | COLAB_QUICKSTART.md |
| 需要多次运行 | Google Drive | COLAB_SETUP.md |
| 本地已打包 | ZIP 上传 | prepare_colab.py |

**立即开始：**
```python
# 复制这段代码到 Colab 新笔记本
!git clone https://github.com/ishikisiko/MSL.git
%cd MSL/MLS3
!pip install -q -r requirements.txt
!python run_optimizations.py
```

祝顺利完成作业！🚀

---

**需要帮助？**
- 查看详细文档
- 检查常见问题
- 查看运行日志
- 使用快速测试脚本
