# Google Colab 常见问题解决方案

## 问题 1: NumPy 版本冲突错误

### 错误信息
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

或者：
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
opencv-python requires numpy>=2.0, but you have numpy 1.26.4
```

### 原因分析
Colab 环境（2025年10月）中的包版本要求：
- **TensorFlow 2.16+**: 需要 `numpy>=2.0`（Colab 已不再提供 2.15）
- **所有现代包**: 都已支持 NumPy 2.0

**好消息**: 冲突已在新版本中解决！只需确保使用最新版本即可。

### 解决方案

#### 方案 A: 使用最新版本（推荐）⭐

在 Colab 新单元格中运行：

```python
# 直接安装最新版本（已解决冲突）
!pip install --upgrade tensorflow keras numpy

# 验证
import numpy as np
import tensorflow as tf
print(f"NumPy 版本: {np.__version__}")
print(f"TensorFlow 版本: {tf.__version__}")
print("✓ 版本兼容，无冲突")
```

#### 方案 B: 重启运行时（如果仍有问题）

```python
# 1. 重启运行时（清除所有冲突）
import os
os.kill(os.getpid(), 9)  # 这将重启 Colab 运行时
```

运行时重启后，继续：

```python
# 2. 安装兼容的版本
!pip install tensorflow>=2.16 keras numpy>=2.0

# 3. 安装其他依赖
!pip install pandas matplotlib seaborn plotly
!pip install psutil memory-profiler tensorflow-model-optimization
!pip install onnx onnxruntime scikit-learn tqdm pyyaml

print("✓ 所有依赖安装完成")
```

#### 方案 C: 使用自动修复脚本

```python
# 运行修复脚本（会自动处理版本问题）
!python colab_numpy_fix.py
```

### 快速修复命令（一键运行）

将以下代码复制到 Colab 新单元格：

```python
# 🔧 依赖安装快速修复（2025-10 更新）
import subprocess
import sys

print("正在安装兼容的依赖包...")

# 安装最新兼容版本
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", 
                "tensorflow", "keras", "numpy"])

# 验证
import numpy as np
import tensorflow as tf
print(f"✓ NumPy 版本: {np.__version__}")
print(f"✓ TensorFlow 版本: {tf.__version__}")
print(f"✓ 安装完成，无冲突")
```

---

## 问题 2: GPU 未启用

### 症状
```
GPU 可用: []
⚠ 未检测到 GPU，将使用 CPU 运行
```

### 解决方案

1. 在 Colab 顶部菜单选择：**运行时** → **更改运行时类型**
2. 在 "硬件加速器" 下拉菜单中选择 **T4 GPU** 或 **A100 GPU**
3. 点击 **保存**
4. 运行以下代码验证：

```python
import tensorflow as tf
print("GPU 设备:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("✓ GPU 已成功启用")
else:
    print("✗ GPU 未启用，请检查运行时设置")
```

---

## 问题 3: 文件未找到 (baseline_mobilenetv2.keras)

### 错误信息
```
FileNotFoundError: baseline_mobilenetv2.keras not found
```

### 解决方案

#### 选项 1: 训练新模型
```python
!python part1_baseline_model.py
```
⏱️ 耗时：30-60 分钟（使用 GPU）

#### 选项 2: 上传已有模型

1. 点击 Colab 左侧的 📁 文件图标
2. 点击 ⬆️ 上传按钮
3. 选择你本地的 `baseline_mobilenetv2.keras` 文件
4. 验证文件已上传：
   ```python
   import os
   print("模型文件存在:", os.path.exists('baseline_mobilenetv2.keras'))
   ```

#### 选项 3: 从 Google Drive 加载

```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 从 Drive 复制模型
!cp "/content/drive/MyDrive/MLS3_Models/baseline_mobilenetv2.keras" .

# 验证
import os
print("模型已复制:", os.path.exists('baseline_mobilenetv2.keras'))
```

---

## 问题 4: 内存不足 (OOM)

### 错误信息
```
ResourceExhaustedError: OOM when allocating tensor
```

### 解决方案

#### 方法 1: 启用动态内存增长
```python
import tensorflow as tf

# 设置 GPU 内存动态增长
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ GPU 动态内存已启用")
```

#### 方法 2: 减小批次大小

编辑项目配置文件或在代码中修改：
```python
# 在 part1_baseline_model.py 或相关文件中
BATCH_SIZE = 16  # 从 32 降低到 16
```

#### 方法 3: 使用 Colab Pro

如果免费版内存不足，考虑升级到 Colab Pro：
- 更多 RAM (25GB → 50GB)
- 更长运行时间
- 优先访问 GPU

---

## 问题 5: 运行时断开连接

### 症状
长时间训练后连接断开，进度丢失

### 解决方案

#### 方法 1: 保持浏览器活跃
在浏览器控制台（F12）运行：
```javascript
function KeepClicking(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click();
}
setInterval(KeepClicking, 60000);  // 每分钟点击一次
```

#### 方法 2: 使用检查点保存

在训练代码中添加：
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'checkpoint_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    save_best_only=False
)

# 训练时使用
model.fit(train_data, callbacks=[checkpoint], ...)
```

#### 方法 3: 使用 Google Drive 自动保存

```python
from google.colab import drive
drive.mount('/content/drive')

# 定期保存到 Drive
import shutil
shutil.copy('baseline_mobilenetv2.keras', 
            '/content/drive/MyDrive/MLS3_Backup/')
```

---

## 问题 6: 包导入错误

### 错误信息
```
ModuleNotFoundError: No module named 'part1_baseline_model'
```

### 解决方案

确保在正确的目录：
```python
import os
print("当前目录:", os.getcwd())

# 应该显示: /content/MSL/MLS3
# 如果不是，切换目录：
os.chdir('/content/MSL/MLS3')

# 验证文件存在
print("\n项目文件:")
!ls -1 *.py
```

---

## 有用的调试命令

### 检查环境状态
```python
import sys
import tensorflow as tf
import numpy as np

print("=== 环境信息 ===")
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print(f"工作目录: {os.getcwd()}")
```

### 查看内存使用
```python
import psutil

mem = psutil.virtual_memory()
print(f"总内存: {mem.total / 1e9:.2f} GB")
print(f"已使用: {mem.used / 1e9:.2f} GB ({mem.percent}%)")
print(f"可用: {mem.available / 1e9:.2f} GB")
```

### 测试 TensorFlow GPU
```python
import tensorflow as tf

# 创建测试张量
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("GPU 计算结果:\n", c)
print("✓ GPU 工作正常")
```

---

## 获取帮助

如果以上方案都无法解决问题，请：

1. **检查日志**: 完整错误信息和堆栈跟踪
2. **环境信息**: Python/TensorFlow/NumPy 版本
3. **复现步骤**: 从头到尾的操作流程
4. **提交 Issue**: 到项目 GitHub 仓库

📧 支持渠道：
- GitHub Issues: https://github.com/ishikisiko/MSL/issues
- 项目文档: 查看 `README.md` 和 `HOWTO_RUN.md`
