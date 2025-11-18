# 量化生成的文件说明

## 概述
`part2_quantization.py` 在量化过程中会生成以下TFLite和Keras模型文件。

---

## 生成的文件列表

### 1. **Post-Training Quantization (PTQ)**
- **文件**: `models/ptq_quantized_model.tflite`
- **类型**: TensorFlow Lite 模型
- **量化方式**: 动态范围量化 (Dynamic Range Quantization)
- **生成方法**: `_implement_post_training_quantization()`
- **特点**:
  - 权重量化为INT8
  - 激活保持FLOAT32
  - 无需重新训练
  - 模型大小: ~6 MB (压缩约4倍)
  - 准确率: ~83% (CIFAR-100)

---

### 2. **Dynamic Range Quantization**
- **文件**: `models/dynamic_range_quantized_model.tflite`
- **类型**: TensorFlow Lite 模型
- **量化方式**: 动态范围量化
- **生成方法**: `_implement_dynamic_range_quantization()`
- **特点**:
  - 与PTQ类似，但明确使用动态范围量化
  - 权重INT8，激活FLOAT32
  - 最快的量化方式
  - 模型大小: ~6 MB
  - 准确率: ~83% (CIFAR-100)

---

### 3. **Quantization-Aware Training (QAT) - TFLite**
- **文件**: `models/qat_quantized_model.tflite`
- **类型**: TensorFlow Lite 模型
- **量化方式**: 量化感知训练
- **生成方法**: `implement_standard_qat()`
- **特点**:
  - 训练时模拟量化效果
  - 权重和激活都量化
  - 需要重新训练/微调
  - 模型大小: ~6 MB
  - 准确率: 预期 > PTQ (通过训练补偿量化损失)
  - **修复**: 现在包含representative_dataset，支持完整整数量化

---

### 4. **Quantization-Aware Training (QAT) - Keras**
- **文件**: `models/qat_fine_tuned_model.keras`
- **类型**: Keras 模型文件
- **量化方式**: 包含QAT fake-quantization节点
- **生成方法**: `implement_standard_qat()`
- **特点**:
  - 保存QAT训练后的完整Keras模型
  - 包含量化参数和fake-quant层
  - 可用于进一步微调
  - 可转换为不同格式的量化模型
  - 模型大小: ~24 MB (未压缩)

---

## 修复的问题 (2025-11-19)

### 问题1: QAT训练数据耗尽
**错误信息**:
```
WARNING:tensorflow:Your input ran out of data; interrupting training.
Make sure that your dataset or generator can generate at least
`steps_per_epoch * epochs` batches
```

**修复方案**:
```python
# 之前 (错误)
train_ds_for_qat = train_ds_for_qat.repeat(epochs)  # 有限repeat

# 现在 (正确)
train_ds_for_qat = train_ds_for_qat.repeat()  # 无限repeat
```

---

### 问题2: QAT TFLite转换失败
**错误信息**:
```
QAT implementation failed: For full integer quantization,
a `representative_dataset` must be specified.
```

**修复方案**:
现在在QAT转换时添加representative_dataset:
```python
def qat_representative_dataset():
    """Representative dataset for QAT TFLite conversion"""
    for sample in calibration_samples:
        yield [np.expand_dims(sample, axis=0).astype(np.float32)]

converter.representative_dataset = qat_representative_dataset
```

---

## 文件存放位置

所有量化模型文件统一存放在:
```
MLS4/
  ├── models/
  │   ├── ptq_quantized_model.tflite           # PTQ量化模型
  │   ├── dynamic_range_quantized_model.tflite  # 动态范围量化模型
  │   ├── qat_quantized_model.tflite           # QAT量化模型(TFLite)
  │   └── qat_fine_tuned_model.keras           # QAT微调模型(Keras)
```

**注意**: `models/` 目录会自动创建（如果不存在）

---

## 量化类型对比

| 量化方式 | 文件 | 大小 | 准确率 | 训练时间 | 适用场景 |
|---------|------|------|--------|----------|---------|
| **PTQ** | ptq_quantized_model.tflite | ~6 MB | ~83% | 0秒 | 快速部署 |
| **Dynamic Range** | dynamic_range_quantized_model.tflite | ~6 MB | ~83% | 0秒 | CPU推理 |
| **QAT (TFLite)** | qat_quantized_model.tflite | ~6 MB | >83% | ~15分钟 | 精度优先 |
| **QAT (Keras)** | qat_fine_tuned_model.keras | ~24 MB | >83% | ~15分钟 | 进一步微调 |

---

## 使用示例

### 加载PTQ模型
```python
import tensorflow as tf

# 加载TFLite模型
interpreter = tf.lite.Interpreter(model_path="models/ptq_quantized_model.tflite")
interpreter.allocate_tensors()

# 获取输入输出
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 推理
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

### 加载QAT Keras模型
```python
import tensorflow as tf

# 加载Keras模型
model = tf.keras.models.load_model("models/qat_fine_tuned_model.keras")

# 推理
predictions = model.predict(input_data)
```

---

## 日志验证

根据最新运行日志:
- ✓ PTQ成功生成: `models/ptq_quantized_model.tflite` (6.05 MB, 83% 准确率)
- ✓ Dynamic Range成功生成: `models/dynamic_range_quantized_model.tflite` (6.05 MB, 83% 准确率)
- ⚠ QAT部分失败: 缺少representative_dataset (已修复)

修复后，运行 `python main.py --onlyq` 应该会成功生成所有4个文件。

---

## 验证文件是否生成

在Linux服务器上运行:
```bash
cd ~/autodl-tmp/MSL/MLS4
ls -lh models/*.tflite models/*.keras
```

期望输出:
```
-rw-r--r-- 1 root root 6.0M Nov 19 05:49 models/dynamic_range_quantized_model.tflite
-rw-r--r-- 1 root root 6.0M Nov 19 05:48 models/ptq_quantized_model.tflite
-rw-r--r-- 1 root root 6.0M Nov 19 XX:XX models/qat_quantized_model.tflite       # 修复后生成
-rw-r--r-- 1 root root  24M Nov 19 XX:XX models/qat_fine_tuned_model.keras       # 修复后生成
```

---

## 总结

修复后的量化管道将生成 **4个文件**:
1. PTQ TFLite模型 (无训练，快速量化)
2. Dynamic Range TFLite模型 (权重量化)
3. QAT TFLite模型 (训练时量化，精度更高) ✨ **新增representative_dataset**
4. QAT Keras模型 (可继续微调)

所有文件保存在 `MLS4/models/` 目录下。
