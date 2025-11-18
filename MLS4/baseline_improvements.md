# 基线模型改进方案

## 当前问题分析

### 训练结果
- 最佳验证准确率: 54.34% @ epoch 89
- 训练准确率: 72.71% @ epoch 100
- 过拟合差距: ~18%
- **未达到目标准确率 >75%**

### 主要问题
1. **严重过拟合** - 训练集/验证集准确率差距大
2. **模型容量不足或优化不当** - 验证准确率远低于目标
3. **训练不稳定** - epoch 9/14/19等出现验证准确率下降
4. **学习率调度可能不当** - 后期学习率过低限制收敛

---

## 改进策略

### 1. 数据增强优化 (预期提升: +5-8%)

**当前问题**: 增强可能不够充分
```python
# 增强当前的数据增强管道
_AUGMENTATION_PIPELINE = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    # 新增:
    tf.keras.layers.RandomContrast(0.2),  # 对比度调整
    tf.keras.layers.RandomBrightness(0.2),  # 亮度调整
], name="enhanced_aug")

# 添加MixUp或CutMix增强
def mixup_augmentation(images, labels, alpha=0.2):
    """MixUp数据增强"""
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([batch_size, 1, 1, 1], 0, alpha)
    
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1 - lam) * tf.gather(images, indices)
    mixed_labels = lam[:, 0, 0, 0:1] * labels + (1 - lam[:, 0, 0, 0:1]) * tf.gather(labels, indices)
    
    return mixed_images, mixed_labels
```

### 2. 正则化增强 (预期提升: +3-5%)

**当前问题**: Dropout 0.3可能不够
```python
# 方案A: 增加Dropout
dropout_rate = 0.4  # 从0.3提升到0.4

# 方案B: 添加Stochastic Depth
def stochastic_depth_layer(x, survival_prob=0.8):
    if not tf.keras.backend.learning_phase():
        return x
    keep_prob = survival_prob
    random_tensor = keep_prob + tf.random.uniform([tf.shape(x)[0], 1, 1, 1])
    binary_tensor = tf.floor(random_tensor)
    return x / keep_prob * binary_tensor

# 方案C: 增加Weight Decay
weight_decay = 2e-4  # 从1e-4提升到2e-4
```

### 3. 模型架构调整 (预期提升: +5-10%)

**当前问题**: EfficientNet-B0可能容量不足

#### 选项A: 切换到更强的架构
```python
# 使用EfficientNet-B1或B2
backbone = tf.keras.applications.EfficientNetB1(
    include_top=False,
    weights=None,  # 或使用'imagenet'预训练
    input_tensor=x,
    drop_connect_rate=0.2,
)
```

#### 选项B: 增加头部容量
```python
# 当前bottleneck_units = 1280
bottleneck_units = 1536  # 增加到1536

# 添加多层分类头
x = tf.keras.layers.Dense(1536, activation="swish")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(768, activation="swish")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(100, activation="softmax")(x)
```

#### 选项C: 使用预训练权重
```python
# 最简单但可能最有效的改进
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',  # 使用ImageNet预训练
    input_tensor=x,
)
# 冻结前面的层,只训练后面的层
for layer in backbone.layers[:100]:
    layer.trainable = False
```

### 4. 学习率调度优化 (预期提升: +2-4%)

**当前问题**: 
- 最小学习率0.02可能过高
- warmup占比10%可能不够

```python
lr_schedule = WarmupCosineDecay(
    base_learning_rate=1e-3,  # 从5e-4提升
    total_steps=total_steps,
    warmup_steps=int(total_steps * 0.15),  # 从10%增加到15%
    min_lr_ratio=0.005,  # 从0.02降低到0.005
)

# 或使用余弦重启
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

lr_schedule = CosineDecayRestarts(
    initial_learning_rate=1e-3,
    first_decay_steps=steps_per_epoch * 20,  # 每20个epoch重启
    t_mul=1.5,
    m_mul=0.8,
    alpha=1e-5,
)
```

### 5. 训练策略优化 (预期提升: +3-6%)

#### 增加训练轮数
```python
epochs = 150  # 从100增加到150
# 配合更长的patience
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=25,  # 从15增加到25
    restore_best_weights=True,
)
```

#### 使用渐进式训练
```python
# 阶段1: 冻结backbone,训练头部 (30 epochs)
# 阶段2: 解冻部分backbone,小学习率训练 (50 epochs)  
# 阶段3: 全模型训练 (70 epochs)
```

### 6. 标签平滑优化

**当前**: label_smoothing=0.1
```python
# 尝试调整
loss = AdaptiveCategoricalCrossentropy(
    num_classes=100, 
    label_smoothing=0.15  # 从0.1增加到0.15
)
```

### 7. 批量大小和优化器调整

```python
# 当前: batch_size=256, AdamW
# 方案A: 减小批量大小以增加梯度更新频率
batch_size = 128  # 从256降到128

# 方案B: 使用SGD with Momentum (通常在CIFAR上表现更好)
optimizer_name = "sgdw"
learning_rate = 0.1  # SGD需要更大的学习率
momentum = 0.9
```

---

## 推荐实施顺序

### 第一优先级 (立即实施)
1. ✅ **使用ImageNet预训练权重** - 最简单有效
2. ✅ **增加训练轮数到150** - 模型还在学习
3. ✅ **添加MixUp/CutMix** - 显著减少过拟合

### 第二优先级 (快速见效)
4. ✅ **调整学习率调度** - 提高base_lr,降低min_lr
5. ✅ **增强正则化** - 提高dropout到0.4
6. ✅ **切换到EfficientNet-B1** - 增加模型容量

### 第三优先级 (精细调优)
7. ⚙️ **优化数据增强** - 添加对比度/亮度调整
8. ⚙️ **尝试不同优化器** - 测试SGD vs AdamW
9. ⚙️ **多阶段训练** - 渐进式解冻

---

## 快速改进代码模板

```python
# 最小改动版本 - 预期提升到65-70%
def create_improved_baseline_v1():
    """使用预训练权重 + 调整超参数"""
    
    # 使用预训练权重
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',  # 关键改进
        input_shape=(128, 128, 3),
    )
    
    # 冻结早期层
    for layer in backbone.layers[:100]:
        layer.trainable = False
    
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Resizing(128, 128)(inputs)
    x = backbone(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # 增加dropout
    x = tf.keras.layers.Dense(1280, activation="swish")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(100, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # 调整学习率
    lr_schedule = WarmupCosineDecay(
        base_learning_rate=1e-3,  # 提高
        total_steps=total_steps,
        warmup_steps=int(total_steps * 0.15),
        min_lr_ratio=0.005,  # 降低
    )
    
    return model

# 进阶版本 - 预期提升到70-75%
def create_improved_baseline_v2():
    """预训练 + 更强架构 + MixUp"""
    
    # 使用更强的EfficientNet-B1
    backbone = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights='imagenet',
        input_shape=(128, 128, 3),
    )
    
    # 添加MixUp到训练流程
    # (需要在训练循环中实现)
    
    return model
```

---

## 预期结果

| 改进方案 | 预期验证准确率 | 实施难度 | 训练时间增加 |
|---------|--------------|---------|-------------|
| 当前基线 | 54.34% | - | - |
| + 预训练权重 | 62-65% | 低 | 0% |
| + 增加训练轮数 | 64-67% | 低 | +50% |
| + MixUp增强 | 66-70% | 中 | +10% |
| + EfficientNet-B1 | 68-72% | 低 | +30% |
| + 优化学习率 | 70-74% | 低 | 0% |
| 全部组合 | **74-78%** | 中 | +60% |

---

## 调试检查清单

- [ ] 检查数据预处理是否正确 (均值/方差归一化)
- [ ] 验证数据增强是否正常工作
- [ ] 确认标签格式正确 (one-hot vs sparse)
- [ ] 检查学习率是否合理
- [ ] 监控训练/验证loss曲线
- [ ] 检查是否有梯度爆炸/消失
- [ ] 验证EarlyStopping是否过早触发
- [ ] 确认模型保存/加载正确

---

## 下一步行动

1. **立即执行**: 修改baseline_model.py,添加预训练权重
2. **快速验证**: 训练30 epochs验证改进效果
3. **迭代优化**: 根据结果逐步添加其他改进
4. **文档记录**: 记录每个改进的具体效果

**目标**: 在3-5次迭代内达到>75%准确率
