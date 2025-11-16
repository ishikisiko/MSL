# 训练准确率修复说明

## 问题诊断

从您提供的训练日志来看，存在严重的**过拟合**问题：
- 训练集准确率：持续上升至 76.84%
- 验证集准确率：停滞在 5-7% 左右（接近随机猜测 1/100 = 1%）
- 验证损失：不断增大（从 6.06 → 7.98）

这表明模型在训练集上学习过度，但完全无法泛化到验证集。

## 已实施的修复

### 1. **降低学习率** ✅
```python
# 修改前: learning_rate=1e-3
# 修改后: learning_rate=5e-4
optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=5e-5)
```
- 更保守的学习率避免训练集上过快收敛
- 降低权重衰减系数 (1e-4 → 5e-5) 减少正则化压力

### 2. **增加Label Smoothing** ✅
```python
# 修改前: label_smoothing=0.05
# 修改后: label_smoothing=0.1
loss=AdaptiveCategoricalCrossentropy(num_classes=num_classes, label_smoothing=0.1)
```
- 更强的标签平滑防止模型对训练样本过度自信

### 3. **简化数据增强** ✅
```python
# 移除了过度的增强操作:
# ❌ RandomZoom
# ❌ RandomContrast
# ✅ 保留: RandomFlip, RandomTranslation, RandomRotation (降低强度)
```
- 减少了可能导致训练/验证不一致的增强操作

### 4. **增加Dropout** ✅
```python
# 修改前: dropout_rate=0.2
# 修改后: dropout_rate=0.3
```
- 更强的正则化防止过拟合

### 5. **优化学习率调度** ✅
- 添加了 **5 epoch warmup** 阶段
- 调整 ReduceLROnPlateau:
  - 监控指标: `val_accuracy` → `val_loss`
  - patience: 5 → 3 (更快响应验证集性能下降)
  - min_lr: 1e-5 → 1e-6

### 6. **调整Early Stopping** ✅
```python
# 监控 val_loss 而非 val_accuracy
# patience 从 10 增加到 12
```

### 7. **增加批次大小** ✅
```python
# 修改前: batch_size=128
# 修改后: batch_size=256
```
- 更大的批次提供更稳定的梯度估计
- 减少训练噪声

### 8. **延长训练周期** ✅
```python
# 修改前: epochs=30
# 修改后: epochs=50
```
- 给模型更多时间以较低学习率收敛

## 验证和测试

### 运行数据验证脚本
```bash
python verify_data.py
```

此脚本会检查：
- ✅ 训练/验证/测试集归一化是否正确
- ✅ 标签分布是否均匀
- ✅ 数据集之间没有重叠
- ✅ 数据管道正确配置（验证集无增强）
- ✅ 模型可以正常推理

### 重新训练模型
```bash
# 删除旧的checkpoint
rm -rf checkpoints/baseline_best.keras baseline_model.keras

# 重新训练
python main.py --train-baseline --batch-size 256
```

## 预期结果

修复后，您应该看到：
- ✅ 验证集准确率稳步上升（目标 >40% @ epoch 30）
- ✅ 训练/验证准确率差距缩小（<15%）
- ✅ 验证损失先下降再趋于稳定
- ✅ Top-5 准确率 >70%

典型的健康训练曲线：
```
Epoch 10: train_acc=0.35, val_acc=0.28, val_loss=2.8
Epoch 20: train_acc=0.55, val_acc=0.48, val_loss=2.1
Epoch 30: train_acc=0.68, val_acc=0.58, val_loss=1.8
Epoch 40: train_acc=0.75, val_acc=0.65, val_loss=1.6
```

## 如果问题仍然存在

如果验证准确率仍然很低，请检查：

### 1. 数据加载问题
```python
# 在 baseline_model.py 中添加调试代码
print(f"Train labels unique: {np.unique(y_train)}")
print(f"Val labels unique: {np.unique(y_val)}")
```

### 2. 标签编码问题
```python
# 确认 one-hot 编码正确
for x, y in val_ds.take(1):
    print(f"Label shape: {y.shape}")  # 应该是 (batch_size, 100)
    print(f"Label sum: {tf.reduce_sum(y, axis=-1)}")  # 应该全是 1.0
```

### 3. 尝试更简单的模型
```python
# 暂时减少模型复杂度测试
model = create_baseline_model(width_multiplier=0.5, dropout_rate=0.4)
```

## 进一步优化建议

如果基础训练稳定后，可以考虑：
1. **Mixup/CutMix** 数据增强
2. **Cosine Annealing** 学习率调度
3. **Gradient Clipping** (clip_norm=1.0)
4. **SAM optimizer** (Sharpness Aware Minimization)
5. **Test-Time Augmentation**

## 参考

- [CIFAR-100 SOTA](https://paperswithcode.com/sota/image-classification-on-cifar-100)
- [Label Smoothing Paper](https://arxiv.org/abs/1512.00567)
- [Bag of Tricks for Image Classification](https://arxiv.org/abs/1812.01187)
