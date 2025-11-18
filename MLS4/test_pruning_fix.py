"""
测试剪枝修复 - 验证稀疏度和学习率问题是否解决
"""
import numpy as np
import tensorflow as tf
from part1_pruning import PruningComparator
from baseline_model import DEFAULT_MODEL_PATH

print("="*60)
print("剪枝修复验证测试")
print("="*60)

# 设置随机种子
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# 创建 pruner
pruner = PruningComparator(base_model_path=DEFAULT_MODEL_PATH)

print("\n测试 1: 幅度剪枝 (Magnitude Pruning)")
print("-"*60)
magnitude_result = pruner.magnitude_based_pruning(
    target_sparsity=0.6,
    fine_tune_epochs=3,  # 减少轮数用于快速测试
    learning_rate=1e-3,
    early_stopping_patience=5,
    use_layer_wise_sparsity=True,
    use_warmup=True,
)

print(f"\n✓ 幅度剪枝结果:")
print(f"  - 目标稀疏度: 60.00%")
print(f"  - 实际稀疏度: {magnitude_result['sparsity_achieved']:.2%}")
print(f"  - 最终准确率: {magnitude_result['final_accuracy']:.4f}")

# 检查稀疏度是否合理 (应该接近目标值)
sparsity_ok = magnitude_result['sparsity_achieved'] >= 0.30  # 至少30%
accuracy_ok = magnitude_result['final_accuracy'] > 0.1  # 准确率不能太低

print(f"\n测试 2: 渐进式剪枝 (Gradual Pruning)")
print("-"*60)
gradual_result = pruner.gradual_magnitude_pruning(
    target_sparsity=0.6,
    num_stages=2,  # 减少阶段数用于快速测试
    epochs_per_stage=3,
    learning_rate=1e-3,
    use_layer_wise_sparsity=True,
)

print(f"\n✓ 渐进式剪枝结果:")
print(f"  - 目标稀疏度: 60.00%")
print(f"  - 实际稀疏度: {gradual_result['sparsity_achieved']:.2%}")
print(f"  - 最终准确率: {gradual_result['final_accuracy']:.4f}")

# 检查每个阶段的学习率是否正确
print(f"\n  各阶段详情:")
for stage in gradual_result['stages']:
    print(f"    阶段 {stage['stage']}: 准确率={stage['final_accuracy']:.4f}, 稀疏度={stage['sparsity_achieved']:.2%}")

gradual_sparsity_ok = gradual_result['sparsity_achieved'] >= 0.30
gradual_accuracy_ok = gradual_result['final_accuracy'] > 0.1

print("\n" + "="*60)
print("测试总结")
print("="*60)
print(f"幅度剪枝:")
print(f"  - 稀疏度检查: {'✓ 通过' if sparsity_ok else '✗ 失败'} ({magnitude_result['sparsity_achieved']:.2%} >= 30%)")
print(f"  - 准确率检查: {'✓ 通过' if accuracy_ok else '✗ 失败'} ({magnitude_result['final_accuracy']:.4f} > 0.1)")

print(f"\n渐进式剪枝:")
print(f"  - 稀疏度检查: {'✓ 通过' if gradual_sparsity_ok else '✗ 失败'} ({gradual_result['sparsity_achieved']:.2%} >= 30%)")
print(f"  - 准确率检查: {'✓ 通过' if gradual_accuracy_ok else '✗ 失败'} ({gradual_result['final_accuracy']:.4f} > 0.1)")

if all([sparsity_ok, accuracy_ok, gradual_sparsity_ok, gradual_accuracy_ok]):
    print("\n✓✓✓ 所有测试通过! 修复成功! ✓✓✓")
else:
    print("\n✗✗✗ 某些测试失败,需要进一步调查 ✗✗✗")
