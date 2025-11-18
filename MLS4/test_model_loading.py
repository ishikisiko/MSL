"""
快速测试模型加载是否正常工作
"""
import tensorflow as tf
from baseline_model import load_baseline_model, DEFAULT_MODEL_PATH, CUSTOM_OBJECTS

print("="*60)
print("模型加载测试")
print("="*60)

print(f"\n自定义对象: {list(CUSTOM_OBJECTS.keys())}")

print(f"\n正在加载模型: {DEFAULT_MODEL_PATH}")
try:
    model = load_baseline_model(DEFAULT_MODEL_PATH)
    print(f"\n✓✓✓ 模型加载成功! ✓✓✓")
    print(f"  模型名称: {model.name}")
    print(f"  输入形状: {model.input_shape}")
    print(f"  输出形状: {model.output_shape}")
    print(f"  层数: {len(model.layers)}")
    print(f"  可训练参数: {model.count_params():,}")
    
    # 检查是否有 TFOpLambda 层
    tfop_layers = [layer for layer in model.layers if 'TFOpLambda' in type(layer).__name__ or 'multiply' in layer.name]
    if tfop_layers:
        print(f"\n  发现 TFOpLambda 层:")
        for layer in tfop_layers:
            print(f"    - {layer.name}: {type(layer).__name__}")
    
except Exception as e:
    print(f"\n✗✗✗ 模型加载失败! ✗✗✗")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
