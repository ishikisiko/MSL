"""
快速测试 - 仅测试模型是否能加载
"""
import sys
print("开始测试模型加载...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow 版本: {tf.__version__}")
    
    from baseline_model import load_baseline_model, DEFAULT_MODEL_PATH, CUSTOM_OBJECTS
    print(f"✓ 导入成功,自定义对象: {list(CUSTOM_OBJECTS.keys())}")
    
    print(f"\n正在加载模型: {DEFAULT_MODEL_PATH}")
    model = load_baseline_model(DEFAULT_MODEL_PATH)
    
    print(f"\n{'='*60}")
    print("✓✓✓ 模型加载成功! ✓✓✓")
    print(f"{'='*60}")
    print(f"模型名称: {model.name}")
    print(f"输入形状: {model.input_shape}")
    print(f"输出形状: {model.output_shape}")
    print(f"总层数: {len(model.layers)}")
    
    # 统计不同类型的层
    layer_types = {}
    for layer in model.layers:
        layer_type = type(layer).__name__
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    print(f"\n层类型统计:")
    for layer_type, count in sorted(layer_types.items()):
        print(f"  {layer_type}: {count}")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n{'='*60}")
    print("✗✗✗ 模型加载失败! ✗✗✗")
    print(f"{'='*60}")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    
    import traceback
    print(f"\n完整错误堆栈:")
    traceback.print_exc()
    
    sys.exit(1)
