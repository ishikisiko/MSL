"""
快速验证改进后的知识蒸馏效果

对比改进前后的性能，验证以下方面：
1. 蒸馏损失是否增大到合理范围
2. 验证准确率是否显著提升
3. 过拟合是否得到缓解
4. 训练是否更加稳定
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少 TensorFlow 日志

import tensorflow as tf
import numpy as np
from baseline_model import create_baseline_mobilenetv2, prepare_compression_datasets
from part3_distillation import DistillationFramework
import json
from datetime import datetime


def create_student_architecture(width_multiplier: float) -> tf.keras.Model:
    """创建学生模型架构"""
    return create_baseline_mobilenetv2(
        input_shape=(32, 32, 3),
        num_classes=10,
        width_multiplier=width_multiplier
    )


def quick_test_improvements():
    """快速测试改进效果 - 使用小规模配置"""
    
    print("=" * 80)
    print("知识蒸馏改进效果验证")
    print("=" * 80)
    
    # 1. 创建或加载教师模型
    print("\n[1/4] 准备教师模型...")
    teacher_path = "results/baseline_model.keras"
    
    if os.path.exists(teacher_path):
        print(f"✓ 加载现有教师模型: {teacher_path}")
        teacher = tf.keras.models.load_model(teacher_path)
    else:
        print("⚠ 教师模型不存在，创建新模型...")
        teacher = create_baseline_mobilenetv2(
            input_shape=(32, 32, 3),
            num_classes=10,
            width_multiplier=1.0
        )
        # 快速训练教师模型（仅用于演示）
        (x_train, y_train, x_val, y_val, _, _, _) = prepare_compression_datasets()
        
        train_ds = tf.data.Dataset.from_tensor_slices((x_train[:5000], y_train[:5000]))
        train_ds = train_ds.shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)
        
        val_ds = tf.data.Dataset.from_tensor_slices((x_val[:1000], y_val[:1000]))
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        
        teacher.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        print("  训练教师模型 (5 epochs, 子集数据)...")
        teacher.fit(train_ds, epochs=5, validation_data=val_ds, verbose=0)
        
        os.makedirs("results", exist_ok=True)
        teacher.save(teacher_path)
        print(f"✓ 教师模型已保存: {teacher_path}")
    
    # 评估教师性能
    (_, _, x_val, y_val, _, _, _) = prepare_compression_datasets()
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    teacher_acc = teacher.evaluate(val_ds, verbose=0)[1]
    print(f"  教师模型验证准确率: {teacher_acc:.2%}")
    
    # 2. 初始化蒸馏框架
    print("\n[2/4] 初始化蒸馏框架...")
    framework = DistillationFramework(
        teacher_model=teacher,
        student_architecture=create_student_architecture,
        cache_datasets=True,
        batch_size=32
    )
    print("✓ 框架初始化完成")
    
    # 3. 快速测试温度优化（小规模）
    print("\n[3/4] 运行改进后的温度优化...")
    print("  配置: 温度范围 (3.0-10.0), 3 trials, 5 epochs, 50 steps/epoch")
    
    results = framework.temperature_optimization(
        temperature_range=(3.0, 10.0),
        num_trials=3,
        width_multiplier=0.5,
        epochs=5,  # 快速测试使用较少 epochs
        steps_per_epoch=50,
        save_path="results/improved_student_quick.keras"
    )
    
    # 4. 分析结果
    print("\n[4/4] 分析结果...")
    print("=" * 80)
    print("结果总结")
    print("=" * 80)
    
    # 提取关键指标
    temp_acc = results['temperature_accuracy_curve']
    best_temp = results['optimal_temperature']
    best_acc = max(temp_acc.values())
    
    print(f"\n📊 温度-准确率曲线:")
    for temp, acc in sorted(temp_acc.items(), key=lambda x: float(x[0])):
        marker = " ← 最优" if abs(float(temp) - best_temp) < 0.01 else ""
        print(f"  T={float(temp):5.2f}: {acc:6.2%}{marker}")
    
    print(f"\n✨ 最优配置:")
    print(f"  温度: {best_temp:.2f}")
    print(f"  准确率: {best_acc:.2%}")
    
    # 分析蒸馏损失
    print(f"\n🔍 蒸馏损失分析:")
    for idx, trial in enumerate(results['knowledge_transfer_metrics']):
        temp = trial['temperature']
        history = trial['history']
        
        # 获取最后一个 epoch 的损失
        final_student_loss = history['student_loss'][-1]
        final_distill_loss = history['distillation_loss'][-1]
        
        print(f"  T={temp:.2f}:")
        print(f"    学生损失: {final_student_loss:.4f}")
        print(f"    蒸馏损失: {final_distill_loss:.4f}")
        print(f"    比例: 1:{final_student_loss/max(final_distill_loss, 1e-8):.1f}")
    
    # 检查改进指标
    print(f"\n✅ 改进检查:")
    checks = []
    
    # 1. 验证准确率应 > 30% (改进前约 10%)
    if best_acc > 0.30:
        checks.append("✓ 验证准确率 > 30%")
        status_acc = True
    else:
        checks.append(f"✗ 验证准确率仅 {best_acc:.2%} (目标 > 30%)")
        status_acc = False
    
    # 2. 蒸馏损失应在合理范围 (0.01-0.1)
    avg_distill_loss = np.mean([
        trial['history']['distillation_loss'][-1] 
        for trial in results['knowledge_transfer_metrics']
    ])
    if 0.01 <= avg_distill_loss <= 0.1:
        checks.append(f"✓ 蒸馏损失在合理范围 ({avg_distill_loss:.4f})")
        status_loss = True
    else:
        checks.append(f"✗ 蒸馏损失异常 ({avg_distill_loss:.4f}, 目标 0.01-0.1)")
        status_loss = False
    
    # 3. 训练应该稳定（最后 3 epochs 准确率持续上升或稳定）
    sample_history = results['knowledge_transfer_metrics'][0]['history']['accuracy']
    if len(sample_history) >= 3:
        last_3_trend = sample_history[-1] - sample_history[-3]
        if last_3_trend >= -0.02:  # 允许轻微下降
            checks.append("✓ 训练稳定（准确率平稳上升）")
            status_stable = True
        else:
            checks.append(f"✗ 训练不稳定（最后3轮下降 {abs(last_3_trend):.2%}）")
            status_stable = False
    else:
        status_stable = False
    
    for check in checks:
        print(f"  {check}")
    
    # 总体评估
    print(f"\n{'='*80}")
    if status_acc and status_loss:
        print("🎉 改进验证成功！所有关键指标均达标。")
        print("   建议：可以运行完整实验（15 epochs, 100 steps/epoch）")
    elif status_acc:
        print("⚠ 改进部分成功：准确率提升明显，但需检查损失平衡。")
    else:
        print("❌ 改进未达预期，请检查：")
        print("   1. 教师模型性能是否足够好 (> 60%)")
        print("   2. 数据预处理是否正确")
        print("   3. GPU 内存是否充足")
    
    # 保存详细报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "teacher_accuracy": float(teacher_acc),
        "best_temperature": float(best_temp),
        "best_student_accuracy": float(best_acc),
        "temperature_accuracy_curve": {
            str(k): float(v) for k, v in temp_acc.items()
        },
        "checks": {
            "accuracy_pass": status_acc,
            "loss_balance_pass": status_loss,
            "training_stable": status_stable
        }
    }
    
    report_path = "results/improvement_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 详细报告已保存: {report_path}")
    
    return results


def full_test_improvements():
    """完整测试 - 使用推荐的配置"""
    
    print("\n" + "=" * 80)
    print("运行完整改进测试（这将需要较长时间）")
    print("=" * 80)
    
    # 加载教师
    teacher_path = "results/baseline_model.keras"
    if not os.path.exists(teacher_path):
        print("❌ 请先运行 quick_test_improvements() 创建教师模型")
        return
    
    teacher = tf.keras.models.load_model(teacher_path)
    
    # 初始化框架
    framework = DistillationFramework(
        teacher_model=teacher,
        student_architecture=create_student_architecture,
        batch_size=32
    )
    
    # 运行所有蒸馏方法
    all_results = {}
    
    print("\n[1/4] 温度优化...")
    all_results['temperature_opt'] = framework.temperature_optimization(
        save_path="results/student_temp_opt.keras"
    )
    
    print("\n[2/4] 渐进蒸馏...")
    all_results['progressive'] = framework.progressive_distillation(
        save_path="results/student_progressive.keras"
    )
    
    print("\n[3/4] 注意力转移...")
    all_results['attention'] = framework.attention_transfer(
        save_path="results/student_attention.keras"
    )
    
    print("\n[4/4] 特征匹配...")
    all_results['feature_matching'] = framework.feature_matching_distillation(
        save_path="results/student_feature.keras"
    )
    
    # 总结对比
    print("\n" + "=" * 80)
    print("完整测试结果对比")
    print("=" * 80)
    
    summary = {
        "温度优化": max(all_results['temperature_opt']['temperature_accuracy_curve'].values()),
        "渐进蒸馏": all_results['progressive']['final_student'].evaluate(
            framework._get_dataset("val"), verbose=0
        )[1] if all_results['progressive']['final_student'] else 0.0,
        "注意力转移": all_results['attention']['combined_distillation_results']['accuracy'],
        "特征匹配": all_results['feature_matching']['accuracy'],
    }
    
    for method, acc in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:12s}: {acc:6.2%}")
    
    print(f"\n🏆 最佳方法: {max(summary, key=summary.get)} ({max(summary.values()):.2%})")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    print("知识蒸馏改进验证脚本")
    print("=" * 80)
    print("选项:")
    print("  1. 快速测试 (推荐首次运行，约 5-10 分钟)")
    print("  2. 完整测试 (所有方法，约 1-2 小时)")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        choice = "2"
    else:
        choice = input("请选择 [1/2, 默认 1]: ").strip() or "1"
    
    if choice == "1":
        results = quick_test_improvements()
    elif choice == "2":
        results = full_test_improvements()
    else:
        print("❌ 无效选择")
        sys.exit(1)
    
    print("\n✓ 测试完成！")
