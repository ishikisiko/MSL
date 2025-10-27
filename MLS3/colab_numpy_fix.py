"""
快速修复 Colab 环境中的 NumPy 版本冲突

使用方法（在 Colab 笔记本中）:
    !python colab_numpy_fix.py
    
或者直接在代码单元格中导入:
    import colab_numpy_fix
    colab_numpy_fix.fix_numpy()
"""

import subprocess
import sys
import os


def fix_numpy(target_version="2.0.0", verbose=True):
    """
    修复 NumPy 版本冲突
    
    Args:
        target_version: 目标 NumPy 版本（默认 2.0.0，与 TensorFlow 2.16+ 兼容）
        verbose: 是否输出详细信息
    """
    if verbose:
        print("🔧 正在修复 NumPy 版本冲突...")
        print(f"目标版本: NumPy {target_version}")
        print("-" * 50)
    
    try:
        # 1. 检查当前 NumPy 版本
        try:
            import numpy as np
            current_version = np.__version__
            if verbose:
                print(f"当前 NumPy 版本: {current_version}")
            
            if current_version == target_version:
                if verbose:
                    print(f"✓ NumPy 版本已经是 {target_version}，无需修复")
                return True
        except ImportError:
            if verbose:
                print("⚠ NumPy 未安装")
        
        # 2. 卸载现有 NumPy（静默）
        if verbose:
            print("\n步骤 1/3: 卸载冲突的 NumPy 版本...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "numpy"],
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None
        )
        
        # 3. 安装目标版本（静默大部分输出）
        if verbose:
            print(f"步骤 2/3: 安装 NumPy {target_version}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", f"numpy=={target_version}"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"✗ 安装失败: {result.stderr}")
            return False
        
        # 4. 验证安装
        if verbose:
            print("步骤 3/3: 验证安装...")
        
        # 清除已导入的 numpy 模块
        if 'numpy' in sys.modules:
            del sys.modules['numpy']
        if 'np' in sys.modules:
            del sys.modules['np']
        
        # 重新导入并验证
        import numpy as np
        installed_version = np.__version__
        
        if installed_version == target_version:
            if verbose:
                print(f"\n✓ NumPy 修复成功！")
                print(f"✓ 当前版本: {installed_version}")
            return True
        else:
            if verbose:
                print(f"\n⚠ 版本不匹配: 期望 {target_version}, 实际 {installed_version}")
            return False
            
    except Exception as e:
        if verbose:
            print(f"\n✗ 修复过程出错: {e}")
        return False


def check_environment(verbose=True):
    """检查当前环境状态"""
    if verbose:
        print("\n" + "=" * 50)
        print("📊 环境检查")
        print("=" * 50)
    
    try:
        # Python 版本
        if verbose:
            print(f"Python: {sys.version.split()[0]}")
        
        # NumPy 版本
        try:
            import numpy as np
            if verbose:
                print(f"NumPy: {np.__version__} ✓")
        except ImportError:
            if verbose:
                print("NumPy: 未安装 ✗")
        
        # TensorFlow 版本
        try:
            import tensorflow as tf
            if verbose:
                print(f"TensorFlow: {tf.__version__} ✓")
        except ImportError:
            if verbose:
                print("TensorFlow: 未安装 ✗")
        
        # GPU 检测
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                if verbose:
                    print(f"GPU: {len(gpus)} 个设备 ✓")
                    for i, gpu in enumerate(gpus):
                        print(f"  - GPU {i}: {gpu.name}")
            else:
                if verbose:
                    print("GPU: 未检测到 ⚠")
        except:
            if verbose:
                print("GPU: 无法检测")
        
        if verbose:
            print("=" * 50)
        
    except Exception as e:
        if verbose:
            print(f"环境检查出错: {e}")


def restart_runtime():
    """重启 Colab 运行时（清除所有状态）"""
    print("\n⚠ 即将重启 Colab 运行时...")
    print("这将清除所有变量和状态，但可以彻底解决冲突。")
    print("运行时重启后，请重新运行修复脚本。")
    
    import os
    os.kill(os.getpid(), 9)


def main():
    """命令行入口"""
    print("=" * 50)
    print("🔧 Colab NumPy 冲突修复工具")
    print("=" * 50)
    print()
    
    # 显示当前状态
    check_environment(verbose=True)
    
    # 执行修复
    print()
    success = fix_numpy(verbose=True)
    
    if success:
        print("\n" + "=" * 50)
        print("✅ 修复完成！现在可以安全导入 TensorFlow")
        print("=" * 50)
        print("\n建议的下一步：")
        print("  !pip install -q tensorflow keras")
        print("  import tensorflow as tf")
        print("  print('TensorFlow:', tf.__version__)")
    else:
        print("\n" + "=" * 50)
        print("⚠ 自动修复失败")
        print("=" * 50)
        print("\n请尝试手动方案：")
        print("1. 重启运行时: 运行时 -> 重启运行时")
        print("2. 运行以下命令:")
        print("   !pip install --force-reinstall numpy==1.26.4")
        print("   !pip install tensorflow==2.15.0")
        print("\n或查看完整文档: COLAB_TROUBLESHOOTING.md")


if __name__ == "__main__":
    main()
