"""
快速打包脚本 - 准备上传到 Google Colab

运行此脚本将项目文件打包成 ZIP，方便上传到 Colab。

Usage:
    python prepare_colab.py
"""

import os
import zipfile
from datetime import datetime
import shutil


def create_colab_package():
    """创建适合 Colab 运行的项目包."""
    
    print("=" * 70)
    print("MLS3 Colab 打包工具")
    print("=" * 70)
    
    # 定义需要打包的文件
    required_files = [
        "part1_baseline_model.py",
        "part2_optimizations.py",
        "part3_modeling.py",
        "part3_deployment.py",
        "performance_profiler.py",
        "optimization_framework.py",
        "run_optimizations.py",
        "requirements.txt",
        "README.md",
        "HOWTO_RUN.md",
        "COLAB_SETUP.md",
    ]
    
    optional_files = [
        "baseline_mobilenetv2.keras",  # 如果已训练
        "quick_test.py",
        "demo_notebook.ipynb",
    ]
    
    # 检查文件
    print("\n检查必需文件...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠ 警告: {len(missing_files)} 个必需文件缺失:")
        for file in missing_files:
            print(f"    - {file}")
        response = input("\n是否继续打包? (y/n): ")
        if response.lower() != 'y':
            print("打包已取消")
            return
    
    print("\n检查可选文件...")
    available_optional = []
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
            available_optional.append(file)
        else:
            print(f"  ○ {file} (跳过)")
    
    # 创建 ZIP 文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"MLS3_colab_{timestamp}.zip"
    
    print(f"\n创建压缩包: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加必需文件
        for file in required_files:
            if os.path.exists(file):
                zipf.write(file, f"MLS3/{file}")
                print(f"  + {file}")
        
        # 添加可选文件
        for file in available_optional:
            zipf.write(file, f"MLS3/{file}")
            print(f"  + {file}")
        
        # 创建必要的目录结构（空目录）
        dirs_to_create = [
            "MLS3/optimized_models/",
            "MLS3/logs/",
            "MLS3/results/",
            "MLS3/platform_configs/",
        ]
        
        for dir_path in dirs_to_create:
            # ZIP 中添加空目录需要在路径末尾加 /
            zipf.writestr(dir_path, '')
            print(f"  + {dir_path}")
    
    # 显示文件信息
    zip_size = os.path.getsize(zip_filename)
    zip_size_mb = zip_size / (1024 * 1024)
    
    print("\n" + "=" * 70)
    print("✓ 打包完成!")
    print("=" * 70)
    print(f"\n文件名: {zip_filename}")
    print(f"大小: {zip_size_mb:.2f} MB")
    print(f"位置: {os.path.abspath(zip_filename)}")
    
    # 显示使用说明
    print("\n" + "-" * 70)
    print("📤 上传到 Colab 的步骤:")
    print("-" * 70)
    print("1. 访问 https://colab.research.google.com")
    print("2. 新建笔记本或打开 colab_setup.ipynb")
    print("3. 运行上传单元格:")
    print("   from google.colab import files")
    print("   uploaded = files.upload()")
    print(f"4. 选择并上传: {zip_filename}")
    print("5. 解压并运行:")
    print("   !unzip -q MLS3_colab_*.zip")
    print("   %cd MLS3")
    print("   !pip install -q -r requirements.txt")
    print("   !python run_optimizations.py")
    
    # 生成快速启动代码
    print("\n" + "-" * 70)
    print("🚀 快速启动代码（复制到 Colab）:")
    print("-" * 70)
    print("""
# 上传并解压
from google.colab import files
uploaded = files.upload()

!unzip -q MLS3_colab_*.zip
%cd MLS3

# 安装依赖
!pip install -q tensorflow keras numpy pandas matplotlib seaborn
!pip install -q psutil memory-profiler tensorflow-model-optimization
!pip install -q onnx onnxruntime scikit-learn tqdm pyyaml

# 检查 GPU
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))

# 运行流程（如果已有基线模型）
!python run_optimizations.py

# 或先训练基线（30-60分钟）
# !python part1_baseline_model.py
# !python run_optimizations.py
    """)
    
    print("\n" + "=" * 70)
    print("详细说明请查看: COLAB_SETUP.md")
    print("=" * 70)


def create_lightweight_package():
    """创建轻量级包（不含模型文件）."""
    
    print("\n创建轻量级包（不含已训练模型）...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"MLS3_colab_lite_{timestamp}.zip"
    
    files_to_include = [
        "part1_baseline_model.py",
        "part2_optimizations.py",
        "part3_modeling.py",
        "part3_deployment.py",
        "performance_profiler.py",
        "optimization_framework.py",
        "run_optimizations.py",
        "requirements.txt",
        "quick_test.py",
        "COLAB_SETUP.md",
    ]
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file, f"MLS3/{file}")
    
    zip_size = os.path.getsize(zip_filename)
    zip_size_kb = zip_size / 1024
    
    print(f"✓ 轻量级包创建完成: {zip_filename}")
    print(f"  大小: {zip_size_kb:.2f} KB")
    print("  注意: 需要在 Colab 中训练基线模型")


def main():
    """主函数."""
    print("\nMLS3 项目打包工具\n")
    print("请选择打包方式:")
    print("1. 完整包（包含所有文件，如果有已训练模型）")
    print("2. 轻量级包（仅代码文件，需在 Colab 训练）")
    print("3. 两者都创建")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        create_colab_package()
    elif choice == '2':
        create_lightweight_package()
    elif choice == '3':
        create_colab_package()
        print("\n" + "=" * 70 + "\n")
        create_lightweight_package()
    else:
        print("无效选择，退出")
        return
    
    print("\n✓ 打包完成！准备好上传到 Colab 了。")


if __name__ == "__main__":
    main()
