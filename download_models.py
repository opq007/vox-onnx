#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 VoxCPM-1.5 模型的脚本 - 使用 ModelScope
"""
import os
import sys

def download_voxcpm_models():
    """下载所需的 VoxCPM 模型"""
    
    # 创建模型目录
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 导入 ModelScope
    try:
        from modelscope import snapshot_download
        print("[OK] ModelScope 库加载成功")
    except ImportError:
        print("[ERROR] 请先安装 ModelScope: pip install modelscope")
        return False
    
    # 下载 VoxCPM-1.5 模型
    print("正在从 ModelScope 下载 VoxCPM-1.5 模型...")
    try:
        snapshot_download(
            'AI-ModelScope/VoxCPM-1.5', 
            local_dir=os.path.join(models_dir, "VoxCPM-1.5"),
            cache_dir=os.path.join(models_dir, ".cache")
        )
        print("[OK] VoxCPM-1.5 模型下载完成")
    except Exception as e:
        print(f"[ERROR] VoxCPM-1.5 下载失败: {e}")
        print("[INFO] 尝试备用源...")
        try:
            # 备用：如果 ModelScope 没有该模型，尝试从 HuggingFace 下载
            from huggingface_hub import snapshot_download as hf_download
            hf_download(
                "openbmb/VoxCPM1.5", 
                local_dir=os.path.join(models_dir, "VoxCPM-1.5"),
                local_dir_use_symlinks=False
            )
            print("[OK] VoxCPM-1.5 从 HuggingFace 下载完成")
        except Exception as e2:
            print(f"[ERROR] 备用源也失败: {e2}")
            return False
    
    # 下载 VoxCPM-0.5B 模型（用于对比）
    print("正在从 ModelScope 下载 VoxCPM-0.5B 模型...")
    try:
        snapshot_download(
            'AI-ModelScope/VoxCPM-0.5B', 
            local_dir=os.path.join(models_dir, "VoxCPM-0.5B"),
            cache_dir=os.path.join(models_dir, ".cache")
        )
        print("[OK] VoxCPM-0.5B 模型下载完成")
    except Exception as e:
        print(f"[ERROR] VoxCPM-0.5B 下载失败: {e}")
        print("[INFO] 尝试备用源...")
        try:
            # 备用：从 HuggingFace 下载
            from huggingface_hub import snapshot_download as hf_download
            hf_download(
                "openbmb/VoxCPM-0.5B", 
                local_dir=os.path.join(models_dir, "VoxCPM-0.5B"),
                local_dir_use_symlinks=False
            )
            print("[OK] VoxCPM-0.5B 从 HuggingFace 下载完成")
        except Exception as e2:
            print(f"[ERROR] 备用源也失败: {e2}")
    
    # 下载增强模型（可选）
    print("正在下载增强模型...")
    try:
        snapshot_download(
            'iic/speech_zipenhancer_ans_multiloss_16k_base',
            local_dir=os.path.join(models_dir, "speech_zipenhancer"),
            cache_dir=os.path.join(models_dir, ".cache")
        )
        print("[OK] ZipEnhancer 下载完成")
        
        snapshot_download(
            'iic/SenseVoiceSmall',
            local_dir=os.path.join(models_dir, "SenseVoiceSmall"),
            cache_dir=os.path.join(models_dir, ".cache")
        )
        print("[OK] SenseVoiceSmall 下载完成")
    except Exception as e:
        print(f"[WARNING] 增强模型下载失败（可选）: {e}")
    
    # 验证下载结果
    print("\n验证下载结果...")
    required_files = {
        "VoxCPM-1.5": ["config.json", "tokenizer.json"],
        "VoxCPM-0.5B": ["config.json", "tokenizer.json"]
    }
    
    all_success = True
    for model_name, files in required_files.items():
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            missing_files = []
            for file_name in files:
                file_path = os.path.join(model_path, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                print(f"[WARNING] {model_name} 缺少文件: {missing_files}")
                all_success = False
            else:
                print(f"[OK] {model_name} 文件完整")
        else:
            print(f"[ERROR] {model_name} 目录不存在")
            all_success = False
    
    return all_success

def main():
    """主函数"""
    print("=" * 60)
    print("VoxCPM 模型下载工具 (ModelScope)")
    print("=" * 60)
    
    # 设置控制台编码
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    
    # 检查网络连接
    try:
        import requests
        response = requests.get("https://www.modelscope.cn", timeout=5)
        if response.status_code == 200:
            print("[OK] 网络连接正常")
        else:
            print("[WARNING] 网络连接可能有问题")
    except Exception as e:
        print(f"[WARNING] 无法检查网络连接: {e}")
    
    # 执行下载
    success = download_voxcpm_models()
    
    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] 所有模型下载完成！")
        print("下一步:")
        print("1. 运行 bash export_voxcpm1.5.sh 导出 ONNX 模型")
        print("2. 使用 python infer_v15.py 测试推理功能")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[PARTIAL] 部分模型下载失败")
        print("请检查网络连接或手动下载缺失的模型")
        print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)