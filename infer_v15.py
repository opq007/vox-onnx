#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure NumPy + ONNXRuntime inference script for VoxCPM-1.5

- Tokenizer: uses the same multi-character Chinese token splitting logic
  as mask_multichar_chinese_tokens in the original implementation
- Audio I/O: soundfile for reading/writing; resampling done via NumPy
- Models: ONNXRuntime sessions for prefill, decode step, VAE encoder/decoder
- VoxCPM-1.5 Support: 44.1kHz audio, patch_size=4, 6.25Hz token rate

This script avoids any torch/torchaudio dependency.
"""

import os
import sys
import argparse
import time
import json
import soundfile as sf
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.onnx_infer.constants import get_constants_for_version
from src.onnx_infer.runtime import (
    create_session_options,
    create_run_options,
    configure_providers,
    create_session,
    get_device_info_from_providers,
)
from src.onnx_infer.tokenize import load_tokenizer
from src.onnx_infer.inputs_v15 import build_inputs_v15
from src.onnx_infer.infer_loop_v15 import run_inference
from src.onnx_infer.vae import decode_audio
from src.voxcpm.version_config import get_model_config


def get_numpy_dtype(dtype_str: str) -> np.dtype:
    """Convert string dtype to numpy dtype"""
    dtype_map = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": np.float32,  # Fallback to fp32
    }
    return dtype_map.get(dtype_str, np.float32)


def load_model_config(models_dir: str) -> Dict[str, Any]:
    """加载模型配置信息"""
    config_path = os.path.join(models_dir, "voxcpm_version_info.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 回退到检测模型路径
    model_config = get_model_config(models_dir)
    return {
        "version": model_config.version,
        "sample_rate": model_config.sample_rate,
        "patch_size": model_config.patch_size,
        "token_rate": model_config.token_rate,
        "latent_dim": model_config.latent_dim,
        "timesteps": 5 if model_config.version == "1.5" else 10,  # VoxCPM-1.5 默认使用 5 timesteps
        "cfg_value": 2.0,
    }


def print_model_info(config: Dict[str, Any]):
    """打印模型信息"""
    print("=" * 60)
    print(f"VoxCPM-{config['version']} 推理引擎")
    print("=" * 60)
    print(f"采样率: {config['sample_rate']} Hz")
    print(f"Patch Size: {config['patch_size']}")
    print(f"Token 速率: {config['token_rate']} Hz")
    print(f"潜在维度: {config['latent_dim']}")
    print(f"Diffusion Timesteps: {config['timesteps']}")
    print(f"CFG Value: {config['cfg_value']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VoxCPM-1.5 ONNX-only inference (NumPy)")
    parser.add_argument("--text", type=str, required=True, help="Target text to synthesize")
    parser.add_argument("--output", type=str, default="output_v15.wav", help="Output WAV path")
    parser.add_argument("--prompt-audio", type=str, default="", help="Reference audio path (optional)")
    parser.add_argument("--prompt-text", type=str, default="", help="Reference text (optional)")
    parser.add_argument("--models-dir", type=str, default="./models/onnx_models_v15", help="ONNX models dir")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum generated length in patches")
    parser.add_argument("--max-len", type=int, default=2000, help="Maximum generated length in patches")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG value")
    parser.add_argument("--timesteps", type=int, default=0, help="Inference timesteps (0=auto from config)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda/tensorrt/trt/cpu/openvino/dml)")
    parser.add_argument("--device-id", type=int, default=0, help="Device id for GPU/NPU EPs")
    parser.add_argument("--max-threads", type=int, default=0, help="Max threads for ONNX Runtime (0=auto)")
    parser.add_argument("--optimize", action="store_true", help="Enable ONNX optimization")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Data type for inference (fp32/fp16/bf16)")
    parser.add_argument("--version", type=str, default="auto", choices=["auto", "0.5B", "1.5"], help="Force model version (auto=detect)")
    
    args = parser.parse_args()

    # 加载模型配置
    model_config = load_model_config(args.models_dir)
    
    # 版本检测和参数调整
    if args.version != "auto":
        model_config["version"] = args.version
    
    # 根据版本调整参数
    if model_config["version"] == "1.5":
        constants = get_constants_for_version("1.5")
        if args.timesteps == 0:
            args.timesteps = model_config.get("timesteps", 5)  # VoxCPM-1.5 默认 5 timesteps
        if args.max_threads == 0:
            args.max_threads = max(1, os.cpu_count() or 1)
    else:
        constants = get_constants_for_version("0.5B")
        if args.timesteps == 0:
            args.timesteps = model_config.get("timesteps", 10)  # VoxCPM-0.5B 默认 10 timesteps
        if args.max_threads == 0:
            args.max_threads = max(1, os.cpu_count() or 1)
    
    # 打印模型信息
    print_model_info(model_config)
    
    # 更新配置
    model_config.update({
        "cfg_value": args.cfg_value,
        "timesteps": args.timesteps,
        "device": args.device,
        "device_id": args.device_id,
    })

    models_dir = args.models_dir
    
    # 检查模型文件是否存在
    required_files = [
        "voxcpm_prefill.onnx",
        "voxcpm_decode_step.onnx", 
        "audio_vae_encoder.onnx",
        "audio_vae_decoder.onnx"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(models_dir, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"[ERROR] 缺少模型文件: {missing_files}")
        print("请先运行导出脚本生成 ONNX 模型文件")
        sys.exit(1)
    
    # Load tokenizer
    print("[INFO] 加载分词器...")
    tokenizer = load_tokenizer(models_dir)

    # Configure EPs and provider options per device
    providers, provider_options = configure_providers(args.device, args.max_threads, args.device_id)

    # Build session options
    session_opts = create_session_options(args.max_threads, args.optimize)
    run_opts = create_run_options()

    # Load ONNX sessions with timing
    print("[INFO] 加载 ONNX 模型...")
    t0 = time.perf_counter()
    prefill_sess = create_session(os.path.join(models_dir, "voxcpm_prefill.onnx"), session_opts, providers, provider_options)
    print(f"[OK] Prefill模型加载耗时: {time.perf_counter()-t0:.3f}s")
    
    t0 = time.perf_counter()
    decode_sess = create_session(os.path.join(models_dir, "voxcpm_decode_step.onnx"), session_opts, providers, provider_options)
    print(f"[OK] Decode模型加载耗时: {time.perf_counter()-t0:.3f}s")
    
    t0 = time.perf_counter()
    vae_enc_sess = create_session(os.path.join(models_dir, "audio_vae_encoder.onnx"), session_opts, providers, provider_options)
    print(f"[OK] VAE编码器加载耗时: {time.perf_counter()-t0:.3f}s")
    
    t0 = time.perf_counter()
    vae_dec_sess = create_session(os.path.join(models_dir, "audio_vae_decoder.onnx"), session_opts, providers, provider_options)
    print(f"[OK] VAE解码器加载耗时: {time.perf_counter()-t0:.3f}s")
    
    print("[OK] 所有模型加载完成！")

    # Load config for patch_size if available
    patch_size = model_config.get("patch_size", 2)
    try:
        config_path = os.path.join(models_dir, "prefill_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                patch_size = int(cfg.get("patch_size", patch_size))
    except Exception:
        pass

    # Get device info for OrtValue creation and numpy dtype
    device_type, device_id = get_device_info_from_providers(providers, args.device_id)
    inference_dtype = get_numpy_dtype(args.dtype)
    print(f"[CONFIG] 使用数据类型: {args.dtype} ({inference_dtype})")
    
    # 构建输入
    print("[INFO] 构建输入...")
    t_inp0 = time.perf_counter()
    
    # 使用 VoxCPM-1.5 专用的输入构建函数
    if model_config["version"] == "1.5":
        text_token, text_mask, audio_feat, audio_mask = build_inputs_v15(
            tokenizer=tokenizer,
            target_text=args.text,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_audio if args.prompt_audio else None,
            vae_enc_sess=vae_enc_sess,
            patch_size=patch_size,
            inference_dtype=inference_dtype,
            run_options=run_opts,
        )
    else:
        # 回退到通用函数 (用于 VoxCPM-0.5B)
        from src.onnx_infer.inputs import build_inputs
        text_token, text_mask, audio_feat, audio_mask = build_inputs(
            tokenizer=tokenizer,
            target_text=args.text,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_audio if args.prompt_audio else None,
            vae_enc_sess=vae_enc_sess,
            patch_size=patch_size,
            inference_dtype=inference_dtype,
            run_options=run_opts,
            sample_rate=constants["SAMPLE_RATE"],
            chunk_size=constants["CHUNK_SIZE"],
        )
    
    print(f"[OK] 输入构建总耗时: {time.perf_counter()-t_inp0:.3f}s")

    print("[INFO] 开始推理...")
    # Run inference
    t_inf0 = time.perf_counter()
    latents = run_inference(
        prefill_sess,
        decode_sess,
        text_token,
        text_mask,
        audio_feat,
        audio_mask,
        min_len=args.min_len,
        max_len=args.max_len,
        cfg_value=args.cfg_value,
        timesteps=args.timesteps,
        device_type=device_type,
        device_id=device_id,
        inference_dtype=inference_dtype,
        run_options=run_opts,
        patch_size=patch_size,
    )
    inference_time = time.perf_counter() - t_inf0
    print(f"[OK] 推理完成，耗时: {inference_time:.3f}s")

    print("[INFO] 解码音频...")
    # Decode audio
    t_dec0 = time.perf_counter()
    audio = decode_audio(
        vae_dec_sess, 
        latents, 
        device_type=device_type, 
        device_id=device_id, 
        inference_dtype=inference_dtype, 
        run_options=run_opts
    )
    decode_time = time.perf_counter() - t_dec0
    print(f"[OK] 音频解码完成，耗时: {decode_time:.3f}s")

    # Save audio and report stats
    sf.write(args.output, audio, constants["SAMPLE_RATE"])
    
    # 计算统计信息
    duration = len(audio) / float(constants["SAMPLE_RATE"])
    total_time = inference_time + decode_time
    rtf = total_time / duration  # Real-Time Factor
    
    print("=" * 60)
    print("[OK] 语音合成完成！")
    print("=" * 60)
    print(f"输出文件: {args.output}")
    print(f"音频时长: {duration:.2f}s")
    print(f"推理耗时: {inference_time:.3f}s")
    print(f"解码耗时: {decode_time:.3f}s")
    print(f"总耗时: {total_time:.3f}s")
    print(f"实时率 (RTF): {rtf:.3f}")
    print(f"采样率: {constants['SAMPLE_RATE']} Hz")
    print("=" * 60)


if __name__ == "__main__":
    main()