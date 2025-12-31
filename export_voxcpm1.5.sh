#!/usr/bin/env bash

# =====================================================================================
# VoxCPM-1.5 ONNX 导出与优化一键脚本
#
# 功能：
#   1) 依次导出四个模块到 ONNX：
#      - audio_vae_encoder.onnx (44.1kHz 支持)
#      - audio_vae_decoder.onnx (适配新的 patch_size)
#      - voxcpm_prefill.onnx (patch_size=4)
#      - voxcpm_decode_step.onnx (6.25Hz token 速率)
#   2) 调用 opt.sh 对导出的模型进行 onnxsim/onnxoptimizer/onnxslim 处理。
#
# 使用：
#   在仓库根目录执行：
#     bash export_voxcpm1.5.sh
#
#   可通过环境变量覆盖默认参数，例如：
#     MODEL_PATH=./models/VoxCPM-1.5 OUTPUT_DIR=./models/onnx_models_v15 TIMESTEPS=5 CFG_VALUE=2.0 bash export_voxcpm1.5.sh
#
# 依赖：
#   - Python 环境可用，且安装了 torch、onnx、onnxruntime
#   - opt.sh 所需的 onnxsim、onnxoptimizer、onnxslim 已在 PATH 中
# =====================================================================================
set PYTHONIOENCODING=utf-8
set TORCH_LOGS=+dynamic
set -euxo pipefail

# --- 参数配置（可用环境变量覆盖） ---
MODEL_PATH=${MODEL_PATH:-./models/VoxCPM-1.5}
OUTPUT_DIR=${OUTPUT_DIR:-./models/onnx_models_v15}
OPSET_VERSION=${OPSET_VERSION:-20}

# AudioVAE - VoxCPM-1.5 特定参数
AUDIO_LENGTH=${AUDIO_LENGTH:-44100}  # 44.1kHz 采样率
LATENT_LENGTH=${LATENT_LENGTH:-400}  # 适配更长的潜在序列
LATENT_DIM=${LATENT_DIM:-64}

# Prefill - VoxCPM-1.5 特定参数
PATCH_SIZE=${PATCH_SIZE:-4}  # VoxCPM-1.5 使用 patch_size=4
SEQ_LENGTH=${SEQ_LENGTH:-8}

# Decode - VoxCPM-1.5 特定参数
TIMESTEPS=${TIMESTEPS:-5}  # VoxCPM-1.5 可以使用更少的 timesteps
CFG_VALUE=${CFG_VALUE:-2.0}

# 验证参数
RTOL=${RTOL:-1e-3}
ATOL=${ATOL:-1e-4}
NUM_TESTS=${NUM_TESTS:-5}

echo "========================================="
echo "VoxCPM-1.5 ONNX 导出开始"
echo "========================================="
echo "[Init] 模型路径: ${MODEL_PATH}"
echo "[Init] 导出目录: ${OUTPUT_DIR}"
echo "[Init] VoxCPM-1.5 配置:"
echo "  - 采样率: ${AUDIO_LENGTH}Hz (1秒音频)"
echo "  - Patch Size: ${PATCH_SIZE}"
echo "  - 潜在长度: ${LATENT_LENGTH}"
echo "  - Diffusion Timesteps: ${TIMESTEPS}"
echo "  - CFG Value: ${CFG_VALUE}"
echo "========================================="

mkdir -p "${OUTPUT_DIR}"

# --- 1) 导出 AudioVAE Encoder ---
echo "[Step 1/4] 导出 audio_vae_encoder.onnx (44.1kHz)"
python onnx/export_audio_vae_encoder_v15.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --audio_length "${AUDIO_LENGTH}" \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 2) 导出 AudioVAE Decoder ---
echo "[Step 2/4] 导出 audio_vae_decoder.onnx (适配 patch_size=${PATCH_SIZE})"
python onnx/export_audio_vae_decoder_v15.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_length "${LATENT_LENGTH}" \
  --latent_dim "${LATENT_DIM}" \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 3) 导出 VoxCPM Prefill ---
echo "[Step 3/4] 导出 voxcpm_prefill.onnx (patch_size=${PATCH_SIZE})"
python onnx/export_voxcpm_prefill_v15.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --seq_length "${SEQ_LENGTH}" \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 4) 导出 VoxCPM Decode Step ---
echo "[Step 4/4] 导出 voxcpm_decode_step.onnx (timesteps=${TIMESTEPS})"
python onnx/export_voxcpm_decode_v15.py \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --timesteps "${TIMESTEPS}" \
  --cfg_value "${CFG_VALUE}" \
  --validate --num_tests "${NUM_TESTS}" --rtol "${RTOL}" --atol "${ATOL}" --opset_version "${OPSET_VERSION}"

# --- 5) 拷贝配置文件 ---
echo "[Step 5/5] 拷贝配置文件到输出目录..."
cp "${MODEL_PATH}/tokenizer.json" "${OUTPUT_DIR}"

# 创建版本信息文件
cat > "${OUTPUT_DIR}/voxcpm_version_info.json" << EOF
{
  "version": "1.5",
  "sample_rate": ${AUDIO_LENGTH},
  "patch_size": ${PATCH_SIZE},
  "token_rate": 6.25,
  "latent_dim": ${LATENT_DIM},
  "timesteps": ${TIMESTEPS},
  "cfg_value": ${CFG_VALUE},
  "export_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "========================================="
echo "VoxCPM-1.5 导出完成！"
echo "========================================="
echo "导出的模型文件："
ls -la "${OUTPUT_DIR}"/.onnx

echo ""
echo "配置文件："
ls -la "${OUTPUT_DIR}"/*.json

echo ""
echo "下一步："
echo "1. 运行 'bash opt.sh' 对模型进行优化"
echo "2. 使用 'python infer_v15.py' 测试推理功能"
echo "3. 启动服务进行 API 测试"
echo "========================================="