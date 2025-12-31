# VoxCPM-ONNX 项目开发指南

## 项目概述

当前项目为社区维护的 bluryar/VoxCPM-ONNX 项目（仓库地址：https://github.com/bluryar/VoxCPM-ONNX.git），专注于将 VoxCPM 文本转语音模型导出为 ONNX 格式并提供高效的推理服务。

### 官方项目信息
- **官方 VoxCPM 项目仓库**: https://github.com/OpenBMB/VoxCPM
- **VoxCPM-0.5B 模型**: https://huggingface.co/openbmb/VoxCPM-0.5B
- **VoxCPM-1.5 最新模型**: https://huggingface.co/openbmb/VoxCPM1.5 (发布于 2025年12月5日)

## 当前项目状态

### 已实现功能
1. **ONNX 导出模块**:
   - AudioVAE 编码器导出 (`onnx/export_audio_vae_encoder.py`)
   - AudioVAE 解码器导出 (`onnx/export_audio_vae_decoder.py`)
   - VoxCPM 预填充模型导出 (`onnx/export_voxcpm_prefill.py`)
   - VoxCPM 解码步骤模型导出 (`onnx/export_voxcpm_decode.py`)

2. **推理引擎** (`src/onnx_infer/`):
   - ONNX Runtime 会话管理
   - 音频处理和 VAE 推理
   - 分词和输入构建
   - 完整推理循环

3. **服务接口** (`src/server/`):
   - FastAPI REST API 服务
   - 参考音频管理
   - TTS 语音生成

4. **部署支持**:
   - Docker 容器化部署
   - GPU 和 CPU 推理支持
   - 模型优化脚本

### 项目架构
```
VoxCPM-ONNX/
├── onnx/                    # ONNX 导出脚本
├── src/
│   ├── onnx_infer/          # ONNX 推理引擎
│   ├── server/              # FastAPI 服务
│   └── voxcpm/              # 核心模型实现
├── export.sh               # ONNX 导出主脚本
├── opt.sh                  # 模型优化脚本
├── infer.py               # 独立推理脚本
└── 配置文件和依赖管理
```

## VoxCPM-1.5 新特性

### 关键改进
| 特性 | VoxCPM | VoxCPM-1.5 |
|------|---------|------------|
| Audio VAE 采样率 | 16kHz | **44.1kHz** |
| LM Token 速率 | 12.5Hz | **6.25Hz** |
| Patch Size | 2 | **4** |
| SFT 支持 | ✅ | ✅ |
| LoRA 支持 | ✅ | ✅ |

### 主要优势
1. **更高音质**: 44.1kHz 采样率保留更多高频细节，提升语音克隆质量
2. **更高效率**: Token 速率降低至 6.25Hz，在保持性能的同时降低计算成本
3. **微调支持**: 新增 SFT 和 LoRA 微调脚本功能，提供更灵活的定制化选项
4. **稳定性增强**: 通过有效控制音频伪影和优化长文本音频生成效果

## 开发任务：VoxCPM-1.5 适配

### 目标
参考当前 bluryar/VoxCPM-ONNX 项目的已有实现，完成对 VoxCPM-1.5 最新模型的 ONNX 转换和推理实现。

### 主要挑战
1. **采样率变更**: 从 16kHz 升级到 44.1kHz，需要调整音频处理流程
2. **Patch Size 变化**: 从 2 增加到 4，影响模型输入输出维度
3. **Token 速率优化**: 从 12.5Hz 降低到 6.25Hz，提升生成效率
4. **架构调整**: 适配新版本的模型结构和参数

### 实施计划

#### 阶段 1: 模型分析和准备
- [ ] 下载和分析 VoxCPM-1.5 模型结构
- [ ] 对比 VoxCPM-0.5B 和 VoxCPM-1.5 的架构差异
- [ ] 识别需要修改的导出脚本和推理代码

#### 阶段 2: ONNX 导出适配
- [ ] 更新 `export_audio_vae_encoder.py` 支持 44.1kHz
- [ ] 更新 `export_audio_vae_decoder.py` 支持新的 patch size
- [ ] 修改 `export_voxcpm_prefill.py` 适配新模型结构
- [ ] 调整 `export_voxcpm_decode.py` 支持 6.25Hz token 速率

#### 阶段 3: 推理引擎更新
- [ ] 修改 `src/onnx_infer/audio.py` 处理 44.1kHz 音频
- [ ] 更新 `src/onnx_infer/inputs.py` 支持新的 patch size
- [ ] 调整 `src/onnx_infer/infer_loop.py` 适配新的 token 速率
- [ ] 更新相关常量和配置参数

#### 阶段 4: 服务和部署
- [ ] 测试新模型的推理性能
- [ ] 更新 API 接口支持新特性
- [ ] 优化 Docker 配置
- [ ] 更新文档和示例

### 技术要点

#### 音频处理调整
```python
# 需要更新的常量
SAMPLE_RATE = 44100  # 从 16000 更新到 44100
PATCH_SIZE = 4       # 从 2 更新到 4
TOKEN_RATE = 6.25    # 从 12.5 更新到 6.25
```

#### 导出参数调整
```bash
# 新的导出参数
AUDIO_LENGTH=44100   # 适配新的采样率
LATENT_LENGTH=200    # 可能需要调整潜变量长度
TIMESTEPS=10         # 保持或优化
```

#### 推理优化
- 利用更低的 token 速率提升推理速度
- 优化内存使用以支持更大的音频数据
- 调整批处理策略以适应新的模型特性

## 开发环境配置

### 依赖管理
项目使用 `uv` 进行环境管理，确保依赖一致性：
```bash
# 同步环境
uv sync

# 安装依赖
uv pip install -r requirement.txt  # CPU 版本
uv pip install -r requirement-gpu.txt  # GPU 版本
```

### 模型获取
```python
# 下载 VoxCPM-1.5 模型
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM1.5")

# 下载增强模型（可选）
from modelscope import snapshot_download
snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
snapshot_download('iic/SenseVoiceSmall')
```

## 验证和测试

### 功能验证
1. **ONNX 导出验证**: 确保所有模块正确导出
2. **推理精度验证**: 对比 PyTorch 和 ONNX 推理结果
3. **性能测试**: 测试推理速度和内存使用
4. **音质评估**: 评估 44.1kHz 音频质量提升

### 测试用例
- 基础文本转语音测试
- 参考音频语音克隆测试
- 长文本生成稳定性测试
- 不同语言支持测试

## 部署和使用

### Docker 部署
```bash
# GPU 版本
docker-compose up voxcpm-gpu

# CPU 版本  
docker-compose up voxcpm-cpu
```

### API 使用
```bash
# 健康检查
curl http://localhost:8100/health

# 上传参考音频
curl -X POST http://localhost:8100/ref_feat \
  -F "feat_id=my_voice" \
  -F "prompt_audio=@reference.wav"

# 语音合成
curl -X POST http://localhost:8100/tts \
  -F "input=测试文本" \
  -F "voice=my_voice" \
  -F "response_format=wav"
```

## 注意事项

1. **向后兼容性**: 确保 VoxCPM-0.5B 模型继续可用
2. **性能优化**: 利用新模型的效率优势
3. **质量保证**: 验证 44.1kHz 音频质量提升
4. **文档更新**: 及时更新相关文档和示例

## 资源链接

- 项目仓库: https://github.com/bluryar/VoxCPM-ONNX
- 官方 VoxCPM: https://github.com/OpenBMB/VoxCPM
- VoxCPM-1.5 模型: https://huggingface.co/openbmb/VoxCPM1.5
- API 文档: http://localhost:8100/docs (部署后访问)
