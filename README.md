# Qwen3-TTS Server (Docker)

基于 [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) 的文本转语音（TTS）服务，使用 GGML + Vulkan 后端，可在 Intel / AMD 集成显卡上本地推理，提供 OpenAI 兼容的 HTTP API。

- 24 kHz 单声道输出，支持 11 种语言（含普通话及方言）
- 流式 PCM 实时返回，或一次性 WAV 文件
- 支持声音克隆（零样本 x-vector / 带文案 ICL），克隆特征可持久化为 `.spk` / `.rvq` 文件
- Vulkan 后端，适配 Intel iGPU（ANV）/ AMD（RADV）

> 本镜像仅支持 `linux/amd64`。

---

## 三种模型模式

Qwen3-TTS 的 Talker 模型有三种模式，按文件名区分，加载时只需指定一个 Talker + 一个共享 Codec：

| 模式 | 文件名示例 | 说明 | 体积 |
|------|-----------|------|------|
| `base` | `qwen-talker-1.7b-base-Q8_0.gguf` | 默认音色；支持声音克隆（参考音频或预编码 `.spk`/`.rvq`） | 0.6B / 1.7B |
| `customvoice` | `qwen-talker-1.7b-customvoice-Q8_0.gguf` | 内置命名音色 + 情感/方言（vivian、eric 四川话、dylan 北京话 等） | 0.6B / 1.7B |
| `voicedesign` | `qwen-talker-1.7b-voicedesign-Q8_0.gguf` | 通过自然语言属性指令设计音色（性别/年龄/音高/风格） | 仅 1.7B |

Codec（共享）：`qwen-tokenizer-12hz-Q8_0.gguf`，所有模式通用。

---

## 快速开始

### 1. 下载模型

镜像不含模型，需自行下载放入 `./models/` 目录。使用 HF Mirror（国内加速）：

```bash
mkdir -p models
# 国内用户使用镜像站
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"

# Talker（按需选择一种模式 + 一种体积）
hf download Serveurperso/Qwen3-TTS-GGUF \
  qwen-talker-0.6b-base-Q8_0.gguf \
  --local-dir models

# Codec（所有模式共享）
hf download Serveurperso/Qwen3-TTS-GGUF \
  qwen-tokenizer-12hz-Q8_0.gguf \
  --local-dir models
```

也可直接从 <https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF> 手动下载。

### 2. 启动服务

```bash
docker compose up -d

# 或本地构建：
docker build -t qwen3-tts .
docker run -d --name qwen3-tts -p 8080:8080 \
  --device /dev/dri:/dev/dri \
  -v "$PWD/models:/app/models" \
  -v "$PWD/voices:/app/voices" \
  -e TALKER_MODEL=/app/models/qwen-talker-0.6b-base-Q8_0.gguf \
  -e CODEC_MODEL=/app/models/qwen-tokenizer-12hz-Q8_0.gguf \
  qwen3-tts
```

### 3. 验证

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

---

## API 文档

### `POST /v1/audio/speech`（OpenAI 兼容）

文本转语音。`response_format` 默认 `pcm`（流式 s16le 24kHz 单声道，边生成边推送），设为 `wav` 则返回完整 RIFF 文件。

请求体（JSON）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `input` | string | **必填**，要朗读的文本 |
| `voice` | string | 内置音色名（仅 `customvoice` 模式有效，如 `vivian`） |
| `instructions` | string | 属性指令（仅 `voicedesign` 模式有效，如 `"male, young adult, moderate pitch"`） |
| `response_format` | string | `pcm`（默认，流式）或 `wav`（一次性文件） |
| `speed` | number | 预留字段，当前不参与变速 |

示例：

```bash
# 流式 PCM
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，这是一段测试。","voice":"vivian"}' \
  --output out.pcm

# 一次性 WAV
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world.","response_format":"wav"}' \
  --output out.wav
```

### `GET /v1/models`

返回当前加载的模型信息。

```bash
curl http://localhost:8080/v1/models
```

### `POST /v1/voices/clone`（声音克隆 + 持久化）

上传参考音频，提取声纹特征并持久化保存。后续合成时通过 `voice` 参数指定克隆音色名称即可复用，无需重复提取。

请求体（`multipart/form-data`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `file` | binary | **必填**，参考音频文件（WAV 格式，建议 5-15 秒） |
| `voice_id` | string | **必填**，自定义音色名称（如 `yuehua`），后续合成时作为 `voice` 参数使用 |
| `ref_text` | string | 可选，参考音频对应的文本内容（提高克隆质量） |

```bash
curl -X POST http://localhost:8080/v1/voices/clone \
  -F "file=@reference.wav" \
  -F "voice_id=yuehua" \
  -F "ref_text=这是参考音频的文本内容"
# {"voice_id":"yuehua","status":"created"}
```

克隆后即可用于合成：

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，我是月华。","voice":"yuehua"}' \
  --output out.pcm
```

### `GET /v1/voices`

返回内置音色和已克隆音色列表。

```bash
curl http://localhost:8080/v1/voices
# {"builtin":["vivian","eric","ryan"],"cloned":["yuehua","xiaoxing"]}
```

### `DELETE /v1/voices/{voice_id}`

删除已克隆的音色。

```bash
curl -X DELETE http://localhost:8080/v1/voices/yuehua
# {"voice_id":"yuehua","status":"deleted"}
```

### `GET /health`

存活探针，固定返回 `{"status":"ok"}`。

---

## 声音克隆工作流

```
克隆阶段（一次性）：
  上传参考音频 WAV
       │
       ▼
  ECAPA-TDNN 提取声纹嵌入 → 保存 {voice_id}.spk
  RVQ 编码器提取码本     → 保存 {voice_id}.rvq
  参考文本（可选）        → 保存 {voice_id}.txt
       │
       ▼
  持久化到 ./voices/ 目录（Docker 卷挂载）

合成阶段（重复调用）：
  POST /v1/audio/speech  voice="yuehua"
       │
       ▼
  查找 voices/yuehua.spk + voices/yuehua.rvq
       │
       ▼
  注入声纹到推理管线 → 生成语音（跳过声纹提取，省 ~100ms）
```

> 克隆音色与 CLI 工具（`qwen-codec` / `qwen-tts`）产出的 `.spk` / `.rvq` 文件二进制兼容，可互通。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TALKER_MODEL` | `/app/models/talker.gguf` | Talker LM 的 GGUF 路径 |
| `CODEC_MODEL` | `/app/models/codec.gguf` | Codec（12Hz tokenizer）的 GGUF 路径 |
| `VOICES_DIR` | `/app/voices` | 克隆音色文件目录（server 自动读取，持久化卷挂载） |
| `PORT` | `8080` | 监听端口 |
| `LD_LIBRARY_PATH` | `/app` | 运行时库搜索路径（ggml 后端 `.so`） |

> 监听地址固定为 `0.0.0.0`（容器内必须，否则端口不可达），语言固定 `auto`，均与 `examples/server.sh` 对齐。

---

## Intel iGPU / Vulkan 说明

- 镜像已安装 `mesa-vulkan-drivers`，提供 Intel（ANV）与 AMD（RADV）的 Vulkan ICD。
- `docker-compose.yml` 挂载 `/dev/dri:/dev/dri` 以访问渲染节点。
- 宿主机需有 Vulkan 可用的 Intel/AMD 驱动；可用 `vulkaninfo`（镜像内已含）排查。
- 纯 CPU 运行时可去掉 `devices` 挂载，但推理速度会显著下降。

---

## 项目结构

完整的模型转换、CLI 参数、推理管线与公共 ABI 文档见上游 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 与上游仓库 <https://github.com/ServeurpersoCom/qwentts.cpp>。

## 许可证

MIT。上游模型 Qwen3-TTS（Apache 2.0，Alibaba / Qwen 团队）。
