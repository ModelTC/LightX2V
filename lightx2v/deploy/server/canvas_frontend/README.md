# Omni-AI-Canvas

<div align="center">

一个强大的可视化 AI 工作流编辑器，支持多模态 AI 模型集成，让您通过拖拽式界面轻松构建复杂的 AI 创作流程。

[![React](https://img.shields.io/badge/React-19.2.3-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8.2-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.2.0-purple.svg)](https://vitejs.dev/)

</div>

## ✨ 功能特性

### 🎨 可视化工作流编辑
- **拖拽式界面**：直观的节点连接方式，轻松构建复杂工作流
- **实时预览**：节点结果实时预览，支持图片、视频、音频等多种格式
- **并行执行**：智能识别工作流层级，自动并行执行独立节点（最多3个并发）
- **工作流暂停/恢复**：支持中途暂停和恢复工作流执行
- **画布缩放**：支持鼠标滚轮和触控板双指缩放，灵活调整画布视图
- **节点替换**：支持替换兼容节点，保持连接关系

### 🤖 AI 模型集成
- **多模态大语言模型**：DeepSeek V3.2、Doubao Vision Pro、Gemini 3 Pro/Flash
- **图像生成**：文生图、图生图，支持多种模型和宽高比
- **视频生成**：文生视频、图生视频、首尾帧视频、数字人视频
- **语音合成**：支持多种音色和情感控制
- **音色克隆**：从音频克隆音色并生成语音
- **角色替换**：视频中的角色替换功能
- **图像处理**：水印去除等工具

### 📋 预设工作流
- **文字-首尾帧生视频工作流**：从文本描述生成首尾帧并创建过渡视频
- **文字-数字人视频工作流**：完整的数字人视频生成流程
- **人物图片+文字-数字人视频工作流**：基于人物图片的数字人视频
- **9分镜故事板视频工作流**：生成9个分镜的连贯故事板视频
- **多机位唱歌**：9个不同机位的唱歌视频生成
- **大师级运镜一镜到底视频工作流**：电影级运镜的一镜到底视频

### 🛠️ 高级功能
- **AI 工作流生成**：通过自然语言描述自动生成工作流
- **多字段输出编辑**：AI 生成的多字段结果可在下游节点中查看和编辑
- **音频编辑器**：内置音频剪辑工具，支持波形可视化和精确裁剪
- **结果预览优化**：预览显示在节点右侧，避免遮挡
- **可折叠界面**：侧边栏和结果面板可折叠，优化工作空间

## 🚀 快速开始

### 环境要求

- Node.js 18+
- npm 或 yarn

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd Omni-AI-Canvas
```

2. **安装依赖**
```bash
npm install
```

3. **配置环境变量**

创建 `.env.local` 文件并设置以下环境变量：

```env
# 必需
GEMINI_API_KEY=your_gemini_api_key

# 可选 - DeepSeek 模型
DEEPSEEK_API_KEY=your_deepseek_api_key

# 可选 - PP Chat Gemini 模型
PPCHAT_API_KEY=your_ppchat_api_key

# 可选 - LightX2V 服务（接自建后端时使用）
LIGHTX2V_TOKEN=your_lightx2v_token
LIGHTX2V_URL=https://x2v.light-ai.top

# 仅前端部署时：启用独立模式并配置云端 LightX2V
# VITE_STANDALONE=true
# LIGHTX2V_CLOUD_URL=https://x2v.light-ai.top
# LIGHTX2V_CLOUD_TOKEN=your_lightx2v_cloud_token
```

4. **启动开发服务器**
```bash
npm run dev
```

5. **访问应用**

打开浏览器访问 `http://localhost:5173`

## 📦 构建和部署

### 本地构建

```bash
npm run build
```

构建产物将输出到 `dist` 目录。

### 仅前端部署（Standalone）

当不接自建后端、仅部署前端并直连 LightX2V 云端时，需：

1. **设置环境变量**（构建前写入 `.env.local` 或 CI 环境）：
   - `VITE_STANDALONE=true`：启用仅前端模式（无自建后端）
   - `LIGHTX2V_CLOUD_URL`：LightX2V 云端 API 地址（如 `https://x2v.light-ai.top`）
   - `LIGHTX2V_CLOUD_TOKEN`：LightX2V 云端访问令牌

2. **构建**：
   ```bash
   VITE_STANDALONE=true npm run build
   ```
   或先在 `.env.local` 中写好上述变量后再执行 `npm run build`。

3. **行为说明**：该模式下工作流数据仅存于浏览器（localStorage / IndexedDB），视频、TTS、图生图等能力通过上述云端地址与 token 调用。

### Railway 部署

项目已配置 Railway 部署支持，详细步骤请参考 [RAILWAY_DEPLOY.md](./RAILWAY_DEPLOY.md)。

## 🎯 支持的工具和模型

### 输入工具
- **文本输入**：提供文本提示
- **图像输入**：上传单张或多张图片
- **音频输入**：上传音频文件（支持内置编辑器剪辑）
- **视频输入**：上传视频文件

### AI 模型

#### 大语言模型 (LLM)
- **DeepSeek V3.2** (`deepseek-v3-2-251201`)
- **Doubao Vision Pro** (`doubao-1-5-vision-pro-32k-250115`) - 支持图像输入
- **PP Chat Gemini 2.5 Pro** (`ppchat-gemini-2.5-pro`) - 使用 PP Chat API 端点，支持图文输入
- **PP Chat Gemini 3 Pro** (`ppchat-gemini-3-pro-preview`) - 使用 PP Chat API 端点
- **Gemini 3 Pro** (`gemini-3-pro-preview`)
- **Gemini 3 Flash** (`gemini-3-flash-preview`)

#### 图像生成
- **LightX2V Qwen Image 2.5** (`Qwen-Image-2512`) - 文生图
- **LightX2V Qwen Image Edit** (`Qwen-Image-Edit-2511`) - 图生图
- **Gemini Flash Image** (`gemini-2.5-flash-image`)

支持的宽高比：`1:1`, `16:9`, `9:16`, `4:3`, `3:4`

#### 视频生成
- **Wan 2.2 T2V** (`Wan2.2_T2V_A14B_distilled`) - 文生视频
- **Wan 2.2 I2V** (`Wan2.2_I2V_A14B_distilled`) - 图生视频
- **Wan 2.2 Animate** (`wan2.2_animate`) - 角色替换

#### 语音合成 (TTS)
- **LightX2V TTS** - 支持多种音色和情感控制
- **Gemini 2.5 TTS** (`gemini-2.5-flash-preview-tts`)

#### 数字人视频
- **SekoTalk** - 音频驱动的数字人播报视频

#### 图像处理
- **Gemini 水印去除** - 使用反向 Alpha 混合算法去除水印

## 📚 预设工作流详解

### 1. 文字-首尾帧生视频工作流

从文本描述生成首尾两张图像，然后创建平滑过渡视频。

**流程**：
1. 文本输入 → AI 规划器生成首尾帧提示词和运动描述
2. 文生图生成起始帧
3. 图生图基于起始帧生成结束帧
4. 首尾帧视频生成器创建过渡视频

**适用场景**：场景转换、时间过渡、状态变化等视频创作

### 2. 文字-数字人视频工作流

完整的数字人视频生成流程，从文本到最终视频。

**流程**：
1. 文本输入 → AI 生成脚本（语音文本、语调、肖像提示、动作提示）
2. 文生图生成数字人肖像
3. TTS 生成语音
4. 数字人视频生成器合成最终视频

**适用场景**：知识讲解、产品介绍、新闻播报等

### 3. 人物图片+文字-数字人视频工作流

基于已有的人物图片生成数字人视频。

**流程**：
1. 人物图片 + 文本输入 → AI 生成场景编辑提示、脚本、语调
2. 图生图编辑人物场景
3. TTS 生成语音
4. 数字人视频生成器合成视频

**适用场景**：个性化数字人视频、虚拟主播等

### 4. 9分镜故事板视频工作流

生成9个连贯分镜的故事板视频，保持人物和场景一致性。

**流程**：
1. 人物图片 + 故事描述 → AI 规划器生成9个分镜的提示词
2. 图生图序列生成（每个分镜基于前一个分镜和人物图片）
3. 图生视频为每个分镜生成视频

**特点**：
- 严格保持人物一致性（面部、服装、发型等）
- 场景一致性（同一场景的元素保持一致）
- 支持服装和发型变化追踪

**适用场景**：故事叙述、动画分镜、连续场景视频

### 5. 多机位唱歌

生成9个不同机位的唱歌视频，结合数字人视频和图生视频。

**机位配置**：
- **数字人视频机位**：特写、中景、侧方位、正面半身、极特写
- **图生视频机位**：全景、俯拍、极大景、过肩（带运镜和转场效果）

**流程**：
1. 角色图片 + 音频 + 可选文本 → AI 规划器生成9个机位的图像和视频提示词
2. 图生图生成各机位图像
3. 数字人视频生成器生成特写机位视频
4. 图生视频生成全景机位视频（带运镜效果）

**特点**：
- 多机位切换，专业MV效果
- 图生视频机位包含运镜提示词，用于转场衔接
- 保持角色一致性

**适用场景**：音乐MV、表演视频、多角度展示

### 6. 大师级运镜一镜到底视频工作流

电影级运镜的一镜到底视频，运用多种运镜技巧。

**流程**：
1. 场景描述 → AI 规划器生成5个分镜的图像和运镜提示词
2. 文生图生成第一个分镜
3. 图生图序列生成后续分镜（保持场景连贯性）
4. 图生视频为每个分镜生成视频（带运镜效果）

**运镜技巧**：推拉镜头、跟随镜头、环绕镜头、升降镜头、俯仰镜头等

**适用场景**：电影级视频、宏大场景展示、一镜到底创作

## 🎨 使用指南

### 创建工作流

1. **从预设开始**：点击"预设工作流"标签，选择一个预设模板
2. **自定义工作流**：从左侧工具箱拖拽工具到画布
3. **连接节点**：点击节点的输出端口，连接到目标节点的输入端口
4. **配置节点**：点击节点打开配置面板，设置模型、参数等

### AI 工作流生成

1. 点击"AI 生成工作流"按钮
2. 详细描述您想要的工作流
3. AI 会自动生成节点和连接关系
4. 可以在此基础上进行修改和优化

### 编辑 AI 输出

当 AI 生成多字段输出时：
1. 在下游节点的配置面板中查看"连接的 AI 输出"
2. 可以直接编辑字段内容
3. 点击"恢复原始值"可撤销修改

### 音频编辑

1. 在音频输入节点点击"编辑"按钮
2. 在波形上拖动开始和结束滑块进行裁剪
3. 预览裁剪后的音频
4. 应用裁剪或重新编辑

### 节点替换

1. 选中要替换的节点
2. 点击"替换"按钮
3. 选择兼容的替代工具（输出类型和数量需匹配）
4. 连接关系会自动保留

## 🔧 高级配置

### 环境变量说明

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `GEMINI_API_KEY` | ✅ | Gemini API 密钥 |
| `DEEPSEEK_API_KEY` | ❌ | DeepSeek API 密钥（使用 DeepSeek 模型时必需） |
| `PPCHAT_API_KEY` | ❌ | PP Chat API 密钥（使用 PP Chat Gemini 模型时必需） |
| `LIGHTX2V_TOKEN` | ❌ | LightX2V 访问令牌（接自建后端时使用） |
| `LIGHTX2V_URL` | ❌ | LightX2V API 地址（接自建后端时使用，默认：https://x2v.light-ai.top） |
| `VITE_STANDALONE` | ❌ | 设为 `true` 时启用仅前端部署模式（无自建后端） |
| `LIGHTX2V_CLOUD_URL` | 仅前端时 | 仅前端部署时 LightX2V 云端 API 地址（如 https://x2v.light-ai.top） |
| `LIGHTX2V_CLOUD_TOKEN` | 仅前端时 | 仅前端部署时 LightX2V 云端访问令牌（调用视频/TTS/图生图等必需） |

### 工作流执行

- **并行执行**：同一层级的节点会自动并行执行（最多3个并发）
- **暂停/恢复**：点击暂停按钮可暂停执行，再次点击恢复
- **错误处理**：执行失败会显示详细错误信息，不会导致页面崩溃

### 性能优化

- **懒加载**：视频预览采用懒加载，减少初始资源消耗
- **历史限制**：工作流历史记录限制为5次运行
- **批量更新**：状态更新采用批量处理，减少重渲染

## 📖 技术栈

- **前端框架**：React 19.2.3
- **构建工具**：Vite 6.2.0
- **语言**：TypeScript 5.8.2
- **UI 组件**：Lucide React（图标）
- **AI SDK**：@google/genai

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[添加许可证信息]

## 🙏 致谢

- [Google Gemini](https://ai.google.dev/)
- [LightX2V](https://light-ai.top/)
- [DeepSeek](https://www.deepseek.com/)

---

<div align="center">

**Omni-AI-Canvas** - 让 AI 创作变得简单而强大

</div>
