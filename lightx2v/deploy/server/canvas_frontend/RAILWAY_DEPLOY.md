# Railway 部署指南

## 快速部署步骤

### 1. 在 Railway 上创建新项目

1. 访问 [Railway](https://railway.app)
2. 登录你的账户（支持 GitHub 登录）
3. 点击 "New Project"
4. 选择 "Deploy from GitHub repo"
5. 选择 `Omni-AI-Canvas` 仓库

### 2. 配置环境变量

在 Railway 项目的 "Variables" 标签页中添加以下环境变量：

**必需的环境变量：**
- `GEMINI_API_KEY` - Google Gemini API Key（用于 AI 对话和图像生成）

**可选的环境变量：**
- `LIGHTX2V_TOKEN` - LightX2V API Token（用于 TTS 和视频生成）
- `LIGHTX2V_URL` - LightX2V API URL（默认: `https://x2v.light-ai.top`）
- `DEEPSEEK_API_KEY` - DeepSeek API Key（用于 DeepSeek AI 对话）

### 3. 部署配置

Railway 会自动：
1. 检测到这是一个 Node.js 项目
2. 运行 `npm install` 安装依赖
3. 运行 `npm run build` 构建项目
4. 运行 `npm start` 启动服务器

### 4. 访问应用

部署成功后，Railway 会提供一个 URL（格式类似：`https://xxx.up.railway.app`）

## 部署配置说明

### railway.json
Railway 会自动读取 `railway.json` 配置文件，配置了：
- 构建命令：`npm run build`
- 启动命令：`npm start`

### package.json
- `build`: 使用 Vite 构建生产版本（生产环境默认 `base: '/'`，根路径部署）
- `start`: 使用 `serve -s dist -l 0.0.0.0:${PORT}` 提供静态文件服务，**必须绑定 0.0.0.0** 否则 Railway 无法访问

### 端口配置
Railway 会自动设置 `PORT` 环境变量，应用会在该端口上启动。如果未设置，默认使用 3000 端口。

## 故障排除

### 构建失败
- 检查环境变量是否都已正确设置
- 查看 Railway 日志中的错误信息
- 确保 `package.json` 中的依赖都已正确安装

### 运行时错误
- 检查所有必需的环境变量是否已设置
- 查看浏览器控制台的错误信息
- 检查 Railway 日志

### 端口问题
- Railway 会自动设置 PORT，应用通过 `serve -l 0.0.0.0:${PORT}` 监听
- **必须监听 0.0.0.0**，不能只监听 localhost，否则公网无法访问

### 页面空白 / Failed to fetch dynamically imported module
- 独立部署（根路径）时，构建已使用 `base: '/'`，资源路径为 `/assets/...`
- 若部署在子路径（如 `/canvas/`），在 Railway 环境变量中设置 `VITE_BASE_URL=/canvas/` 后重新构建

## 自定义域名

1. 在 Railway 项目设置中点击 "Settings"
2. 找到 "Domains" 部分
3. 点击 "Generate Domain" 生成免费域名，或添加自己的自定义域名

## 持续部署

Railway 默认启用自动部署：
- 当你推送到 GitHub 仓库的主分支时
- Railway 会自动触发新的构建和部署

你可以在项目的 "Settings" > "Deployments" 中配置分支和部署策略。
