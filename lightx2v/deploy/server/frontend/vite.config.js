import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
    // 自定义插件，避免 Vite 尝试优化 /canvas/ 路径下的文件
    {
      name: 'ignore-canvas-assets',
      configureServer(server) {
        // 拦截对 /canvas/assets/ 的请求，直接返回文件，不进行依赖优化
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/canvas/assets/')) {
            // 让静态文件中间件处理，不进行依赖优化
            return next()
          }
          next()
        })
      },
    },
  ],
  // 配置服务器
  server: {
    fs: {
      // 允许访问 canvas 目录
      strict: false,
    },
    // 监听配置：排除 canvas 目录，避免触发依赖优化
    watch: {
      ignored: ['**/public/canvas/**', '**/canvas/**'],
    },
  },
  // 确保 /canvas/ 路径下的资源不会被 Vite 拦截
  publicDir: 'public',
})
