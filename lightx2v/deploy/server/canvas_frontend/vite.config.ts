import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import qiankun from 'vite-plugin-qiankun';

export default defineConfig(({ mode }) => {
    // 优先使用系统环境变量，如果没有则从 .env 文件读取
    const env = loadEnv(mode, '.', '');
    // 合并系统环境变量（如果存在）
    const mergedEnv = {
      ...env,
      // 如果系统环境变量存在，优先使用系统环境变量
      LIGHTX2V_URL: process.env.LIGHTX2V_URL || env.LIGHTX2V_URL || '',
      LIGHTX2V_TOKEN: process.env.LIGHTX2V_TOKEN || env.LIGHTX2V_TOKEN || '',
      LIGHTX2V_CLOUD_URL: process.env.LIGHTX2V_CLOUD_URL || env.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top',
      LIGHTX2V_CLOUD_TOKEN: process.env.LIGHTX2V_CLOUD_TOKEN || env.LIGHTX2V_CLOUD_TOKEN || '',
      GEMINI_API_KEY: process.env.GEMINI_API_KEY || env.GEMINI_API_KEY || '',
      DEEPSEEK_API_KEY: process.env.DEEPSEEK_API_KEY || env.DEEPSEEK_API_KEY || '',
      PPCHAT_API_KEY: process.env.PPCHAT_API_KEY || env.PPCHAT_API_KEY || '',
      VITE_STANDALONE: process.env.VITE_STANDALONE || env.VITE_STANDALONE || '',
    };

    // 构建时打印环境变量状态（用于调试）
    if (mode === 'production') {
      console.log('[Vite Build] 环境变量检查:');
      console.log('  DEEPSEEK_API_KEY:', mergedEnv.DEEPSEEK_API_KEY ? `${mergedEnv.DEEPSEEK_API_KEY.substring(0, 10)}...` : '未设置');
      console.log('  GEMINI_API_KEY:', mergedEnv.GEMINI_API_KEY ? `${mergedEnv.GEMINI_API_KEY.substring(0, 10)}...` : '未设置');
      console.log('  PPCHAT_API_KEY:', mergedEnv.PPCHAT_API_KEY ? `${mergedEnv.PPCHAT_API_KEY.substring(0, 10)}...` : '未设置');
      console.log('  LIGHTX2V_URL:', mergedEnv.LIGHTX2V_URL ? `${mergedEnv.LIGHTX2V_URL.substring(0, 10)}...` : '未设置');
      console.log('  LIGHTX2V_TOKEN:', mergedEnv.LIGHTX2V_TOKEN ? `${mergedEnv.LIGHTX2V_TOKEN.substring(0, 10)}...` : '未设置');
      console.log('  LIGHTX2V_CLOUD_URL:', mergedEnv.LIGHTX2V_CLOUD_URL ? `${mergedEnv.LIGHTX2V_CLOUD_URL.substring(0, 10)}...` : '未设置');
      console.log('  LIGHTX2V_CLOUD_TOKEN:', mergedEnv.LIGHTX2V_CLOUD_TOKEN ? `${mergedEnv.LIGHTX2V_CLOUD_TOKEN.substring(0, 10)}...` : '未设置');
    }
    const basePath =
      mode === 'production'
        ? (process.env.VITE_BASE_URL || env.VITE_BASE_URL || '/')
        : '/';
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        cors: true, // 允许跨域，支持微前端
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
      },
      plugins: [
        react(),
        // qiankun 插件，自动处理微前端生命周期
        qiankun('react-canvas', {
          useDevMode: mode === 'development'
        }),
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(mergedEnv.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(mergedEnv.GEMINI_API_KEY),
        'process.env.LIGHTX2V_TOKEN': JSON.stringify(mergedEnv.LIGHTX2V_TOKEN || ''),
        'process.env.LIGHTX2V_URL': JSON.stringify(mergedEnv.LIGHTX2V_URL || ''),
        'process.env.LIGHTX2V_CLOUD_URL': JSON.stringify(mergedEnv.LIGHTX2V_CLOUD_URL || 'https://x2v.light-ai.top'),
        'process.env.LIGHTX2V_CLOUD_TOKEN': JSON.stringify(mergedEnv.LIGHTX2V_CLOUD_TOKEN || ''),
        'process.env.DEEPSEEK_API_KEY': JSON.stringify(mergedEnv.DEEPSEEK_API_KEY || ''),
        'process.env.PPCHAT_API_KEY': JSON.stringify(mergedEnv.PPCHAT_API_KEY || ''),
        'import.meta.env.VITE_STANDALONE': JSON.stringify(mergedEnv.VITE_STANDALONE || '')
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        esbuildOptions: {
          // 忽略 source map 错误
          logOverride: { 'this-is-undefined-in-esm': 'silent' },
          // 跳过 source map 处理
          sourcemap: false
        }
      },
      build: {
        sourcemap: false,
        // 确保资源路径使用相对路径，支持微前端
        rollupOptions: {
          output: {
            entryFileNames: 'assets/[name].[hash].js',
            chunkFileNames: 'assets/[name].[hash].js',
            assetFileNames: 'assets/[name].[hash].[ext]',
          },
        },
      },
      // 生产环境默认使用根路径，可通过 VITE_BASE_URL 覆盖
      base: basePath,
    };
});
