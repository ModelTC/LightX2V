<template>
  <div class="w-full h-full fixed inset-0">
    <!-- qiankun 子应用容器 -->
    <div id="canvas-subapp-container" class="w-full h-full overflow-hidden"></div>
    <!-- 加载状态 -->
    <div v-if="loading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-50">
      <div class="text-center">
        <div class="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
        <p class="text-gray-600 dark:text-gray-400">加载 Canvas 应用中...</p>
      </div>
    </div>
    <!-- 错误状态 -->
    <div v-if="error" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-50">
      <div class="text-center max-w-md mx-auto p-6">
        <div class="text-red-500 text-6xl mb-4">⚠️</div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">加载失败</h2>
        <p class="text-gray-600 dark:text-gray-400 mb-6">{{ error }}</p>
        <button
          @click="loadSubApp"
          class="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
        >
          重试
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { loadMicroApp } from 'qiankun'
import { sharedStore } from '../utils/sharedStore'
import { apiClient } from '../utils/apiClient'

const loading = ref(true)
const error = ref(null)
let microApp = null

// 根据环境选择子应用入口
// 注意：开发环境下 Vite 会注入 React Refresh 等脚本，qiankun 无法正确处理
// 所以即使在开发环境，也使用构建后的静态文件（需要先构建 Canvas 应用）
// 或者使用环境变量 VITE_USE_CANVAS_DEV_SERVER=true 来强制使用开发服务器
const getCanvasEntry = () => {
  // 如果明确设置了使用开发服务器
  if (import.meta.env.DEV && import.meta.env.VITE_USE_CANVAS_DEV_SERVER === 'true') {
    return import.meta.env.VITE_CANVAS_APP_URL || '//localhost:3000'
  }
  // 默认使用静态文件（需要先构建 Canvas 应用）
  return '/canvas/index.html'
}

// React 子应用配置
const subAppConfig = {
  name: 'react-canvas',
  entry: getCanvasEntry(),
  container: '#canvas-subapp-container',
  // 完全禁用沙箱，避免执行 ES 模块时的错误
  sandbox: false,
  // 排除不属于子应用的资源，防止 qiankun 加载主应用的脚本
  excludeAssetFilter: (assetUrl) => {
    // 排除主应用的资源（/assets/ 但不在 /canvas/ 下）
    // 只允许 Canvas 子应用的资源通过
    const url = assetUrl.toLowerCase();
    // 如果是以 /assets/ 开头但不包含 /canvas/，或者是主应用的路径，排除
    if (url.startsWith('/assets/') && !url.includes('/canvas/')) {
      console.log('[主应用] 排除主应用资源:', assetUrl);
      return true; // 排除主应用的资源
    }
    // 排除根路径下的资源（主应用的资源）
    if (url.startsWith('/assets/index-') || url.match(/^\/assets\/index-[a-z0-9]+\.js$/i)) {
      console.log('[主应用] 排除主应用入口脚本:', assetUrl);
      return true; // 排除主应用的入口脚本
    }
    // 如果包含 /canvas/ 或 /canvas，允许通过
    if (url.includes('/canvas/') || url.includes('/canvas')) {
      return false; // 允许 Canvas 应用的资源
    }
    // 默认排除（安全起见）
    console.log('[主应用] 默认排除资源:', assetUrl);
    return true;
  },
  // vite-plugin-qiankun 需要这个配置来正确处理
  // 确保在加载前初始化 window.proxy
  beforeLoad: (app) => {
    console.log('[主应用] beforeLoad', app.name)
    // 初始化 vite-plugin-qiankun 需要的全局对象
    if (!window.proxy) {
      window.proxy = {}
    }
    // 确保 window.moudleQiankunAppLifeCycles 存在
    if (!window.moudleQiankunAppLifeCycles) {
      window.moudleQiankunAppLifeCycles = {}
    }
  },
  beforeMount: (app) => {
    console.log('[主应用] beforeMount', app.name)
  },
  afterMount: (app) => {
    console.log('[主应用] afterMount', app.name)
    // 在挂载后添加包装器样式
    const addWrapperClasses = () => {
      const wrapper = document.getElementById('__qiankun_microapp_wrapper_for_react_canvas__')
      if (wrapper) {
        wrapper.classList.add('w-full', 'h-full')
        console.log('[主应用] 已为 qiankun 包装器添加 w-full h-full 类')
        return true
      }
      return false
    }

    // 立即尝试添加
    if (!addWrapperClasses()) {
      // 如果包装器还未创建，使用 MutationObserver 监听
      const observer = new MutationObserver(() => {
        if (addWrapperClasses()) {
          observer.disconnect()
        }
      })
      observer.observe(document.getElementById('canvas-subapp-container'), {
        childList: true,
        subtree: true
      })
      // 10秒后停止监听
      setTimeout(() => observer.disconnect(), 10000)
    }
  },
  // 自定义 fetch，过滤掉 ES 模块脚本的执行
  fetch: (url, options = {}) => {
    // 如果是 JS 文件且可能是 ES 模块，添加标识
    if (url.endsWith('.js') && !url.includes('vendor')) {
      return fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'Accept': 'application/javascript, application/json'
        }
      })
    }
    return fetch(url, options)
  },
  // 自定义 HTML 处理，确保模块脚本不会被作为 entry 执行
  getTemplate: (html) => {
    // 处理 HTML，确保 entry 脚本被正确标记，模块脚本被忽略
    const closeScript = '</' + 'script>';
    const openScript = '<' + 'script';

    // 1. 标记包含生命周期注册的脚本为 entry（应该已经有 entry 标记了）
    // 2. 标记所有包含 import() 的模块脚本为 ignore
    const modulePattern = new RegExp(
      openScript + '([^>]*?)>([\\s\\S]*?import\\([^)]+\\)[\\s\\S]*?)' + closeScript,
      'g'
    );
    let processedHtml = html.replace(
      modulePattern,
      (match, attrs, content) => {
        // 如果已经有 entry 标记，保留（这是生命周期注册脚本）
        if (attrs.includes('entry')) {
          return match;
        }
        // 模块脚本标记为 ignore
        if (attrs.includes('data-qiankun-ignore')) {
          return match;
        }
        let newAttrs = attrs;
        if (!newAttrs.includes('type="module"') && !newAttrs.includes("type='module'")) {
          newAttrs = ` type="module"${newAttrs}`;
        }
        if (!newAttrs.includes('data-qiankun-ignore')) {
          newAttrs = `${newAttrs} data-qiankun-ignore`;
        }
        return openScript + newAttrs + '>' + content + closeScript;
      }
    );

    // 确保 body 中的生命周期脚本有 entry 标记
    const lifecyclePattern = new RegExp(
      openScript + '([^>]*?)>([\\s\\S]*?createDeffer[\\s\\S]*?global\\[\\\'react-canvas\\\'\\][\\s\\S]*?)' + closeScript,
      'g'
    );
    processedHtml = processedHtml.replace(
      lifecyclePattern,
      (match, attrs, content) => {
        if (!attrs.includes('entry')) {
          return openScript + ' entry' + (attrs ? ' ' + attrs.trim() : '') + '>' + content + closeScript;
        }
        return match;
      }
    );

    return processedHtml;
  },
  props: {
    // 传递共享状态和方法给子应用
    sharedStore,
    apiClient,
    routerBase: '/canvas',
    // 同步主应用状态到子应用
    getInitialState: () => ({
      user: sharedStore.getState('user'),
      token: sharedStore.getState('token'),
      theme: sharedStore.getState('theme'),
      language: sharedStore.getState('language'),
    })
  }
}

// 加载子应用
const loadSubApp = async () => {
  try {
    loading.value = true
    error.value = null

    // 如果已有应用实例，先卸载
    if (microApp) {
      await microApp.unmount()
      microApp = null
    }

    // 确保 window.proxy 和 moudleQiankunAppLifeCycles 在加载前就存在
    if (!window.proxy) {
      window.proxy = {}
    }
    if (!window.moudleQiankunAppLifeCycles) {
      window.moudleQiankunAppLifeCycles = {}
    }


    // 加载子应用
    microApp = loadMicroApp(subAppConfig)

    // 等待应用加载和挂载完成
    // qiankun 的 loadMicroApp 会返回一个 promise，但它不会自动挂载
    // 需要手动调用 mount
    try {
      // 等待 HTML 和脚本加载完成（这会触发 bootstrap）
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          // 输出调试信息
          console.error('[主应用] 超时调试信息:')
          console.error('  window.proxy:', window.proxy)
          console.error('  window.moudleQiankunAppLifeCycles:', window.moudleQiankunAppLifeCycles)
          console.error('  window["react-canvas"]:', window['react-canvas'])
          console.error('  microApp status:', microApp?.getStatus?.())
          reject(new Error('应用加载超时：无法检测到应用生命周期'))
        }, 15000) // 15秒超时检测生命周期

        // 轮询检查生命周期是否已注册
        const checkInterval = setInterval(() => {
          // 检查多种可能的生命周期注册位置
          const hasLifeCycle = (
            (window.moudleQiankunAppLifeCycles && window.moudleQiankunAppLifeCycles['react-canvas']) ||
            (window['react-canvas'] && typeof window['react-canvas'].bootstrap === 'function') ||
            (typeof window.bootstrap === 'function' && typeof window.mount === 'function' && typeof window.unmount === 'function')
          )

          if (hasLifeCycle) {
            clearInterval(checkInterval)
            clearTimeout(timeout)
            console.log('[主应用] 检测到子应用生命周期已注册')
            console.log('  生命周期对象:', window.moudleQiankunAppLifeCycles?.['react-canvas'] || window['react-canvas'])
            resolve()
          }
        }, 100)
      })

      // 检查应用状态，如果已经挂载则不需要再次挂载
      const appStatus = microApp?.getStatus?.()
      console.log('[主应用] 应用当前状态:', appStatus)

      // 添加包装器样式的函数
      const addWrapperClasses = () => {
        const wrapper = document.getElementById('__qiankun_microapp_wrapper_for_react_canvas__')
        if (wrapper) {
          wrapper.classList.add('w-full', 'h-full')
          console.log('[主应用] 已为 qiankun 包装器添加 w-full h-full 类')
        } else {
          console.warn('[主应用] 未找到 qiankun 包装器元素')
        }
      }

      if (appStatus === 'MOUNTED') {
        // 应用已经自动挂载（通过 proxy.vitemount）
        console.log('[主应用] 子应用已自动挂载，无需手动挂载')
        // 延迟一下确保 DOM 已渲染
        setTimeout(addWrapperClasses, 100)
        loading.value = false
      } else if (appStatus === 'NOT_MOUNTED' || appStatus === 'NOT_LOADED') {
        // 应用未挂载，手动挂载
        console.log('[主应用] 开始手动挂载子应用')
        await microApp.mount()
        console.log('[主应用] 子应用挂载成功')
        // 延迟一下确保 DOM 已渲染
        setTimeout(addWrapperClasses, 100)
        loading.value = false
      } else {
        // 等待应用状态变为 MOUNTED
        console.log('[主应用] 等待应用挂载完成...')
        await new Promise((resolve) => {
          const checkInterval = setInterval(() => {
            const status = microApp?.getStatus?.()
            if (status === 'MOUNTED') {
              clearInterval(checkInterval)
              console.log('[主应用] 子应用挂载完成')
              // 延迟一下确保 DOM 已渲染
              setTimeout(addWrapperClasses, 100)
              resolve()
            }
          }, 100)
          // 最多等待 10 秒
          setTimeout(() => {
            clearInterval(checkInterval)
            addWrapperClasses()
            resolve()
          }, 10000)
        })
        loading.value = false
      }
    } catch (mountError) {
      console.error('[主应用] 挂载失败:', mountError)
      const status = microApp?.getStatus?.() || 'unknown'
      throw new Error(`应用挂载失败 (状态: ${status}): ${mountError.message}`)
    }
  } catch (err) {
    console.error('加载子应用失败:', err)
    error.value = err.message || '加载 Canvas 应用失败，请检查应用是否正常运行。如果问题持续，请检查 Canvas 应用是否已正确构建。'
    loading.value = false
  }
}

onMounted(() => {
  loadSubApp()
})

onUnmounted(() => {
  // 卸载子应用
  if (microApp) {
    microApp.unmount()
    microApp = null
  }
})
</script>
