<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import router from './router'
import { init, handleLoginCallback, handleClickOutside, validateToken } from './utils/other'
import { initLanguage } from './utils/i18n'
import { startHintRotation, stopHintRotation } from './utils/other'
import { currentUser,
  isLoading,
  applyMobileStyles,
  isLoggedIn,
  loginLoading,
  initLoading,
  pollingInterval,
  pollingTasks,
  showAlert
 } from './utils/other'
import { useI18n } from 'vue-i18n'
import Loading from './components/Loading.vue'
const { t, locale } = useI18n()
let source = null

// 页面加载时应用移动端样式
onMounted(() => {
    applyMobileStyles();
    window.addEventListener('resize', applyMobileStyles);
});

// 组件卸载时移除事件监听器
onUnmounted(() => {
    window.removeEventListener('resize', applyMobileStyles);
});

// 生命周期：页面加载
onMounted(async () => {
  // 1. 初始化语言
  isLoading.value = true
  await initLanguage()
  initLoading.value = true

  // 2. 启动提示滚动
  startHintRotation()

  // 3. 添加全局点击事件监听器
  document.addEventListener('click', handleClickOutside)

  try {
    // 检查是否有登录回调参数
    const urlParams = new URLSearchParams(window.location.search)
    const code = urlParams.get('code')

    if (code) {
      // 处理登录回调
      isLoading.value = true
      source = localStorage.getItem('loginSource')
      await handleLoginCallback(code, source)
      return
    }

    // 检查本地存储的登录状态
    const savedToken = localStorage.getItem('accessToken')
    const savedUser = localStorage.getItem('currentUser')

    if (savedToken && savedUser) {
      // 验证token是否过期
      const isValidToken = await validateToken(savedToken)
      if (isValidToken) {
        currentUser.value = JSON.parse(savedUser)
        isLoggedIn.value = true
        await init();
        console.log('用户已登录，初始化完成')
      } else {
        // Token已过期，清除本地存储
        localStorage.removeItem('accessToken')
        localStorage.removeItem('currentUser')
        isLoggedIn.value = false
        console.log('Token已过期')
        showAlert('请重新登录', 'warning')
      }
    } else {
      isLoggedIn.value = false
      console.log('用户未登录')
    }
  } catch (error) {
    console.error('初始化失败', error)
    showAlert('初始化失败，请刷新页面重试', 'danger')
    isLoggedIn.value = false
  } finally {
    loginLoading.value = false
    initLoading.value = false
    isLoading.value = false
  }

  // 6. 移动端样式适配
  applyMobileStyles()
  window.addEventListener('resize', applyMobileStyles)
})

// 生命周期：页面卸载
onUnmounted(() => {
  // 清理轮询
  if (pollingInterval.value) clearInterval(pollingInterval.value)
  pollingTasks.value.clear()

  // 清理提示滚动
  stopHintRotation()

  // 移除事件监听器
  window.removeEventListener('resize', applyMobileStyles)
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
    <router-view></router-view>
    <!-- 全局路由跳转Loading覆盖层 -->
    <div v-show="isLoading" class="fixed inset-0 bg-gradient-main flex items-center justify-center">
        <Loading />
    </div>
</template>
