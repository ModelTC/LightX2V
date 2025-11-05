<script setup>
import LoginCard from '../components/LoginCard.vue'
import Alert from '../components/Alert.vue'
import Loading from '../components/Loading.vue'
import TemplateDisplay from '../components/TemplateDisplay.vue'
import { isLoading, featuredTemplates, loadFeaturedTemplates, getRandomFeaturedTemplates } from '../utils/other'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
const { t } = useI18n()

// 当前显示的精选模版
const currentFeaturedTemplates = ref([])

// 获取随机精选模版
const refreshRandomTemplates = async () => {
    try {
        const randomTemplates = await getRandomFeaturedTemplates(5) // 获取5个模版
        currentFeaturedTemplates.value = randomTemplates
    } catch (error) {
        console.error('刷新随机模版失败:', error)
    }
}

// 组件挂载时初始化
onMounted(async () => {
    // 加载精选模版数据
    isLoading.value = true
    await loadFeaturedTemplates(true)
    // 获取随机精选模版
    const randomTemplates = await getRandomFeaturedTemplates(5) // 获取5个模版
    currentFeaturedTemplates.value = randomTemplates
    isLoading.value = false
})
</script>

<template>
  <!-- Apple 极简风格登录页面 -->
  <div class="h-full w-full bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center p-4 sm:p-6 md:p-8 transition-colors duration-300">

    <!-- 主内容区域 - Apple 风格 -->
    <div class="w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 items-stretch">

      <!-- 左侧：登录区域 -->
      <div class="flex flex-col items-center justify-center">
        <LoginCard />
      </div>

      <!-- 右侧：模版展示区域 - Apple 风格，等高布局 -->
      <div v-if="currentFeaturedTemplates.length > 0"
           class="hidden lg:flex flex-col h-full">

        <!-- 区域头部 - Apple 风格 -->
        <div class="flex-shrink-0 mb-6">
          <div class="flex items-center justify-center gap-3">
            <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('templatesGeneratedByLightX2V') }}</p>
            <button @click="refreshRandomTemplates"
                    class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-full transition-all duration-200 hover:scale-110 active:scale-100 hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/25"
                    :title="t('refreshRandomTemplates')">
              <i class="fas fa-random text-sm"></i>
            </button>
          </div>
        </div>

        <!-- 模版展示区域 - 与登录卡片等高 -->
        <div class="flex-1 overflow-y-auto main-scrollbar pr-2">
          <TemplateDisplay
            :templates="currentFeaturedTemplates"
            :show-actions="false"
            layout="waterfall"
            :max-templates="4"
          />
        </div>
      </div>

      <!-- 如果没有模版数据，显示占位区域 - Apple 风格 -->
      <div v-else class="hidden lg:flex items-center justify-center">
        <Loading />
      </div>

    </div>

    <!-- GitHub 仓库链接 - Apple 极简风格 -->
    <div class="fixed bottom-6 right-6 z-50">
      <a href="https://github.com/ModelTC/LightX2V"
         target="_blank"
         rel="noopener noreferrer"
         class="flex items-center gap-2.5 px-4 py-2.5 bg-white/85 dark:bg-[#1e1e1e]/85 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.3)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.4)] hover:scale-105 active:scale-100 transition-all duration-200 group"
         title="Star us on GitHub">
        <i class="fab fa-github text-lg text-[#1d1d1f] dark:text-[#f5f5f7] transition-transform duration-200 group-hover:rotate-12"></i>
        <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">LightX2V</span>
        <i class="fas fa-external-link-alt text-xs text-[#86868b] dark:text-[#98989d] transition-all duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"></i>
      </a>
    </div>

    <Alert />

    <!-- 全局加载覆盖层 - Apple 风格 -->
    <div v-show="isLoading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-[9999] transition-opacity duration-300">
      <Loading />
    </div>
  </div>
</template>

<style scoped>
/* Apple 风格极简设计 - 所有样式已通过 Tailwind CSS 在 template 中定义 */
</style>
