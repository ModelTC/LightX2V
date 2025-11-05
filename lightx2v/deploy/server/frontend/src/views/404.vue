<script setup>
import { switchToCreateView } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
  <!-- Apple 极简风格 404 页面 -->
  <div class="w-full min-h-screen flex items-center justify-center p-6 sm:p-8 md:p-12">
    <div class="max-w-2xl w-full text-center">
      <!-- 404 数字 - Apple 风格 -->
      <div class="mb-6">
        <div class="inline-flex items-center justify-center w-32 h-32 sm:w-40 sm:h-40 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-3xl shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.3)] mb-6">
          <span class="text-5xl sm:text-6xl font-bold text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] tracking-tight">
            404
          </span>
        </div>
      </div>

      <!-- 标题 - Apple 风格 -->
      <h1 class="text-4xl sm:text-5xl md:text-6xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-4 tracking-tight">
        {{ t('pageNotFound') || 'Page Not Found' }}
      </h1>

      <!-- 描述 - Apple 风格 -->
      <p class="text-lg sm:text-xl text-[#86868b] dark:text-[#98989d] mb-10 sm:mb-12 max-w-md mx-auto leading-relaxed tracking-tight">
        {{ t('pageNotFoundDescription') || "Sorry, we couldn't find the page you're looking for." }}
      </p>

      <!-- 操作按钮 - Apple 风格 -->
      <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
        <!-- 返回首页按钮 -->
        <RouterLink
          to="/generate"
          @click="switchToCreateView"
          class="inline-flex items-center justify-center gap-2 px-8 py-3.5 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full text-[15px] font-semibold tracking-tight transition-all duration-200 hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100">
          <i class="fas fa-home text-sm"></i>
          <span>{{ t('goBackHome') || 'Go Back Home' }}</span>
        </RouterLink>

        <!-- 返回上一页按钮 -->
        <button
          @click="$router.go(-1)"
          class="inline-flex items-center justify-center gap-2 px-8 py-3.5 bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full text-[15px] font-medium tracking-tight transition-all duration-200 hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98]">
          <i class="fas fa-arrow-left text-sm"></i>
          <span>{{ t('goBack') || 'Go Back' }}</span>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 在 template 中定义 */
/* 不需要额外的 CSS 规则 */
</style>
