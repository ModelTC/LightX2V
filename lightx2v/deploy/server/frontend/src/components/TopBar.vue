<script setup>
import { onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
const { t, locale } = useI18n()
const router = useRouter()
import { initLanguage,loadLanguageAsync, switchLang, languageOptions } from '../utils/i18n'
import {
            currentUser,
            logout,
            showTemplateDetailModal,
            showTaskDetailModal,
            login,
            theme,
            initTheme,
            toggleTheme,
            getThemeIcon,
} from '../utils/other'

// 导航到主页面
const goToHome = () => {
    showTemplateDetailModal.value = false
    showTaskDetailModal.value = false
    router.push({ name: 'Generate' })
}

// 初始化主题
onMounted(() => {
    initTheme()
})

</script>

<template>
            <!-- Apple 风格顶部栏 - Tailwind 深浅色 -->
            <div class="sticky top-0 z-[100] bg-white/80 dark:bg-[#1e1e1e]/80 backdrop-blur-[20px] backdrop-saturate-[180%] border-b border-black/8 dark:border-white/8 shadow-[0_1px_3px_0_rgba(0,0,0,0.05)] dark:shadow-[0_1px_3px_0_rgba(0,0,0,0.3)] transition-all duration-300 flex-shrink-0">
                <div class="flex justify-between items-center max-w-full mx-auto px-6 py-3">
                    <!-- 左侧 Logo -->
                    <div class="flex items-center">
                        <button @click="goToHome"
                                class="flex items-center gap-2.5 px-3 py-2 bg-transparent border-0 rounded-[10px] cursor-pointer transition-all duration-200 hover:bg-black/4 dark:hover:bg-white/6 hover:-translate-y-px active:scale-[0.97]"
                                :title="t('goToHome')">
                            <img src="../../public/logo.svg" alt="LightX2V" class="w-6 h-6 sm:w-6 sm:h-6 md:w-8 md:h-8 lg:w-8 lg:h-8" loading="lazy" />
                            <span class="inline-flex items-baseline text-[20px] font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-[-0.025em]">
                                <span>Light</span>
                                <span class="text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]">X2V</span>
                            </span>
                        </button>
                    </div>

                    <!-- 右侧用户信息和控制 -->
                    <div class="flex items-center gap-4">
                        <!-- 主题切换按钮 -->
                        <button @click="toggleTheme"
                                class="flex items-center justify-center w-9 h-9 p-0 bg-transparent border-0 rounded-lg cursor-pointer transition-all duration-200 hover:bg-black/4 dark:hover:bg-white/8 hover:scale-105 active:scale-95"
                                :title="'切换主题'">
                            <i :class="getThemeIcon()" class="text-base text-[#86868b] dark:text-[#98989d] transition-all duration-200"></i>
                        </button>

                        <!-- 语言切换按钮 - Apple 精致风格 -->
                                <button @click="switchLang"
                                class="relative flex items-center justify-center w-9 h-9 p-0 bg-black/2 dark:bg-white/4 border border-black/6 dark:border-white/8 rounded-full cursor-pointer transition-all duration-200 hover:bg-black/6 dark:hover:bg-white/10 hover:border-black/10 dark:hover:border-white/15 hover:scale-110 hover:shadow-[0_2px_8px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_2px_8px_rgba(0,0,0,0.3)] active:scale-100"
                                    :title="t('switchLanguage')">
                            <span class="text-base leading-none filter grayscale-0 hover:grayscale-0 transition-all">{{ languageOptions.find(lang => lang.code === (locale === 'zh' ? 'en' : 'zh'))?.flag }}</span>
                                </button>

                        <!-- 用户信息卡片 - Apple 精致风格 -->
                        <div class="flex items-center gap-2.5 px-3 py-1.5 bg-black/2 dark:bg-white/4 border border-black/6 dark:border-white/8 rounded-[20px] transition-all duration-200 hover:bg-black/4 dark:hover:bg-white/8 hover:border-black/8 dark:hover:border-white/12 hover:shadow-[0_2px_8px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_2px_8px_rgba(0,0,0,0.3)]">
                            <!-- 用户头像 -->
                            <div class="flex items-center justify-center w-8 h-8 flex-shrink-0">
                                <img v-if="currentUser.avatar_url"
                                     :src="currentUser.avatar_url"
                                     :alt="currentUser.username"
                                     class="w-full h-full rounded-full object-cover border border-black/8 dark:border-white/12 shadow-[0_1px_3px_rgba(0,0,0,0.1)] dark:shadow-[0_1px_3px_rgba(0,0,0,0.3)]">
                                <!-- 默认头像 - Apple 风格圆形图标 -->
                                <div v-else class="w-full h-full rounded-full bg-gradient-to-br from-[#86868b]/20 to-[#86868b]/10 dark:from-[#98989d]/20 dark:to-[#98989d]/10 border border-black/8 dark:border-white/12 flex items-center justify-center">
                                    <i class="fas fa-user text-[14px] text-[#86868b] dark:text-[#98989d]"></i>
                                </div>
                            </div>

                            <!-- 用户名 -->
                            <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-[-0.01em] whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px]">
                                <span v-if="currentUser">
                                    {{ currentUser.username || currentUser.email || '用户' }}
                                </span>
                                <span v-else>未登录</span>
                            </div>

                            <!-- 登录/登出按钮 - Apple 精致风格 -->
                            <button v-if="currentUser.username"
                                    @click="logout"
                                    class="flex items-center justify-center w-7 h-7 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-red-500/10 dark:hover:bg-red-400/15 hover:scale-110 active:scale-100 flex-shrink-0 group"
                                    :title="t('logout')">
                                <i class="fas fa-arrow-right-from-bracket text-[13px] text-[#86868b] dark:text-[#98989d] group-hover:text-red-500 dark:group-hover:text-red-400 transition-colors"></i>
                                </button>
                            <button v-else
                                    @click="login"
                                    class="flex items-center justify-center w-7 h-7 p-0 bg-transparent border-0 rounded-full cursor-pointer transition-all duration-200 hover:bg-[color:var(--brand-primary)]/10 dark:hover:bg-[color:var(--brand-primary-light)]/15 hover:scale-110 active:scale-100 flex-shrink-0 group"
                                    :title="t('login')">
                                <i class="fas fa-arrow-right-to-bracket text-[13px] text-[#86868b] dark:text-[#98989d] group-hover:text-[color:var(--brand-primary)] dark:group-hover:text-[color:var(--brand-primary-light)] transition-colors"></i>
                                </button>
                        </div>
                    </div>
        </div>
    </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* 不需要额外的 CSS 规则 */
</style>
