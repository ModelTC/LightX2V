<script setup>
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
} from '../utils/other'

// 导航到主页面
const goToHome = () => {
    showTemplateDetailModal.value = false
    showTaskDetailModal.value = false
    router.push({ name: 'Generate' })
}

</script>

<template>
            <!-- 顶部栏 -->
            <div class="top-bar">
                <div class="top-bar-content">
                    <div class="top-bar-left">
                        <button @click="goToHome" class="logo-button" :title="t('goToHome')">
                            <i class="fas fa-film text-gradient-primary mr-2 text-xl"></i>
                            <span class="text-lg text-white">LightX2V</span>
                        </button>
                    </div>

                    <!-- 右侧用户信息 -->
                    <div class="top-bar-right">

                            <!-- 语言切换按钮 -->
                            <div class="language-switcher mr-6">
                                <button @click="switchLang"
                                    class="w-10 h-10 text-gradient-primary items-center px-1 py-1
                                           rounded-lg bg-laser-purple border border-laser-purple hover:border-laser-purple
                                           transition-all duration-200 text-sm hover:scale-105"
                                    :title="t('switchLanguage')">
                                    <span class="text-lg">{{ languageOptions.find(lang => lang.code ===
                                        (locale === 'zh' ? 'en' : 'zh'))?.flag }}</span>
                                </button>
                            </div>

                        <div class="user-info">
                            <div>
                                    <avatar v-if="currentUser.avatar" :src="getUserAvatarUrl(currentUser)"
                                        :alt="currentUser.username" class="size-10">
                                    </avatar>
                                    <i v-else class="fi fi-rr-circle-user text-2xl"></i>

                            </div>
                            <div class="user-details">
                                <span v-if="currentUser">
                                    {{ currentUser.username || currentUser.email || '用户' }}
                                </span>
                                <span v-else>未登录</span>
                            </div>
                            <div v-if="currentUser.username">
                                <button @click="logout" class="text-gradient-primary" :title="t('logout')">
                                    <i class="fas fa-sign-out-alt"></i>
                                </button>
                            </div>
                            <div v-else>
                                <button @click="login" class="text-gradient-primary" :title="t('login')">
                                    <i class="fas fa-sign-in-alt"></i>
                                </button>
                            </div>
                        </div>
                    </div>
        </div>
    </div>
</template>

<style scoped>
.logo-button {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    display: flex;
    align-items: center;
    transition: all 0.2s ease;
    border-radius: 0.5rem;
    padding: 0.5rem;
}

.logo-button:hover {
    background: rgba(139, 92, 246, 0.1);
    transform: scale(1.05);
}

.logo-button:active {
    transform: scale(0.95);
}
</style>
