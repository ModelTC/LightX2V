<script setup>
import { showTemplateDetailModal,
        closeTemplateDetailModal,
        useTemplate,
        getTemplateFileUrl,
        onVideoLoaded,
        selectedTemplate,
        applyTemplateAudio,
        applyTemplateImage,
        applyTemplatePrompt,
        showImageZoom,
        copyPrompt,
        generateTemplateShareUrl,
        copyShareLink,
        shareTemplateToSocial,
         } from '../utils/other'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { ref, onMounted, onUnmounted } from 'vue'
const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()

// 添加响应式变量
const showDetails = ref(false)

// 获取图片素材
const getImageMaterials = () => {
    if (!selectedTemplate.value?.inputs?.input_image) return []
    return [['input_image', getTemplateFileUrl(selectedTemplate.value.inputs.input_image, 'images')]]
}

// 获取音频素材
const getAudioMaterials = () => {
    if (!selectedTemplate.value?.inputs?.input_audio) return []
    return [['input_audio', getTemplateFileUrl(selectedTemplate.value.inputs.input_audio, 'audios')]]
}

// 路由关闭功能
const closeWithRoute = () => {
    closeTemplateDetailModal()
    // 只有当前路由是模板详情页面时才进行路由跳转
    // 如果在其他页面（如 generate）打开的弹窗，关闭时保持在原页面
    if (route.path.startsWith('/template/')) {
        // 从模板详情路由进入的，返回到上一页或首页
        if (window.history.length > 1) {
            router.go(-1)
        } else {
            router.push('/')
        }
    }
    // 如果不是模板详情路由，不做任何路由跳转，保持在当前页面
}

// 滚动到生成区域（仅在 generate 页面）
const scrollToCreationArea = () => {
    const creationArea = document.querySelector('#task-creator')
    if (creationArea) {
        creationArea.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        })
    }
}

// 包装 useTemplate 函数，在 generate 页面时滚动到生成区域
const handleUseTemplate = async () => {
    await useTemplate(selectedTemplate.value)
    // 如果当前在 generate 页面，滚动到生成区域
    if (route.path === '/generate' || route.name === 'Generate') {
        // 等待 DOM 更新和展开动画完成
        setTimeout(() => {
            scrollToCreationArea()
        }, 300)
    }
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'Escape' && showTemplateDetailModal.value) {
        closeWithRoute()
    }
}

// 生命周期钩子
onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
})
</script>
<template>
            <!-- 模板详情弹窗 - Apple 极简风格 -->
            <div v-cloak>
                <div v-if="showTemplateDetailModal"
                    class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-2 sm:p-1"
                    @click="closeWithRoute">
                    <div class="w-full h-full max-w-7xl max-h-[100vh] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden flex flex-col" @click.stop>
                        <!-- 弹窗头部 - Apple 风格 -->
                        <div class="flex items-center justify-between px-8 py-5 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
                            <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
                                <i class="fas fa-star text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                {{ t('templateDetail') }}
                            </h3>
                            <div class="flex items-center gap-2">
                                <button @click="closeWithRoute"
                                        class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-red-500 dark:hover:text-red-400 hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100"
                                        :title="t('close')">
                                    <i class="fas fa-times text-sm"></i>
                                </button>
                            </div>
                        </div>

                        <!-- 主要内容区域 - Apple 风格 -->
                        <div class="flex-1 overflow-y-auto main-scrollbar">
                            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 p-8 lg:p-12">
                                <!-- 左侧视频区域 -->
                                <div class="flex items-center justify-center">
                                    <div class="w-full max-w-[400px] aspect-[9/16] bg-black dark:bg-[#000000] rounded-2xl overflow-hidden shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.5)]">
                                        <!-- 视频播放器 -->
                                        <video
                                            v-if="selectedTemplate?.outputs?.output_video"
                                            :src="getTemplateFileUrl(selectedTemplate.outputs.output_video,'videos')"
                                            :poster="getTemplateFileUrl(selectedTemplate.inputs.input_image,'images')"
                                            class="w-full h-full object-contain"
                                            controls
                                            loop
                                            preload="metadata"
                                            @loadeddata="onVideoLoaded">
                                            {{ t('browserNotSupported') }}
                                        </video>
                                        <div v-else class="w-full h-full flex flex-col items-center justify-center bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                            <div class="w-16 h-16 rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center mb-4">
                                                <i class="fas fa-video text-3xl text-[#86868b] dark:text-[#98989d]"></i>
                                            </div>
                                            <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('videoNotAvailable') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 - Apple 风格 -->
                                <div class="flex items-center justify-center">
                                    <div class="w-full max-w-[400px]">
                                        <!-- 标题 - Apple 风格 -->
                                        <h1 class="text-3xl sm:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-4 tracking-tight">
                                            {{ t('template') }}
                                        </h1>

                                        <!-- 描述 - Apple 风格 -->
                                        <p class="text-sm sm:text-base text-[#86868b] dark:text-[#98989d] mb-8 tracking-tight">
                                            {{ t('templateDescription') }}
                                        </p>

                                        <!-- 快速操作 - Apple 风格 -->
                                        <div class="grid grid-cols-2 gap-2 mb-8">
                                            <button @click="applyTemplateImage(selectedTemplate)"
                                                    class="flex items-center gap-2 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] active:scale-[0.98]">
                                                <div class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-lg flex-shrink-0">
                                                    <i class="fas fa-image text-sm text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                </div>
                                                <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('onlyUseImage') }}</span>
                                            </button>
                                            <button @click="applyTemplateAudio(selectedTemplate)"
                                                    class="flex items-center gap-2 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] active:scale-[0.98]">
                                                <div class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-lg flex-shrink-0">
                                                    <i class="fas fa-music text-sm text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                </div>
                                                <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('onlyUseAudio') }}</span>
                                            </button>
                                        </div>

                                        <!-- 操作按钮 - Apple 风格 -->
                                        <div class="space-y-2.5">
                                            <button @click="handleUseTemplate"
                                                    class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-6 py-3 text-[15px] font-semibold text-white hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100 transition-all duration-200 ease-out tracking-tight flex items-center justify-center gap-2">
                                                <i class="fas fa-magic text-sm"></i>
                                                <span>{{ t('useTemplate') }}</span>
                                            </button>

                                            <button @click="copyShareLink(selectedTemplate?.task_id, 'template')"
                                                    class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                <i class="fas fa-share-alt text-sm"></i>
                                                <span>{{ t('shareTemplate') }}</span>
                                            </button>

                                            <button @click="showDetails = !showDetails"
                                                    class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="text-sm"></i>
                                                <span>{{ showDetails ? t('hideDetails') : t('showDetails') }}</span>
                                            </button>
                                        </div>

                                        <!-- 技术信息 - Apple 风格 -->
                                        <div class="text-center pt-6 mt-6 border-t border-black/8 dark:border-white/8">
                                            <a href="https://github.com/ModelTC/LightX2V"
                                               target="_blank"
                                               rel="noopener noreferrer"
                                               class="inline-flex items-center gap-2 text-sm text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors tracking-tight">
                                                <i class="fab fa-github text-base"></i>
                                                <span>{{ t('poweredByLightX2V') }}</span>
                                                <i class="fas fa-external-link-alt text-xs"></i>
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 详细信息面板 - Apple 风格 -->
                            <div v-if="showDetails && selectedTemplate" class="bg-[#f5f5f7] dark:bg-[#1c1c1e] border-t border-black/8 dark:border-white/8 py-12">
                                <div class="max-w-6xl mx-auto px-8">
                                    <!-- 输入素材标题 - Apple 风格 -->
                                    <h2 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center justify-center gap-3 mb-8 tracking-tight">
                                        <i class="fas fa-upload text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <span>{{ t('inputMaterials') }}</span>
                                    </h2>

                                    <!-- 三个并列的分块卡片 - Apple 风格 -->
                                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        <!-- 图片卡片 - Apple 风格 -->
                                        <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                            <!-- 卡片头部 -->
                                            <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                                <div class="flex items-center gap-3">
                                                    <i class="fas fa-image text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                    <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('image') }}</h3>
                                                </div>
                                                <button v-if="selectedTemplate?.inputs?.input_image"
                                                        @click="applyTemplateImage(selectedTemplate)"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                        :title="t('applyImage')">
                                                    <i class="fas fa-magic text-xs"></i>
                                                </button>
                                            </div>
                                            <!-- 卡片内容 -->
                                            <div class="p-6 min-h-[200px]">
                                                <div v-if="getImageMaterials().length > 0">
                                                    <div v-for="[inputName, url] in getImageMaterials()" :key="inputName"
                                                         class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8 cursor-pointer hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all duration-200"
                                                         @click="showImageZoom(url)">
                                                        <img :src="url" :alt="inputName" class="w-full h-auto object-contain">
                                                    </div>
                                                </div>
                                                <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                                    <i class="fas fa-image text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                                    <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noImage') }}</p>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- 音频卡片 - Apple 风格 -->
                                        <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                            <!-- 卡片头部 -->
                                            <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                                <div class="flex items-center gap-3">
                                                    <i class="fas fa-music text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                    <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('audio') }}</h3>
                                                </div>
                                                <button v-if="selectedTemplate?.inputs?.input_audio"
                                                        @click="applyTemplateAudio(selectedTemplate)"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                        :title="t('applyAudio')">
                                                    <i class="fas fa-magic text-xs"></i>
                                                </button>
                                            </div>
                                            <!-- 卡片内容 -->
                                            <div class="p-6 min-h-[200px]">
                                                <div v-if="getAudioMaterials().length > 0" class="space-y-4">
                                                    <div v-for="[inputName, url] in getAudioMaterials()" :key="inputName">
                                                        <audio :src="url" controls class="w-full rounded-xl"></audio>
                                                    </div>
                                                </div>
                                                <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                                    <i class="fas fa-music text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                                    <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noAudio') }}</p>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- 提示词卡片 - Apple 风格 -->
                                        <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                            <!-- 卡片头部 -->
                                            <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                                <div class="flex items-center gap-3">
                                                    <i class="fas fa-file-alt text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                    <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('prompt') }}</h3>
                                                </div>
                                                <div class="flex items-center gap-1">
                                                    <button v-if="selectedTemplate?.params?.prompt"
                                                            @click="copyPrompt(selectedTemplate?.params?.prompt)"
                                                            class="w-8 h-8 flex items-center justify-center bg-[#86868b]/10 dark:bg-[#98989d]/15 border border-[#86868b]/20 dark:border-[#98989d]/20 text-[#86868b] dark:text-[#98989d] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                            :title="t('copy')">
                                                        <i class="fas fa-copy text-xs"></i>
                                                    </button>
                                                    <button v-if="selectedTemplate?.params?.prompt"
                                                            @click="applyTemplatePrompt(selectedTemplate)"
                                                            class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                            :title="t('applyPrompt')">
                                                        <i class="fas fa-magic text-xs"></i>
                                                    </button>
                                                </div>
                                            </div>
                                            <!-- 卡片内容 -->
                                            <div class="p-6 min-h-[200px]">
                                                <div v-if="selectedTemplate?.params?.prompt" class="bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-xl p-4">
                                                    <p class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight break-words">{{ selectedTemplate.params.prompt }}</p>
                                                </div>
                                                <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                                    <i class="fas fa-file-alt text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                                    <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noPrompt') }}</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* Apple 风格极简黑白设计 */
</style>
