<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import topMenu from '../components/TopBar.vue'
import Loading from '../components/Loading.vue'
import {
    isLoading,
    selectedTaskId,
    getCurrentForm,
    setCurrentImagePreview,
    setCurrentAudioPreview,
    getTemplateFileUrl,
    isCreationAreaExpanded,
    switchToCreateView,
    showAlert,
    login,
    copyPrompt
} from '../utils/other'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

const shareId = computed(() => route.params.shareId)
const shareData = ref(null)
const error = ref(null)
const videoUrl = ref('')
const inputUrls = ref({})
const showDetails = ref(false)
const videoLoading = ref(false)
const videoError = ref(false)

// 获取分享数据
const fetchShareData = async () => {
    try {
        const response = await fetch(`/api/v1/share/${shareId.value}`)

        if (!response.ok) {
            throw new Error('分享不存在或已过期')
        }

        const data = await response.json()
        shareData.value = data

        // 设置视频URL
        if (data.output_video_url) {
            videoUrl.value = data.output_video_url
        }

        // 设置输入素材URL
        if (data.input_urls) {
            inputUrls.value = data.input_urls
            console.log('设置输入素材URL:', data.input_urls)
        }
    } catch (err) {
        error.value = err.message
        console.error('获取分享数据失败:', err)
    }
}

// 获取分享标题
const getShareTitle = () => {
    if (shareData.value?.share_type === 'task') {
        // 获取用户名，如果没有则显示默认文本
        const username = shareData.value?.username || '用户'
        return `${username}${t('userGeneratedVideo')}`
    }
    return t('templateVideo')
}

// 获取分享描述
const getShareDescription = () => {
    return t('description')
}

// 获取分享按钮文本
const getShareButtonText = () => {
    switch (shareData.value?.share_type) {
        case 'template':
            return t('useTemplate')
        default:
            return t('createSimilar')
    }
}

// 视频事件处理
const onVideoLoadStart = () => {
    videoLoading.value = true
    videoError.value = false
}

const onVideoCanPlay = () => {
    videoLoading.value = false
    videoError.value = false
}

const onVideoError = () => {
    videoLoading.value = false
    videoError.value = true
}

// 获取图片素材
const getImageMaterials = () => {
    if (!inputUrls.value) return []
    const imageMaterials = Object.entries(inputUrls.value).filter(([name, url]) =>
        name.includes('image') && url
    )
    console.log('图片素材:', imageMaterials)
    return imageMaterials
}

// 获取音频素材
const getAudioMaterials = () => {
    if (!inputUrls.value) return []
    const audioMaterials = Object.entries(inputUrls.value).filter(([name, url]) =>
        name.includes('audio') && url
    )
    console.log('音频素材:', audioMaterials)
    return audioMaterials
}

// 处理图片加载错误
const handleImageError = (event, inputName, url) => {
    console.log('图片加载失败:', inputName, url)
    console.log('错误详情:', event)
    console.log('图片元素:', event.target)

    // 尝试移除crossorigin属性重新加载
    const img = event.target
    if (img.crossOrigin) {
        console.log('尝试移除crossorigin属性重新加载')
        img.crossOrigin = null
        img.src = url + '?retry=' + Date.now()
    }
}

// 做同款功能
const createSimilar = async () => {
    const token = localStorage.getItem('accessToken')
    if (!token) {
        // 未登录，跳转到登录页面，并保存分享ID
        localStorage.setItem('shareData', JSON.stringify({ shareId: shareId.value }))
        login()
        return
    }

    if (!shareData.value) {
        showAlert('分享数据不完整', 'danger')
        return
    }

    console.log('使用分享数据:', shareData.value)

    try {
        // 先设置任务类型
        selectedTaskId.value = shareData.value.task_type

        // 获取当前表单
        const currentForm = getCurrentForm()

        // 设置表单数据
        currentForm.prompt = shareData.value.prompt || ''
        currentForm.negative_prompt = shareData.value.negative_prompt || ''
        currentForm.seed = 42 // 默认种子
        currentForm.model_cls = shareData.value.model_cls || ''
        currentForm.stage = shareData.value.stage || 'single_stage'

        // 如果有输入图片，先设置URL，延迟加载文件
        if (shareData.value.inputs && shareData.value.inputs.input_image) {
            let imageUrl
            if (shareData.value.share_type === 'template') {
                // 对于模板，使用模板文件URL
                imageUrl = getTemplateFileUrl(shareData.value.inputs.input_image, 'images')
            } else {
                // 对于任务，使用分享数据中的URL
                imageUrl = shareData.value.input_urls?.input_image || shareData.value.input_urls?.[Object.keys(shareData.value.input_urls).find(key => key.includes('image'))]
            }

            if (imageUrl) {
                currentForm.imageUrl = imageUrl
                setCurrentImagePreview(imageUrl) // 直接使用URL作为预览
                console.log('分享输入图片:', imageUrl)

                // 异步加载图片文件（不阻塞UI）
                setTimeout(async () => {
                    try {
                        const imageResponse = await fetch(imageUrl)
                        if (imageResponse.ok) {
                            const blob = await imageResponse.blob()
                            const filename = shareData.value.inputs.input_image
                            const file = new File([blob], filename, { type: blob.type })
                            currentForm.imageFile = file
                            console.log('分享图片文件已加载')
                        }
                    } catch (error) {
                        console.warn('Failed to load share image file:', error)
                    }
                }, 100)
            }
        }

        // 如果有输入音频，先设置URL，延迟加载文件
        if (shareData.value.inputs && shareData.value.inputs.input_audio) {
            let audioUrl
            if (shareData.value.share_type === 'template') {
                // 对于模板，使用模板文件URL
                audioUrl = getTemplateFileUrl(shareData.value.inputs.input_audio, 'audios')
            } else {
                // 对于任务，使用分享数据中的URL
                audioUrl = shareData.value.input_urls?.input_audio || shareData.value.input_urls?.[Object.keys(shareData.value.input_urls).find(key => key.includes('audio'))]
            }

            if (audioUrl) {
                currentForm.audioUrl = audioUrl
                setCurrentAudioPreview(audioUrl) // 直接使用URL作为预览
                console.log('分享输入音频:', audioUrl)

                // 异步加载音频文件（不阻塞UI）
                setTimeout(async () => {
                    try {
                        const audioResponse = await fetch(audioUrl)
                        if (audioResponse.ok) {
                            const blob = await audioResponse.blob()
                            const filename = shareData.value.inputs.input_audio

                            // 根据文件扩展名确定正确的MIME类型
                            let mimeType = blob.type
                            if (!mimeType || mimeType === 'application/octet-stream') {
                                const ext = filename.toLowerCase().split('.').pop()
                                const mimeTypes = {
                                    'mp3': 'audio/mpeg',
                                    'wav': 'audio/wav',
                                    'mp4': 'audio/mp4',
                                    'aac': 'audio/aac',
                                    'ogg': 'audio/ogg',
                                    'm4a': 'audio/mp4'
                                }
                                mimeType = mimeTypes[ext] || 'audio/mpeg'
                            }

                            const file = new File([blob], filename, { type: mimeType })
                            currentForm.audioFile = file
                            console.log('分享音频文件已加载')
                            // 使用FileReader生成data URL，与正常上传保持一致
                            const reader = new FileReader()
                            reader.onload = (e) => {
                                setCurrentAudioPreview(e.target.result)
                                console.log('分享音频预览已设置:', e.target.result.substring(0, 50) + '...')
                            }
                            reader.readAsDataURL(file)
                        }
                    } catch (error) {
                        console.warn('Failed to load share audio file:', error)
                    }
                }, 100)
            }
        }

        // 切换到创建视图
        isCreationAreaExpanded.value = true
        switchToCreateView()

        showAlert(`已应用分享数据`, 'success')
    } catch (error) {
        console.error('应用分享数据失败:', error)
        showAlert(`应用分享数据失败: ${error.message}`, 'danger')
    }
}

onMounted(async () => {
    await fetchShareData()
    isLoading.value = false
})
</script>

<template>
    <!-- Apple 极简风格分享页面 -->
    <div class="min-h-screen w-full bg-[#f5f5f7] dark:bg-[#000000]">
        <!-- TopBar -->
        <topMenu />

        <!-- 主要内容区域 -->
        <div class="w-full min-h-[calc(100vh-80px)] overflow-y-auto main-scrollbar">
            <!-- 错误状态 - Apple 风格 -->
            <div v-if="error" class="flex items-center justify-center min-h-[60vh] px-6">
                <div class="text-center max-w-md">
                    <div class="inline-flex items-center justify-center w-20 h-20 bg-red-500/10 dark:bg-red-400/10 rounded-3xl mb-6">
                        <i class="fas fa-exclamation-triangle text-3xl text-red-500 dark:text-red-400"></i>
                    </div>
                    <h2 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-4 tracking-tight">{{ t('shareNotFound') }}</h2>
                    <p class="text-base text-[#86868b] dark:text-[#98989d] mb-8 tracking-tight">{{ error }}</p>
                    <button @click="router.push('/')"
                            class="inline-flex items-center justify-center gap-2 px-8 py-3 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full text-[15px] font-semibold tracking-tight transition-all duration-200 hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100">
                        <i class="fas fa-home text-sm"></i>
                        <span>{{ t('backToHome') }}</span>
                    </button>
                </div>
            </div>

            <!-- 分享内容 - Apple 风格 -->
            <div v-else-if="shareData" class="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 w-full max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-12 lg:py-16 items-center">
                <!-- 左侧视频区域 -->
                <div class="flex justify-center items-center">
                    <div class="w-full max-w-[400px] aspect-[9/16] bg-black dark:bg-[#000000] rounded-2xl overflow-hidden shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.5)] relative">
                        <!-- 视频加载占位符 - Apple 风格 -->
                        <div v-if="!videoUrl" class="w-full h-full flex flex-col items-center justify-center bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                            <div class="relative w-12 h-12 mb-6">
                                <div class="absolute inset-0 rounded-full border-2 border-black/8 dark:border-white/8"></div>
                                <div class="absolute inset-0 rounded-full border-2 border-transparent border-t-[color:var(--brand-primary)] dark:border-t-[color:var(--brand-primary-light)] animate-spin"></div>
                            </div>
                            <p class="text-sm font-medium text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loadingVideo') }}...</p>
                        </div>

                        <!-- 视频播放器 -->
                        <video
                            v-if="videoUrl"
                            :src="videoUrl"
                            class="w-full h-full object-contain"
                            controls
                            autoplay
                            loop
                            preload="metadata"
                            @loadstart="onVideoLoadStart"
                            @canplay="onVideoCanPlay"
                            @error="onVideoError">
                            {{ t('browserNotSupported') }}
                        </video>

                        <!-- 视频错误状态 - Apple 风格 -->
                        <div v-if="videoError" class="w-full h-full flex flex-col items-center justify-center bg-[#fef2f2] dark:bg-[#2c1b1b]">
                            <div class="w-16 h-16 rounded-full bg-red-500/10 dark:bg-red-400/10 flex items-center justify-center mb-4">
                                <i class="fas fa-exclamation-triangle text-3xl text-red-500 dark:text-red-400"></i>
                            </div>
                            <p class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('videoNotAvailable') }}</p>
                        </div>
                    </div>
                </div>

                <!-- 右侧信息区域 - Apple 风格 -->
                <div class="flex items-center justify-center">
                    <div class="w-full max-w-[500px]">
                        <!-- 标题 - Apple 风格 -->
                        <h1 class="text-4xl sm:text-5xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-4 tracking-tight leading-tight">
                            {{ getShareTitle() }}
                        </h1>

                        <!-- 描述 - Apple 风格 -->
                        <p class="text-lg text-[#86868b] dark:text-[#98989d] mb-8 leading-relaxed tracking-tight">
                            {{ getShareDescription() }}
                        </p>

                        <!-- 特性列表 - Apple 风格 -->
                        <div class="grid grid-cols-1 gap-3 mb-8">
                            <div class="flex items-center gap-3 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]">
                                <div class="w-10 h-10 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-lg flex-shrink-0">
                                    <i class="fas fa-rocket text-base text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                </div>
                                <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('latestAIModel') }}</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]">
                                <div class="w-10 h-10 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-lg flex-shrink-0">
                                    <i class="fas fa-bolt text-base text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                </div>
                                <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('oneClickReplication') }}</span>
                            </div>
                            <div class="flex items-center gap-3 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]">
                                <div class="w-10 h-10 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-lg flex-shrink-0">
                                    <i class="fas fa-user-cog text-base text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                </div>
                                <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('customizableCharacter') }}</span>
                            </div>
                        </div>

                        <!-- 操作按钮 - Apple 风格 -->
                        <div class="space-y-3 mb-8">
                            <button @click="createSimilar"
                                    class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-8 py-3.5 text-[15px] font-semibold text-white hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100 transition-all duration-200 ease-out tracking-tight flex items-center justify-center gap-2">
                                <i class="fas fa-magic text-sm"></i>
                                <span>{{ getShareButtonText() }}</span>
                            </button>

                            <!-- 详细信息按钮 -->
                            <button @click="showDetails = !showDetails"
                                    class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-8 py-3 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="text-sm"></i>
                                <span>{{ showDetails ? t('hideDetails') : t('showDetails') }}</span>
                            </button>
                        </div>

                        <!-- 技术信息 - Apple 风格 -->
                        <div class="text-center pt-6 border-t border-black/8 dark:border-white/8">
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
            <div v-if="showDetails && shareData" class="w-full bg-white dark:bg-[#1c1c1e] border-t border-black/8 dark:border-white/8 py-16">
                <div class="max-w-6xl mx-auto px-6 sm:px-8 lg:px-12">
                    <!-- 输入素材标题 - Apple 风格 -->
                    <h2 class="text-2xl sm:text-3xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center justify-center gap-3 mb-10 tracking-tight">
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
                            </div>
                            <!-- 卡片内容 -->
                            <div class="p-6 min-h-[200px]">
                                <div v-if="getImageMaterials().length > 0">
                                    <div v-for="[inputName, url] in getImageMaterials()" :key="inputName" class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8">
                                        <img :src="url" :alt="inputName"
                                             class="w-full h-auto object-contain"
                                             @load="console.log('图片加载成功:', inputName, url)"
                                             @error="handleImageError($event, inputName, url)">
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
                                <button v-if="shareData.prompt"
                                        @click="copyPrompt(shareData.prompt)"
                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                        :title="t('copy')">
                                    <i class="fas fa-copy text-xs"></i>
                                </button>
                            </div>
                            <!-- 卡片内容 -->
                            <div class="p-6 min-h-[200px]">
                                <div v-if="shareData.prompt" class="bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-xl p-4">
                                    <p class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight break-words">{{ shareData.prompt }}</p>
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

    <!-- 全局路由跳转Loading覆盖层 - Apple 风格 -->
    <div v-show="isLoading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-[9999]">
        <Loading />
    </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* Apple 风格极简黑白设计 */
</style>
