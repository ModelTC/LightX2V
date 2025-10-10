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
    login
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
    <div class="landing-page">
        <!-- TopBar -->
        <topMenu />

        <!-- 主要内容区域 -->
        <div class="main-content main-scrollbar overflow-y-auto">
            <!-- 错误状态 -->
            <div v-if="error" class="error-container">
                <div class="error-content">
                    <div class="error-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <h2 class="error-title">{{ t('shareNotFound') }}</h2>
                    <p class="error-message">{{ error }}</p>
                    <button @click="router.push('/')" class="error-button">
                        <i class="fas fa-home mr-2"></i>
                        {{ t('backToHome') }}
                    </button>
                </div>
            </div>

            <!-- 分享内容 -->
            <div v-else-if="shareData" class="content-grid">
                <!-- 左侧视频区域 -->
                <div class="video-section">
                    <div class="video-container">
                        <!-- 视频加载占位符 -->
                        <div v-if="!videoUrl" class="video-placeholder">
                            <div class="loading-spinner">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <p class="loading-text">{{ t('loadingVideo') }}...</p>
                        </div>

                        <!-- 视频播放器 -->
                        <video
                            v-if="videoUrl"
                            :src="videoUrl"
                            class="video-player"
                            controls
                            autoplay
                            loop
                            preload="metadata"
                            @loadstart="onVideoLoadStart"
                            @canplay="onVideoCanPlay"
                            @error="onVideoError">
                            {{ t('browserNotSupported') }}
                        </video>

                        <!-- 视频错误状态 -->
                        <div v-if="videoError" class="video-error">
                            <i class="fas fa-exclamation-triangle"></i>
                            <p>{{ t('videoNotAvailable') }}</p>
                        </div>
                    </div>
                </div>

                <!-- 右侧信息区域 -->
                <div class="info-section">
                    <div class="info-content">
                        <!-- 标题 -->
                        <h1 class="main-title">
                            {{ getShareTitle() }}
                        </h1>

                        <!-- 描述 -->
                        <p class="main-description">
                            {{ getShareDescription() }}
                        </p>

                        <!-- 特性列表 -->
                        <div class="features-list">
                            <div class="feature-item">
                                <i class="fas fa-rocket feature-icon"></i>
                                <span class="feature-text">{{ t('latestAIModel') }}</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-bolt feature-icon"></i>
                                <span class="feature-text">{{ t('oneClickReplication') }}</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-user-cog feature-icon"></i>
                                <span class="feature-text">{{ t('customizableCharacter') }}</span>
                            </div>
                        </div>

                        <!-- 操作按钮 -->
                        <div class="action-buttons">
                            <button @click="createSimilar" class="primary-button">
                                <i class="fas fa-magic mr-2"></i>
                                {{ getShareButtonText() }}
                            </button>

                            <!-- 详细信息按钮 -->
                            <button @click="showDetails = !showDetails" class="secondary-button">
                                <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="mr-2"></i>
                                {{ showDetails ? t('hideDetails') : t('showDetails') }}
                            </button>
                        </div>

                        <!-- 技术信息 -->
                        <div class="tech-info">
                            <p class="tech-text">
                                <a href="https://github.com/ModelTC/LightX2V" target="_blank" rel="noopener noreferrer" class="tech-link">
                                    {{ t('poweredByLightX2V') }}
                                </a>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 详细信息面板 -->
        <div v-if="showDetails && shareData" class="details-panel">
            <div class="details-content">
                <!-- 输入素材标题 -->
                <div class="materials-header">
                    <h2 class="materials-title">
                        <i class="fas fa-upload mr-2"></i>
                        {{ t('inputMaterials') }}
                    </h2>
                </div>

                <!-- 三个并列的分块卡片 -->
                <div class="materials-cards">
                    <!-- 图片卡片 -->
                    <div class="material-card">
                        <div class="card-header">
                            <i class="fas fa-image card-icon"></i>
                            <h3 class="card-title">{{ t('image') }}</h3>
                        </div>
                        <div class="card-content">
                            <div v-if="getImageMaterials().length > 0" class="image-grid">
                                <div v-for="[inputName, url] in getImageMaterials()" :key="inputName" class="image-item">
                                    <div class="image-container">
                                        <img :src="url" :alt="inputName" class="image-preview"
                                             @load="console.log('图片加载成功:', inputName, url)"
                                             @error="handleImageError($event, inputName, url)">
                                        <div class="image-placeholder" v-if="!url">
                                            <i class="fas fa-image"></i>
                                        </div>
                                        <div class="image-error-placeholder" v-if="false">
                                            <i class="fas fa-exclamation-triangle"></i>
                                            <p>图片加载失败</p>
                                        </div>
                                    </div>
\\
                                </div>
                            </div>
                            <div v-else class="empty-state">
                                <i class="fas fa-image empty-icon"></i>
                                <p class="empty-text">{{ t('noImage') }}</p>
                            </div>
                        </div>
                    </div>

                    <!-- 音频卡片 -->
                    <div class="material-card">
                        <div class="card-header">
                            <i class="fas fa-music card-icon"></i>
                            <h3 class="card-title">{{ t('audio') }}</h3>
                        </div>
                        <div class="card-content">
                            <div v-if="getAudioMaterials().length > 0" class="audio-list">
                                <div v-for="[inputName, url] in getAudioMaterials()" :key="inputName" class="audio-item">
                                    <audio :src="url" controls class="audio-player"></audio>
                                </div>
                            </div>
                            <div v-else class="empty-state">
                                <i class="fas fa-music empty-icon"></i>
                                <p class="empty-text">{{ t('noAudio') }}</p>
                            </div>
                        </div>
                    </div>

                    <!-- 提示词卡片 -->
                    <div class="material-card">
                        <div class="card-header">
                            <i class="fas fa-file-alt card-icon"></i>
                            <h3 class="card-title">{{ t('prompt') }}</h3>
                        </div>
                        <div class="card-content">
                            <div v-if="shareData.prompt" class="prompt-content">
                                <p class="prompt-text">{{ shareData.prompt }}</p>
                            </div>
                            <div v-else class="empty-state">
                                <i class="fas fa-file-alt empty-icon"></i>
                                <p class="empty-text">{{ t('noPrompt') }}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 全局路由跳转Loading覆盖层 -->
    <div v-show="isLoading" class="loading-overlay">
        <Loading />
    </div>
</template>

<style scoped>
/* Landing Page 样式 */
.landing-page {
    min-height: 100vh;
    width: 100%;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: white;
}

.main-content {
    width: 100%;
    padding: 2rem 0;
    min-height: calc(100vh - 80px);
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}

/* 错误状态 */
.error-container {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
}

.error-content {
    text-align: center;
    max-width: 500px;
    padding: 2rem;
}

.error-icon {
    width: 80px;
    height: 80px;
    margin: 0 auto 1.5rem;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: #ef4444;
}

.error-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: white;
}

.error-message {
    color: #9ca3af;
    margin-bottom: 2rem;
    line-height: 1.6;
}

.error-button {
    padding: 0.75rem 1.5rem;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 0.75rem;
    color: white;
    font-weight: 500;
    transition: all 0.2s;
    cursor: pointer;
}

.error-button:hover {
    background: rgba(139, 92, 246, 0.3);
    border-color: rgba(139, 92, 246, 0.6);
    transform: translateY(-1px);
}

/* 内容网格布局 */
.content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4rem;
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
    align-items: center;
    min-height: 60vh;
}

/* 视频区域 */
.video-section {
    display: flex;
    justify-content: center;
    align-items: center;
}

.video-container {
    width: 100%;
    max-width: 500px;
    aspect-ratio: 9/16;
    background: #000;
    border-radius: 1rem;
    overflow: hidden;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
    position: relative;
}

.video-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #1f2937;
}

.loading-spinner {
    font-size: 2rem;
    color: #8b5cf6;
    margin-bottom: 1rem;
}

.loading-text {
    color: #9ca3af;
    font-size: 0.875rem;
}

.video-player {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.video-error {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #1f2937;
    color: #ef4444;
}

.video-error i {
    font-size: 2rem;
    margin-bottom: 1rem;
}

/* 信息区域 */
.info-section {
    display: flex;
    align-items: center;
    justify-content: center;
}

.info-content {
    max-width: 500px;
    padding: 2rem 0;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.main-description {
    font-size: 1.25rem;
    color: #d1d5db;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* 特性列表 */
.features-list {
    margin-bottom: 2.5rem;
}

.feature-item {
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
    padding: 0.75rem 0;
}

.feature-icon {
    width: 40px;
    height: 40px;
    background: rgba(139, 92, 246, 0.1);
    border-radius: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    color: #8b5cf6;
    font-size: 1.125rem;
}

.feature-text {
    font-size: 1rem;
    color: #e5e7eb;
    font-weight: 500;
}

/* 操作按钮 */
.action-buttons {
    margin-bottom: 2rem;
}

.primary-button {
    width: 100%;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    border: none;
    border-radius: 0.75rem;
    color: white;
    font-size: 1.125rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.primary-button:hover {
    background: linear-gradient(135deg, #7c3aed, #9333ea);
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(139, 92, 246, 0.4);
}

.secondary-button {
    width: 100%;
    padding: 0.75rem 1.5rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.5rem;
    color: white;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    justify-content: center;
}

.secondary-button:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
}

/* 技术信息 */
.tech-info {
    text-align: center;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.tech-text {
    color: #9ca3af;
    font-size: 0.875rem;
    font-weight: 500;
}

.tech-link {
    color: #9ca3af;
    text-decoration: underline;
    transition: color 0.3s ease;
}

.tech-link:hover {
    color: #8b5cf6;
}

/* 详细信息面板 */
.details-panel {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);

    padding: 5rem 0;
}

.details-content {
    width: 100%;
    margin: 0 auto;
    padding: 0 2rem;
}

/* 输入素材标题 */
.materials-header {
    text-align: center;
    margin-bottom: 2rem;
}

.materials-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
}

/* 三个并列的卡片 */
.materials-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

.material-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    overflow: hidden;
    transition: all 0.3s ease;
}

.material-card:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(139, 92, 246, 0.3);
    transform: translateY(-2px);
}

.card-header {
    background: rgba(139, 92, 246, 0.1);
    padding: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.card-icon {
    font-size: 1.25rem;
    color: #8b5cf6;
}

.card-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: white;
    margin: 0;
}

.card-content {
    padding: 1.5rem;
    min-height: 200px;
}

/* 图片网格 */
.image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
}

.image-item {
    text-align: center;
}

.image-container {
    position: relative;
    width: 100%;
    min-height: 120px;
    margin-bottom: 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    justify-content: center;
}

.image-preview {
    max-width: 100%;
    min-height: 80px;
    height: auto;
    width: auto;
    object-fit: contain;
    display: block;
    position: relative !important;
}

.image-placeholder {
    width: 100%;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.1);
    color: #9ca3af;
    font-size: 1.5rem;
}

.image-error-placeholder {
    width: 100%;
    height: 80px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    font-size: 0.875rem;
    text-align: center;
}

.image-error-placeholder i {
    font-size: 1.5rem;
    margin-bottom: 0.25rem;
}

.image-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 0;
    word-break: break-all;
}

.debug-url {
    font-size: 0.6rem;
    color: #6b7280;
    margin: 0.25rem 0 0 0;
    word-break: break-all;
    opacity: 0.7;
}

/* 音频列表 */
.audio-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.audio-item {
    text-align: center;
}

.audio-player {
    width: 100%;
    height: 40px;
    margin-bottom: 0.5rem;
}

.audio-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 0;
    word-break: break-all;
}

/* 提示词内容 */
.prompt-content {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.5rem;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.prompt-text {
    color: #d1d5db;
    line-height: 1.6;
    margin: 0;
    word-break: break-word;
}

/* 空状态 */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 120px;
    color: #6b7280;
}

.empty-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}

.empty-text {
    font-size: 0.875rem;
    margin: 0;
    opacity: 0.7;
}

.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

/* 响应式设计 */
@media (max-width: 1024px) {
    .content-grid {
        gap: 3rem;
        padding: 0 1.5rem;
    }

    .main-title {
        font-size: 2.5rem;
    }

    .video-container {
        max-width: 400px;
    }

    /* 卡片响应式 */
    .materials-cards {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
}

@media (max-width: 768px) {
    .main-content {
        padding: 1rem 0;
    }

    .content-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
        padding: 0 1rem;
    }

    .main-title {
        font-size: 2rem;
    }

    .main-description {
        font-size: 1.125rem;
    }

    .video-container {
        max-width: 300px;
    }

    .info-content {
        padding: 1rem 0;
    }

    .details-content {
        padding: 0 1rem;
    }

    /* 移动端卡片调整 */
    .materials-cards {
        gap: 1rem;
    }

    .card-content {
        padding: 1rem;
        min-height: 150px;
    }

    .image-grid {
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }

    .materials-title {
        font-size: 1.25rem;
    }
}
</style>
