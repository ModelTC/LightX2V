<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import topMenu from '../components/TopBar.vue'
import Loading from '../components/Loading.vue'
import Alert from '../components/Alert.vue'
import Confirm from '../components/Confirm.vue'
import {
    isLoading,
    selectedTaskId,
    getCurrentForm,
    setCurrentImagePreview,
    setCurrentAudioPreview,
    setCurrentVideoPreview,
    getTemplateFileUrl,
    getTaskFileUrl,
    i2iForm,
    i2iImagePreviews,
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
const imageUrl = ref('')
const inputUrls = ref({})
const showDetails = ref(false)
const videoLoading = ref(false)
const videoError = ref(false)
const imageLoading = ref(false)
const imageError = ref(false)

// 音频播放器相关（支持多个音频，参考TaskDetails实现）
const audioElements = ref({}) // 使用对象存储多个音频元素，key 为 inputName
const audioStates = ref({}) // 存储每个音频的状态，key 为 inputName

// 判断是否是图片输出任务（i2i 或 t2i）
const isImageTask = computed(() => {
    return shareData.value?.task_type === 'i2i' || shareData.value?.task_type === 't2i'
})

// 获取分享数据
const fetchShareData = async () => {
    try {
        const response = await fetch(`/api/v1/share/${shareId.value}`)

        if (!response.ok) {
            throw new Error('分享不存在或已过期')
        }

        const data = await response.json()
        shareData.value = data

        // 根据任务类型设置输出URL
        if (data.task_type === 'i2i' || data.task_type === 't2i') {
            // 图片输出任务：设置图片URL
            if (data.output_image_url) {
                imageLoading.value = true
                imageError.value = false
                imageUrl.value = data.output_image_url
            }
        } else {
            // 视频输出任务：设置视频URL
            if (data.output_video_url) {
                videoUrl.value = data.output_video_url
            }
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
        // 根据任务类型显示不同的文本
        if (isImageTask.value) {
            return `${username}${t('userGeneratedImage')}`
        }
        return `${username}${t('userGeneratedVideo')}`
    }
    // 模板类型也根据任务类型区分
    if (isImageTask.value) {
        return t('templateImage')
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

// 切换详细信息显示
const toggleDetails = () => {
    showDetails.value = !showDetails.value
    if (showDetails.value) {
        // 等待DOM更新后滚动到详细信息面板
        setTimeout(() => {
            const detailsPanel = document.getElementById('share-details-panel')
            if (detailsPanel) {
                detailsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }
        }, 100)
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

// 根据任务类型获取应该显示的内容类型
const getVisibleMaterials = computed(() => {
    if (!shareData.value?.task_type) {
        return { image: false, video: false, audio: false, prompt: false }
    }

    const taskType = shareData.value.task_type

    // 根据任务类型定义应该显示的内容
    const visibilityMap = {
        't2v': {
            image: false,
            video: false,
            audio: false,
            prompt: true
        },
        'i2v': {
            image: true,
            video: false,
            audio: false,
            prompt: true
        },
        'i2i': {
            image: true,
            video: false,
            audio: false,
            prompt: true
        },
        't2i': {
            image: false,  // t2i 不需要输入图片
            video: false,
            audio: false,
            prompt: true
        },
        's2v': {
            image: true,
            video: false,
            audio: true,
            prompt: true
        },
        'animate': {
            image: true,
            video: true,
            audio: false,
            prompt: false  // animate 任务不显示 prompt
        }
    }

    return visibilityMap[taskType] || {
        image: true,
        video: false,
        audio: true,
        prompt: true
    }
})

// 获取图片素材（支持多图，参考TaskDetails实现）
const getImageMaterials = computed(() => {
    if (!shareData.value?.inputs?.input_image || !inputUrls.value) return []

    const inputImage = shareData.value.inputs.input_image
    const imageMaterials = []

    // 检查是否是逗号分隔的多图路径
    if (typeof inputImage === 'string' && inputImage.includes(',')) {
        // 按逗号拆分路径
        const imagePaths = inputImage.split(',').map(path => path.trim()).filter(path => path)

        // 为每个路径生成 URL
        imagePaths.forEach((path, index) => {
            // 使用 input_image_0, input_image_1, input_image_2 等作为 key
            const inputName = `input_image_${index}`
            const url = inputUrls.value[inputName] || inputUrls.value[`input_image_${index}`]

            if (url) {
                imageMaterials.push([inputName, url, index])
            }
        })
    } else {
        // 单图情况：优先使用 input_image_0，如果没有则使用 input_image
        const url0 = inputUrls.value.input_image_0
        const url = url0 || inputUrls.value.input_image
        if (url) {
            imageMaterials.push(['input_image_0', url, 0])
        }
    }

    return imageMaterials
})

// 获取视频素材
const getVideoMaterials = () => {
    if (!shareData.value?.inputs?.input_video || !inputUrls.value) return []
    const videoUrl = inputUrls.value.input_video
    if (videoUrl) {
        return [['input_video', videoUrl]]
    }
    return []
}

// 获取音频素材
const getAudioMaterials = () => {
    if (!inputUrls.value) return []
    const audioMaterials = Object.entries(inputUrls.value).filter(([name, url]) =>
        name.includes('audio') && url
    )
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

// 格式化音频时间
const formatAudioTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

// 设置音频元素 ref（安全版本）
const setAudioElement = (inputName, el) => {
    if (!audioElements.value) {
        audioElements.value = {}
    }
    if (el) {
        audioElements.value[inputName] = el
    } else if (audioElements.value[inputName]) {
        // 元素被卸载时，清理 ref
        delete audioElements.value[inputName]
    }
}

// 获取音频元素
const getAudioElement = (inputName) => {
    if (!audioElements.value) {
        audioElements.value = {}
    }
    return audioElements.value[inputName]
}

// 获取音频状态
const getAudioState = (inputName) => {
    if (!audioStates.value) {
        audioStates.value = {}
    }
    if (!audioStates.value[inputName]) {
        audioStates.value[inputName] = {
            isPlaying: false,
            duration: 0,
            currentTime: 0,
            isDragging: false
        }
    }
    return audioStates.value[inputName]
}

// 切换播放/暂停
const toggleAudioPlayback = (inputName) => {
    const audio = getAudioElement(inputName)
    if (!audio) {
        console.warn('Audio element not found for:', inputName)
        return
    }

    const state = getAudioState(inputName)

    if (audio.paused) {
        audio.play().catch(error => {
            console.error('播放失败:', error)
            showAlert(t('audioPlaybackFailed') + ': ' + error.message, 'error')
        })
    } else {
        audio.pause()
    }
}

// 音频加载完成
const onAudioLoaded = (inputName) => {
    const audio = getAudioElement(inputName)
    const state = getAudioState(inputName)
    if (audio && state) {
        state.duration = audio.duration || 0
    }
}

// 时间更新
const onTimeUpdate = (inputName) => {
    const audio = getAudioElement(inputName)
    const state = getAudioState(inputName)
    if (audio && state && !state.isDragging) {
        state.currentTime = audio.currentTime || 0
    }
}

// 进度条变化处理
const onProgressChange = (event, inputName) => {
    const audio = getAudioElement(inputName)
    const state = getAudioState(inputName)
    if (state && state.duration > 0 && audio && event.target) {
        const newTime = parseFloat(event.target.value)
        state.currentTime = newTime
        audio.currentTime = newTime
    }
}

// 进度条拖拽结束处理
const onProgressEnd = (event, inputName) => {
    const audio = getAudioElement(inputName)
    const state = getAudioState(inputName)
    if (audio && state && state.duration > 0 && event.target) {
        const newTime = parseFloat(event.target.value)
        audio.currentTime = newTime
        state.currentTime = newTime
    }
    if (state) {
        state.isDragging = false
    }
}

// 播放结束
const onAudioEnded = (inputName) => {
    const state = getAudioState(inputName)
    if (state) {
        state.isPlaying = false
        state.currentTime = 0
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
        showAlert(t('shareDataIncomplete'), 'danger')
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

        // t2i 任务不需要输入图片，直接返回
        if (shareData.value.task_type === 't2i') {
            // 切换到创建视图
            isCreationAreaExpanded.value = true
            switchToCreateView()
            showAlert(`已应用分享数据`, 'success')
            return
        }

        // 如果有输入图片，先设置URL，延迟加载文件（i2i 任务需要）
        if (shareData.value.inputs && shareData.value.inputs.input_image) {
            let imageUrl
            if (shareData.value.share_type === 'template' && shareData.value.inputs?.input_image) {
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
                        // 检查是否是 i2i 任务且是多图场景
                        const isI2IMultiImage = shareData.value.task_type === 'i2i' &&
                            shareData.value.inputs &&
                            shareData.value.inputs.input_image &&
                            typeof shareData.value.inputs.input_image === 'string' &&
                            shareData.value.inputs.input_image.includes(',');

                        if (isI2IMultiImage) {
                            // i2i 多图场景：加载所有图片
                            try {
                                // 解析逗号分隔的图片路径
                                const imagePaths = shareData.value.inputs.input_image.split(',').map(path => path.trim()).filter(path => path);

                                // 初始化数组
                                if (!i2iForm.value.imageFiles) {
                                    i2iForm.value.imageFiles = [];
                                }
                                i2iImagePreviews.value = [];

                                // 加载每张图片
                                const imageLoadPromises = imagePaths.map(async (imagePath, index) => {
                                    try {
                                        // 获取图片 URL（使用 input_image_0, input_image_1, input_image_2 等）
                                        const inputName = `input_image_${index}`;
                                        let singleImageUrl;

                                        // 优先使用 input_urls 中的 URL（模板和任务都可能有）
                                        singleImageUrl = shareData.value.input_urls?.[inputName];

                                        if (!singleImageUrl) {
                                            if (shareData.value.share_type === 'template') {
                                                // 模板：使用模板文件URL
                                                singleImageUrl = getTemplateFileUrl(imagePath, 'images');
                                            } else {
                                                // 任务：从API获取
                                                singleImageUrl = await getTaskFileUrl(shareData.value.task_id, inputName);
                                            }
                                        }

                                        if (singleImageUrl) {
                                            const imageResponse = await fetch(singleImageUrl);
                                            if (imageResponse && imageResponse.ok) {
                                                const blob = await imageResponse.blob();
                                                const filename = shareData.value.inputs[inputName] || `image_${index}.png`;
                                                const file = new File([blob], filename, { type: blob.type });

                                                // 读取为 data URL 用于预览
                                                const dataUrl = await new Promise((resolve, reject) => {
                                                    const reader = new FileReader();
                                                    reader.onload = (e) => resolve(e.target.result);
                                                    reader.onerror = reject;
                                                    reader.readAsDataURL(file);
                                                });

                                                return { file, dataUrl };
                                            }
                                        }
                                    } catch (error) {
                                        console.warn(`Failed to load image ${index}:`, error);
                                        return null;
                                    }
                                    return null;
                                });

                                // 等待所有图片加载完成
                                const imageData = await Promise.all(imageLoadPromises);
                                const validImageData = imageData.filter(item => item !== null);

                                // 添加到表单和预览
                                validImageData.forEach(({ file, dataUrl }) => {
                                    i2iForm.value.imageFiles.push(file);
                                    i2iImagePreviews.value.push(dataUrl);
                                });

                                // 同步更新 imageFile 以保持兼容性
                                if (i2iForm.value.imageFiles.length > 0) {
                                    i2iForm.value.imageFile = i2iForm.value.imageFiles[0];
                                }

                                console.log(`分享 - 从后端加载 ${validImageData.length} 张图片（i2i 多图模式）`);
                            } catch (error) {
                                console.warn('Failed to load multiple images:', error);
                            }
                        } else {
                            // 单图场景：原有逻辑
                            const imageResponse = await fetch(imageUrl);
                            if (imageResponse.ok) {
                                const blob = await imageResponse.blob();
                                const filename = shareData.value.inputs.input_image
                                const file = new File([blob], filename, { type: blob.type });

                                if (shareData.value.task_type === 'i2i') {
                                    // i2i 单图：也使用 imageFiles 数组（保持一致性）
                                    if (!i2iForm.value.imageFiles) {
                                        i2iForm.value.imageFiles = [];
                                    }
                                    i2iForm.value.imageFiles = [file];
                                    i2iForm.value.imageFile = file;

                                    // 读取为 data URL 用于预览
                                    const reader = new FileReader();
                                    reader.onload = (e) => {
                                        if (!i2iImagePreviews.value) {
                                            i2iImagePreviews.value = [];
                                        }
                                        i2iImagePreviews.value = [e.target.result];
                                    };
                                    reader.readAsDataURL(file);
                                } else {
                                    // 其他任务类型：单图模式
                                    currentForm.imageFile = file;
                                    setCurrentImagePreview(URL.createObjectURL(file));
                                }

                                console.log('分享图片文件已加载');
                            }
                        }
                    } catch (error) {
                        console.warn('Failed to load share image file:', error)
                    }
                }, 100)
            }
        }

        // 如果有输入视频，先设置URL，延迟加载文件（仅 animate 任务）
        if (shareData.value.inputs && shareData.value.inputs.input_video &&
            shareData.value.task_type === 'animate') {
            let videoUrl
            if (shareData.value.share_type === 'template' && shareData.value.inputs?.input_video) {
                // 对于模板，使用模板文件URL
                videoUrl = getTemplateFileUrl(shareData.value.inputs.input_video, 'videos')
            } else {
                // 对于任务，使用分享数据中的URL
                videoUrl = shareData.value.input_urls?.input_video || shareData.value.input_urls?.[Object.keys(shareData.value.input_urls).find(key => key.includes('video'))]
            }

            if (videoUrl) {
                currentForm.videoUrl = videoUrl
                setCurrentVideoPreview(videoUrl) // 直接使用URL作为预览
                console.log('分享输入视频:', videoUrl)

                // 异步加载视频文件（不阻塞UI）
                setTimeout(async () => {
                    try {
                        const videoResponse = await fetch(videoUrl)
                        if (videoResponse.ok) {
                            const blob = await videoResponse.blob()
                            const filename = shareData.value.inputs.input_video || 'input_video.mp4'

                            // 根据文件扩展名确定正确的MIME类型
                            let mimeType = blob.type
                            if (!mimeType || mimeType === 'application/octet-stream') {
                                const ext = filename.toLowerCase().split('.').pop()
                                const mimeTypes = {
                                    'mp4': 'video/mp4',
                                    'm4v': 'video/x-m4v',
                                    'mpeg': 'video/mpeg',
                                    'webm': 'video/webm',
                                    'mov': 'video/quicktime'
                                }
                                mimeType = mimeTypes[ext] || 'video/mp4'
                            }

                            const file = new File([blob], filename, { type: mimeType })
                            currentForm.videoFile = file

                            // 读取为 data URL 用于预览
                            const reader = new FileReader()
                            reader.onload = (e) => {
                                setCurrentVideoPreview(e.target.result)
                            }
                            reader.readAsDataURL(file)

                            console.log('分享视频文件已加载:', {
                                name: file.name,
                                type: file.type,
                                size: file.size
                            })
                        }
                    } catch (error) {
                        console.warn('Failed to load share video file:', error)
                    }
                }, 100)
            }
        }

        // 如果有输入音频，先设置URL，延迟加载文件（i2i 和 t2i 任务不需要音频）
        if (shareData.value.inputs && shareData.value.inputs.input_audio &&
            shareData.value.task_type !== 'i2i' && shareData.value.task_type !== 't2i') {
            let audioUrl
            if (shareData.value.share_type === 'template' && shareData.value.inputs?.input_audio) {
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

onUnmounted(() => {
    // 清理所有音频资源
    Object.values(audioElements.value).forEach(audio => {
        if (audio) {
            audio.pause()
        }
    })
    audioElements.value = {}
    audioStates.value = {}
})
</script>

<template>
    <!-- Apple 极简风格分享页面 -->
    <div class="min-h-screen w-full bg-[#f5f5f7] dark:bg-[#000000] flex flex-col">
        <!-- TopBar -->
        <topMenu />

        <!-- 主要内容区域 -->
        <div class="flex-1 overflow-y-auto main-scrollbar">
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
            <div v-else-if="shareData" class="w-full">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 w-full max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-12 lg:py-16 items-center">
                <!-- 左侧视频/图片区域 -->
                <div class="flex justify-center items-center">
                    <div class="w-full max-w-[400px] aspect-[9/16] bg-black dark:bg-[#000000] rounded-2xl overflow-hidden shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.5)] relative">
                        <!-- 图片输出任务：显示图片 -->
                        <template v-if="isImageTask">
                            <!-- 图片加载占位符 - Apple 风格 -->
                            <div v-if="!imageUrl || (imageLoading && !imageError)" class="w-full h-full flex flex-col items-center justify-center bg-[#f5f5f7] dark:bg-[#1c1c1e] absolute inset-0 z-10">
                                <div class="relative w-12 h-12 mb-6">
                                    <div class="absolute inset-0 rounded-full border-2 border-black/8 dark:border-white/8"></div>
                                    <div class="absolute inset-0 rounded-full border-2 border-transparent border-t-[color:var(--brand-primary)] dark:border-t-[color:var(--brand-primary-light)] animate-spin"></div>
                                </div>
                                <p class="text-sm font-medium text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loadingImage') || '加载图片中' }}...</p>
                            </div>

                            <!-- 图片显示 - 始终渲染以便触发加载事件 -->
                            <img
                                v-if="imageUrl"
                                :src="imageUrl"
                                class="w-full h-full object-contain"
                                :class="{ 'opacity-0': imageLoading && !imageError }"
                                @load="imageLoading = false; imageError = false"
                                @error="imageLoading = false; imageError = true"
                                alt="Generated Image">

                            <!-- 图片错误状态 - Apple 风格 -->
                            <div v-if="imageError" class="w-full h-full flex flex-col items-center justify-center bg-[#fef2f2] dark:bg-[#2c1b1b] absolute inset-0 z-10">
                                <div class="w-16 h-16 rounded-full bg-red-500/10 dark:bg-red-400/10 flex items-center justify-center mb-4">
                                    <i class="fas fa-exclamation-triangle text-3xl text-red-500 dark:text-red-400"></i>
                                </div>
                                <p class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('imageNotAvailable') || '图片不可用' }}</p>
                            </div>
                        </template>

                        <!-- 视频输出任务：显示视频 -->
                        <template v-else>
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
                        </template>
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
                            <button @click="toggleDetails"
                                    class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-8 py-3 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="text-sm"></i>
                                <span>{{ showDetails ? t('hideDetails') : t('showDetails') }}</span>
                            </button>
                        </div>

                        <!-- 技术信息 - Apple 风格 - 响应式 -->
                        <div class="text-center pt-4 sm:pt-6 border-t border-black/8 dark:border-white/8">
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
                <div v-if="showDetails" id="share-details-panel" class="w-full bg-white dark:bg-[#1c1c1e] border-t border-black/8 dark:border-white/8 py-16">
                    <div class="max-w-6xl mx-auto px-6 sm:px-8 lg:px-12">
                        <!-- 输入素材标题 - Apple 风格 -->
                        <h2 class="text-2xl sm:text-3xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center justify-center gap-3 mb-10 tracking-tight">
                            <i class="fas fa-upload text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                            <span>{{ t('inputMaterials') }}</span>
                        </h2>

                        <!-- 根据任务类型显示相应的素材卡片 - Apple 风格 -->
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <!-- 图片卡片 - Apple 风格 -->
                            <div v-if="getVisibleMaterials.image" class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                <!-- 卡片头部 -->
                                <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                    <div class="flex items-center gap-3">
                                        <i class="fas fa-image text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('image') }}</h3>
                                    </div>
                                </div>
                                <!-- 卡片内容 -->
                                <div class="p-6 min-h-[200px]">
                                    <div v-if="getImageMaterials.length > 0">
                                        <div v-for="([inputName, url, index]) in getImageMaterials" :key="inputName || index" class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8 mb-3 last:mb-0">
                                            <div class="relative">
                                                <img v-if="url" :src="url" :alt="inputName || `图片 ${index + 1}`" class="w-full h-auto object-contain" @error="handleImageError($event, inputName, url)">
                                                <!-- 多图时显示序号 -->
                                                <div v-if="getImageMaterials.length > 1 && url" class="absolute top-2 left-2 px-2 py-1 bg-black/50 dark:bg-black/70 text-white text-xs rounded backdrop-blur-sm">
                                                    {{ index + 1 }} / {{ getImageMaterials.length }}
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                        <i class="fas fa-image text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                        <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noImage') }}</p>
                                    </div>
                                </div>
                            </div>

                            <!-- 视频卡片 - Apple 风格 -->
                            <div v-if="getVisibleMaterials.video" class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                <!-- 卡片头部 -->
                                <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                    <div class="flex items-center gap-3">
                                        <i class="fas fa-video text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('video') }}</h3>
                                    </div>
                                </div>
                                <!-- 卡片内容 -->
                                <div class="p-6 min-h-[200px]">
                                    <div v-if="getVideoMaterials().length > 0">
                                        <div v-for="[inputName, url] in getVideoMaterials()" :key="inputName" class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8">
                                            <video :src="url" :alt="inputName" class="w-full h-auto object-contain" controls preload="metadata">
                                                {{ t('browserNotSupported') }}
                                            </video>
                                        </div>
                                    </div>
                                    <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                        <i class="fas fa-video text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                        <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noVideo') }}</p>
                                    </div>
                                </div>
                            </div>

                            <!-- 音频卡片 - Apple 风格 -->
                            <div v-if="getVisibleMaterials.audio" class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
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
                                            <!-- 音频播放器卡片 - Apple 风格 -->
                                            <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)] w-full p-4">
                                                <div class="relative flex items-center mb-3">
                                                    <!-- 头像容器 -->
                                                    <div class="relative mr-3 flex-shrink-0">
                                                        <!-- 透明白色头像 -->
                                                        <div class="w-12 h-12 rounded-full bg-white/40 dark:bg-white/20 border border-white/30 dark:border-white/20 transition-all duration-200"></div>
                                                        <!-- 播放/暂停按钮 -->
                                                        <button
                                                            @click="toggleAudioPlayback(inputName)"
                                                            class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white cursor-pointer hover:scale-110 transition-all duration-200 z-20 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]"
                                                        >
                                                            <i :class="getAudioState(inputName).isPlaying ? 'fas fa-pause' : 'fas fa-play'" class="text-xs ml-0.5"></i>
                                                        </button>
                                                    </div>

                                                    <!-- 音频信息 -->
                                                    <div class="flex-1 min-w-0">
                                                        <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate">
                                                            {{ t('audio') }}
                                                        </div>
                                                    </div>

                                                    <!-- 音频时长 -->
                                                    <div class="text-xs font-medium text-[#86868b] dark:text-[#98989d] tracking-tight flex-shrink-0">
                                                        {{ formatAudioTime(getAudioState(inputName).currentTime) }} / {{ formatAudioTime(getAudioState(inputName).duration) }}
                                                    </div>
                                                </div>

                                                <!-- 进度条 -->
                                                <div class="flex items-center gap-2" v-if="getAudioState(inputName).duration > 0">
                                                    <input
                                                        type="range"
                                                        :min="0"
                                                        :max="getAudioState(inputName).duration"
                                                        :value="getAudioState(inputName).currentTime"
                                                        @input="(e) => onProgressChange(e, inputName)"
                                                        @change="(e) => onProgressChange(e, inputName)"
                                                        @mousedown="getAudioState(inputName).isDragging = true"
                                                        @mouseup="(e) => onProgressEnd(e, inputName)"
                                                        @touchstart="getAudioState(inputName).isDragging = true"
                                                        @touchend="(e) => onProgressEnd(e, inputName)"
                                                        class="flex-1 h-1 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                                                    />
                                                </div>
                                            </div>

                                            <!-- 隐藏的音频元素 -->
                                            <audio
                                                :ref="(el) => setAudioElement(inputName, el)"
                                                :src="url"
                                                @loadedmetadata="() => onAudioLoaded(inputName)"
                                                @timeupdate="() => onTimeUpdate(inputName)"
                                                @ended="() => onAudioEnded(inputName)"
                                                @play="() => getAudioState(inputName).isPlaying = true"
                                                @pause="() => getAudioState(inputName).isPlaying = false"
                                                @error="(e) => { console.error('Audio error:', e, url); showAlert(t('audioLoadFailed'), 'error') }"
                                                preload="metadata"
                                                class="hidden"
                                            ></audio>
                                        </div>
                                    </div>
                                    <div v-else class="flex flex-col items-center justify-center h-[150px]">
                                        <i class="fas fa-music text-3xl text-[#86868b]/30 dark:text-[#98989d]/30 mb-3"></i>
                                        <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('noAudio') }}</p>
                                    </div>
                                </div>
                            </div>

                            <!-- 提示词卡片 - Apple 风格 -->
                            <div v-if="getVisibleMaterials.prompt" class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl overflow-hidden transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
                                <!-- 卡片头部 -->
                                <div class="flex items-center justify-between px-5 py-4 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 border-b border-black/8 dark:border-white/8">
                                    <div class="flex items-center gap-3">
                                        <i class="fas fa-file-alt text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <h3 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('prompt') }}</h3>
                                    </div>
                                    <button v-if="shareData?.params?.prompt || shareData?.prompt"
                                            @click="copyPrompt(shareData?.params?.prompt || shareData?.prompt)"
                                            class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                            :title="t('copy')">
                                        <i class="fas fa-copy text-xs"></i>
                                    </button>
                                </div>
                                <!-- 卡片内容 -->
                                <div class="p-6 min-h-[200px]">
                                    <div v-if="shareData?.params?.prompt || shareData?.prompt" class="bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-xl p-4">
                                        <p class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight break-words">{{ shareData?.params?.prompt || shareData?.prompt }}</p>
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
        <Alert />
        <Confirm />
    </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* Apple 风格极简黑白设计 */
</style>
