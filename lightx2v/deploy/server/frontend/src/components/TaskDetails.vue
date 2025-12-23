<script setup>
import { ref, watch, onMounted, onUnmounted, computed, nextTick } from 'vue'
import { showTaskDetailModal,
        modalTask,
        closeTaskDetailModal,
        cancelTask,
        reuseTask,
        handleDownloadFile,
        deleteTask,
        getTaskTypeName,
        showFailureDetails,
        formatTime,
        getTaskStatusDisplay,
        getStatusTextClass,
        getProgressTitle,
        getProgressInfo,
        getOverallProgress,
        getSubtaskStatusText,
        getSubtaskProgress,
        formatEstimatedTime,
        generateShareUrl,
        copyShareLink,
        shareToSocial,
        copyPrompt,
        getTaskFileUrlSync,
        getTaskFileFromCache,
        getTaskFileUrl,
        getTaskFileUrlFromApi,
        getTaskInputAudio,
        downloadLoading,
        showAlert,
        apiRequest,
        startPollingTask,
        resumeTask,
         } from '../utils/other'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()

// 添加响应式变量
const showDetails = ref(false)
const loadingTaskFiles = ref(false)

// 音频播放器相关（支持多个音频）
const audioElements = ref({}) // 使用对象存储多个音频元素，key 为 inputName
const audioStates = ref({}) // 存储每个音频的状态，key 为 inputName
const currentAudioUrl = ref('')

// 音频素材 URL（响应式，支持异步加载）
const audioMaterials = ref([])

// 图片素材 URL 缓存（用于异步加载）
const imageMaterialsCache = ref({})

// 获取图片素材（支持多图，逗号分隔）
const getImageMaterials = computed(() => {
    const taskId = modalTask.value?.task_id
    const taskType = modalTask.value?.task_type
    const inputs = modalTask.value?.inputs || {}

    // flf2v 任务：同时显示首帧和尾帧
    if (taskType === 'flf2v') {
        const imageMaterials = []

        // 首帧图片
        if (inputs.input_image) {
            const cacheKey = `${taskId}_input_image`
            let url = getTaskFileUrlSync(taskId, 'input_image')
            if (!url && imageMaterialsCache.value[cacheKey]) {
                url = imageMaterialsCache.value[cacheKey]
            }
            if (!url) {
                getTaskFileUrl(taskId, 'input_image').then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to load image URL for input_image:`, err)
                })
            }
            imageMaterials.push(['input_image', url, 0, 'firstFrameImage'])
        }

        // 尾帧图片
        if (inputs.input_last_frame) {
            const cacheKey = `${taskId}_input_last_frame`
            let url = getTaskFileUrlSync(taskId, 'input_last_frame')
            if (!url && imageMaterialsCache.value[cacheKey]) {
                url = imageMaterialsCache.value[cacheKey]
            }
            if (!url) {
                getTaskFileUrl(taskId, 'input_last_frame').then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to load image URL for input_last_frame:`, err)
                })
            }
            imageMaterials.push(['input_last_frame', url, 1, 'lastFrameImage'])
        }

        return imageMaterials
    }

    // 其他任务类型：原有的多图逻辑
    if (!inputs.input_image) return []

    const inputImage = inputs.input_image
    // multi-images
    const inputImages = Object.keys(inputs).filter(key => key.startsWith("input_image/"));

    if (inputImages.length > 0) {
        // 为每个路径生成 URL
        const imageMaterials = []
        // 多图输入时，名字分别为：input_image/xxx
        inputImages.forEach((inputName, index) => {
            const cacheKey = `${taskId}_${inputName}`

            // 先尝试从同步缓存获取
            let url = getTaskFileUrlSync(taskId, inputName)

            // 如果同步缓存中没有，尝试从异步缓存获取
            if (!url && imageMaterialsCache.value[cacheKey]) {
                url = imageMaterialsCache.value[cacheKey]
            }

            // 如果还是没有，异步加载（不阻塞渲染）
            if (!url) {
                // 异步加载 URL
                getTaskFileUrl(taskId, inputName).then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to load image URL for ${inputName}:`, err)
                })
            }

            // 返回结果（url 可能为 null，模板会处理）
            imageMaterials.push([inputName, url, index])
        })

        // 如果所有图片都获取到了 URL，返回结果；否则回退到单图模式
        return imageMaterials.length > 0 ? imageMaterials : [['input_image', getTaskFileUrlSync(taskId, 'input_image')]]
    } else {
        // 单图情况：使用 input_image
        const url = getTaskFileUrlSync(taskId, 'input_image')
        return [['input_image', url]]
    }
})

// 监听 imageMaterialsCache 变化，触发重新计算
watch(imageMaterialsCache, () => {
    // 当缓存更新时，computed 会自动重新计算
}, { deep: true })

// 获取视频素材
const getVideoMaterials = () => {
    if (!modalTask.value?.inputs?.input_video) return []
    return [['input_video', getTaskFileUrlSync(modalTask.value.task_id, 'input_video')]]
}

// 获取音频素材（使用响应式 ref）
const getAudioMaterials = () => {
    return audioMaterials.value
}

// 判断是否是图片输出任务（i2i 或 t2i）
const isImageTask = computed(() => {
    return modalTask.value?.task_type === 'i2i' || modalTask.value?.task_type === 't2i'
})

// 保持向后兼容
const isI2ITask = computed(() => {
    return modalTask.value?.task_type === 'i2i'
})

// 处理图片加载错误
const handleImageError = async (event, taskId, inputName) => {
    console.warn(`图片加载失败: ${inputName}`, event)
    // 尝试异步加载 URL
    try {
        const url = await getTaskFileUrl(taskId, inputName)
        if (url) {
            const cacheKey = `${taskId}_${inputName}`
            imageMaterialsCache.value[cacheKey] = url
            // 更新图片源
            if (event.target) {
                event.target.src = url
            }
        }
    } catch (error) {
        console.error(`无法加载图片 URL: ${inputName}`, error)
    }
}

// 监听 modalTask 变化，预加载所有图片 URL
watch(() => modalTask.value?.task_id, async (taskId) => {
    if (!taskId) return

    const taskType = modalTask.value?.task_type
    const inputs = modalTask.value?.inputs || {}

    // flf2v 任务：预加载首帧和尾帧
    if (taskType === 'flf2v') {
        // 预加载首帧
        if (inputs.input_image) {
            const cacheKey = `${taskId}_input_image`
            if (!getTaskFileUrlSync(taskId, 'input_image') && !imageMaterialsCache.value[cacheKey]) {
                getTaskFileUrl(taskId, 'input_image').then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to preload image URL for input_image:`, err)
                })
            }
        }

        // 预加载尾帧
        if (inputs.input_last_frame) {
            const cacheKey = `${taskId}_input_last_frame`
            if (!getTaskFileUrlSync(taskId, 'input_last_frame') && !imageMaterialsCache.value[cacheKey]) {
                getTaskFileUrl(taskId, 'input_last_frame').then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to preload image URL for input_last_frame:`, err)
                })
            }
        }
        return
    }

    // 其他任务类型：原有的预加载逻辑
    if (!inputs.input_image) return

    const inputImage = inputs.input_image
    // multi-images
    const inputImages = Object.keys(inputs).filter(key => key.startsWith("input_image/"));

    if (inputImages.length > 0) {
        // 为每个路径生成 URL
        const imageMaterials = []
        // 多图输入时，名字分别为：input_image/xxx
        inputImages.forEach((inputName, index) => {
            const cacheKey = `${taskId}_${inputName}`

            // 如果缓存中没有，异步加载
            if (!getTaskFileUrlSync(taskId, inputName) && !imageMaterialsCache.value[cacheKey]) {
                getTaskFileUrl(taskId, inputName).then(loadedUrl => {
                    if (loadedUrl) {
                        imageMaterialsCache.value[cacheKey] = loadedUrl
                    }
                }).catch(err => {
                    console.warn(`Failed to preload image URL for ${inputName}:`, err)
                })
            }
        })
    } else {
        // 单图情况：使用 input_image
        const inputName = 'input_image';
        const cacheKey = `${taskId}_${inputName}`
        if (!getTaskFileUrlSync(taskId, inputName) && !imageMaterialsCache.value[cacheKey]) {
            getTaskFileUrl(taskId, inputName).then(loadedUrl => {
                if (loadedUrl) {
                    imageMaterialsCache.value[cacheKey] = loadedUrl
                }
            }).catch(err => {
                console.warn(`Failed to preload image URL for ${inputName}:`, err)
            })
        }
    }
}, { immediate: true })

// 根据任务类型获取应该显示的内容类型
const getVisibleMaterials = computed(() => {
    if (!modalTask.value?.task_type) {
        return { image: false, video: false, audio: false, prompt: false }
    }

    const taskType = modalTask.value.task_type

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
        'flf2v': {
            image: true,  // flf2v 显示首帧和尾帧
            video: false,
            audio: false,
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

// 异步加载音频素材 URL（支持目录模式）
const loadAudioMaterials = async () => {
    if (!modalTask.value?.inputs?.input_audio) {
        audioMaterials.value = []
        return
    }

    try {
        // 使用 getTaskInputAudio 来获取音频 URL，它会自动处理目录情况
        const audioUrl = await getTaskInputAudio(modalTask.value)
        if (audioUrl) {
            audioMaterials.value = [['input_audio', audioUrl]]
        } else {
            audioMaterials.value = []
        }
    } catch (error) {
        console.error('Failed to load audio materials:', error)
        audioMaterials.value = []
    }
}

// 路由关闭功能
const closeWithRoute = () => {
    closeTaskDetailModal()
    modalTask.value = null
    // 只有当前路由是 /task/:id 时才进行路由跳转
    // 如果在其他页面（如 /generate）打开的弹窗，关闭时保持在原页面
    if (route.path.startsWith('/task/')) {
        // 从任务详情路由进入的，返回到上一页或首页
        if (window.history.length > 1) {
            router.go(-1)
        } else {
            router.push('/')
        }
    }
    // 如果不是任务详情路由，不做任何路由跳转，保持在当前页面
}

// 滚动到生成区域（仅在 generate 页面）
const scrollToCreationArea = () => {
    const mainScrollable = document.querySelector('.main-scrollbar')
    if (mainScrollable) {
        mainScrollable.scrollTo({
            top: 0,
            behavior: 'smooth'
        })
    }
}

// 包装 reuseTask 函数，复用任务后回到生成区域
const handleReuseTask = () => {
    const task = modalTask.value
    if (!task) {
        return
    }
    void reuseTask(task)
    if (route.path === '/generate' || route.name === 'Generate') {
        setTimeout(() => {
            scrollToCreationArea()
        }, 300)
    }
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'Escape' && showTaskDetailModal.value) {
        closeWithRoute()
    }
}

// 获取文件扩展名
const getFileExtension = (fileKey) => {
    if (fileKey.includes('video')) return 'mp4'
    if (fileKey.includes('image')) return 'jpg'
    if (fileKey.includes('audio')) return 'mp3'
    return 'file'
}


const getTaskFailureInfo = (task) => {
                if (!task) return null;

                // 检查子任务的失败信息
                if (!task.fail_msg && task.subtasks && task.subtasks.length > 0) {
                    const failedSubtasks = task.subtasks.filter(subtask =>
                        (subtask.extra_info && subtask.extra_info.fail_msg) || subtask.fail_msg
                    );
                    if (failedSubtasks.length > 0) {
                        const msg = failedSubtasks.map(subtask =>
                            (subtask.extra_info && subtask.extra_info.fail_msg) || subtask.fail_msg
                        ).join('\n');
                        task.fail_msg = msg;
                    }
                }
                console.log('task.fail_msg', task.fail_msg);
                return task.fail_msg;
            };

const viewTaskDetail = async (task) => {
    try {
        const response = await apiRequest(`/api/v1/task/query?task_id=${task.task_id}`);
        console.log('viewTaskDetail: response=', response);
        if (response && response.ok) {
            modalTask.value = await response.json();
            console.log('updated task data:', modalTask.value);
        }
    } catch (error) {
        console.warn(`Failed to fetch updated task data: task_id=${task.task_id}`, error.message);
    }
    // 如果任务还在进行中，开始轮询状态
    if (['CREATED', 'PENDING', 'RUNNING'].includes(task.status)) {

        startPollingTask(task.task_id);
    }
    if (['FAILED'].includes(task.status)) {
        modalTask.value.fail_msg = getTaskFailureInfo(task);
    }
};

// 监听modalTask的第一次变化，确保任务详情正确加载
const hasLoadedTask = ref(false);
watch(modalTask, async (newTask) => {
    if (newTask && !hasLoadedTask.value) {
        console.log('modalTask第一次变化，加载任务详情:', newTask);
        viewTaskDetail(newTask);
        hasLoadedTask.value = true;
    }
    // 加载音频素材（支持目录模式）
    if (newTask) {
        await loadAudioMaterials();
    }
}, { immediate: true });

// 生命周期钩子
onMounted(async () => {
    document.addEventListener('keydown', handleKeydown)
    console.log('TaskDetails组件已挂载，当前modalTask:', modalTask.value);
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
    // 清理所有音频资源
    Object.values(audioElements.value).forEach(audio => {
        if (audio) {
            audio.pause()
        }
    })
    audioElements.value = {}
    audioStates.value = {}
})

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

// 视频事件处理函数
const onVideoLoadStart = () => {
    // 视频开始加载
}

const onVideoCanPlay = () => {
    // 视频可以播放
}

const onVideoError = () => {
    // 视频加载错误
    console.error('Video load error')
}

// 监听音频URL变化
watch(audioMaterials, (newMaterials) => {
    if (newMaterials && newMaterials.length > 0) {
        currentAudioUrl.value = newMaterials[0][1]
        // 确保 audioStates.value 存在
        if (!audioStates.value) {
            audioStates.value = {}
        }
        // 为每个音频初始化状态
        newMaterials.forEach(([inputName, url]) => {
            if (!audioStates.value[inputName]) {
                audioStates.value[inputName] = {
                    isPlaying: false,
                    duration: 0,
                    currentTime: 0,
                    isDragging: false
                }
            }
        })
        // 加载所有音频
        nextTick(() => {
            newMaterials.forEach(([inputName]) => {
                const audio = getAudioElement(inputName)
                if (audio) {
                    audio.load()
                }
            })
        })
    } else {
        currentAudioUrl.value = ''
        audioStates.value = {}
    }
}, { immediate: true })
</script>
<template>
            <!-- 任务详情弹窗 - Apple 极简风格 -->
            <div v-cloak>
                <div v-if="showTaskDetailModal"
                    class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-2 sm:p-1"
                    @click="closeWithRoute">
                    <!-- 任务完成时的大弹窗 - Apple 风格 -->
                    <div v-if="modalTask?.status === 'SUCCEED'"
                        class="w-full h-full max-w-7xl max-h-[100vh] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden flex flex-col" @click.stop>
                        <!-- 弹窗头部 - Apple 风格 -->
                        <div class="flex items-center justify-between px-8 py-5 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
                            <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
                                <i class="fas fa-check-circle text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                {{ t('taskDetail') }}
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
                                        <!-- 图片输出任务（i2i 或 t2i）：显示输出图片 -->
                                        <img
                                            v-if="isImageTask && modalTask?.outputs?.output_image"
                                            :src="getTaskFileUrlSync(modalTask.task_id, 'output_image')"
                                            :alt="getTaskTypeName(modalTask?.task_type)"
                                            class="w-full h-full object-contain">

                                        <!-- 其他任务：显示视频播放器 -->
                                        <video
                                            v-else-if="!isImageTask && modalTask?.outputs?.output_video"
                                            :src="getTaskFileUrlSync(modalTask.task_id, 'output_video')"
                                            :poster="getTaskFileUrlSync(modalTask.task_id, 'input_image')"
                                            class="w-full h-full object-contain"
                                            controls
                                            loop
                                            preload="metadata"
                                            @loadstart="onVideoLoadStart"
                                            @canplay="onVideoCanPlay"
                                            @error="onVideoError">
                                            {{ t('browserNotSupported') }}
                                        </video>

                                        <!-- 无输出内容时显示占位符 -->
                                        <div v-else class="w-full h-full flex flex-col items-center justify-center bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                            <div class="w-16 h-16 rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center mb-4">
                                                <i :class="isImageTask ? 'fas fa-image' : 'fas fa-video'" class="text-3xl text-[#86868b] dark:text-[#98989d]"></i>
                                            </div>
                                            <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ isImageTask ? (t('imageNotAvailable') || t('videoNotAvailable')) : t('videoNotAvailable') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 -->
                                <div class="flex items-center justify-center">
                                    <div class="w-full max-w-[400px] aspect-[9/16] relative flex flex-col">

                                        <!-- 居中的内容区域 -->
                                        <div class="flex-1 flex items-center justify-center px-8 py-6">
                                            <div class="w-full">
                                                <div class="flex flex-col items-center gap-3 mb-6">
                                                    <div class="flex items-center gap-3">
                                                        <button @click="copyShareLink(modalTask.task_id, 'task')"
                                                            class="w-12 h-12 flex items-center justify-center bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.12)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.4)] hover:scale-110 active:scale-100 transition-all duration-200"
                                                            :title="t('share')">
                                                            <i class="fas fa-share-alt text-base"></i>
                                                        </button>
                                                        <button @click="deleteTask(modalTask.task_id, true)"
                                                            class="w-12 h-12 flex items-center justify-center bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.12)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.4)] hover:scale-110 active:scale-100 transition-all duration-200"
                                                            :title="t('delete')">
                                                            <i class="fas fa-trash text-base"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                                <!-- 标题 -->
                                                <div class="text-center mb-6">
                                                    <h1 class="text-3xl sm:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">
                                                        {{ t('taskCompleted') }}
                                                    </h1>
                                                    <p class="text-sm sm:text-base text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                        {{ t('taskCompletedSuccessfully') }}
                                                    </p>
                                                </div>

                                                <!-- 特性列表 - Apple 风格 -->
                                                <div class="grid grid-cols-3 gap-2 mb-6">
                                                    <div class="flex flex-col items-center gap-1.5 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl">
                                                        <i class="fas fa-toolbox text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span class="text-[11px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight">{{ getTaskTypeName(modalTask) }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-1.5 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl">
                                                        <i class="fas fa-robot text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span class="text-[11px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight truncate max-w-full">{{ modalTask.model_cls }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-1.5 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl">
                                                        <i class="fas fa-clock text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span class="text-[11px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight">{{ Math.round(modalTask.extra_info?.active_elapse || 0) }}s</span>
                                                    </div>
                                                </div>

                                                <!-- 操作按钮 - Apple 风格 -->
                                                <div class="space-y-2.5">
                                                    <!-- 图片输出任务（i2i 或 t2i）：下载图片按钮 -->
                                                    <button v-if="isImageTask && modalTask?.outputs?.output_image"
                                                            @click="handleDownloadFile(modalTask.task_id, 'output_image', modalTask.outputs.output_image)"
                                                            :disabled="downloadLoading"
                                                            class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-6 py-3 text-[15px] font-semibold text-white transition-all duration-200 ease-out tracking-tight flex items-center justify-center gap-2"
                                                            :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100'">
                                                        <i class="fas fa-download text-sm"></i>
                                                        <span>{{ t('downloadImage') || t('downloadVideo') }}</span>
                                                    </button>

                                                    <!-- 其他任务：下载视频按钮 -->
                                                    <button v-else-if="!isImageTask && modalTask?.outputs?.output_video"
                                                            @click="handleDownloadFile(modalTask.task_id, 'output_video', modalTask.outputs.output_video)"
                                                            :disabled="downloadLoading"
                                                            class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-6 py-3 text-[15px] font-semibold text-white transition-all duration-200 ease-out tracking-tight flex items-center justify-center gap-2"
                                                            :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100'">
                                                        <i class="fas fa-download text-sm"></i>
                                                        <span>{{ t('downloadVideo') }}</span>
                                                    </button>
                                                    <button @click="handleReuseTask"
                                                            class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                        <i class="fas fa-magic text-sm"></i>
                                                        <span>{{ t('reuseTask') }}</span>
                                                    </button>
                                                    <button @click="showDetails = !showDetails"
                                                            class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                        <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="text-sm"></i>
                                                        <span>{{ showDetails ? t('hideDetails') : t('showDetails') }}</span>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 详细信息面板 - Apple 风格 （成功状态）-->
                            <div v-if="showDetails && modalTask" class="bg-[#f5f5f7] dark:bg-[#1c1c1e] border-t border-black/8 dark:border-white/8 py-12">
                                <div class="max-w-6xl mx-auto px-8">
                                    <!-- 输入素材标题 - Apple 风格 -->
                                    <h2 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center justify-center gap-3 mb-8 tracking-tight">
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
                                                <button v-if="getImageMaterials.length > 0"
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_image', modalTask.inputs.input_image)"
                                                        :disabled="downloadLoading"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                        :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                        :title="t('download')">
                                                    <i class="fas fa-download text-xs"></i>
                                                </button>
                                            </div>
                                            <!-- 卡片内容 -->
                                            <div class="p-6 min-h-[200px]">
                                            <div v-if="getImageMaterials.length > 0">
                                                <div v-for="([inputName, url, index, label]) in getImageMaterials" :key="inputName || index" class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8 mb-3 last:mb-0">
                                                    <div class="relative">
                                                        <img v-if="url" :src="url" :alt="inputName || label || `图片 ${index + 1}`" class="w-full h-auto object-contain" @error="handleImageError($event, modalTask.task_id, inputName)">
                                                        <!-- 加载中或错误状态 -->
                                                        <div v-else class="flex flex-col items-center justify-center h-[150px] bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                                            <i class="fas fa-spinner fa-spin text-2xl text-[#86868b]/50 dark:text-[#98989d]/50 mb-2"></i>
                                                            <p class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loading') || '加载中...' }}</p>
                                                        </div>
                                                        <!-- 标签：flf2v 任务显示"首帧"/"尾帧"，其他多图显示序号 -->
                                                        <div v-if="url && (label || getImageMaterials.length > 1)" class="absolute top-2 left-2 px-2 py-1 bg-black/50 dark:bg-black/70 text-white text-xs rounded backdrop-blur-sm">
                                                            {{ label ? t(label) : `${index + 1} / ${getImageMaterials.length}` }}
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
                                                <button v-if="getVideoMaterials().length > 0"
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_video', modalTask.inputs.input_video)"
                                                        :disabled="downloadLoading"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                        :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                        :title="t('download')">
                                                    <i class="fas fa-download text-xs"></i>
                                                </button>
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
                                                <button v-if="getAudioMaterials().length > 0"
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_audio', modalTask.inputs.input_audio)"
                                                        :disabled="downloadLoading"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                        :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                        :title="t('download')">
                                                    <i class="fas fa-download text-xs"></i>
                                                </button>
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
                                                <button v-if="modalTask?.params?.prompt"
                                                        @click="copyPrompt(modalTask?.params?.prompt)"
                                                        class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                        :title="t('copy')">
                                                    <i class="fas fa-copy text-xs"></i>
                                                </button>
                                            </div>
                                            <!-- 卡片内容 -->
                                            <div class="p-6 min-h-[200px]">
                                                <div v-if="modalTask?.params?.prompt" class="bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-xl p-4">
                                                    <p class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight break-words">{{ modalTask.params.prompt }}</p>
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

                    <!-- 其他状态的弹窗 - Apple 风格 -->
                    <div v-else class="w-full max-w-7xl max-h-[95vh] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden flex flex-col" @click.stop>
                        <!-- 弹窗头部 - Apple 风格 -->
                        <div class="flex items-center justify-between px-8 py-5 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
                            <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
                                <i v-if="modalTask?.status === 'FAILED'" class="fas fa-exclamation-triangle text-red-500 dark:text-red-400"></i>
                                <i v-else-if="modalTask?.status === 'CANCEL'" class="fas fa-ban text-[#86868b] dark:text-[#98989d]"></i>
                                <i v-else class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                <span>{{ t('taskDetail') }}</span>
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
                                <!-- 左侧占位图区域 - Apple 风格 -->
                                <div class="flex items-center justify-center">
                                    <div class="w-full max-w-[400px] aspect-[9/16] bg-black dark:bg-[#000000] rounded-2xl overflow-hidden shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.5)] relative">
                                        <!-- 根据状态显示不同的占位图 -->
                                        <!-- 进行中状态 -->
                                        <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)" class="w-full h-full flex flex-col items-center justify-center relative bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                            <!-- 如果有图像输入，显示为背景（使用第一张图片） -->
                                            <div v-if="getImageMaterials.length > 0" class="absolute top-0 left-0 w-full h-full z-[1]">
                                                <img :src="getImageMaterials[0][1]" :alt="getImageMaterials[0][0] || '输入图片'" class="w-full h-full object-cover opacity-20 blur-sm">
                                            </div>
                                            <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2]">
                                                <div class="relative w-12 h-12 mb-6">
                                                    <div class="absolute inset-0 rounded-full border-2 border-black/8 dark:border-white/8"></div>
                                                    <div class="absolute inset-0 rounded-full border-2 border-transparent border-t-[color:var(--brand-primary)] dark:border-t-[color:var(--brand-primary-light)] animate-spin"></div>
                                                </div>
                                                <p class="text-sm font-medium text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('videoGenerating') }}...</p>
                                            </div>
                                        </div>
                                        <!-- 失败状态 -->
                                        <div v-else-if="modalTask?.status === 'FAILED'" class="w-full h-full flex flex-col items-center justify-center relative bg-[#fef2f2] dark:bg-[#2c1b1b]">
                                            <div v-if="getImageMaterials.length > 0" class="absolute top-0 left-0 w-full h-full z-[1]">
                                                <img :src="getImageMaterials[0][1]" :alt="getImageMaterials[0][0]" class="w-full h-full object-cover opacity-10 blur-sm">
                                            </div>
                                            <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2]">
                                                <div class="w-16 h-16 rounded-full bg-red-500/10 dark:bg-red-400/10 flex items-center justify-center mb-4">
                                                    <i class="fas fa-exclamation-triangle text-3xl text-red-500 dark:text-red-400"></i>
                                                </div>
                                                <p class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('videoGeneratingFailed') }}</p>
                                            </div>
                                        </div>
                                        <!-- 取消状态 -->
                                        <div v-else-if="modalTask?.status === 'CANCEL'" class="w-full h-full flex flex-col items-center justify-center relative bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                            <div v-if="getImageMaterials.length > 0" class="absolute top-0 left-0 w-full h-full z-[1]">
                                                <img :src="getImageMaterials[0][1]" :alt="getImageMaterials[0][0]" class="w-full h-full object-cover opacity-10 blur-sm">
                                            </div>
                                            <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2]">
                                                <div class="w-16 h-16 rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center mb-4">
                                                    <i class="fas fa-ban text-3xl text-[#86868b] dark:text-[#98989d]"></i>
                                                </div>
                                                <p class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('taskCancelled') }}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 - Apple 风格 -->
                                <div class="flex items-center justify-center">
                                    <div class="w-full max-w-[400px] aspect-[9/16] relative flex flex-col">
                                        <!-- 右上角删除按钮 - Apple 极简风格 -->
                                        <div class="absolute top-0 right-0 z-10">
                                            <button v-if="['FAILED', 'CANCEL'].includes(modalTask?.status)"
                                                    @click="deleteTask(modalTask.task_id, true)"
                                                    class="w-8 h-8 flex items-center justify-center bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] border border-black/10 dark:border-white/10 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-red-500 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100 transition-all duration-200"
                                                    :title="t('delete')">
                                                <i class="fas fa-trash text-xs"></i>
                                            </button>
                                        </div>

                                        <!-- 居中的内容区域 -->
                                        <div class="flex-1 flex items-center justify-center px-8 py-6">
                                            <div class="w-full">
                                                <!-- 标题和状态 - Apple 风格 -->
                                                <div class="text-center mb-6">
                                                    <h1 class="text-3xl sm:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">
                                                        <span v-if="modalTask?.status === 'SUCCEED'">{{ t('taskCompleted') }}</span>
                                                        <span v-else-if="modalTask?.status === 'FAILED'">{{ t('taskFailed') }}</span>
                                                        <span v-else-if="modalTask?.status === 'CANCEL'">{{ t('taskCancelled') }}</span>
                                                        <span v-else-if="modalTask?.status === 'RUNNING'">{{ t('taskRunning') }}</span>
                                                        <span v-else-if="modalTask?.status === 'PENDING'">{{ t('taskPending') }}</span>
                                                        <span v-else>{{ t('taskDetail') }}</span>
                                                    </h1>
                                                </div>

                                                <!-- 进度条 - Apple 风格 -->
                                                <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)" class="mb-6">
                                            <div v-for="(subtask, index) in (modalTask.subtasks || [])" :key="index">
                                                <!-- PENDING状态：Apple 风格排队显示 -->
                                                <div v-if="subtask.status === 'PENDING'" class="text-center">
                                                    <div v-if="subtask.estimated_pending_order !== null && subtask.estimated_pending_order !== undefined && subtask.estimated_pending_order >= 0" class="flex flex-col items-center gap-3">
                                                        <!-- 排队图标 -->
                                                        <div class="flex flex-wrap justify-center gap-1.5 mb-2">
                                                            <i v-for="n in Math.min(Math.max(subtask.estimated_pending_order, 0), 10)"
                                                               :key="n"
                                                               class="fas fa-circle text-[8px] text-[#86868b] dark:text-[#98989d] opacity-60"></i>
                                                            <span v-if="subtask.estimated_pending_order > 10" class="text-xs text-[#86868b] dark:text-[#98989d] font-medium ml-0.5">
                                                                +{{ subtask.estimated_pending_order - 10 }}
                                                            </span>
                                                        </div>
                                                        <!-- 排队文字 -->
                                                        <span class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight">
                                                            {{ t('queuePosition') }}: {{ subtask.estimated_pending_order }}
                                                        </span>
                                                    </div>
                                                </div>

                                                <!-- RUNNING状态：Apple 风格进度条 -->
                                                <div v-else-if="subtask.status === 'RUNNING'" class="w-full">
                                                    <!-- 进度条 -->
                                                    <div class="mb-4">
                                                        <div class="relative w-full h-1 bg-black/8 dark:bg-white/8 rounded-full overflow-hidden">
                                                            <div class="absolute top-0 left-0 h-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full transition-all duration-500 ease-out" :style="{ width: getSubtaskProgress(subtask) + '%' }"></div>
                                                        </div>
                                                    </div>
                                                    <!-- 百分比显示 -->
                                                    <div class="flex justify-center items-center">
                                                        <span class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                            {{ getSubtaskProgress(subtask) }}%
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                                <!-- 描述 - Apple 风格 -->
                                                <div class="text-sm sm:text-base text-[#86868b] dark:text-[#98989d] text-center mb-6 tracking-tight">
                                                    <p v-if="['RUNNING'].includes(modalTask?.status)" class="mb-0">
                                                        {{ isImageTask ? (t('aiIsGeneratingYourImage') || t('aiIsGeneratingYourVideo')) : t('aiIsGeneratingYourVideo') }}
                                                    </p>
                                                    <p v-else-if="['CREATED'].includes(modalTask?.status)" class="mb-0">
                                                        {{ t('taskSubmittedSuccessfully') }}
                                                    </p>
                                                    <p v-else-if="['PENDING'].includes(modalTask?.status)" class="mb-0">
                                                        {{ t('taskQueuePleaseWait') }}
                                                    </p>
                                                    <div v-else-if="modalTask?.status === 'FAILED'">
                                                        <p class="mb-4">{{ t('sorryYourVideoGenerationTaskFailed') }}</p>
                                                        <button v-if="modalTask?.fail_msg"
                                                                @click="showFailureDetails = !showFailureDetails"
                                                                class="text-sm text-[#86868b] dark:text-[#98989d] hover:text-red-500 dark:hover:text-red-400 transition-colors underline underline-offset-2">
                                                            {{ showFailureDetails ? t('hideDetails') : t('viewErrorDetails') }}
                                                        </button>
                                                        <div v-if="showFailureDetails && modalTask?.fail_msg" class="mt-4 p-4 bg-black/2 dark:bg-white/2 border border-black/6 dark:border-white/6 rounded-xl text-left">
                                                            <p class="text-xs text-[#86868b] dark:text-[#98989d] whitespace-pre-wrap leading-relaxed">{{ modalTask?.fail_msg }}</p>
                                                        </div>
                                                    </div>
                                                    <p v-else-if="modalTask?.status === 'CANCEL'" class="mb-0">
                                                        {{ t('thisTaskHasBeenCancelledYouCanRegenerateOrViewTheMaterialsYouUploadedBefore') }}
                                                    </p>
                                                </div>

                                                <!-- 特性列表 - Apple 风格 -->
                                                <div class="grid grid-cols-2 gap-2 mb-6">
                                                    <div class="flex flex-col items-center gap-1.5 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl">
                                                        <i class="fas fa-toolbox text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span class="text-[11px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight">{{ getTaskTypeName(modalTask) }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-1.5 p-3 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl">
                                                        <i class="fas fa-robot text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span class="text-[11px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium tracking-tight truncate max-w-full">{{ modalTask.model_cls }}</span>
                                                    </div>
                                                </div>

                                                <!-- 操作按钮 - Apple 风格 -->
                                                <div class="space-y-2.5">
                                                    <!-- 进行中状态：取消按钮 -->
                                                    <button v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                            @click="cancelTask(modalTask.task_id, true)"
                                                            class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-3 text-[15px] font-semibold text-red-500 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 hover:border-red-500/30 dark:hover:border-red-400/30 hover:shadow-[0_8px_24px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_8px_24px_rgba(248,113,113,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                        <i class="fas fa-times text-sm"></i>
                                                        <span>{{ t('cancelTask') }}</span>
                                                    </button>

                                                    <!-- 失败或取消状态：重试按钮 -->
                                                    <button v-if="modalTask?.status === 'FAILED' || modalTask?.status === 'CANCEL'"
                                                            @click="resumeTask(modalTask.task_id, true)"
                                                            class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-6 py-3 text-[15px] font-semibold text-white hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100 transition-all duration-200 ease-out tracking-tight flex items-center justify-center gap-2">
                                                        <i class="fas fa-redo text-sm"></i>
                                                        <span>{{ modalTask?.status === 'CANCEL' ? t('regenerateTask') : t('retryTask') }}</span>
                                                    </button>

                                                    <!-- 通用按钮 -->
                                                    <button v-if="['SUCCEED', 'FAILED', 'CANCEL','CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                            @click="handleReuseTask"
                                                            class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                        <i class="fas fa-copy text-sm"></i>
                                                        <span>{{ t('reuseTask') }}</span>
                                                    </button>

                                                    <button @click="showDetails = !showDetails"
                                                            class="w-full rounded-full bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 px-6 py-2.5 text-[15px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98] transition-all duration-200 tracking-tight flex items-center justify-center gap-2">
                                                        <i :class="showDetails ? 'fas fa-chevron-up' : 'fas fa-info-circle'" class="text-sm"></i>
                                                        <span>{{ showDetails ? t('hideDetails') : t('showDetails') }}</span>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 详细信息面板 - Apple 风格（其他状态）-->
                        <div v-if="showDetails && modalTask" class="bg-[#f5f5f7] dark:bg-[#1c1c1e] border-t border-black/8 dark:border-white/8 py-12">
                            <div class="max-w-6xl mx-auto px-8">
                                <!-- 输入素材标题 - Apple 风格 -->
                                <h2 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center justify-center gap-3 mb-8 tracking-tight">
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
                                            <button v-if="getImageMaterials.length > 0"
                                                    @click="handleDownloadFile(modalTask.task_id, 'input_image', modalTask.inputs.input_image)"
                                                    :disabled="downloadLoading"
                                                    class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                    :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                    :title="t('download')">
                                                <i class="fas fa-download text-xs"></i>
                                            </button>
                                        </div>
                                        <!-- 卡片内容 -->
                                        <div class="p-6 min-h-[200px]">
                                            <div v-if="getImageMaterials.length > 0">
                                                <div v-for="([inputName, url, index]) in getImageMaterials" :key="inputName || index" class="rounded-xl overflow-hidden border border-black/8 dark:border-white/8 mb-3 last:mb-0">
                                                    <div class="relative">
                                                        <img v-if="url" :src="url" :alt="inputName || `图片 ${index + 1}`" class="w-full h-auto object-contain" @error="handleImageError($event, modalTask.task_id, inputName)">
                                                        <!-- 加载中或错误状态 -->
                                                        <div v-else class="flex flex-col items-center justify-center h-[150px] bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                                            <i class="fas fa-spinner fa-spin text-2xl text-[#86868b]/50 dark:text-[#98989d]/50 mb-2"></i>
                                                            <p class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loading') || '加载中...' }}</p>
                                                        </div>
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
                                            <button v-if="getVideoMaterials().length > 0"
                                                    @click="handleDownloadFile(modalTask.task_id, 'input_video', modalTask.inputs.input_video)"
                                                    :disabled="downloadLoading"
                                                    class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                    :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                    :title="t('download')">
                                                <i class="fas fa-download text-xs"></i>
                                            </button>
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
                                            <button v-if="getAudioMaterials().length > 0"
                                                    @click="handleDownloadFile(modalTask.task_id, 'input_audio', modalTask.inputs.input_audio)"
                                                    :disabled="downloadLoading"
                                                    class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200"
                                                    :class="downloadLoading ? 'opacity-60 cursor-not-allowed' : 'hover:scale-110 active:scale-100'"
                                                    :title="t('download')">
                                                <i class="fas fa-download text-xs"></i>
                                            </button>
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
                                            <button v-if="modalTask?.params?.prompt"
                                                    @click="copyPrompt(modalTask?.params?.prompt)"
                                                    class="w-8 h-8 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-lg transition-all duration-200 hover:scale-110 active:scale-100"
                                                    :title="t('copy')">
                                                <i class="fas fa-copy text-xs"></i>
                                            </button>
                                        </div>
                                        <!-- 卡片内容 -->
                                        <div class="p-6 min-h-[200px]">
                                            <div v-if="modalTask?.params?.prompt" class="bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-xl p-4">
                                                <p class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight break-words">{{ modalTask.params.prompt }}</p>
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
/* Apple 风格极简样式 - 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
</style>
