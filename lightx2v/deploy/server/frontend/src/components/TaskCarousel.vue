<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import {
    getTaskFileUrlSync,
    getTaskFileUrl,
    getTaskTypeName,
    formatTime,
    getTaskStatusDisplay,
    getStatusTextClass,
    getProgressTitle,
    getProgressInfo,
    getOverallProgress,
    getSubtaskStatusText,
    getSubtaskProgress,
    formatEstimatedTime,
    showAlert,
    cancelTask,
    resumeTask,
    downloadFile,
    handleDownloadFile,
    getTaskFileFromCache,
    apiRequest,
    copyShareLink,
    currentTask,
    startPollingTask,
    openTaskDetailModal,
} from '../utils/other'

const { t } = useI18n()

// Props
const props = defineProps({
    tasks: {
        type: Array,
        required: true,
        default: () => []
    }
})

// 响应式数据å
const isVideoLoaded = ref(false)
const isVideoError = ref(false)
const videoElement = ref(null)

// 计算属性
const sortedTasks = computed(() => {
    // 按创建时间排序，最新的在前
    return [...props.tasks].sort((a, b) => {
        const timeA = new Date(a.created_at || a.task_id).getTime()
        const timeB = new Date(b.created_at || b.task_id).getTime()
        return timeB - timeA
    })
})

const taskStatus = computed(() => currentTask.value?.status || 'CREATED')
const isCompleted = computed(() => taskStatus.value === 'SUCCEED')
const isRunning = computed(() => ['CREATED', 'PENDING', 'RUNNING'].includes(taskStatus.value))
const isFailed = computed(() => taskStatus.value === 'FAILED')
const isCancelled = computed(() => taskStatus.value === 'CANCEL')

// 当前任务索引（用于显示）
const currentTaskIndex = computed(() => {
    return sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
})

// 获取视频URL
const videoUrl = computed(() => {
    if (!isCompleted.value || !currentTask.value) return null
    return getTaskFileUrlSync(currentTask.value.task_id, 'output_video')
})

// 获取图片URL（用于缩略图）
const imageUrl = computed(() => {
    if (!currentTask.value) return null
    return getTaskFileUrlSync(currentTask.value.task_id, 'input_image')
})

// 更新当前任务数据并启动轮询
const updateCurrentTaskData = async (task) => {
    if (!task?.task_id) return

    try {
        const response = await apiRequest(`/api/v1/task/query?task_id=${task.task_id}`)
        if (response && response.ok) {
            const updatedTask = await response.json()
            // 更新全局currentTask
            currentTask.value = updatedTask
            console.log('TaskCarousel: 更新任务数据', updatedTask)

            // 如果任务还在进行中，开始轮询状态
            if (['CREATED', 'PENDING', 'RUNNING'].includes(updatedTask.status)) {
                startPollingTask(updatedTask.task_id)
            }
        }
    } catch (error) {
        console.warn(`TaskCarousel: 获取任务数据失败 task_id=${task.task_id}`, error.message)
    }
}



// 任务切换方法
const goToPreviousTask = () => {
    if (sortedTasks.value.length <= 1) return

    const currentIndex = sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
    if (currentIndex === -1) return

    const newIndex = currentIndex > 0 ? currentIndex - 1 : sortedTasks.value.length - 1
    const newTask = sortedTasks.value[newIndex]
    currentTask.value = newTask
    resetVideoState()
    // 更新新任务的数据并启动轮询
    updateCurrentTaskData(newTask)
}

const goToNextTask = () => {
    if (sortedTasks.value.length <= 1) return

    const currentIndex = sortedTasks.value.findIndex(task => task.task_id === currentTask.value?.task_id)
    if (currentIndex === -1) return

    const newIndex = currentIndex < sortedTasks.value.length - 1 ? currentIndex + 1 : 0
    const newTask = sortedTasks.value[newIndex]
    currentTask.value = newTask
    resetVideoState()
    // 更新新任务的数据并启动轮询
    updateCurrentTaskData(newTask)
}

// 处理任务指示器点击
const handleTaskIndicatorClick = (task) => {
    currentTask.value = task
    resetVideoState()
    // 更新任务数据并启动轮询
    updateCurrentTaskData(task)
}

// 重置视频状态
const resetVideoState = () => {
    isVideoLoaded.value = false
    isVideoError.value = false
}

// 视频加载事件
const onVideoLoaded = () => {
    isVideoLoaded.value = true
    isVideoError.value = false
}

const onVideoError = () => {
    isVideoError.value = true
    isVideoLoaded.value = false
}


// 处理取消任务
const handleCancel = async () => {
    if (!currentTask.value?.task_id) return

    try {
        await cancelTask(currentTask.value.task_id)
    } catch (error) {
        console.error('取消任务失败:', error)
        showAlert('取消任务失败，请重试', 'danger')
    }
}

// 处理分享任务
const handleShareTask = async () => {
    if (!currentTask.value?.task_id) return

    try {
        await copyShareLink(currentTask.value.task_id, 'task')
        // copyShareLink 函数内部已经显示了带"查看"按钮的 alert，不需要再次调用
    } catch (error) {
        console.error('分享失败:', error)
        showAlert('分享失败，请重试', 'danger')
    }
}

// 处理重试任务
const handleRetry = async () => {
    if (!currentTask.value?.task_id) return

    try {
        await resumeTask(currentTask.value.task_id)
    } catch (error) {
        console.error('重试任务失败:', error)
        showAlert('重试任务失败，请重试', 'danger')
    }
}


// 获取文件扩展名
const getFileExtension = (fileKey) => {
    if (fileKey.includes('video')) return 'mp4'
    if (fileKey.includes('image')) return 'jpg'
    if (fileKey.includes('audio')) return 'mp3'
    return 'file'
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'ArrowLeft') {
        goToPreviousTask()
    } else if (event.key === 'ArrowRight') {
        goToNextTask()
    }
}

// 生命周期
onMounted(() => {
    document.addEventListener('keydown', handleKeydown)
    // 初始化时设置第一个任务为当前任务
    if (sortedTasks.value.length > 0 && !currentTask.value) {
        const firstTask = sortedTasks.value[0]
        currentTask.value = firstTask
        // 更新任务数据并启动轮询
        updateCurrentTaskData(firstTask)
    }
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
    <!-- Apple 风格任务轮播 -->
    <div class="w-full max-w-[500px] mx-auto">
        <!-- 任务计数器 - Apple 风格 -->
        <div class="flex justify-center items-center text-sm font-medium text-[#86868b] dark:text-[#98989d] mb-4 tracking-tight">
                {{ currentTaskIndex + 1 }} / {{ sortedTasks.length }}
            </div>

        <!-- 视频区域 -->
        <div class="flex flex-col items-center gap-6 relative">
            <!-- 左侧导航箭头 - Apple 极简风格 -->
            <button
                v-if="sortedTasks.length > 1"
                @click="goToPreviousTask"
                class="absolute top-1/2 -translate-y-1/2 left-[-10px] sm:left-[-20px] md:left-[-40px] lg:left-[-60px] w-[44px] h-[44px] rounded-full border-0 cursor-pointer flex items-center justify-center text-base transition-all duration-200 ease-out z-10 bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 active:scale-95 disabled:opacity-20 disabled:cursor-not-allowed"
                :disabled="sortedTasks.length <= 1">
                <i class="fas fa-chevron-left"></i>
            </button>

            <!-- 右侧导航箭头 - Apple 极简风格 -->
            <button
                v-if="sortedTasks.length > 1"
                @click="goToNextTask"
                class="absolute top-1/2 -translate-y-1/2 right-[-10px] sm:right-[-20px] md:right-[-40px] lg:right-[-60px] w-[44px] h-[44px] rounded-full border-0 cursor-pointer flex items-center justify-center text-base transition-all duration-200 ease-out z-10 bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 active:scale-95 disabled:opacity-20 disabled:cursor-not-allowed"
                :disabled="sortedTasks.length <= 1">
                <i class="fas fa-chevron-right"></i>
            </button>

            <!-- 视频容器 - Apple 圆角和阴影 -->
            <div class="w-full max-w-[280px] sm:max-w-[300px] md:max-w-[400px] lg:max-w-[400px] aspect-[9/16] bg-black dark:bg-[#000000] rounded-[16px] overflow-hidden shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.5)] relative cursor-pointer transition-all duration-200 hover:shadow-[0_12px_32px_rgba(0,0,0,0.2)] dark:hover:shadow-[0_12px_32px_rgba(0,0,0,0.6)]"
                @click="openTaskDetailModal(currentTask)"
                :title="t('viewTaskDetails')">
                <!-- 已完成：显示视频播放器 -->
                <video
                    v-if="isCompleted && videoUrl"
                    :src="videoUrl"
                    :poster="imageUrl"
                    class="w-full h-full object-contain"
                    controls
                    preload="metadata"
                    @loadeddata="onVideoLoaded"
                    @error="onVideoError"
                    ref="videoElement">
                    {{ t('browserNotSupported') }}
                </video>

                <!-- 进行中：Apple 风格加载状态 -->
                <div v-else-if="isRunning" class="w-full h-full flex flex-col items-center justify-center relative bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="absolute top-0 left-0 w-full h-full z-[1]">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="w-full h-full object-cover opacity-20 blur-sm">
                    </div>

                    <!-- 进度内容覆盖层 -->
                    <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2] p-8 md:p-6 sm:p-4">
                        <div class="w-full max-w-[280px] text-center">

                            <!-- 进度条 -->
                            <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(taskStatus)">
                                <div v-for="(subtask, index) in (currentTask?.subtasks || [])" :key="index">
                                    <!-- PENDING状态：Apple 风格排队显示 -->
                                    <div v-if="subtask.status === 'PENDING'" class="mt-4 text-center">
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
                                    <div v-else-if="subtask.status === 'RUNNING'" class="w-full text-center">
                                        <!-- 进度条 -->
                                        <div class="mb-4">
                                            <div class="relative w-full h-1 bg-black/8 dark:bg-white/8 rounded-full overflow-hidden">
                                                <div class="absolute top-0 left-0 h-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full transition-all duration-500 ease-out" :style="{ width: getSubtaskProgress(subtask) + '%' }"></div>
                                            </div>
                                        </div>
                                        <!-- 百分比显示 -->
                                        <div class="flex justify-center items-center">
                                            <span class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight animate-progress">
                                                {{ getSubtaskProgress(subtask) }}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 失败：Apple 风格错误状态 -->
                <div v-else-if="isFailed" class="w-full h-full flex flex-col items-center justify-center relative bg-[#fef2f2] dark:bg-[#2c1b1b]">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="absolute top-0 left-0 w-full h-full z-[1]">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="w-full h-full object-cover opacity-10 blur-sm">
                    </div>

                    <!-- 错误信息 -->
                    <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2] p-8 md:p-6 sm:p-4">
                        <div class="w-12 h-12 rounded-full bg-red-500/10 dark:bg-red-400/10 flex items-center justify-center mb-4">
                            <i class="fas fa-exclamation-triangle text-2xl text-red-500 dark:text-red-400"></i>
                        </div>
                        <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-sm text-center font-medium tracking-tight">{{ t('videoGeneratingFailed') }}</p>
                    </div>
                </div>

                <!-- 已取消：Apple 风格取消状态 -->
                <div v-else-if="isCancelled" class="w-full h-full flex flex-col items-center justify-center relative bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                    <!-- 背景图片 -->
                    <div v-if="imageUrl" class="absolute top-0 left-0 w-full h-full z-[1]">
                        <img :src="imageUrl" :alt="getTaskTypeName(currentTask?.task_type)" class="w-full h-full object-cover opacity-10 blur-sm">
                    </div>

                    <!-- 取消信息 -->
                    <div class="absolute top-0 left-0 w-full h-full flex flex-col justify-center items-center z-[2] p-8 md:p-6 sm:p-4">
                        <div class="w-12 h-12 rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center mb-4">
                            <i class="fas fa-ban text-2xl text-[#86868b] dark:text-[#98989d]"></i>
                        </div>
                        <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-sm text-center font-medium tracking-tight">{{ t('taskCancelled') }}</p>
                    </div>
                </div>

                <!-- 默认状态：Apple 风格 -->
                <div v-else class="w-full h-full flex flex-col items-center justify-center relative bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                    <div class="w-16 h-16 rounded-full bg-black/5 dark:bg-white/5 flex items-center justify-center mb-4 z-[2]">
                        <i class="fas fa-video text-3xl text-[#86868b] dark:text-[#98989d]"></i>
                    </div>
                    <p class="text-[#86868b] dark:text-[#98989d] text-sm z-[2] tracking-tight">{{ t('videoNotAvailable') }}</p>
                </div>
            </div>

            <!-- Apple 风格操作按钮 -->
            <div class="flex justify-center gap-3">
                <!-- 已完成：下载按钮 -->
                <button
                    v-if="isCompleted && currentTask?.outputs?.output_video"
                    @click="handleDownloadFile(currentTask.task_id, 'output_video', currentTask.outputs.output_video)"
                    class="w-[40px] h-[40px] sm:w-[44px] sm:h-[44px] rounded-full flex items-center justify-center text-base transition-all duration-200 ease-out border-0 cursor-pointer bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 active:scale-95"
                    :title="t('download')">
                    <i class="fas fa-download"></i>
                </button>

                <!-- 已完成：分享按钮 -->
                <button
                    v-if="isCompleted && currentTask?.outputs?.output_video"
                    @click="handleShareTask"
                    class="w-[40px] h-[40px] sm:w-[44px] sm:h-[44px] rounded-full flex items-center justify-center text-base transition-all duration-200 ease-out border-0 cursor-pointer bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:scale-105 active:scale-95"
                    :title="t('share')">
                    <i class="fas fa-share-alt"></i>
                </button>

                <!-- 进行中：取消按钮 -->
                <button
                    v-if="isRunning"
                    @click="handleCancel"
                    class="w-[40px] h-[40px] sm:w-[44px] sm:h-[44px] rounded-full flex items-center justify-center text-base transition-all duration-200 ease-out border-0 cursor-pointer bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-red-500 dark:text-red-400 hover:scale-105 active:scale-95"
                    :title="t('cancel')">
                    <i class="fas fa-times"></i>
                </button>

                <!-- 失败或取消：重试按钮 -->
                <button
                    v-if="isFailed || isCancelled"
                    @click="handleRetry"
                    class="w-[40px] h-[40px] sm:w-[44px] sm:h-[44px] rounded-full flex items-center justify-center text-base transition-all duration-200 ease-out border-0 cursor-pointer bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:scale-105 active:scale-95"
                    :title="t('retry')">
                    <i class="fas fa-redo"></i>
                </button>
            </div>
        </div>

        <!-- Apple 风格任务指示器 -->
        <div v-if="sortedTasks.length > 1" class="flex justify-center gap-2 mt-5">
            <div
                v-for="(task, index) in sortedTasks"
                :key="task.task_id"
                @click="handleTaskIndicatorClick(task)"
                class="w-2 h-2 rounded-full cursor-pointer transition-all duration-200 ease-out"
                :class="index === currentTaskIndex
                    ? 'bg-[#1d1d1f] dark:bg-[#f5f5f7] scale-110'
                    : 'bg-[#86868b]/30 dark:bg-[#98989d]/30 hover:bg-[#86868b]/50 dark:hover:bg-[#98989d]/50 hover:scale-105'">
            </div>
        </div>
    </div>
</template>

<style scoped>
/* Apple 风格动画 */
@keyframes progress {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.85;
    }
}

.animate-progress {
    animation: progress 1.5s ease-in-out infinite;
    }

/* 所有其他样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
</style>
