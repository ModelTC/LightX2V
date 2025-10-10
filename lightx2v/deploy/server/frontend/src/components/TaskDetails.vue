<script setup>
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'
import { showTaskDetailModal,
        modalTask,
        closeTaskDetailModal,
        cancelTask,
        reuseTask,
        downloadFile,
        deleteTask,
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
        generateShareUrl,
        copyShareLink,
        shareToSocial,
        copyPrompt,
        getTaskFileUrlSync,
        getTaskFileFromCache,
        getTaskFileUrl,
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

// 获取图片素材
const getImageMaterials = () => {
    if (!modalTask.value?.inputs?.input_image) return []
    return [['input_image', getTaskFileUrlSync(modalTask.value.task_id, 'input_image')]]
}

// 获取音频素材
const getAudioMaterials = () => {
    if (!modalTask.value?.inputs?.input_audio) return []
    return [['input_audio', getTaskFileUrlSync(modalTask.value.task_id, 'input_audio')]]
}

// 路由关闭功能
const closeWithRoute = () => {
    closeTaskDetailModal()
    modalTask.value = null
    // 如果是从路由进入的，可以返回到上一页
    if (window.history.length > 1) {
        router.go(-1)
    } else {
        router.push('/')
    }
}

// 键盘事件处理
const handleKeydown = (event) => {
    if (event.key === 'Escape' && showTaskDetailModal.value) {
        closeWithRoute()
    }
}

// 处理文件下载
const handleDownloadFile = async (taskId, fileKey, fileName) => {
    try {
        console.log('开始下载文件:', { taskId, fileKey, fileName })
        
        // 处理文件名，确保有正确的后缀名
        let finalFileName = fileName
        if (fileName && typeof fileName === 'string') {
            // 检查是否已有后缀名
            const hasExtension = /\.[a-zA-Z0-9]+$/.test(fileName)
            if (!hasExtension) {
                // 没有后缀名，根据文件类型添加
                const extension = getFileExtension(fileKey)
                finalFileName = `${fileName}.${extension}`
                console.log('添加后缀名:', finalFileName)
            }
        } else {
            // 没有文件名，使用默认名称
            finalFileName = `${fileKey}.${getFileExtension(fileKey)}`
        }
        
        // 先尝试从缓存获取
        let fileData = getTaskFileFromCache(taskId, fileKey)
        console.log('缓存中的文件数据:', fileData)
        
        if (fileData && fileData.blob) {
            // 缓存中有blob数据，直接使用
            console.log('使用缓存中的文件数据')
            downloadFile({ ...fileData, name: finalFileName })
            return
        }
        
        if (fileData && fileData.url) {
            // 缓存中有URL，使用URL下载
            console.log('使用缓存中的URL下载:', fileData.url)
            try {
                const response = await fetch(fileData.url)
                console.log('文件响应状态:', response.status, response.ok)
                
                if (response.ok) {
                    const blob = await response.blob()
                    console.log('文件blob大小:', blob.size)
                    
                    const downloadData = {
                        blob: blob,
                        name: finalFileName
                    }
                    console.log('构造的文件数据:', downloadData)
                    downloadFile(downloadData)
                    return
                } else {
                    console.error('文件响应失败:', response.status, response.statusText)
                }
            } catch (error) {
                console.error('使用缓存URL下载失败:', error)
            }
        }
        
        if (!fileData) {
            console.log('缓存中没有文件，尝试异步获取...')
            // 缓存中没有，尝试异步获取
            const url = await getTaskFileUrl(taskId, fileKey)
            console.log('获取到的文件URL:', url)
            
            if (url) {
                const response = await fetch(url)
                console.log('文件响应状态:', response.status, response.ok)
                
                if (response.ok) {
                    const blob = await response.blob()
                    console.log('文件blob大小:', blob.size)
                    
                    fileData = {
                        blob: blob,
                        name: finalFileName
                    }
                    console.log('构造的文件数据:', fileData)
                } else {
                    console.error('文件响应失败:', response.status, response.statusText)
                }
            } else {
                console.error('无法获取文件URL')
            }
        }
        
        if (fileData && fileData.blob) {
            console.log('开始下载文件:', fileData.name)
            downloadFile(fileData)
        } else {
            console.error('文件数据无效:', fileData)
            showAlert(t('fileUnavailableAlert'), 'danger')
        }
    } catch (error) {
        console.error('下载失败:', error)
        showAlert(t('downloadFailedAlert'), 'danger')
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
watch(modalTask, (newTask) => {
    if (newTask && !hasLoadedTask.value) {
        console.log('modalTask第一次变化，加载任务详情:', newTask);
        viewTaskDetail(newTask);
        hasLoadedTask.value = true;
    }
}, { immediate: true });

// 生命周期钩子
onMounted(async () => {
    document.addEventListener('keydown', handleKeydown)
    console.log('TaskDetails组件已挂载，当前modalTask:', modalTask.value);
})

onUnmounted(() => {
    document.removeEventListener('keydown', handleKeydown)
})
</script>
<template>
            <!-- 任务详情弹窗 -->
            <div v-cloak>
                <div v-if="showTaskDetailModal"
                    class="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4"
                    @click="closeWithRoute">
                    <!-- 任务完成时的大弹窗 -->
                    <div v-if="modalTask?.status === 'SUCCEED'"
                        class="landing-page" @click.stop>
                        <!-- 弹窗头部 -->
                        <div class="modal-header">
                            <h3 class="modal-title">
                                <i class="fas fa-info-circle mr-2"></i>
                                {{ t('taskDetails') }}
                            </h3>
                            <div class="header-actions">
                                <button @click="closeWithRoute" class="action-button back-button" :title="t('back')">
                                    <i class="fas fa-arrow-left"></i>
                                </button>
                                <button @click="closeWithRoute" class="action-button close-button" :title="t('close')">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                        </div>

                        <!-- 主要内容区域 -->
                        <div class="main-content main-scrollbar overflow-y-auto">
                            <!-- 分享内容 -->
                            <div class="content-grid">
                                <!-- 左侧视频区域 -->
                                <div class="video-section">
                                    <div class="video-container">
                                        <!-- 视频播放器 -->
                                        <video
                                            v-if="modalTask?.outputs?.output_video"
                                            :src="getTaskFileUrlSync(modalTask.task_id, 'output_video')"
                                            :poster="getTaskFileUrlSync(modalTask.task_id, 'input_image')"
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
                                        <div v-else class="video-placeholder">
                                            <div class="loading-spinner">
                                                <i class="fas fa-video"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoNotAvailable') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 -->
                                <div class="info-section">
                                    <div class="info-content">
                                        <!-- 标题 -->
                                        <div class="title-container">
                                            <h1 class="main-title">
                                                <span v-if="modalTask?.status === 'SUCCEED'">{{ t('taskCompleted') }}</span>
                                                <span v-else-if="modalTask?.status === 'FAILED'">{{ t('taskFailed') }}</span>
                                                <span v-else-if="modalTask?.status === 'CANCEL'">{{ t('taskCancelled') }}</span>
                                                <span v-else-if="modalTask?.status === 'RUNNING'">{{ t('taskRunning') }}</span>
                                                <span v-else-if="modalTask?.status === 'PENDING'">{{ t('taskPending') }}</span>
                                                <span v-else>{{ t('taskDetails') }}</span>
                                            </h1>
                                        </div>
                                        
                                        <!-- 描述 -->
                                        <p class="main-description">
                                            {{ t('taskCompletedSuccessfully') }}
                                        </p>
                                        
                                        <div class="features-list justify-between">
                                            <div class="feature-item">
                                                <i class="fas fa-toolbox feature-icon"></i>
                                                <span>{{ getTaskTypeName(modalTask) }}</span>
                                            </div>
                                            <div class="feature-item cursor-pointer">
                                                <i class="fas fa-robot feature-icon"></i>
                                                <span>{{ modalTask.model_cls }}</span>
                                            </div>
                                            <div class="feature-item">
                                                <i class="fas fa-clock feature-icon"></i>
                                                <span>{{ t('timeCost')}}{{ Math.round(modalTask.extra_info?.active_elapse || 0) }}s</span>
                                            </div>
                                        </div>
                                        
                                        <!-- 操作按钮 -->
                                        <div class="action-buttons">
                                            <button v-if="modalTask?.outputs?.output_video"
                                                    @click="handleDownloadFile(modalTask.task_id, 'output_video', modalTask.outputs.output_video)" class="primary-button">
                                                <i class="fas fa-download mr-2"></i>
                                                {{ t('downloadVideo') }}
                                            </button>
                                            <button @click="reuseTask(modalTask); closeTaskDetailModal()" class="primary-button">
                                                <i class="fas fa-magic mr-2"></i>
                                                {{ t('reuseTask') }}
                                            </button>
                                            
                                            <button
                                                    @click="copyShareLink(modalTask.task_id, 'task')" class="secondary-button">
                                                <i class="fas fa-share-alt mr-2"></i>
                                                {{ t('copyShareLink') }}
                                        </button>

                                            <button
                                                    @click="deleteTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="secondary-button">
                                                <i class="fas fa-trash mr-2"></i>
                                                {{ t('deleteTask') }}
                                            </button>
                                            
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
                        <div v-if="showDetails && modalTask" class="details-panel">
                            <div class="details-content">
                                <!-- 输入素材标题 -->
                                    <h2 class="materials-title">
                                        <i class="fas fa-upload mr-2"></i>
                                        {{ t('inputMaterials') }}
                                    </h2>
                                
                                <!-- 三个并列的分块卡片 -->
                                <div class="materials-cards">
                                    <!-- 图片卡片 -->
                                    <div class="material-card">
                                        <div class="card-header">
                                            <i class="fas fa-image card-icon"></i>
                                            <h3 class="card-title">{{ t('image') }}</h3>
                                            <div class="card-actions">
                                                <button v-if="getImageMaterials().length > 0" 
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_image', modalTask.inputs.input_image)"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="getImageMaterials().length > 0" class="image-grid">
                                                <div v-for="[inputName, url] in getImageMaterials()" :key="inputName" class="image-item">
                                                    <div class="image-container">
                                                        <img :src="url" :alt="inputName" class="image-preview">
                                                    </div>
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
                                            <div class="card-actions">
                                                <button v-if="getAudioMaterials().length > 0" 
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_audio', modalTask.inputs.input_audio)"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
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
                                            <div class="card-actions">
                                                <button v-if="modalTask?.params?.prompt" 
                                                        @click="copyPrompt(modalTask?.params?.prompt)"
                                                        class="action-btn copy-btn" 
                                                        :title="t('copy')">
                                                    <i class="fas fa-copy"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="modalTask?.params?.prompt" class="prompt-content">
                                                <p class="prompt-text">{{ modalTask.params.prompt }}</p>
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

                    <!-- 其他状态的弹窗 -->
                    <div v-else class="landing-page" @click.stop>
                        <!-- 弹窗头部 -->
                        <div class="modal-header">
                            <h3 class="modal-title">
                                <i class="fas fa-info-circle mr-2"></i>
                                {{ t('taskDetails') }}
                            </h3>
                            <div class="header-actions">
                                <button @click="closeWithRoute" class="action-button back-button" :title="t('back')">
                                    <i class="fas fa-arrow-left"></i>
                                </button>
                                <button @click="closeTaskDetailModal" class="action-button close-button" :title="t('close')">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                        </div>

                        <!-- 主要内容区域 -->
                        <div class="main-content main-scrollbar overflow-y-auto">
                            <div class="content-grid">
                                <!-- 左侧占位图区域 -->
                                <div class="video-section">
                                    <div class="video-container">
                                        <!-- 根据状态显示不同的占位图 -->
                                        <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)" class="video-placeholder">
                                            <!-- 如果有图像输入，显示灰一点的图像作为背景 -->
                                            <div v-if="getImageMaterials().length > 0" class="background-image">
                                                <img :src="getImageMaterials()[0][1]" :alt="getImageMaterials()[0][0]" class="dimmed-image">
                                            </div>
                                            <div class="loading-spinner">
                                                <i class="fas fa-spinner fa-spin"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoGenerating') }}...</p>
                                        </div>
                                        <div v-else-if="modalTask?.status === 'FAILED'" class="video-placeholder error-placeholder">
                                            <div class="error-icon">
                                                <i class="fas fa-exclamation-triangle"></i>
                                            </div>
                                            <p class="loading-text">{{ t('videoGeneratingFailed') }}</p>
                                        </div>
                                        <div v-else-if="modalTask?.status === 'CANCEL'" class="video-placeholder cancel-placeholder">
                                            <div class="cancel-icon">
                                                <i class="fas fa-ban"></i>
                                            </div>
                                            <p class="loading-text">{{ t('taskCancelled') }}</p>
                                        </div>
                                    </div>
                                </div>

                                <!-- 右侧信息区域 -->
                                <div class="info-section">
                                    <div class="info-content">
                                        <!-- 标题和状态 -->
                                        <div class="title-container">
                                            <h1 class="main-title">
                                                <span v-if="modalTask?.status === 'SUCCEED'">{{ t('taskCompleted') }}</span>
                                                <span v-else-if="modalTask?.status === 'FAILED'">{{ t('taskFailed') }}</span>
                                                <span v-else-if="modalTask?.status === 'CANCEL'">{{ t('taskCancelled') }}</span>
                                                <span v-else-if="modalTask?.status === 'RUNNING'">{{ t('taskRunning') }}</span>
                                                <span v-else-if="modalTask?.status === 'PENDING'">{{ t('taskPending') }}</span>
                                                <span v-else>{{ t('taskDetails') }}</span>
                                            </h1>
                                        </div>
                                        <!-- 进度条 -->
                                        <div v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)">
                                            <div v-for="(subtask, index) in (modalTask.subtasks || [])" :key="index">
                                                
                                                <!-- PENDING状态：显示排队信息 -->
                                                <div v-if="subtask.status === 'PENDING'" class="queue-info">
                                                    <div v-if="subtask.estimated_pending_order !== null" class="queue-visualization">
                                                        <div class="queue-people">
                                                            <i v-for="n in Math.min(subtask.estimated_pending_order, 10)" 
                                                               :key="n" 
                                                               class="fas fa-user queue-person"></i>
                                                            <span v-if="subtask.estimated_pending_order > 10" class="queue-more">
                                                                +{{ subtask.estimated_pending_order - 10 }}
                                                            </span>
                                                        </div>
                                                        <span class="queue-text">
                                                            {{ t('queuePosition') }}: {{ subtask.estimated_pending_order }}
                                                        </span>
                                                    </div>
                                                </div>
                                                
                                                <!-- RUNNING状态：显示进度条 -->
                                                <div v-else-if="subtask.status === 'RUNNING'" class="progress-container">
                                                    <div class="minimal-progress-bar">
                                                        <div class="progress-line">
                                                            <div class="progress-fill" :style="{ width: getSubtaskProgress(subtask) + '%' }"></div>
                                                            <div class="moving-dot" :style="{ left: getSubtaskProgress(subtask) + '%' }"></div>
                                                        </div>
                                                    </div>
                                                    <div class="progress-info">
                                                        <span class="estimated-time">
                                                            {{ getSubtaskProgress(subtask) }}%
                                                        </span>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <!-- 描述 -->
                                        <p class="main-description mt-4">
                                            <span v-if="['RUNNING'].includes(modalTask?.status)">
                                                {{ t('aiIsGeneratingYourVideo') }}
                                            </span>
                                            <span v-else-if="['CREATED'].includes(modalTask?.status)">
                                                {{ t('taskSubmittedSuccessfully') }}
                                            </span>
                                            <span v-else-if="['PENDING'].includes(modalTask?.status)">
                                                {{ t('taskQueuePleaseWait') }}
                                            </span>
                                            <span v-else-if="modalTask?.status === 'FAILED'">
                                                {{ t('sorryYourVideoGenerationTaskFailed') }}
                                                <button v-if="modalTask?.fail_msg" @click="showFailureDetails = !showFailureDetails" class="text-red-400 hover:text-red-300 transition-colors">
                                                    <i class="fas fa-exclamation-triangle text-xs"></i>
                                                </button>
                                                <div v-if="showFailureDetails && modalTask?.fail_msg" class="mt-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
                                                    <div class="flex items-start gap-2">
                                                        <i class="fas fa-exclamation-triangle text-red-400 text-sm mt-0.5"></i>
                                                        <div class="flex-1">
                                                            <p class="text-xs text-red-300 font-medium mb-1">{{ t('failureReason') }}:</p>
                                                            <p class="text-xs text-red-200 whitespace-pre-wrap">{{ modalTask?.fail_msg }}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            </span>
                                            <span v-else-if="modalTask?.status === 'CANCEL'">
                                                {{ t('thisTaskHasBeenCancelledYouCanRegenerateOrViewTheMaterialsYouUploadedBefore') }}
                                            </span>
                                        </p>
                                        
                                        <div class="features-list justify-between">
                                            <div class="feature-item">
                                                <i class="fas fa-toolbox feature-icon"></i>
                                                <span>{{ getTaskTypeName(modalTask) }}</span>
                                            </div>
                                            <div class="feature-item cursor-pointer">
                                                <i class="fas fa-robot feature-icon"></i>
                                                <span>{{ modalTask.model_cls }}</span>
                                            </div>
                                        </div>

                                        <!-- 操作按钮 -->
                                        <div class="action-buttons">
                                            <!-- 进行中状态 -->
                                            <button v-if="['CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                    @click="cancelTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-times mr-2"></i>
                                                {{ t('cancelTask') }}
                                            </button>
                                            
                                            <!-- 失败状态 -->
                                            <button v-if="modalTask?.status === 'FAILED'"
                                                    @click="resumeTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-redo mr-2"></i>
                                                {{ t('retryTask') }}
                                            </button>
                                            
                                            <!-- 取消状态 -->
                                            <button v-if="modalTask?.status === 'CANCEL'"
                                                    @click="resumeTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="primary-button">
                                                <i class="fas fa-redo mr-2"></i>
                                                {{ t('regenerateTask') }}
                                            </button>
                                            
                                            <!-- 通用按钮 -->
                                            <button v-if="['SUCCEED', 'FAILED', 'CANCEL','CREATED', 'PENDING', 'RUNNING'].includes(modalTask?.status)"
                                                    @click="reuseTask(modalTask); closeTaskDetailModal()" 
                                                    class="secondary-button">
                                                <i class="fas fa-copy mr-2"></i>
                                                {{ t('reuseTask') }}
                                            </button>
                                            
                                            <button v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(modalTask?.status)"
                                                    @click="deleteTask(modalTask.task_id, true); closeTaskDetailModal()" 
                                                    class="secondary-button">
                                                <i class="fas fa-trash mr-2"></i>
                                                {{ t('deleteTask') }}
                                            </button>

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
                        <div v-if="showDetails && modalTask" class="details-panel">
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
                                            <div class="card-actions">
                                                <button v-if="getImageMaterials().length > 0" 
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_image', modalTask.inputs.input_image)"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="getImageMaterials().length > 0" class="image-grid">
                                                <div v-for="[inputName, url] in getImageMaterials()" :key="inputName" class="image-item">
                                                    <div class="image-container">
                                                        <img :src="url" :alt="inputName" class="image-preview">
                                                    </div>
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
                                            <div class="card-actions">
                                                <button v-if="getAudioMaterials().length > 0" 
                                                        @click="handleDownloadFile(modalTask.task_id, 'input_audio', modalTask.inputs.input_audio)"
                                                        class="action-btn download-btn" 
                                                        :title="t('download')">
                                                    <i class="fas fa-download"></i>
                                                </button>
                                            </div>
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
                                            <div class="card-actions">
                                                <button v-if="modalTask?.params?.prompt" 
                                                        @click="copyPrompt(modalTask?.params?.prompt)"
                                                        class="action-btn copy-btn" 
                                                        :title="t('copy')">
                                                    <i class="fas fa-copy"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-content">
                                            <div v-if="modalTask?.params?.prompt" class="prompt-content">
                                                <p class="prompt-text">{{ modalTask.params.prompt }}</p>
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
    </div>
</div>
</template>

<style scoped>
/* Landing Page 样式 */
.landing-page {
    min-height: calc(100vh - 60px);
    width: 100%;
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    color: white;
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 60;
}

.modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: white;
    margin: 0;
    display: flex;
    align-items: center;
}

.header-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

.action-button {
    background: none;
    border: none;
    color: #9ca3af;
    font-size: 1.25rem;
    cursor: pointer;
    transition: all 0.2s;
    padding: 0.5rem;
    border-radius: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.action-button:hover {
    color: white;
    background: rgba(255, 255, 255, 0.1);
}

.back-button:hover {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
}

.close-button:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.main-content {
    width: 100%;
    padding: 2rem 0;
    min-height: calc(100vh - 80px);
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
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
    position: relative;
}

/* 占位图背景图像 */
.background-image {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    border-radius: 1rem;
}

.dimmed-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: brightness(0.5);
    opacity: 0.5;
}

.error-placeholder {
    background: #1f2937;
}

.error-icon {
    font-size: 2rem;
    color: #ef4444;
    margin-bottom: 1rem;
}

.cancel-placeholder {
    background: #1f2937;
}

.cancel-icon {
    font-size: 2rem;
    color: #f59e0b;
    margin-bottom: 1rem;
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

/* 标题容器 */
.title-container {
    text-align: center;
    margin-bottom: 2rem;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.progress-status {
    font-size: 0.75rem;
    color: #8b5cf6;
    background: rgba(139, 92, 246, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
}

.main-description {
    font-size: 1.25rem;
    color: #d1d5db;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* 进度条样式 */
.progress-section {
    margin-bottom: 2rem;
}

.subtask-progress {
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.75rem;
}

.progress-header {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 0.75rem;
}

.progress-status {
    font-size: 0.75rem;
    color: #8b5cf6;
    background: rgba(139, 92, 246, 0.1);
    padding: 0.25rem 0.5rem;
    border-radius: 0.375rem;
}

.progress-container {
    margin-top: 0.75rem;
}

.minimal-progress-bar {
    margin-bottom: 0.75rem;
}

.progress-line {
    position: relative;
    width: 100%;
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 1px;
}

.progress-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #8b5cf6, #a855f7);
    border-radius: 1px;
    transition: width 0.5s ease;
}

.moving-dot {
    position: absolute;
    top: -4px;
    width: 10px;
    height: 10px;
    background: linear-gradient(45deg, #8b5cf6, #a855f7);
    border-radius: 50%;
    box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    transition: left 0.5s ease;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.6);
    }
    50% {
        transform: scale(1.2);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.8);
    }
}

.progress-info {
    display: flex;
    justify-content: center;
    font-size: 0.75rem;
    color: #9ca3af;
}

.queue-info {
    margin-top: 0.75rem;
    text-align: center;
}

.queue-visualization {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
}

.queue-people {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.25rem;
    margin-bottom: 0.5rem;
}

.queue-person {
    font-size: 0.875rem;
    color: #f59e0b;
    opacity: 0.8;
}

.queue-more {
    font-size: 0.75rem;
    color: #f59e0b;
    font-weight: 600;
    margin-left: 0.25rem;
}

.queue-text {
    font-size: 0.75rem;
    color: #f59e0b;
    font-weight: 500;
}

.estimated-wait-time {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    color: #22c55e;
    font-weight: 600;
    animation: countdown 1s ease-in-out infinite;
}

.estimated-time {
    display: flex;
    align-items: center;
    font-size: 0.875rem;
    color: #22c55e;
    font-weight: 600;
    animation: countdown 1s ease-in-out infinite;
}

@keyframes countdown {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.8;
        transform: scale(1.05);
    }
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
    margin-bottom: 0.5rem;
}

.secondary-button:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: rgba(255, 255, 255, 0.3);
}

.delete-button {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #ef4444;
}

.delete-button:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.5);
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
    position: relative;
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
    flex: 1;
}

.card-actions {
    display: flex;
    gap: 0.5rem;
    margin-left: auto;
}

.action-btn {
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 0.375rem;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.875rem;
}

.download-btn {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.download-btn:hover {
    background: rgba(34, 197, 94, 0.3);
    border-color: rgba(34, 197, 94, 0.5);
    transform: scale(1.05);
}

.copy-btn {
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
    border: 1px solid rgba(107, 114, 128, 0.3);
}

.copy-btn:hover {
    background: rgba(107, 114, 128, 0.3);
    border-color: rgba(107, 114, 128, 0.5);
    transform: scale(1.05);
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
    
    /* 移动端进度条调整 */
    .subtask-progress {
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .progress-info {
        flex-direction: column;
        gap: 0.5rem;
    }
}
</style>