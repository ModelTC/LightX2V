<script setup>
import {
            submitting,
            templateLoading,
            // 任务类型下拉菜单
            showTaskTypeMenu,
            showModelMenu,

            isLoggedIn,
            loading,
            loginLoading,
            initLoading,
            downloadLoading,

            // 录音相关
            isRecording,
            recordingDuration,
            startRecording,
            stopRecording,
            formatRecordingDuration,
            taskSearchQuery,
            currentUser,
            models,
            tasks,
            alert,
            showErrorDetails,
            showFailureDetails,
            confirmDialog,
            showConfirmDialog,
            showTaskDetailModal,
            modalTask,
            currentTask,
            t2vForm,
            i2vForm,
            s2vForm,
            getCurrentForm,
            i2vImagePreview,
            s2vImagePreview,
            s2vAudioPreview,
            getCurrentImagePreview,
            getCurrentAudioPreview,
            setCurrentImagePreview,
            setCurrentAudioPreview,
            updateUploadedContentStatus,
            availableTaskTypes,
            availableModelClasses,
            currentTaskHints,
            currentHintIndex,
            startHintRotation,
            stopHintRotation,
            filteredTasks,
            selectedTaskId,
            selectedTask,
            selectedModel,
            loadingTaskFiles,
            statusFilter,
            pagination,
            paginationInfo,
            currentTaskPage,
            taskPageSize,
            taskPageInput,
            paginationKey,
            taskMenuVisible,
            toggleTaskMenu,
            closeAllTaskMenus,
            handleClickOutside,
            showAlert,
            setLoading,
            apiCall,
            logout,
            loadModels,
            sidebarCollapsed,
            sidebarWidth,
            showExpandHint,
            showGlow,
            isDefaultStateHidden,
            hideDefaultState,
            showDefaultState,
            isCreationAreaExpanded,
            hasUploadedContent,
            isContracting,
            expandCreationArea,
            contractCreationArea,
            taskFileCache,
            taskFileCacheLoaded,
            templateFileCache,
            templateFileCacheLoaded,
            loadTaskFiles,
            downloadFile,
            viewFile,
            handleImageUpload,
            selectTask,
            selectModel,
            resetForm,
            triggerImageUpload,
            triggerAudioUpload,
            removeImage,
            removeAudio,
            handleAudioUpload,
            loadImageAudioTemplates,
            selectImageTemplate,
            selectAudioTemplate,
            previewAudioTemplate,
            getTemplateFile,
            imageTemplates,
            audioTemplates,
            showImageTemplates,
            showAudioTemplates,
            mediaModalTab,
            templatePagination,
            templatePaginationInfo,
            templateCurrentPage,
            templatePageSize,
            templatePageInput,
            templatePaginationKey,
            imageHistory,
            audioHistory,
            showTemplates,
            showHistory,
            showPromptModal,
            promptModalTab,
            submitTask,
            fileToBase64,
            formatTime,
            refreshTasks,
            goToPage,
            jumpToPage,
            getVisiblePages,
            goToTemplatePage,
            jumpToTemplatePage,
            getVisibleTemplatePages,
            goToInspirationPage,
            jumpToInspirationPage,
            getVisibleInspirationPages,
            preloadTaskFilesUrl,
            preloadTemplateFilesUrl,
            loadTaskFilesFromCache,
            saveTaskFilesToCache,
            getTaskFileFromCache,
            setTaskFileToCache,
            getTaskFileUrlFromApi,
            getTaskFileUrl,
            getTaskFileUrlSync,
            getTemplateFileUrlFromApi,
            getTemplateFileUrl,
            getTemplateFileUrlAsync,
            loadTemplateFilesFromCache,
            saveTemplateFilesToCache,
            loadFromCache,
            saveToCache,
            clearAllCache,
            getStatusBadgeClass,
            viewSingleResult,
            cancelTask,
            resumeTask,
            deleteTask,
            startPollingTask,
            stopPollingTask,
            reuseTask,
            showTaskCreator,
            toggleSidebar,
            clearPrompt,
            getTaskItemClass,
            getStatusIndicatorClass,
            getTaskTypeBtnClass,
            getModelBtnClass,
            getTaskTypeIcon,
            getTaskTypeName,
            getPromptPlaceholder,
            getStatusTextClass,
            getImagePreview,
            getTaskInputUrl,
            getTaskInputImage,
            getTaskInputAudio,
            getHistoryImageUrl,
            getUserAvatarUrl,
            getCurrentImagePreviewUrl,
            getCurrentAudioPreviewUrl,
            handleThumbnailError,
            handleImageError,
            handleImageLoad,
            handleAudioError,
            handleAudioLoad,
            getTaskStatusDisplay,
            getTaskStatusColor,
            getTaskStatusIcon,
            getTaskDuration,
            getRelativeTime,
            getTaskHistory,
            getActiveTasks,
            getOverallProgress,
            getProgressTitle,
            getProgressInfo,
            getSubtaskProgress,
            getSubtaskStatusText,
            formatEstimatedTime,
            formatDuration,
            searchTasks,
            filterTasksByStatus,
            filterTasksByType,
            getAlertClass,
            getAlertBorderClass,
            getAlertTextClass,
            getAlertIcon,
            getAlertIconBgClass,
            getPromptTemplates,
            selectPromptTemplate,
            promptHistory,
            getPromptHistory,
            addTaskToHistory,
            getLocalTaskHistory,
            selectPromptHistory,
            clearPromptHistory,
            getImageHistory,
            getAudioHistory,
            selectImageHistory,
            selectAudioHistory,
            previewAudioHistory,
            clearImageHistory,
            clearAudioHistory,
            getAudioMimeType,
            getAuthHeaders,
            startResize,
            sidebar,
            switchToCreateView,
            switchToProjectsView,
            switchToInspirationView,
            switchToLoginView,
            openTaskDetailModal,
            closeTaskDetailModal,
            // 灵感广场相关
            inspirationSearchQuery,
            selectedInspirationCategory,
            inspirationItems,
            InspirationCategories,
            loadInspirationData,
            selectInspirationCategory,
            handleInspirationSearch,
            loadMoreInspiration,
            inspirationPagination,
            inspirationPaginationInfo,
            inspirationCurrentPage,
            inspirationPageSize,
            inspirationPageInput,
            inspirationPaginationKey,
            // 精选模版相关
            featuredTemplates,
            featuredTemplatesLoading,
            loadFeaturedTemplates,
            getRandomFeaturedTemplates,
            // 工具函数
            formatDate,
            // 模板详情弹窗相关
            showTemplateDetailModal,
            selectedTemplate,
            previewTemplateDetail,
            closeTemplateDetailModal,
            useTemplate,
            // 图片放大弹窗相关
            showImageZoomModal,
            zoomedImageUrl,
            showImageZoom,
            closeImageZoomModal,
            // 模板素材应用相关
            applyTemplateImage,
            applyTemplateAudio,
            applyTemplatePrompt,
            copyPrompt,
            // 视频播放控制
            playVideo,
            pauseVideo,
            toggleVideoPlay,
            pauseAllVideos,
            updateVideoIcon,
            onVideoLoaded,
            onVideoError,
            onVideoEnded,
            generateShareUrl,
            copyShareLink,
            shareToSocial,
            openTaskFromRoute
        } from '../utils/other'

import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { watch, onMounted, computed, ref, nextTick, onUnmounted } from 'vue'
import ModelDropdown from './ModelDropdown.vue'
import MediaTemplate from './MediaTemplate.vue'
import Voice_tts from './Voice_tts.vue'
import TaskCarousel from './TaskCarousel.vue'

// Props
const props = defineProps({
  query: {
    type: Object,
    default: () => ({})
  }
})

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()

// 当前显示的精选模版
const currentFeaturedTemplates = ref([])

// 屏幕尺寸响应式状态
const screenSize = ref('large') // 'small' 或 'large'

// 拖拽状态
const isDragOver = ref(false)

// 语音合成模态框状态
const showVoiceTTSModal = ref(false)

// 处理提交任务并滚动到任务区域
const handleSubmitTask = async () => {
    try {
        // 提交任务，获取新任务ID
        const newTaskId = await submitTask()

        if (!newTaskId) {
            console.error('任务提交失败，未获取到任务ID')
            return
        }

        // 刷新任务列表
        await refreshTasks(true)

        // 等待 Vue 更新 DOM
        await nextTick()

        // 查找新创建的任务并设置为当前任务
        const newTask = tasks.value.find(task => task.task_id === newTaskId)
        if (newTask) {
            currentTask.value = newTask
            console.log('已将新任务设置为当前任务:', newTaskId)
        }

        // 滚动到任务区域
        scrollToTaskArea()
    } catch (error) {
        console.error('提交任务失败:', error)
    }
}

// 滚动到任务区域
const scrollToTaskArea = () => {
    const taskArea = document.querySelector('.task-carousel')
    if (taskArea) {
        taskArea.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        })
    }
}

// 滚动到生成区域
const scrollToCreationArea = () => {
    const creationArea = document.querySelector('#task-creator')
    if (creationArea) {
        creationArea.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        })
    }
}

// 包装 useTemplate 函数，在应用模板后滚动到生成区域
const handleUseTemplate = async (item) => {
    await useTemplate(item)
    // 等待 DOM 更新后再滚动
    await nextTick()
    // 延迟一下确保展开动画完成
    setTimeout(() => {
        scrollToCreationArea()
    }, 100)
}

// 处理语音合成完成后的回调
const handleTTSComplete = (audioBlob) => {
    // 创建File对象
    const audioFile = new File([audioBlob], 'tts_audio.mp3', { type: 'audio/mpeg' })

    // 模拟文件上传事件
    const dataTransfer = new DataTransfer()
    dataTransfer.items.add(audioFile)
    const fileList = dataTransfer.files

    const event = {
        target: {
            files: fileList
        }
    }

    // 处理音频上传
    handleAudioUpload(event)

    // 关闭模态框
    showVoiceTTSModal.value = false

    // 显示成功提示
    showAlert('语音合成完成，已自动添加到音频素材', 'success')
}

// 跳转到项目页面
const goToProjects = () => {
    // 构建查询参数，包含当前的表单数据
    const query = {}

    // 添加任务类型
    if (selectedTaskId.value) {
        query.taskType = selectedTaskId.value
    }

    // 添加模型
    if (selectedModel.value) {
        query.model = selectedModel.value
    }

    // 添加提示词
    if (getCurrentForm().prompt) {
        query.prompt = getCurrentForm().prompt
    }

    // 添加图片（如果有）
    if (getCurrentImagePreview()) {
        query.hasImage = 'true'
    }

    // 添加音频（如果有）
    if (getCurrentAudioPreview()) {
        query.hasAudio = 'true'
    }

    // 跳转到项目页面
    router.push({
        path: '/projects',
        query: query
    })
}

// 获取随机精选模版
const refreshRandomTemplates = async () => {
    const randomTemplates = await getRandomFeaturedTemplates(10) // 获取10个模版
    currentFeaturedTemplates.value = randomTemplates
}

// 随机列布局相关函数
const generateRandomColumnLayout = (templates) => {
    if (!templates || templates.length === 0) return { columns: [], templates: [] }

    // 响应式列数控制
    const getColumnCount = () => {
        if (screenSize.value === 'large') {
            // 大屏幕：4-6列
            return Math.floor(Math.random() * 2) + 4 // 4, 5, 6列
        } else {
            // 小屏幕：2-3列
            return Math.floor(Math.random() * 2) + 2 // 2, 3列
        }
    }

    const numColumns = getColumnCount()

    // 生成随机列宽（总和为100%）
    const columnWidths = []
    let remainingWidth = 100

    for (let i = 0; i < numColumns; i++) {
        if (i === numColumns - 1) {
            // 最后一列使用剩余宽度
            columnWidths.push(remainingWidth)
        } else {
            // 随机宽度：20% 到 50%
            const minWidth = 20
            const maxWidth = Math.min(50, remainingWidth - (numColumns - i - 1) * minWidth)
            const width = Math.random() * (maxWidth - minWidth) + minWidth
            columnWidths.push(Math.round(width))
            remainingWidth -= Math.round(width)
        }
    }

    // 生成每列的起始位置（距离顶部的距离）
    const columnStartPositions = []
    for (let i = 0; i < numColumns; i++) {
        // 随机起始位置：0% 到 20%
        const startPosition = Math.random() * 20
        columnStartPositions.push(Math.round(startPosition))
    }

    // 计算每列的起始left位置
    const columnLeftPositions = []
    let currentLeft = 0
    for (let i = 0; i < numColumns; i++) {
        columnLeftPositions.push(currentLeft)
        currentLeft += columnWidths[i]
    }

    // 将模版分配到各列
    const columnTemplates = Array.from({ length: numColumns }, () => [])
    templates.forEach((template, index) => {
        const columnIndex = index % numColumns
        columnTemplates[columnIndex].push(template)
    })

    // 生成列配置
    const columns = columnWidths.map((width, index) => ({
        width: `${width}%`,
        left: `${columnLeftPositions[index]}%`,
        top: `${columnStartPositions[index]}%`,
        templates: columnTemplates[index]
    }))

    return { columns, templates }
}

// 计算属性：带随机列布局的模版
const templatesWithRandomColumns = computed(() => {
    return generateRandomColumnLayout(currentFeaturedTemplates.value)
})

// 屏幕尺寸监听器
const updateScreenSize = () => {
    screenSize.value = window.innerWidth >= 1024 ? 'large' : 'small'
}

// 监听屏幕尺寸变化
let resizeHandler = null

// 路由监听和URL同步
watch(() => route.query, (newQuery) => {
    // 同步URL参数到组件状态
    if (newQuery.taskType) {
        // 根据URL参数设置任务类型
        const taskType = newQuery.taskType
        if (availableTaskTypes.value.some(type => type.value === taskType)) {
            selectTask(taskType)
        }
    }
    if (newQuery.model) {
        // 根据URL参数设置模型
        const model = newQuery.model
        if (availableModelClasses.value.some(m => m.value === model)) {
            selectModel(model)
        }
    }
    if (newQuery.expanded === 'true') {
        // 展开创建区域
        expandCreationArea()
    }

    // 注意：分享数据导入功能已移至 Share.vue 中的 createSimilar 函数
    // 这里不再需要处理分享数据导入
}, { immediate: true })

// 监听组件状态变化，同步到URL
watch([selectedTaskId, isCreationAreaExpanded, selectedModel], () => {
    const query = {}
    if (selectedTaskId.value) {
        query.taskType = selectedTaskId.value
    }
    if (isCreationAreaExpanded.value) {
        query.expanded = 'true'
    }
    if (selectedModel.value) {
        query.model = selectedModel.value
    }

    // 更新URL但不触发路由监听
    router.replace({ query })
})


// 组件挂载时初始化
onMounted(async () => {
    // 确保URL参数正确同步
    const query = route.query
    if (query.taskType) {
        selectTask(query.taskType)
    }
    if (query.model) {
        selectModel(query.model)
    }
    if (query.expanded === 'true') {
        expandCreationArea()
    }

    // 初始化屏幕尺寸
    updateScreenSize()

    // 添加屏幕尺寸监听器
    resizeHandler = () => {
        updateScreenSize()
    }
    window.addEventListener('resize', resizeHandler)

    // 加载精选模版数据
    await loadFeaturedTemplates(true)
    // 获取随机精选模版
    const randomTemplates = await getRandomFeaturedTemplates(10) // 获取10个模版
    currentFeaturedTemplates.value = randomTemplates
})

// 拖拽处理函数
const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
}

const handleDragEnter = (e) => {
    e.preventDefault()
    e.stopPropagation()
    isDragOver.value = true
}

const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    // 只有当离开整个拖拽区域时才设置为false
    if (!e.currentTarget.contains(e.relatedTarget)) {
        isDragOver.value = false
    }
}

const handleImageDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    isDragOver.value = false

    const files = Array.from(e.dataTransfer.files)
    const imageFile = files.find(file => file.type.startsWith('image/'))

    if (imageFile) {
        // 创建FileList对象来模拟input[type="file"]的change事件
        const dataTransfer = new DataTransfer()
        dataTransfer.items.add(imageFile)
        const fileList = dataTransfer.files

        // 创建模拟的change事件
        const event = {
            target: {
                files: fileList
            }
        }

        handleImageUpload(event)
        showAlert('图片拖拽上传成功', 'success')
    } else {
        showAlert('请拖拽图片文件', 'warning')
    }
}

const handleAudioDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    isDragOver.value = false

    const files = Array.from(e.dataTransfer.files)
    const audioFile = files.find(file => file.type.startsWith('audio/'))

    if (audioFile) {
        // 创建FileList对象来模拟input[type="file"]的change事件
        const dataTransfer = new DataTransfer()
        dataTransfer.items.add(audioFile)
        const fileList = dataTransfer.files

        // 创建模拟的change事件
        const event = {
            target: {
                files: fileList
            }
        }

        handleAudioUpload(event)
        showAlert('音频拖拽上传成功', 'success')
    } else {
        showAlert('请拖拽音频文件', 'warning')
    }
}

// 组件卸载时清理
onUnmounted(() => {
    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler)
    }
})

</script>
<template>
                <!-- 主内容区域 - 响应式布局 -->
                <div class="flex-1 flex flex-col min-h-0 mobile-content">
                    <!-- 生成视频区域 -->
                    <div class="flex-1 flex flex-col">
                        <!-- 内容区域 -->
                        <div class="flex-1 p-6">


                        <!-- 任务创建面板 -->
                        <div class="max-w-4xl mx-auto min-h-[80vh] flex flex-col" id="task-creator">
                            <!-- 合并的创作区域 -->
                            <div class="creation-area-container flex-1 flex flex-col justify-center">

                                <div class="default-state-container">
                                    <!-- 两个并列的下拉菜单 - Apple 风格 -->
                                    <div class="flex justify-center gap-6 mb-10">
                                        <!-- 任务类型下拉菜单 -->
                                        <ModelDropdown
                                            :available-models="availableTaskTypes.map(taskType => ({
                                                value: taskType,
                                                label: getTaskTypeName(taskType),
                                                icon: getTaskTypeIcon(taskType)
                                            }))"
                                            :selected-model="selectedTaskId"
                                            @select-model="selectTask"
                                        />

                                        <!-- 模型选择下拉菜单 -->
                                        <ModelDropdown
                                            :available-models="availableModelClasses"
                                            :selected-model="getCurrentForm().model_cls"
                                            @select-model="selectModel"
                                        />
                                    </div>

                                 <!-- 默认状态：中心文字 -->
                                <div v-show="!isCreationAreaExpanded" class="flex flex-col items-center justify-center">

                                    <div class="text-center">

                                        <h2 class="text-3xl sm:text-3xl md:text-4xl lg:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-8 tracking-tight">{{ t('whatDoYouWantToDo') }}</h2>

                                        <!-- 动态滚动提示 - Apple 风格 -->
                                        <div class="hint-container mb-12 pb-10">
                                            <div class="hint-text text-[#86868b] dark:text-[#98989d] text-base min-h-[60px] flex items-center justify-center tracking-tight">
                                                <transition name="hint-fade" mode="out-in">
                                                    <p :key="currentHintIndex" class="text-center max-w-2xl">
                                                        {{ currentTaskHints[currentHintIndex] }}
                                                    </p>
                                                </transition>
                                            </div>
                                            <!-- 提示指示器 - Apple 风格 -->
                                            <div class="flex justify-center mt-6 gap-2">
                                                <div v-for="(hint, index) in currentTaskHints" :key="index"
                                                    class="w-2 h-2 rounded-full transition-all duration-300"
                                                    :class="index === currentHintIndex ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] scale-110' : 'bg-[#86868b]/30 dark:bg-[#98989d]/30'">
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 展开开关 - Apple 极简风格 -->
                                    <div class="relative group cursor-pointer" @click="expandCreationArea">
                                        <button
                                            class="cursor-pointer relative bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-full px-8 py-4 text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.12)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)] active:scale-100 transition-all duration-200 ease-out min-w-[250px] max-w-[400px] tracking-tight"
                                        >
                                        <i class="fi fi-sr-cursor-finger-click text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] transition-all duration-200 pointer-events-none"></i>
                                        <span class="pl-2 text-base font-semibold transition-all duration-200 pointer-events-none">{{ t('startCreatingVideo') }}</span>
                                        </button>
                                    </div>
                                </div>

                                <!-- 展开状态：素材区域 -->
                                <div v-if="isCreationAreaExpanded" class="mb-8 prompt-input-section">
                                    <!-- 中心文字 - Apple 风格 -->
                                    <div class="text-center">

                                        <h2 class="text-3xl sm:text-3xl md:text-3xl lg:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-6 animate-fade-in tracking-tight">{{ t('whatMaterialsDoYouNeed') }}</h2>

                                        <p class="text-[#86868b] dark:text-[#98989d] text-base mb-10 transition-all duration-300 tracking-tight">
                                            <span v-if="selectedTaskId === 't2v'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseEnterTheMostDetailedVideoScript') }}</span>
                                            <span v-else-if="selectedTaskId === 'i2v'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseUploadAnImageAsTheFirstFrameOfTheVideoAndTheMostDetailedVideoScript') }}</span>
                                            <span v-else-if="selectedTaskId === 's2v'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseUploadARoleImageAnAudioAndTheGeneralVideoRequirements') }}</span>
                                            <span v-else
                                                  class="inline-block animate-fade-in">选择任务类型开始创作您的视频</span>
                                        </p>
                                    </div>

                                 <!-- 收缩开关 - Apple 风格 -->

                                <div
                                    class="creation-area transition-all duration-500 ease-out max-w-10xl mx-auto"
                                    @click.stop>
                                    <!-- 收起按钮 - Apple 风格 -->
                                    <div class="flex justify-center mb-6">
                                        <button @click="contractCreationArea"
                                                class="flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/6 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/10 dark:hover:border-white/12 hover:shadow-[0_2px_8px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_2px_8px_rgba(0,0,0,0.3)] rounded-full transition-all duration-200 ease-out group"
                                                :class="{ 'animate-pulse': isContracting }">
                                            <i class="fas fa-compress-alt text-sm transition-transform duration-200 group-hover:scale-110"
                                               :class="{ 'animate-spin': isContracting }"></i>
                                            <span class="text-sm font-medium tracking-tight">
                                                {{ t('collapseCreationArea') }}
                                            </span>
                                            <i class="fas fa-chevron-up text-xs transition-transform duration-200 group-hover:translate-y-[-2px]"
                                               :class="{ 'animate-bounce': isContracting }"></i>
                                        </button>
                                    </div>

                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v'" class="upload-section">
                                    <!-- 上传图片 - Apple 风格 -->
                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v'">
                                        <!-- 图片标签 -->
                                        <div class="flex justify-between items-center mb-3">
                                                <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                                {{ t('image') }}
                                                </label>
                                        </div>
                                        <!-- 上传图片区域 - Apple 风格 -->
                                        <div class="relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl p-2 min-h-[220px] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_16px_rgba(0,0,0,0.3)]"
                                            @drop="handleImageDrop"
                                            @dragover="handleDragOver"
                                            @dragenter="handleDragEnter"
                                            @dragleave="handleDragLeave"
                                            :class="{
                                                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': isDragOver,
                                                'p-8': !getCurrentImagePreview()
                                            }"
                                            >
                                            <!-- 默认上传界面 - Apple 风格 -->
                                            <div v-if="!getCurrentImagePreview()" class="flex flex-col items-center justify-center h-full">
                                            <p class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ t('uploadImage') }}</p>
                                            <p class="text-xs text-[#86868b] dark:text-[#98989d] mb-6 tracking-tight">{{ t('supportedImageFormats') }}</p>
                                            <div class="flex items-center justify-center gap-4">
                                                        <div class="flex flex-col items-center gap-2">
                                                            <button
                                                                class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.3)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100"
                                                                @click="triggerImageUpload"
                                                                :title="t('uploadImage')">
                                                                <i class="fas fa-upload text-base"></i>
                                                            </button>
                                                            <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('upload') }}</span>
                                                        </div>
                                                        <div class="flex flex-col items-center gap-2">
                                                            <button
                                                                @click.stop="showImageTemplates = true; mediaModalTab = 'history'; getImageHistory()"
                                                                class="w-12 h-12 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                                :title="t('templates')">
                                                                <i class="fas fa-history text-base"></i>
                                                            </button>
                                                            <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('templates') }}</span>
                                                        </div>
                                            </div>
                                            </div>

                                            <!-- 图片预览 - Apple 风格 -->
                                            <div v-if="getCurrentImagePreview()" class="relative w-full min-h-[220px] flex items-center justify-center group">
                                                    <img :src="getCurrentImagePreviewUrl()" alt="t('previewImage')"
                                                        class="max-w-full max-h-[220px] w-auto h-auto object-contain rounded-xl transition-all duration-200">

                                                <!-- 删除按钮 - Apple 风格 -->
                                                    <div
                                                        class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-200">
                                                    <div class="flex gap-3">
                                                            <button @click.stop="removeImage"
                                                            class="w-11 h-11 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100"
                                                                :title="t('deleteImage')">
                                                            <i class="fas fa-trash text-base"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                                <input type="file" ref="imageInput" @change="handleImageUpload" accept="image/*"
                                                style="display: none;">
                                            </div>
                                    </div>

                                    <!-- 上传音频 - Apple 风格 -->
                                    <div v-if="selectedTaskId === 's2v'">
                                        <!-- 音频标签 -->
                                        <div class="flex justify-between items-center mb-3">
                                                        <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                                        {{ t('audio') }}
                                                        </label>
                                        </div>
                                        <!-- 上传音频区域 - Apple 风格 -->
                                        <div class="relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl p-2 min-h-[220px] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_16px_rgba(0,0,0,0.3)]"
                                            @drop="handleAudioDrop"
                                            @dragover="handleDragOver"
                                            @dragenter="handleDragEnter"
                                            @dragleave="handleDragLeave"
                                            :class="{
                                                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': isDragOver,
                                                'p-8': !getCurrentAudioPreview()
                                            }"
                                            >
                                        <!-- 默认上传界面 - Apple 风格 -->
                                            <div v-if="!getCurrentAudioPreview()" class="flex flex-col items-center justify-center h-full"
                                                >
                                                <p class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ t('uploadAudio') }}</p>
                                                <p class="text-xs text-[#86868b] dark:text-[#98989d] mb-6 tracking-tight">{{ t('supportedAudioFormats') }}</p>
                                            <div class="flex items-center justify-center gap-3">
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button
                                                            class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.3)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100"
                                                            @click="triggerAudioUpload"
                                                            :title="t('uploadAudio')">
                                                            <i class="fas fa-upload text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('upload') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button
                                                            @click.stop="showAudioTemplates = true; mediaModalTab = 'history'; getAudioHistory()"
                                                            class="w-12 h-12 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                            :title="t('templates')">
                                                            <i class="fas fa-history text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('templates') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button @click.stop="isRecording ? stopRecording() : startRecording()"
                                                        class="w-12 h-12 flex items-center justify-center rounded-full transition-all duration-200 hover:scale-110 active:scale-100"
                                                            :class="isRecording ? 'bg-red-500 dark:bg-red-400 text-white shadow-[0_4px_12px_rgba(239,68,68,0.3)] dark:shadow-[0_4px_12px_rgba(248,113,113,0.4)]' : 'bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)]'"
                                                            :title="isRecording ? t('stopRecording') : t('recordAudio')">
                                                        <i class="fas fa-microphone text-base" :class="{ 'animate-pulse': isRecording }"></i>
                                                    </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ isRecording ? formatRecordingDuration(recordingDuration) : t('recordAudio') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button @click.stop="showVoiceTTSModal = true"
                                                            class="w-12 h-12 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                            :title="t(' textToSpeech ')">
                                                            <i class="fas fa-volume-up text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('textToSpeech') }}</span>
                                                    </div>

                                        </div>
                                            </div>

                                        <!-- 音频预览 - Apple 风格 -->
                                            <div v-if="getCurrentAudioPreview()" class="relative w-full min-h-[220px] group flex items-center justify-center p-8">
                                            <audio controls class="w-full max-w-md" @error="handleAudioError" @loadstart="console.log('音频开始加载')" @canplay="console.log('音频可以播放')">
                                                    <source :src="getCurrentAudioPreviewUrl()" :type="getAudioMimeType()" preload="metadata">
                                            </audio>

                                            <!-- 删除按钮 - Apple 风格 -->
                                                <div
                                                    class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-200">
                                                <div class="flex gap-3">
                                                        <button @click.stop="removeAudio"
                                                        class="w-11 h-11 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100"
                                                            :title="t('deleteAudio')">
                                                        <i class="fas fa-trash text-base"></i>
                                                    </button>
                                                </div>
                                            </div>
                                            </div>

                                            <input type="file" ref="audioInput" @change="handleAudioUpload" accept="audio/*"
                                            style="display: none;">
                                        </div>
                                    </div>
                                </div>

                                        <!-- 提示词输入区域 - Apple 风格 -->
                                        <div class="flex justify-between items-center mb-3">
                                            <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center tracking-tight">
                                                    {{ t('prompt') }}
                                                    <button @click="showPromptModal = true; promptModalTab = 'templates'"
                                                        class="ml-2 text-xs text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors"
                                                        :title="t('promptTemplates')">
                                                    <i class="fas fa-lightbulb"></i>
                                                </button>
                                            </label>
                                            <div class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                {{ getCurrentForm().prompt?.length || 0 }} / 1000
                                            </div>
                                        </div>
                                        <div class="relative">
                                            <textarea v-model="getCurrentForm().prompt"
                                                class="relative w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] min-h-[120px] transition-all duration-200 resize-none main-scrollbar placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)]"
                                                :placeholder="getPromptPlaceholder()"
                                                rows="4"
                                                maxlength="500"
                                                required></textarea>
                                        </div>

                                        <div class="flex justify-between items-center mt-3">
                                            <button @click="clearPrompt"
                                                class="flex items-center text-sm rounded-lg px-3 py-1.5 transition-all duration-200 text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] hover:bg-black/4 dark:hover:bg-white/6 group tracking-tight">
                                                <i class="fas fa-sync-alt text-sm mr-2 group-hover:rotate-180 transition-transform duration-300"></i>
                                                {{ t('clear') }}
                                            </button>

                                        </div>
                                <!-- 提交按钮 - Apple 极简风格 -->
                                <div class="flex justify-center mt-8">
                                    <button @click="handleSubmitTask" :disabled="submitting || templateLoading"
                                            class="cursor-pointer relative bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-full px-10 py-4 text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.12)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)] active:scale-100 transition-all duration-200 ease-out min-w-[250px] max-w-[400px] tracking-tight disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
                                            :class="{ 'disabled': submitting || templateLoading }">

                                        <i v-if="submitting" class="fas fa-spinner fa-spin text-lg mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <i v-else-if="templateLoading" class="fas fa-spinner fa-spin text-lg mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <i v-else class="fi fi-sr-cursor-finger-click text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] transition-all duration-200 pointer-events-none"></i>
                                        <span class="pl-2 text-base font-semibold transition-all duration-200 pointer-events-none">{{ submitting ? t('submitting') : templateLoading ? '模板加载中...' : t('generateVideo') }}</span>
                                        </button>
                                </div>

                                </div>

                                </div>

                            </div>
                </div>

                        </div>
                </div>
                        </div>

                    <!-- 历史任务区域 - Apple 风格 -->
                    <div v-if="tasks.length > 0" class="task-carousel mt-16">
                        <div class="flex-1 p-6">
                            <div class="relative flex items-center justify-center mb-8">
                                <!-- 标题 - Apple 风格 -->
                                <h2 class="text-3xl sm:text-3xl md:text-4xl lg:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('historyTask') }}</h2>
                                <!-- 更多按钮 - Apple 风格 -->
                                <button @click="switchToProjectsView()"
                                    class="absolute right-0 flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/6 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/10 dark:hover:border-white/12 rounded-full transition-all duration-200"
                                    :title="t('viewMore')">
                                <span class="text-sm font-medium tracking-tight">{{ t('more') }}</span>
                                <i class="fas fa-arrow-right text-xs"></i>
                            </button>
                            </div>
                            <div class="mx-auto">
                                <TaskCarousel :tasks="tasks" />
                            </div>
                        </div>
                    </div>


                    <!-- 精选模版区域 - Apple 风格 -->
                    <div v-if="currentFeaturedTemplates.length > 0" class="flex-1 flex flex-col min-h-0 mt-20">
                        <div class="flex-1 p-6">
                            <!-- 控制区域 - Apple 风格 -->
                            <div class="relative flex items-center justify-center mb-8">
                                <!-- 标题和随机按钮 -->
                                <div class="flex items-center gap-4">
                                    <h2 class="text-3xl sm:text-3xl md:text-4xl lg:text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('discover') }}</h2>
                                    <!-- 随机图标按钮 - Apple 风格 -->
                                    <button @click="refreshRandomTemplates"
                                            :disabled="featuredTemplatesLoading"
                                            class="w-10 h-10 flex items-center justify-center bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 border border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded-full transition-all duration-200 hover:scale-110 hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/25 active:scale-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                                            :title="t('refreshRandomTemplates')">
                                        <i class="fas fa-random text-sm"
                                           :class="{ 'animate-spin': featuredTemplatesLoading }"></i>
                                    </button>
                                </div>
                                <!-- 更多按钮 - Apple 风格 -->
                                <button @click="switchToInspirationView()"
                                    class="absolute right-0 flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/6 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/10 dark:hover:border-white/12 rounded-full transition-all duration-200"
                                    :title="t('viewMore')">
                                <span class="text-sm font-medium tracking-tight">{{ t('more') }}</span>
                                <i class="fas fa-arrow-right text-xs"></i>
                            </button>
                            </div>

                            <!-- 精选模版随机列布局 - Apple 风格 -->
                            <div class="relative min-h-[400px] lg:min-h-[600px]">
                            <!-- 随机列 -->
                            <div v-for="(column, columnIndex) in templatesWithRandomColumns.columns" :key="columnIndex"
                                    class="absolute transition-all duration-500 animate-fade-in"
                                    :style="{
                                        width: column.width,
                                        left: column.left,
                                        top: column.top,
                                        animationDelay: `${columnIndex * 0.2}s`
                                    }">
                                <!-- 列内的模版卡片 - Apple 风格 -->
                                <div v-for="item in column.templates" :key="item.task_id"
                                        class="mb-3 group relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] rounded-2xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.2)]">
                                <!-- 视频缩略图区域 -->
                                <div class="cursor-pointer bg-black/2 dark:bg-white/2 relative flex flex-col"
                                @click="previewTemplateDetail(item, false)"
                                :title="t('viewTemplateDetail')">
                                        <!-- 视频预览 -->
                                        <video v-if="item?.outputs?.output_video"
                                            :src="getTemplateFileUrl(item.outputs.output_video,'videos')"
                                            :poster="getTemplateFileUrl(item.inputs.input_image,'images')"
                                            class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                            preload="auto" playsinline webkit-playsinline
                                            @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                                            @loadeddata="onVideoLoaded($event)"
                                            @ended="onVideoEnded($event)"
                                            @error="onVideoError($event)"></video>
                                    <!-- 图片缩略图 -->
                                        <img v-else
                                        :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                                        :alt="item.params?.prompt || '模板图片'"
                                        class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                        @error="handleThumbnailError" />
                                        <!-- 移动端播放按钮 - Apple 风格 -->
                                        <button v-if="item?.outputs?.output_video"
                                            @click.stop="toggleVideoPlay($event)"
                                            class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.2)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 transition-all duration-200 z-20">
                                            <i class="fas fa-play text-sm"></i>
                                        </button>
                                    <!-- 悬浮操作按钮（下方居中，仅桌面端）- Apple 风格 -->
                                    <div
                                        class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
                                        <div class="flex gap-2 pointer-events-auto">
                                            <button @click.stop="applyTemplateImage(item)"
                                                class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                :title="t('applyImage')">
                                                <i class="fas fa-image text-sm"></i>
                                            </button>
                                            <button @click.stop="applyTemplateAudio(item)"
                                                class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                :title="t('applyAudio')">
                                                <i class="fas fa-music text-sm"></i>
                                            </button>
                                            <button @click.stop="handleUseTemplate(item)"
                                                class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                :title="t('useTemplate')">
                                                <i class="fas fa-clone text-sm"></i>
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

                <MediaTemplate />

                <!-- 语音合成模态框 -->
                <div v-if="showVoiceTTSModal" class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 dark:bg-black/60 backdrop-blur-sm">
                    <div class="relative w-full h-full max-w-6xl max-h-[100vh] mx-4 my-8 bg-gray-900 rounded-xl shadow-2xl overflow-hidden">
                        <Voice_tts @tts-complete="handleTTSComplete" @close-modal="showVoiceTTSModal = false" />
                    </div>
                </div>

                <!-- GitHub 仓库链接 - Apple 极简风格 -->
                <div class="fixed bottom-6 right-6 z-50">
                    <a href="https://github.com/ModelTC/LightX2V"
                       target="_blank"
                       rel="noopener noreferrer"
                       class="flex items-center gap-2.5 px-4 py-2.5 bg-white/85 dark:bg-[#1e1e1e]/85 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.3)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.4)] hover:scale-105 active:scale-100 transition-all duration-200 group"
                       title="Star us on GitHub">
                        <i class="fab fa-github text-lg text-[#1d1d1f] dark:text-[#f5f5f7] transition-transform duration-200 group-hover:rotate-12"></i>
                        <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">LightX2V</span>
                        <i class="fas fa-external-link-alt text-xs text-[#86868b] dark:text-[#98989d] transition-all duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"></i>
                    </a>
                </div>
</template>

<style scoped>
/* Apple 风格极简设计 - 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
</style>
