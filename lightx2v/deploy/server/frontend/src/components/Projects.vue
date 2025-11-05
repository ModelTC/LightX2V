<script setup>
import { useI18n } from 'vue-i18n'
import { watch } from 'vue'
const { t, locale } = useI18n()
import FloatingParticles from './FloatingParticles.vue'
import TopBar from './TopBar.vue'
import LeftBar from './LeftBar.vue'
import Loading from './Loading.vue'

import { useRoute, useRouter } from 'vue-router'

// Props
const props = defineProps({
  query: {
    type: Object,
    default: () => ({})
  },
  taskId: {
    type: String,
    default: null
  }
})
import {
    submitting,

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

// 路由监听
const route = useRoute()
const router = useRouter()

// 处理任务下载
const handleDownloadTask = async (task) => {
    try {
        console.log('开始下载任务文件:', { taskId: task.task_id, outputs: task.outputs })

        // 处理文件名，确保有正确的后缀名
        let fileName = task.outputs?.output_video || 'video.mp4'
        if (fileName && typeof fileName === 'string') {
            // 检查是否已有后缀名
            const hasExtension = /\.[a-zA-Z0-9]+$/.test(fileName)
            if (!hasExtension) {
                // 没有后缀名，添加mp4后缀
                fileName = `${fileName}.mp4`
                console.log('添加后缀名:', fileName)
            }
        }

        // 先尝试从缓存获取
        let fileData = getTaskFileFromCache(task.task_id, 'output_video')
        console.log('缓存中的文件数据:', fileData)

        if (fileData && fileData.blob) {
            // 缓存中有blob数据，直接使用
            console.log('使用缓存中的文件数据')
            downloadFile({ ...fileData, name: fileName })
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
                        name: fileName
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
            const url = await getTaskFileUrl(task.task_id, 'output_video')
            console.log('获取到的文件URL:', url)

            if (url) {
                const response = await fetch(url)
                console.log('文件响应状态:', response.status, response.ok)

                if (response.ok) {
                    const blob = await response.blob()
                    console.log('文件blob大小:', blob.size)

                    fileData = {
                        blob: blob,
                        name: fileName
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

// 监听路由变化，处理任务详情路由
watch(() => route.params.taskId, (newTaskId) => {
    if (newTaskId && route.name === 'TaskDetail') {
        openTaskFromRoute(newTaskId)
    }
}, { immediate: true })

// 监听Projects页面的查询参数
watch(() => route.query, (newQuery) => {
    // 同步URL参数到组件状态
    if (newQuery.search) {
        taskSearchQuery.value = newQuery.search
    }
    if (newQuery.status) {
        statusFilter.value = newQuery.status
    }
    if (newQuery.page) {
        const page = parseInt(newQuery.page)
        if (page > 0 && page !== currentTaskPage.value) {
            goToPage(page)
        }
    }
}, { immediate: true })

// 监听组件状态变化，同步到URL
watch([taskSearchQuery, statusFilter, currentTaskPage], () => {
    const query = {}
    if (taskSearchQuery.value) {
        query.search = taskSearchQuery.value
    }
    if (statusFilter.value && statusFilter.value !== 'ALL') {
        query.status = statusFilter.value
    }
    if (currentTaskPage.value > 1) {
        query.page = currentTaskPage.value.toString()
    }

    // 更新URL但不触发路由监听
    router.replace({ query })
})

</script>
<template>
    <!-- 历史任务区域 - Apple 极简风格 -->
                        <div class="flex-1 flex flex-col min-h-0 mobile-content">
                            <!-- 内容区域 -->
        <div class="flex-1 overflow-y-auto p-6 content-area main-scrollbar">
                                <!-- 历史任务功能区 -->
            <div class="max-w-7xl mx-auto" id="task-creator">
                <!-- 标题区域 - Apple 风格 -->
                <div class="text-center mb-10">
                    <h1 class="text-4xl sm:text-5xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('myProjects') }}</h1>
                                        </div>

                <!-- 搜索和筛选区域 - Apple 风格 -->
                <div class="flex flex-col md:flex-row gap-4 mb-8">
                    <!-- 搜索框 - Apple 风格 -->
                                        <div class="relative flex-1">
                        <i class="fas fa-search absolute left-4 top-1/2 -translate-y-1/2 text-[#86868b] dark:text-[#98989d] pointer-events-none z-10"></i>
                                            <input v-model="taskSearchQuery"
                            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl py-3 pl-11 pr-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200"
                            :placeholder="t('searchTasks')"
                            type="text" />
                            </div>

                    <!-- 状态筛选按钮 - Apple 风格 -->
                    <div class="flex gap-2 flex-wrap items-center">
                                            <button @click="statusFilter = 'ALL'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                                                :class="statusFilter === 'ALL'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                                {{ t('all') }}
                                    </button>
                                            <button @click="statusFilter = 'SUCCEED'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                                                :class="statusFilter === 'SUCCEED'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                                {{ t('success') }}
                                    </button>
                                            <button @click="statusFilter = 'RUNNING'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                                                :class="statusFilter === 'RUNNING'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                                {{ t('running') }}
                                    </button>
                                            <button @click="statusFilter = 'FAILED'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                                                :class="statusFilter === 'FAILED'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                                {{ t('failed') }}
                                    </button>
                        <!-- 刷新按钮 - Apple 风格 -->
                                            <button @click="() => refreshTasks(true)"
                            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100 flex-shrink-0"
                                                :title="t('refreshTasks')">
                            <i class="fas fa-sync-alt text-sm"></i>
                                    </button>
                            </div>
                        </div>

                <!-- 分页组件 - Apple 风格 -->
                <div v-if="paginationInfo" :key="paginationKey" class="mb-6">
                    <div class="flex items-center justify-between text-xs mb-4">
                        <div class="flex items-center space-x-1 text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                    <span>{{ t('total') }} {{ paginationInfo.total }} {{ t('tasks') }}</span>
                        </div>
                            </div>
                                                <div v-if="paginationInfo.total_pages > 1" class="flex justify-center">
                        <nav class="isolate inline-flex gap-1" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToPage(currentTaskPage - 1)"
                                                            :disabled="currentTaskPage <= 1"
                                class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': currentTaskPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                <i class="fas fa-chevron-left text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisiblePages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToPage(page)"
                                                                :class="[
                                        'relative inline-flex items-center justify-center min-w-[36px] h-9 px-3 text-sm font-medium rounded-lg transition-all duration-200',
                                                                    page === currentTaskPage
                                            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                            : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'
                                                                ]"
                                                                :aria-current="page === currentTaskPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                <span v-else class="relative inline-flex items-center px-2 text-sm font-semibold text-[#86868b] dark:text-[#98989d]">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToPage(currentTaskPage + 1)"
                                                            :disabled="currentTaskPage >= paginationInfo.total_pages"
                                class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': currentTaskPage >= paginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                <i class="fas fa-chevron-right text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                        </div>
                                    </div>

                <!-- 任务内容网格 - Apple 风格 -->
                <div v-if="filteredTasks.length === 0" class="flex flex-col items-center justify-center py-16 text-center">
                    <div class="w-20 h-20 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-6">
                        <i class="fas fa-video text-3xl text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                </div>
                    <p class="text-lg font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ t('noHistoryTasks') }}</p>
                    <p class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('startToCreateYourFirstAIVideo') }}</p>
                            </div>

                <div v-else class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4">
                    <!-- 任务卡片 - Apple 风格 -->
                                                <div v-for="task in filteredTasks" :key="task.task_id"
                        class="break-inside-avoid mb-4 group relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] rounded-2xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.2)]">
                                                    <!-- 缩略图区域 -->
                        <div class="cursor-pointer bg-black/2 dark:bg-white/2 relative flex flex-col"
                                                    @click="openTaskDetailModal(task)"
                                                    :title="t('viewTaskDetails')">
                                                            <!-- 成功任务：显示视频动图 -->
                                                            <video v-if="task.status === 'SUCCEED' && task.outputs?.output_video"
                                                                :src="getTaskFileUrlSync(task.task_id, 'output_video')"
                                                                :poster="getTaskFileUrlSync(task.task_id, 'input_image')"
                                class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                                                preload="auto" playsinline webkit-playsinline
                                                                    @mouseenter="playVideo($event)"
                                                                    @mouseleave="pauseVideo($event)"
                                                                    @loadeddata="onVideoLoaded($event)"
                                                                    @ended="onVideoEnded($event)"
                                                                @error="onVideoError($event)"></video>

                                                        <!-- 其他状态：显示输入图片或占位符 -->
                            <img v-else-if="task.inputs?.input_image"
                                                        :src="getTaskFileUrlSync(task.task_id, 'input_image')"
                                class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                                        @error="handleThumbnailError" />

                            <!-- 默认占位符 - Apple 风格 -->
                            <div v-else class="w-full aspect-[9/16] bg-[#f5f5f7] dark:bg-[#1c1c1e] flex items-center justify-center">
                                <i class="fas fa-video text-4xl text-[#86868b]/30 dark:text-[#98989d]/30"></i>
                            </div>

                            <!-- 移动端播放按钮 - Apple 风格 -->
                            <button v-if="task.status === 'SUCCEED' && task.outputs?.output_video"
                                                        @click.stop="toggleVideoPlay($event)"
                                class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.2)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 transition-all duration-200 z-20">
                                                        <i class="fas fa-play text-sm"></i>
                                                    </button>

                            <!-- 状态指示器 - Apple 风格 -->
                            <div class="absolute top-3 right-3">
                                <span :class="[
                                    'px-3 py-1.5 rounded-full text-xs font-medium backdrop-blur-[20px] shadow-sm',
                                    task.status === 'SUCCEED' ? 'bg-green-500/90 dark:bg-green-400/90 text-white' :
                                    task.status === 'RUNNING' ? 'bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 text-white' :
                                    task.status === 'FAILED' ? 'bg-red-500/90 dark:bg-red-400/90 text-white' :
                                    'bg-white/90 dark:bg-[#2c2c2e]/90 text-[#86868b] dark:text-[#98989d] border border-black/8 dark:border-white/8'
                                ]">
                                                                    {{ getTaskStatusDisplay(task.status) }}
                                                                </span>
                                                            </div>

                            <!-- 悬停时显示的操作按钮（桌面端）- Apple 风格 -->
                            <div class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
                                <div class="flex gap-2 pointer-events-auto">
                                    <!-- 复用按钮 - 所有状态可用 -->
                                    <button v-if="['CREATED', 'PENDING', 'RUNNING','SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="reuseTask(task)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                    :title="t('reuseTask')">
                                        <i class="fas fa-copy text-sm"></i>
                                                    </button>

                                    <!-- 取消按钮 - 进行中状态 -->
                                    <button v-if="['CREATED', 'PENDING', 'RUNNING'].includes(task.status)"
                                                                    @click.stop="cancelTask(task.task_id)"
                                        class="w-10 h-10 rounded-full bg-white dark:bg-[#3a3a3c] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-red-500 dark:text-red-400 hover:scale-110 active:scale-100 transition-all duration-200"
                                                                    :title="t('cancelTask')">
                                        <i class="fas fa-times text-sm"></i>
                                                            </button>

                                    <!-- 重试按钮 - 失败/取消状态 -->
                                    <button v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="resumeTask(task.task_id)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                    :title="t('retryTask')">
                                        <i class="fas fa-redo text-sm"></i>
                                                            </button>

                                    <!-- 下载按钮 - 成功状态 -->
                                                                <button v-if="task.status === 'SUCCEED'"
                                                                        @click.stop="handleDownloadTask(task)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                    :title="t('downloadTask')">
                                        <i class="fas fa-download text-sm"></i>
                                                            </button>

                                    <!-- 删除按钮 - 完成/失败/取消状态 -->
                                    <button v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="deleteTask(task.task_id)"
                                        class="w-10 h-10 rounded-full bg-white dark:bg-[#3a3a3c] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-red-500 dark:text-red-400 hover:scale-110 active:scale-100 transition-all duration-200"
                                                                    :title="t('deleteTask')">
                                        <i class="fas fa-trash text-sm"></i>
                                                            </button>
                                                    </div>
                                                </div>
                                            </div>

                        <!-- 任务信息 - Apple 风格 -->
                        <div class="px-4 py-4">
                            <h3 class="text-[#1d1d1f] dark:text-[#f5f5f7] font-medium text-sm mb-2 line-clamp-2 tracking-tight">
                                {{ task.params.prompt.length > 50 ? task.params.prompt.slice(0, 50) + '...' : task.params.prompt }}
                            </h3>
                            <div class="flex items-center justify-between text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">
                                <span class="truncate max-w-[60%]">{{ task.model_cls }}</span>
                                <span class="flex-shrink-0">{{ getRelativeTime(task.create_t) }}</span>
                                    </div>
                                </div>
                                            </div>
                                    </div>
                                </div>
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
