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
                        <!-- 历史任务区域 -->
                        <div class="flex-1 flex flex-col min-h-0 mobile-content">
                            <!-- 内容区域 -->
                            <div class="flex-1 overflow-y-auto p-10 content-area main-scrollbar">
                                <!-- 历史任务功能区 -->
                                <div class="max-w-4xl mx-auto" id="task-creator">
                                    <!-- 搜索和筛选区域 -->
                                    <div class="flex flex-col md:flex-row gap-4 mb-6">
                                <!-- 搜索框 -->
                                        <div class="relative flex-1">
                                            <i
                                                class="fas fa-search absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none z-10"></i>
                                            <input v-model="taskSearchQuery"
                                                class="w-full bg-dark-light border border-laser-purple/30 rounded-lg py-3 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-laser-purple/50 transition-all focus:border-laser focus:shadow-laser"
                                                :placeholder="t('searchTasks')" type="text" />
                            </div>

                                        <!-- 分类筛选 -->
                                        <div class="flex gap-2">
                                            <button @click="statusFilter = 'ALL'"
                                                :class="statusFilter === 'ALL'
                                                    ? 'bg-laser-purple/20 text-white border-laser-purple/40'
                                                    : 'bg-dark-light text-gray-400 hover:text-white hover:bg-dark-light/80'"
                                                class="px-4 py-2 rounded-lg border border-transparent transition-all duration-200 text-sm font-medium">
                                                {{ t('all') }}
                                    </button>
                                            <button @click="statusFilter = 'SUCCEED'"
                                                :class="statusFilter === 'SUCCEED'
                                                    ? 'bg-laser-purple/20 text-white border-laser-purple/40'
                                                    : 'bg-dark-light text-gray-400 hover:text-white hover:bg-dark-light/80'"
                                                class="px-4 py-2 rounded-lg border border-transparent transition-all duration-200 text-sm font-medium">
                                                {{ t('success') }}
                                    </button>
                                            <button @click="statusFilter = 'RUNNING'"
                                                :class="statusFilter === 'RUNNING'
                                                    ? 'bg-laser-purple/20 text-white border-laser-purple/40'
                                                    : 'bg-dark-light text-gray-400 hover:text-white hover:bg-dark-light/80'"
                                                class="px-4 py-2 rounded-lg border border-transparent transition-all duration-200 text-sm font-medium">
                                                {{ t('running') }}
                                    </button>
                                            <button @click="statusFilter = 'FAILED'"
                                                :class="statusFilter === 'FAILED'
                                                    ? 'bg-laser-purple/20 text-white border-laser-purple/40'
                                                    : 'bg-dark-light text-gray-400 hover:text-white hover:bg-dark-light/80'"
                                                class="px-4 py-2 rounded-lg border border-transparent transition-all duration-200 text-sm font-medium">
                                                {{ t('failed') }}
                                    </button>
                                            <button @click="() => refreshTasks(true)"
                                        class="text-gray-400 hover:text-gradient-primary transition-colors flex-shrink-0"
                                                :title="t('refreshTasks')">
                                        <i class="fas fa-sync-alt"></i>
                                    </button>
                            </div>
                        </div>

                                <!-- 分页组件 -->
                                <div v-cloak>
                                                <div v-if="paginationInfo" :key="paginationKey" class="pagination-container mb-3">
                                        <div class="flex items-center justify-between text-xs text-gray-400">
                                            <div class="flex items-center space-x-1 text-gray-500">
                                                    <span>{{ t('total') }} {{ paginationInfo.total }} {{ t('tasks') }}</span>
                            </div>
                                                <div v-if="paginationInfo.total_pages > 1" class="flex justify-center">
                                                    <nav class="isolate inline-flex -space-x-px rounded-md" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToPage(currentTaskPage - 1)"
                                                            :disabled="currentTaskPage <= 1"
                                                            class="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': currentTaskPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                                            <i class="fas fa-chevron-left text-sm" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisiblePages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToPage(page)"
                                                                :class="[
                                                                    'relative inline-flex items-center px-4 py-2 text-sm font-semibold focus:z-20 focus:outline-offset-0',
                                                                    page === currentTaskPage
                                                                        ? 'z-10 text-white focus-visible:outline-2 focus-visible:outline-offset-2 bg-laser-purple focus-visible:outline-laser-purple'
                                                                        : 'text-gray-200 inset-ring inset-ring-gray-700 hover:bg-white/5'
                                                                ]"
                                                                :aria-current="page === currentTaskPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                                            <span v-else class="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-400 inset-ring inset-ring-gray-700 focus:outline-offset-0">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToPage(currentTaskPage + 1)"
                                                            :disabled="currentTaskPage >= paginationInfo.total_pages"
                                                            class="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': currentTaskPage >= paginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                                            <i class="fas fa-chevron-right text-sm" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 可滚动的任务列表区域 -->
                                    <div class="flex-1 overflow-y-auto main-scrollbar min-h-0 p-4"
                                        style="overflow-x: visible;">
                                <div v-cloak>
                                            <div v-if="filteredTasks.length === 0"
                                                class="flex-col items-center justify-center py-12 text-center">
                                                <p class="text-gray-400 text-sm">{{ t('noHistoryTasks') }}</p>
                                                <p class="text-gray-500 text-xs mt-1">{{ t('startToCreateYourFirstAIVideo') }}</p>
                                            </div>
                                            <div v-else
                                                class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-3">
                                                    <!-- 任务卡片 -->
                                                <div v-for="task in filteredTasks" :key="task.task_id"
                                                        class="break-inside-avoid mb-3 group relative bg-dark-light rounded-xl overflow-hidden border border-gray-700/50 hover:border-laser-purple/40 transition-all duration-300 hover:shadow-laser/20">
                                                    <!-- 缩略图区域 -->
                                                    <div class="cursor-pointer bg-gray-800 relative flex flex-col"
                                                    @click="openTaskDetailModal(task)"
                                                    :title="t('viewTaskDetails')">
                                                        <!-- 视频预览 -->
                                                            <!-- 成功任务：显示视频动图 -->
                                                            <video v-if="task.status === 'SUCCEED' && task.outputs?.output_video"
                                                                :src="getTaskFileUrlSync(task.task_id, 'output_video')"
                                                                :poster="getTaskFileUrlSync(task.task_id, 'input_image')"
                                                                    class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                                                                preload="auto" playsinline webkit-playsinline
                                                                    @mouseenter="playVideo($event)"
                                                                    @mouseleave="pauseVideo($event)"
                                                                    @loadeddata="onVideoLoaded($event)"
                                                                    @ended="onVideoEnded($event)"
                                                                @error="onVideoError($event)"></video>

                                                        <!-- 其他状态：显示输入图片或占位符 -->
                                                        <img v-else="task.inputs?.input_image"
                                                        :src="getTaskFileUrlSync(task.task_id, 'input_image')"
                                                        class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                                                        @error="handleThumbnailError" />

                                                        <!-- 移动端播放按钮 -->
                                                        <button v-if="task.status === 'SUCCEED'"
                                                        @click.stop="toggleVideoPlay($event)"
                                                        class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/70 transition-colors z-20">
                                                        <i class="fas fa-play text-sm"></i>
                                                    </button>

                                                            <!-- 状态指示器 -->
                                                            <div class="absolute top-2 right-2">
                                                            <span :class="getTaskStatusColor(task.status)"
                                                                class="px-2 py-1 rounded-full text-xs font-medium bg-black/50 backdrop-blur-sm">
                                                                    {{ getTaskStatusDisplay(task.status) }}
                                                                </span>
                                                            </div>

                                                            <!-- 悬停时显示的操作按钮（桌面端） -->
                                                        <div
                                                            class="hidden md:block absolute bottom-2 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                                                                <div class="flex space-x-2 sm:space-x-3 pointer-events-auto">
                                                    <button
                                                                        v-if="['CREATED', 'PENDING', 'RUNNING','SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="reuseTask(task)"
                                                                        class="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-laser-purple/80 backdrop-blur-sm flex items-center justify-center text-white hover:bg-white/30 transition-colors"
                                                                    :title="t('reuseTask')">
                                                                <i class="fas fa-copy text-sm sm:text-lg"></i>
                                                    </button>
                                                            <button
                                                                    v-if="['CREATED', 'PENDING', 'RUNNING'].includes(task.status)"
                                                                    @click.stop="cancelTask(task.task_id)"
                                                                    class="w-10 h-10 rounded-full bg-red-400/80 backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                                                                    :title="t('cancelTask')">
                                                                <i class="fas fa-times text-sm sm:text-lg"></i>
                                                            </button>
                                                            <button
                                                                        v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="resumeTask(task.task_id)"
                                                                        class="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-laser-purple/80 backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                                                                    :title="t('retryTask')">
                                                                        <i class="fas fa-redo text-sm sm:text-lg"></i>
                                                            </button>
                                                                <button v-if="task.status === 'SUCCEED'"
                                                                        @click.stop="handleDownloadTask(task)"
                                                                        class="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-laser-purple/80 backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                                                                    :title="t('downloadTask')">
                                                                        <i class="fas fa-download text-sm sm:text-lg"></i>
                                                            </button>
                                                            <button
                                                                        v-if="['SUCCEED', 'FAILED', 'CANCEL'].includes(task.status)"
                                                                        @click.stop="deleteTask(task.task_id)"
                                                                        class="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-red-400/80 backdrop-blur-sm flex items-center justify-center text-white hover:bg-red-500 transition-colors"
                                                                    :title="t('deleteTask')">
                                                                        <i class="fas fa-trash text-sm sm:text-lg"></i>
                                                            </button>
                                                    </div>
                                                </div>
                                            </div>

                                                        <!-- 任务信息 -->
                                                        <div class="pl-4 pr-4 pb-4">
                                                        <h3 class="text-white font-medium text-sm mb-2 line-clamp-2">{{
                                                            task.params.prompt.length > 20 ? task.params.prompt.slice(0, 20)
                                                            + '···' : task.params.prompt }}</h3>
                                                        <div
                                                            class="flex items-center justify-between text-xs text-gray-500">
                                                                <span>{{ task.model_cls }}</span>
                                                                <span>{{ getRelativeTime(task.create_t) }}</span>
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
