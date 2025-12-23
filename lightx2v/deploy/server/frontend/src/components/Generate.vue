<script setup>
import {
            submitting,
            templateLoading,
            templateLoadingMessage,
            // 任务类型下拉菜单
            showTaskTypeMenu,
            showModelMenu,

            isLoggedIn,
            loading,
            loginLoading,
            initLoading,
            downloadLoading,
            downloadLoadingMessage,

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
            i2iImagePreview,
            flf2vImagePreview,
            flf2vLastFramePreview,
            getCurrentImagePreview,
            getCurrentAudioPreview,
            getCurrentVideoPreview,
            getCurrentLastFramePreview,
            setCurrentImagePreview,
            setCurrentLastFramePreview,
            setCurrentAudioPreview,
            setCurrentVideoPreview,
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
            detectFacesInImage,
            faceDetecting,
            audioSeparating,
            cropFaceImage,
            updateFaceRoleName,
            toggleFaceEditing,
            saveFaceRoleName,
            handleLastFrameUpload,
            selectTask,
            selectModel,
            resetForm,
            triggerImageUpload,
            triggerLastFrameUpload,
            triggerAudioUpload,
            removeImage,
            removeLastFrame,
            removeAudio,
            removeVideo,
            handleAudioUpload,
            handleVideoUpload,
            separateAudioTracks,
            updateSeparatedAudioRole,
            updateSeparatedAudioName,
            toggleSeparatedAudioEditing,
            saveSeparatedAudioName,
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
            getCurrentVideoPreviewUrl,
            getCurrentLastFramePreviewUrl,
            getI2IImagePreviews,
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
            selectLastFrameImageHistory,
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
            applyTemplateLastFrameImage,
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
            openTaskFromRoute,
            showVoiceTTSModal
        } from '../utils/other'

import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { watch, onMounted, computed, ref, nextTick, onUnmounted } from 'vue'
import ModelDropdown from './ModelDropdown.vue'
import TaskCarousel from './TaskCarousel.vue'
import DropdownMenu from './DropdownMenu.vue'

// Props
const props = defineProps({
  query: {
    type: Object,
    default: () => ({})
  }
})

const { t, tm, locale } = useI18n()
const route = useRoute()
const router = useRouter()

// 当前显示的精选模版
const currentFeaturedTemplates = ref([])

// 屏幕尺寸响应式状态
const screenSize = ref('large') // 'small' 或 'large'

// 精选模版瀑布流容器高度控制
const featuredMasonryRef = ref(null)
const baseMasonryHeight = computed(() => (screenSize.value === 'large' ? 600 : 400))
const masonryHeight = ref(baseMasonryHeight.value)

const updateMasonryHeight = () => {
    const container = featuredMasonryRef.value
    if (!container) {
        masonryHeight.value = baseMasonryHeight.value
        return
    }

    const columnEls = container.querySelectorAll('[data-masonry-column]')
    if (!columnEls.length) {
        masonryHeight.value = baseMasonryHeight.value
        return
    }

    let maxBottom = 0
    columnEls.forEach((column) => {
        const bottom = column.offsetTop + column.offsetHeight
        if (bottom > maxBottom) {
            maxBottom = bottom
        }
    })

    masonryHeight.value = Math.max(Math.ceil(maxBottom) + 32, baseMasonryHeight.value)
}

const scheduleMasonryUpdate = () => {
    requestAnimationFrame(() => updateMasonryHeight())
}

// 拖拽状态
const isDragOver = ref(false)

// 音频预览播放器相关
const audioPreviewElement = ref(null)
const audioPreviewIsPlaying = ref(false)
const audioPreviewDuration = ref(0)
const audioPreviewCurrentTime = ref(0)
const audioPreviewIsDragging = ref(false)

// 分离后的音频播放器相关状态
const separatedAudioElements = ref([]) // Array of audio elements
const separatedAudioPlaying = ref({}) // { index: boolean }
const separatedAudioDuration = ref({}) // { index: number }
const separatedAudioCurrentTime = ref({}) // { index: number }
const separatedAudioIsDragging = ref({}) // { index: boolean }

// 拖拽排序相关状态
const draggedRoleIndex = ref(-1)
const draggedAudioIndex = ref(-1)
const dragOverRoleIndex = ref(-1)
const dragOverAudioIndex = ref(-1)
const dragPreviewElement = ref(null)
const dragOffset = ref({ x: 0, y: 0 })

// 计算属性：当前表单检测到的脸
const currentDetectedFaces = computed(() => {
    const form = getCurrentForm()
    return form?.detectedFaces || []
})

// 计算属性：当前表单分离后的音频
const currentSeparatedAudios = computed(() => {
    const form = getCurrentForm()
    const audios = form?.separatedAudios || []
    // Debug log
    if (audios.length > 0) {
        console.log('currentSeparatedAudios computed:', audios.length, 'audios')
    }
    return audios
})

// 计算属性：当前音频预览
const currentAudioPreview = computed(() => {
    return getCurrentAudioPreview()
})

// 记录上次分离的角色个数，避免重复分离
const lastSeparatedFaceCount = ref(0)
const lastSeparatedAudioUrl = ref('')

// 角色模式：单角色/多角色
const isMultiRoleMode = ref(false) // false = 单角色模式, true = 多角色模式（默认关闭，单人模式）

// 监听任务类型变化，切换任务时重置为单人模式
watch(selectedTaskId, (newTaskId, oldTaskId) => {
    if (newTaskId !== oldTaskId) {
        // 切换任务类型时，重置为单人模式
        isMultiRoleMode.value = false
        // 重置分离记录
        lastSeparatedFaceCount.value = 0
        lastSeparatedAudioUrl.value = ''

        // 对于 t2v 和 t2i 任务，如果 customShape 为 null，自动设置默认值
        const form = getCurrentForm()
        if (form && !form.customShape) {
            if (newTaskId === 't2v') {
                // t2v: 使用 720p（横屏）作为默认值
                const defaultResolution = videoResolutions.find(r => r.label === '720p_landscape')
                if (defaultResolution) {
                    form.customShape = defaultResolution.value
                }
            } else if (newTaskId === 't2i') {
                // t2i: 使用 16:9 宽高比作为默认值
                const ratio16_9 = imageAspectRatios.find(r => r.value === '16:9')
                if (ratio16_9) {
                    selectImageAspectRatio(ratio16_9)
                }
            }
        }
    }
})

// 统一监听 detectedFaces 和音频预览，当两者都存在且角色个数 > 1 时，自动分离音频
// 这样可以覆盖所有场景：上传音频、应用历史音频、使用音频模板、复用任务等
watch([currentDetectedFaces, currentAudioPreview, selectedTaskId], ([newFaces, audioUrl, taskType], [oldFaces, oldAudioUrl, oldTaskType]) => {
    // 只在 s2v 任务下处理
    if (taskType !== 's2v') {
        // 如果不是 s2v 任务，清空分离记录
        if (oldTaskType === 's2v') {
            lastSeparatedFaceCount.value = 0
            lastSeparatedAudioUrl.value = ''
        }
        return
    }

    const faceCount = newFaces?.length || 0
    const oldFaceCount = oldFaces?.length || 0

    // 如果角色个数 <= 1，清空分离的音频
    if (faceCount <= 1) {
        if (oldFaceCount > 1) {
            const form = getCurrentForm()
            if (form) {
                form.separatedAudios = []
            }
            lastSeparatedFaceCount.value = 0
            lastSeparatedAudioUrl.value = ''
        }
        return
    }

    // 如果角色个数 > 1 且有音频预览
    if (faceCount > 1 && audioUrl) {
        // 检查是否需要分离（避免重复分离）
        const needsSeparation =
            faceCount !== lastSeparatedFaceCount.value ||
            audioUrl !== lastSeparatedAudioUrl.value

        if (needsSeparation) {
            console.log(`[自动音频分离] 检测到 ${faceCount} 个角色且有音频，开始分离...`, {
                faceCount,
                audioUrl: audioUrl.substring(0, 50) + '...',
                lastSeparatedFaceCount: lastSeparatedFaceCount.value,
                lastSeparatedAudioUrl: lastSeparatedAudioUrl.value?.substring(0, 50) + '...'
            })

            separateAudioTracks(audioUrl, faceCount)
                .then(() => {
                    // 分离成功，更新记录
                    lastSeparatedFaceCount.value = faceCount
                    lastSeparatedAudioUrl.value = audioUrl
                    console.log(`[自动音频分离] 分离成功，角色个数: ${faceCount}`)
                })
                .catch(error => {
                    console.error('[自动音频分离] 分离失败:', error)
                    // 分离失败，不清空记录，允许重试
                })
        } else {
            console.log(`[自动音频分离] 跳过重复分离，角色个数: ${faceCount}，音频未变化`)
        }
    } else if (faceCount > 1 && !audioUrl) {
        // 有多个角色但没有音频，清空分离的音频
        const form = getCurrentForm()
        if (form && form.separatedAudios && form.separatedAudios.length > 0) {
            form.separatedAudios = []
            lastSeparatedFaceCount.value = 0
            lastSeparatedAudioUrl.value = ''
        }
    }
}, { immediate: true })

// 手动切换模式
const toggleRoleMode = async () => {
    if (selectedTaskId.value !== 's2v') return

    const form = getCurrentForm()
    if (!form) return

    const newMode = !isMultiRoleMode.value

    if (newMode) {
        // 切换到多角色模式
        isMultiRoleMode.value = true

        // 如果还没有检测到角色，调用角色识别功能
        if (!form.detectedFaces || form.detectedFaces.length === 0) {
            const imageUrl = getCurrentImagePreviewUrl()
            if (imageUrl) {
                try {
                    faceDetecting.value = true
                    await detectFacesInImage(imageUrl)
                } catch (error) {
                    console.error('Face detection failed:', error)
                    showAlert(t('faceDetectionFailed') + ': ' + (error.message || t('unknownError')), 'error')
                } finally {
                    faceDetecting.value = false
                }
            } else {
                showAlert(t('pleaseUploadImage'), 'warning')
                isMultiRoleMode.value = false
                return
            }
        }

        // 如果检测后仍然只有0个或1个角色，提示用户
        if (!form.detectedFaces || form.detectedFaces.length <= 1) {
            showAlert(t('multiRoleModeRequires'), 'info')
            return
        }

        // 如果有音频，自动分离
        if (form.audioFile && getCurrentAudioPreview()) {
            const audioDataUrl = getCurrentAudioPreview()
            if (audioDataUrl && form.detectedFaces && form.detectedFaces.length > 1) {
                try {
                    await separateAudioTracks(audioDataUrl, form.detectedFaces.length)
                } catch (error) {
                    console.error('Audio separation failed:', error)
                    showAlert(t('audioSeparationFailed') + ': ' + error.message, 'error')
                }
            }
        }
    } else {
        // 切换到单角色模式
        isMultiRoleMode.value = false

        // 清空分离的音频（单角色模式不需要分离）
        form.separatedAudios = []

        // 如果有多于1个角色的情况下切回单模式，提示用户
        if (form.detectedFaces && form.detectedFaces.length > 1) {
            showAlert(t('singleRoleModeInfo'), 'info')
        }
    }
}

// 处理删除图片，同时重置多角色模式
const handleRemoveImage = () => {
    removeImage()
    // 删除图片后，自动切回单角色模式
    if (selectedTaskId.value === 's2v' && isMultiRoleMode.value) {
        isMultiRoleMode.value = false
        const form = getCurrentForm()
        if (form) {
            form.separatedAudios = []
        }
    }
}

// 脸部放大图片编辑相关状态
const showFaceEditModal = ref(false)
const editingFaceIndex = ref(-1)
const editingFaceBbox = ref([0, 0, 0, 0]) // [x1, y1, x2, y2]
const originalImageUrl = ref('')
const imageContainerRef = ref(null)
const imageLoaded = ref(false) // 图片是否已加载完成
const imageNaturalSize = ref({ width: 0, height: 0 }) // 图片原始尺寸
const isDraggingBbox = ref(false)
const dragType = ref('move') // 'move', 'resize-n', 'resize-s', 'resize-w', 'resize-e', 'resize-nw', 'resize-ne', 'resize-sw', 'resize-se'
const dragStartPos = ref({ x: 0, y: 0 })
const dragStartBbox = ref([0, 0, 0, 0]) // 拖拽开始时的bbox坐标
const bboxOffset = ref({ x: 0, y: 0 })
const isAddingNewFace = ref(false) // 是否在新增角色模式
const faceSaving = ref(false) // 是否正在保存角色（用于显示加载状态）
const showRoleModeInfo = ref(false) // 是否显示角色模式说明

// 打开脸部编辑模态框
const openFaceEditModal = async (faceIndex) => {
    const form = getCurrentForm()
    if (!form) return

    originalImageUrl.value = getCurrentImagePreviewUrl()
    imageLoaded.value = false // 重置图片加载状态
    imageNaturalSize.value = { width: 0, height: 0 } // 重置图片尺寸

    // 如果是新增模式（faceIndex 为 -1）
    if (faceIndex === -1) {
        isAddingNewFace.value = true
        editingFaceIndex.value = -1
        showFaceEditModal.value = true

        // 等待DOM更新，确保图片元素已渲染
        await nextTick()
        await nextTick() // 多等待一次，确保图片元素完全渲染

        // 等待图片加载完成
        const img = imageContainerRef.value?.querySelector('img')
        if (img) {
            // 如果图片已经加载完成（从缓存），立即设置状态
            if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                imageLoaded.value = true
            } else {
                // 确保图片完全加载
                await new Promise((resolve) => {
                    if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                        imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                        imageLoaded.value = true
                        resolve()
                    } else {
                        const onLoad = () => {
                            // 确保图片尺寸已正确设置
                            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                                imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                                imageLoaded.value = true
                                img.removeEventListener('load', onLoad)
                                img.removeEventListener('error', onError)
                                resolve()
                            }
                        }
                        const onError = () => {
                            imageLoaded.value = true
                            img.removeEventListener('load', onLoad)
                            img.removeEventListener('error', onError)
                            resolve() // 即使加载失败也继续
                        }
                        img.addEventListener('load', onLoad)
                        img.addEventListener('error', onError)
                    }
                })
            }

            // 再次等待，确保图片尺寸已正确设置
            await nextTick()

            // 计算图片的原始尺寸
            const imgNaturalWidth = img.naturalWidth
            const imgNaturalHeight = img.naturalHeight

            if (imgNaturalWidth > 0 && imgNaturalHeight > 0) {
                imageNaturalSize.value = { width: imgNaturalWidth, height: imgNaturalHeight }
                // 默认居中，大小为图片的 30%
                const bboxSize = Math.min(imgNaturalWidth, imgNaturalHeight) * 0.3
                const centerX = imgNaturalWidth / 2
                const centerY = imgNaturalHeight / 2

                editingFaceBbox.value = [
                    centerX - bboxSize / 2,
                    centerY - bboxSize / 2,
                    centerX + bboxSize / 2,
                    centerY + bboxSize / 2
                ]

                // 标记图片已加载
                imageLoaded.value = true
                // 再次等待DOM更新，确保边界框已渲染
                await nextTick()
            } else {
                // 如果图片尺寸无效，使用默认值
                editingFaceBbox.value = [0, 0, 100, 100]
                imageLoaded.value = true
                await nextTick()
            }
        } else {
            // 如果图片还没加载，使用默认值
            editingFaceBbox.value = [0, 0, 100, 100]
            imageLoaded.value = true
            await nextTick()
        }
    } else {
        // 编辑现有角色
        if (!form.detectedFaces || !form.detectedFaces[faceIndex]) return
        isAddingNewFace.value = false
        const face = form.detectedFaces[faceIndex]
        editingFaceIndex.value = faceIndex
        editingFaceBbox.value = [...(face.bbox || [0, 0, 0, 0])]
        showFaceEditModal.value = true

        // 等待DOM更新，确保图片元素已渲染
        await nextTick()
        await nextTick() // 多等待一次，确保图片元素完全渲染

        // 等待图片加载完成，确保边界框能正确显示
        const img = imageContainerRef.value?.querySelector('img')
        if (img) {
            // 如果图片已经加载完成（从缓存），立即设置状态
            if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                imageLoaded.value = true
            } else {
                // 确保图片完全加载
                await new Promise((resolve) => {
                    if (img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                        imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                        imageLoaded.value = true
                        resolve()
                    } else {
                        const onLoad = () => {
                            // 确保图片尺寸已正确设置
                            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                                imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
                                imageLoaded.value = true
                                img.removeEventListener('load', onLoad)
                                img.removeEventListener('error', onError)
                                resolve()
                            }
                        }
                        const onError = () => {
                            imageLoaded.value = true
                            img.removeEventListener('load', onLoad)
                            img.removeEventListener('error', onError)
                            resolve() // 即使加载失败也继续
                        }
                        img.addEventListener('load', onLoad)
                        img.addEventListener('error', onError)
                    }
                })
            }

            // 再次等待，确保图片尺寸已正确设置
            await nextTick()
        } else {
            imageLoaded.value = true
        }
    }
}

// 处理脸部编辑模态框中的图片加载
const handleFaceEditImageLoad = () => {
    const img = imageContainerRef.value?.querySelector('img')
    if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
        imageNaturalSize.value = { width: img.naturalWidth, height: img.naturalHeight }
        imageLoaded.value = true
        nextTick()
    }
}

// 处理脸部编辑模态框中的图片加载错误
const handleFaceEditImageError = () => {
    imageLoaded.value = true // 即使加载失败也显示，避免一直显示加载中
}

// 关闭脸部编辑模态框
const closeFaceEditModal = () => {
    showFaceEditModal.value = false
    editingFaceIndex.value = -1
    isDraggingBbox.value = false
    isAddingNewFace.value = false
    imageLoaded.value = false
    imageNaturalSize.value = { width: 0, height: 0 }
}

// 保存边界框更改
const saveFaceBbox = async () => {
    const form = getCurrentForm()
    if (!form) return

    // 保存当前状态（在关闭模态框之前）
    const wasAddingNewFace = isAddingNewFace.value
    const currentEditingIndex = editingFaceIndex.value // 保存编辑索引
    const currentBbox = [...editingFaceBbox.value] // 保存边界框
    const currentImageUrl = originalImageUrl.value // 保存图片URL

    // 立即关闭模态框
    closeFaceEditModal()

    // 如果是新增模式，显示加载状态
    if (wasAddingNewFace) {
        faceSaving.value = true
    }

    try {
        // 如果是新增模式
        if (wasAddingNewFace) {
            if (!form.detectedFaces) {
                form.detectedFaces = []
            }

            // 创建新角色
            const newFaceIndex = form.detectedFaces.length
            const newFace = {
                bbox: [...currentBbox],
                roleName: `角色${newFaceIndex + 1}`,
                roleIndex: newFaceIndex,
                isEditing: false,
                face_image: null
            }

            // 根据新的 bbox 坐标，从原始图片裁剪出新的 face_image
            try {
                let imageUrl = currentImageUrl
                if (imageUrl.startsWith('data:image')) {
                    // 保持 data URL 格式，可以直接使用
                } else if (!imageUrl.startsWith('http') && !imageUrl.startsWith('/')) {
                    imageUrl = currentImageUrl
                }

                // 裁剪出新的 face_image
                const croppedImage = await cropFaceImage(imageUrl, newFace.bbox)

                // 移除 data URL 前缀，只保留 base64 部分（与后端返回的格式一致）
                const base64Data = croppedImage.split(',')[1] || croppedImage
                newFace.face_image = base64Data

            } catch (error) {
                console.error('Failed to crop face image:', error)
            }

            // 添加到角色列表
            form.detectedFaces.push(newFace)
            // 触发响应式更新
            form.detectedFaces = [...form.detectedFaces]

            // 如果是在 s2v 模式下且有上传的音频，自动重新分割音频
            if (selectedTaskId.value === 's2v' && getCurrentAudioPreview()) {
                try {
                    const audioDataUrl = getCurrentAudioPreview()
                    await separateAudioTracks(audioDataUrl, form.detectedFaces.length)
                } catch (error) {
                    console.error('Failed to re-separate audio after adding face:', error)
                }
            }

        } else {
            // 编辑现有角色
            if (!form.detectedFaces || currentEditingIndex < 0 || !form.detectedFaces[currentEditingIndex]) {
                console.error('Invalid editing index or face not found:', currentEditingIndex)
                return
            }

            const face = form.detectedFaces[currentEditingIndex]

            // editingFaceBbox.value 存储的是原始图片坐标 [x1, y1, x2, y2]
            face.bbox = [...currentBbox]

            // 根据新的 bbox 坐标，从原始图片裁剪出新的 face_image
            try {
                let imageUrl = currentImageUrl
                if (imageUrl.startsWith('data:image')) {
                    // 保持 data URL 格式，可以直接使用
                } else if (!imageUrl.startsWith('http') && !imageUrl.startsWith('/')) {
                    imageUrl = currentImageUrl
                }

                // 裁剪出新的 face_image
                const croppedImage = await cropFaceImage(imageUrl, face.bbox)

                // 移除 data URL 前缀，只保留 base64 部分（与后端返回的格式一致）
                const base64Data = croppedImage.split(',')[1] || croppedImage
                face.face_image = base64Data

            } catch (error) {
                console.error('Failed to crop face image:', error)
            }

            // 触发响应式更新
            form.detectedFaces = [...form.detectedFaces]
        }
    } finally {
        // 隐藏加载状态
        faceSaving.value = false
    }
}

// 删除角色
const removeFace = async (faceIndex) => {
    const form = getCurrentForm()
    if (!form || !form.detectedFaces || faceIndex < 0 || faceIndex >= form.detectedFaces.length) return

    // 从角色列表中删除
    form.detectedFaces.splice(faceIndex, 1)
    // 触发响应式更新
    form.detectedFaces = [...form.detectedFaces]

    // 如果是在 s2v 模式下且有上传的音频，自动重新分割音频
    if (selectedTaskId.value === 's2v' && getCurrentAudioPreview() && form.detectedFaces.length > 0) {
        try {
            const audioDataUrl = getCurrentAudioPreview()
            await separateAudioTracks(audioDataUrl, form.detectedFaces.length)
        } catch (error) {
            console.error('Failed to re-separate audio after removing face:', error)
        }
    } else if (selectedTaskId.value === 's2v') {
        // 如果没有角色了，清空分离的音频
        s2vForm.value.separatedAudios = []
    }
}

// 获取图片缩放比例
const getImageScale = () => {
    const container = imageContainerRef.value
    if (!container) return { scaleX: 1, scaleY: 1, imgWidth: 0, imgHeight: 0 }

    const img = container.querySelector('img')
    if (!img || !img.complete) return { scaleX: 1, scaleY: 1, imgWidth: 0, imgHeight: 0 }

    const imgRect = img.getBoundingClientRect()
    const scaleX = img.naturalWidth > 0 ? imgRect.width / img.naturalWidth : 1
    const scaleY = img.naturalHeight > 0 ? imgRect.height / img.naturalHeight : 1

    return { scaleX, scaleY, imgWidth: imgRect.width, imgHeight: imgRect.height }
}

// 获取图片相对于容器的偏移
const getImageOffset = () => {
    const container = imageContainerRef.value
    if (!container) return { offsetX: 0, offsetY: 0 }

    const img = container.querySelector('img')
    if (!img) return { offsetX: 0, offsetY: 0 }

    const containerRect = container.getBoundingClientRect()
    const imgRect = img.getBoundingClientRect()

    return {
        offsetX: imgRect.left - containerRect.left,
        offsetY: imgRect.top - containerRect.top
    }
}

// 开始拖拽边界框
const startDragBbox = (event, type = 'move') => {
    event.preventDefault()
    event.stopPropagation()

    const container = imageContainerRef.value
    if (!container) return

    const img = container.querySelector('img')
    if (!img) return

    // 获取图片的实际显示尺寸和原始尺寸
    const imgRect = img.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const displayWidth = imgRect.width
    const displayHeight = imgRect.height
    const naturalWidth = img.naturalWidth
    const naturalHeight = img.naturalHeight

    // 检查图片是否已加载（通过尺寸判断，而不是 complete 属性）
    // 因为 complete 可能在图片尺寸设置之前就为 true
    if (naturalWidth === 0 || naturalHeight === 0 || displayWidth === 0 || displayHeight === 0) {
        console.warn('Image not ready for dragging:', { naturalWidth, naturalHeight, displayWidth, displayHeight, complete: img.complete })
        return
    }

    // 检查 #app 是否有 transform: scale
    const appElement = document.getElementById('app')
    let appScale = 1
    if (appElement) {
        const appStyle = window.getComputedStyle(appElement)
        const transform = appStyle.transform
        if (transform && transform !== 'none') {
            const matrix = transform.match(/matrix\(([^)]+)\)/)
            if (matrix) {
                const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
                appScale = values[0] || 1
            } else {
                const scaleMatch = transform.match(/scale\(([^)]+)\)/)
                if (scaleMatch) {
                    appScale = parseFloat(scaleMatch[1])
                }
            }
        }
    }

    // 计算缩放比例（补偿 #app 的缩放）
    const scaleX = displayWidth / (naturalWidth * appScale)
    const scaleY = displayHeight / (naturalHeight * appScale)

    // 图片在容器中的偏移
    const offsetX = imgRect.left - containerRect.left
    const offsetY = imgRect.top - containerRect.top

    // 边界框坐标
    const [x1, y1, x2, y2] = editingFaceBbox.value

    // 计算边界框在容器中的显示位置
    const bboxRect = {
        left: offsetX + x1 * scaleX,
        top: offsetY + y1 * scaleY,
        right: offsetX + x2 * scaleX,
        bottom: offsetY + y2 * scaleY
    }

    // 点击位置相对于容器
    const clickX = event.clientX - containerRect.left
    const clickY = event.clientY - containerRect.top

    // 如果是拖拽手柄（type 不是 'move'），直接开始拖拽
    if (type !== 'move') {
        isDraggingBbox.value = true
        dragType.value = type
        dragStartPos.value = { x: clickX, y: clickY }
        dragStartBbox.value = [...editingFaceBbox.value]
        return
    }

    // 检查点击是否在边界框内（移动模式）
    if (clickX < bboxRect.left || clickX > bboxRect.right ||
        clickY < bboxRect.top || clickY > bboxRect.bottom) {
        return
    }

    isDraggingBbox.value = true
    dragType.value = 'move'
    dragStartPos.value = { x: clickX, y: clickY }
    dragStartBbox.value = [...editingFaceBbox.value]
}

// 拖拽边界框
const dragBbox = (event) => {
    if (!isDraggingBbox.value) return

    const container = imageContainerRef.value
    if (!container) return

    const img = container.querySelector('img')
    if (!img || !img.complete) return

    // 获取图片的实际显示尺寸和原始尺寸
    const imgRect = img.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()
    const displayWidth = imgRect.width
    const displayHeight = imgRect.height
    const naturalWidth = img.naturalWidth
    const naturalHeight = img.naturalHeight

    if (naturalWidth === 0 || naturalHeight === 0) return

    // 检查 #app 是否有 transform: scale
    const appElement = document.getElementById('app')
    let appScale = 1
    if (appElement) {
        const appStyle = window.getComputedStyle(appElement)
        const transform = appStyle.transform
        if (transform && transform !== 'none') {
            const matrix = transform.match(/matrix\(([^)]+)\)/)
            if (matrix) {
                const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
                appScale = values[0] || 1
            } else {
                const scaleMatch = transform.match(/scale\(([^)]+)\)/)
                if (scaleMatch) {
                    appScale = parseFloat(scaleMatch[1])
                }
            }
        }
    }

    // 计算缩放比例（补偿 #app 的缩放）
    // displayWidth 已经是经过 appScale 缩放后的尺寸，所以需要除以 appScale 来得到相对于原始图片的缩放比例
    const scaleX = displayWidth / (naturalWidth * appScale)
    const scaleY = displayHeight / (naturalHeight * appScale)

    // 图片在容器中的偏移
    const offsetX = imgRect.left - containerRect.left
    const offsetY = imgRect.top - containerRect.top

    // 鼠标当前位置相对于容器
    const containerRect2 = container.getBoundingClientRect()
    const currentX = event.clientX - containerRect2.left
    const currentY = event.clientY - containerRect2.top

    // 将坐标转换为相对于图片的位置（考虑图片在容器中的偏移）
    const imgCurrentX = currentX - offsetX
    const imgCurrentY = currentY - offsetY
    const imgStartX = dragStartPos.value.x - offsetX
    const imgStartY = dragStartPos.value.y - offsetY

    const deltaX = (imgCurrentX - imgStartX) / scaleX
    const deltaY = (imgCurrentY - imgStartY) / scaleY

    // 获取拖拽开始时的bbox
    const [startX1, startY1, startX2, startY2] = dragStartBbox.value
    const startWidth = startX2 - startX1
    const startHeight = startY2 - startY1

    let newX1 = startX1
    let newY1 = startY1
    let newX2 = startX2
    let newY2 = startY2

    // 根据拖拽类型调整bbox
    const type = dragType.value
    if (type === 'move') {
        // 移动模式：整体移动
        newX1 = startX1 + deltaX
        newY1 = startY1 + deltaY
        newX2 = startX2 + deltaX
        newY2 = startY2 + deltaY
    } else if (type === 'resize-n') {
        // 调整顶部
        newY1 = Math.min(startY1 + deltaY, startY2 - 10) // 最小高度10px
        newX1 = startX1
        newX2 = startX2
        newY2 = startY2
    } else if (type === 'resize-s') {
        // 调整底部
        newY2 = Math.max(startY2 + deltaY, startY1 + 10) // 最小高度10px
        newX1 = startX1
        newY1 = startY1
        newX2 = startX2
    } else if (type === 'resize-w') {
        // 调整左侧
        newX1 = Math.min(startX1 + deltaX, startX2 - 10) // 最小宽度10px
        newY1 = startY1
        newX2 = startX2
        newY2 = startY2
    } else if (type === 'resize-e') {
        // 调整右侧
        newX2 = Math.max(startX2 + deltaX, startX1 + 10) // 最小宽度10px
        newX1 = startX1
        newY1 = startY1
        newY2 = startY2
    } else if (type === 'resize-nw') {
        // 调整左上角
        newX1 = Math.min(startX1 + deltaX, startX2 - 10)
        newY1 = Math.min(startY1 + deltaY, startY2 - 10)
        newX2 = startX2
        newY2 = startY2
    } else if (type === 'resize-ne') {
        // 调整右上角
        newX2 = Math.max(startX2 + deltaX, startX1 + 10)
        newY1 = Math.min(startY1 + deltaY, startY2 - 10)
        newX1 = startX1
        newY2 = startY2
    } else if (type === 'resize-sw') {
        // 调整左下角
        newX1 = Math.min(startX1 + deltaX, startX2 - 10)
        newY2 = Math.max(startY2 + deltaY, startY1 + 10)
        newX2 = startX2
        newY1 = startY1
    } else if (type === 'resize-se') {
        // 调整右下角
        newX2 = Math.max(startX2 + deltaX, startX1 + 10)
        newY2 = Math.max(startY2 + deltaY, startY1 + 10)
        newX1 = startX1
        newY1 = startY1
    }

    // 边界限制：确保bbox在图片范围内
    const minSize = 10 // 最小尺寸

    // X方向边界限制
    if (newX1 < 0) {
        newX1 = 0
        if (type.includes('w') || type === 'resize-nw' || type === 'resize-sw') {
            // 如果是调整左边或左角，需要保持宽度
            newX2 = Math.max(newX2, minSize)
        }
    }
    if (newX2 > naturalWidth) {
        newX2 = naturalWidth
        if (type.includes('e') || type === 'resize-ne' || type === 'resize-se') {
            // 如果是调整右边或右角，需要保持宽度
            newX1 = Math.min(newX1, naturalWidth - minSize)
        }
    }

    // Y方向边界限制
    if (newY1 < 0) {
        newY1 = 0
        if (type.includes('n') || type === 'resize-nw' || type === 'resize-ne') {
            // 如果是调整上边或上角，需要保持高度
            newY2 = Math.max(newY2, minSize)
        }
    }
    if (newY2 > naturalHeight) {
        newY2 = naturalHeight
        if (type.includes('s') || type === 'resize-sw' || type === 'resize-se') {
            // 如果是调整下边或下角，需要保持高度
            newY1 = Math.min(newY1, naturalHeight - minSize)
        }
    }

    // 确保最小尺寸
    if (newX2 - newX1 < minSize) {
        if (type.includes('w') || type === 'resize-nw' || type === 'resize-sw') {
            newX1 = newX2 - minSize
        } else {
            newX2 = newX1 + minSize
        }
    }
    if (newY2 - newY1 < minSize) {
        if (type.includes('n') || type === 'resize-nw' || type === 'resize-ne') {
            newY1 = newY2 - minSize
        } else {
            newY2 = newY1 + minSize
        }
    }

    // 更新边界框坐标
    editingFaceBbox.value = [newX1, newY1, newX2, newY2]
}

// 结束拖拽
const endDragBbox = () => {
    isDraggingBbox.value = false
    dragType.value = 'move'
}

// 计算边界框的样式（用于在放大图片上显示）
const getBboxStyle = computed(() => {
    if (!imageContainerRef.value || editingFaceBbox.value.length !== 4 || !imageLoaded.value) {
        return {}
    }

    const container = imageContainerRef.value
    const img = container.querySelector('img')
    if (!img || !img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
        return {}
    }

    // 获取图片的实际显示尺寸
    // getBoundingClientRect() 返回的是相对于 viewport 的坐标
    // 如果 #app 有 transform: scale(0.8)，那么所有元素都会被缩放 0.8
    const imgRect = img.getBoundingClientRect()
    const containerRect = container.getBoundingClientRect()

    // 图片的实际显示尺寸（已考虑所有CSS样式和可能的缩放，包括 #app 的 0.8 缩放）
    const displayWidth = imgRect.width
    const displayHeight = imgRect.height

    // 图片的原始尺寸（naturalWidth/naturalHeight 是图片文件的真实尺寸）
    const naturalWidth = img.naturalWidth
    const naturalHeight = img.naturalHeight

    if (naturalWidth === 0 || naturalHeight === 0) {
        return {}
    }

    // 检查 #app 是否有 transform: scale
    // 如果模态框在 #app 内部，会受到 #app 的 transform 影响
    const appElement = document.getElementById('app')
    let appScale = 1
    if (appElement) {
        const appStyle = window.getComputedStyle(appElement)
        const transform = appStyle.transform
        if (transform && transform !== 'none') {
            // 解析 transform matrix 或 scale
            const matrix = transform.match(/matrix\(([^)]+)\)/)
            if (matrix) {
                const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
                // matrix(a, b, c, d, tx, ty) 中，a 和 d 是缩放值
                appScale = values[0] || 1
            } else {
                const scaleMatch = transform.match(/scale\(([^)]+)\)/)
                if (scaleMatch) {
                    appScale = parseFloat(scaleMatch[1])
                }
            }
        }
    }

    // 计算缩放比例
    // displayWidth 已经是经过 appScale 缩放后的尺寸
    // 所以相对于原始图片的实际缩放比例是 displayWidth / naturalWidth
    // 但由于 #app 的缩放，边界框在模态框中的尺寸需要补偿这个缩放
    // 如果 appScale = 0.8，那么边界框的尺寸应该是 displayWidth / appScale / naturalWidth = displayWidth / (naturalWidth * appScale)
    const scaleX = displayWidth / (naturalWidth * appScale)
    const scaleY = displayHeight / (naturalHeight * appScale)

    // 图片在容器中的偏移（相对于容器）
    // 如果容器是 inline-block，图片和容器可能在同一位置
    // 我们需要检查容器是否包裹了图片，或者图片就是容器的唯一内容
    let offsetX = imgRect.left - containerRect.left
    let offsetY = imgRect.top - containerRect.top

    // 如果计算出的偏移很小（可能是浮点数误差），或者容器和图片尺寸相同，说明图片填充了整个容器
    // 在这种情况下，offset 应该为 0
    if (Math.abs(offsetX) < 1 && Math.abs(offsetY) < 1) {
        offsetX = 0
        offsetY = 0
    }

    // 边界框坐标（原始图片坐标 [x1, y1, x2, y2]）
    // 这些坐标是基于原始图片尺寸的绝对像素坐标
    const [x1, y1, x2, y2] = editingFaceBbox.value

    // 转换为显示坐标
    // bbox坐标是基于原始图片尺寸的，需要乘以缩放比例得到显示尺寸
    // 注意：这里计算的是边界框在容器中的位置和尺寸
    const left = offsetX + x1 * scaleX
    const top = offsetY + y1 * scaleY
    const width = (x2 - x1) * scaleX
    const height = (y2 - y1) * scaleY

    const indicatorSize = 12

    // 确保边界框的尺寸计算正确（考虑 border 的影响）
    // border-2 = 2px，左右各2px，所以总宽度需要包含 border
    // 但由于使用了 box-sizing: border-box，所以不需要额外调整

    return {
        left: `${left}px`,
        top: `${top}px`,
        width: `${width}px`,
        height: `${height}px`,
        indicatorSize: indicatorSize,
        boxSizing: 'border-box'
    }
})

// 计算角色名字标签的样式（显示在边界框上方）
const getRoleNameLabelStyle = computed(() => {
    const bboxStyle = getBboxStyle.value
    if (!bboxStyle.left || !bboxStyle.top) {
        return {}
    }

    // 获取当前编辑的人脸信息
    const form = getCurrentForm()
    let roleName

    // 如果是新增模式
    if (isAddingNewFace.value) {
        // 计算新角色的序号：当前角色数量 + 1
        const newRoleIndex = (form?.detectedFaces?.length || 0) + 1
        roleName = `角色${newRoleIndex}`
    } else {
        // 编辑现有角色
        const face = form?.detectedFaces?.[editingFaceIndex.value]
        roleName = face?.roleName || `角色${editingFaceIndex.value + 1}`
    }

    // 计算标签位置：在边界框上方居中
    const left = parseFloat(bboxStyle.left) || 0
    const top = parseFloat(bboxStyle.top) || 0
    const width = parseFloat(bboxStyle.width) || 0

    // 标签在边界框上方，水平居中
    const labelLeft = left + width / 2

    return {
        left: `${labelLeft}px`,
        top: `${top - 28}px`,  // 在边界框上方 28px
        transform: 'translateX(-50%)',  // 水平居中
        roleName: roleName
    }
})


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

const handleMasonryVideoLoaded = (event) => {
    onVideoLoaded(event)
    scheduleMasonryUpdate()
}

const handleMasonryVideoEnded = (event) => {
    onVideoEnded(event)
    scheduleMasonryUpdate()
}

const handleMasonryVideoError = (event) => {
    onVideoError(event)
    scheduleMasonryUpdate()
}

const handleMasonryImageLoaded = () => {
    scheduleMasonryUpdate()
}

const handleMasonryImageError = (event) => {
    handleThumbnailError(event)
    scheduleMasonryUpdate()
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
    showAlert(t('ttsCompleted'), 'success')
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

watch(currentFeaturedTemplates, async () => {
    masonryHeight.value = baseMasonryHeight.value
    await nextTick()
    scheduleMasonryUpdate()
}, { deep: true })

watch(templatesWithRandomColumns, async () => {
    await nextTick()
    scheduleMasonryUpdate()
})

watch(baseMasonryHeight, (value) => {
    masonryHeight.value = value
    scheduleMasonryUpdate()
})

// 屏幕尺寸监听器
const updateScreenSize = () => {
    screenSize.value = window.innerWidth >= 1024 ? 'large' : 'small'
}

// 监听屏幕尺寸变化
let resizeHandler = null

// 路由监听和URL同步
// 标记是否正在更新 URL，避免循环更新
let isUpdatingUrl = false
// 标记是否正在从路由恢复状态
let isRestoringFromRoute = false

// 存储待处理的路由参数（当 availableTaskTypes 还未加载完成时）
let pendingRouteRestore = null

// 处理路由参数恢复的函数
const restoreFromRoute = (newQuery, oldQuery) => {
    // 如果正在更新 URL，跳过处理，避免循环更新
    if (isUpdatingUrl) {
        return
    }

    // 如果 availableTaskTypes 还没有加载完成，保存参数等待处理
    if (availableTaskTypes.value.length === 0) {
        pendingRouteRestore = { newQuery, oldQuery }
        return
    }

    // 标记正在从路由恢复状态
    isRestoringFromRoute = true

    // 同步URL参数到组件状态
    // 首次加载时（oldQuery 为 undefined），或者参数真正变化时才更新
    const isInitialLoad = !oldQuery

    // 处理任务类型
    if (newQuery.taskType) {
        const taskType = newQuery.taskType
        const shouldUpdate = isInitialLoad || (newQuery.taskType !== oldQuery?.taskType)
        // availableTaskTypes 是字符串数组，不是对象数组
        if (shouldUpdate && availableTaskTypes.value.includes(taskType)) {
            // 如果当前任务类型不匹配，执行 selectTask
            // 在首次加载时，即使值已经匹配，也执行 selectTask 以确保所有相关状态正确设置
            if (selectedTaskId.value !== taskType || isInitialLoad) {
                selectTask(taskType)

                // 等待 selectTask 完成后再处理 model（因为 selectTask 可能会改变 availableModelClasses）
                // 使用 nextTick 确保 selectTask 的副作用已完成
                nextTick(() => {
                    // 再次等待，确保 availableModelClasses 已更新
                    setTimeout(() => {
                        // 处理模型（在任务类型设置后）
                        if (newQuery.model) {
                            const model = newQuery.model
                            const shouldUpdateModel = isInitialLoad || (newQuery.model !== oldQuery?.model)
                            // availableModelClasses 是字符串数组，不是对象数组
                            // 在首次加载时，即使值已经匹配，也执行 selectModel 以确保所有相关状态正确设置
                            if (shouldUpdateModel && availableModelClasses.value.includes(model) && (selectedModel.value !== model || isInitialLoad)) {
                                selectModel(model)
                            }
                        }
                    }, 100)
                })
            }
        } else if (!shouldUpdate && newQuery.model) {
            // 如果任务类型没有变化，但需要更新模型
            const model = newQuery.model
            const shouldUpdateModel = isInitialLoad || (newQuery.model !== oldQuery?.model)
            // availableModelClasses 是字符串数组，不是对象数组
            // 在首次加载时，即使值已经匹配，也执行 selectModel 以确保所有相关状态正确设置
            if (shouldUpdateModel && availableModelClasses.value.includes(model) && (selectedModel.value !== model || isInitialLoad)) {
                selectModel(model)
            }
        }
    } else if (newQuery.model && selectedTaskId.value) {
        // 如果没有任务类型参数，但任务类型已经设置，直接处理模型
        const model = newQuery.model
        const shouldUpdate = isInitialLoad || (newQuery.model !== oldQuery?.model)
        // availableModelClasses 是字符串数组，不是对象数组
        // 在首次加载时，即使值已经匹配，也执行 selectModel 以确保所有相关状态正确设置
        if (shouldUpdate && availableModelClasses.value.includes(model) && (selectedModel.value !== model || isInitialLoad)) {
            selectModel(model)
        }
    }

    // 处理 expanded 状态
    const shouldBeExpanded = newQuery.expanded === 'true'
    if (shouldBeExpanded && !isCreationAreaExpanded.value) {
        // 展开创建区域
        expandCreationArea()
    } else if (!shouldBeExpanded && isCreationAreaExpanded.value && (isInitialLoad || oldQuery?.expanded === 'true')) {
        // 如果 URL 中 expanded 从 'true' 变为其他值，收缩创建区域
        contractCreationArea()
    }

    // 恢复状态完成，使用 setTimeout 确保所有状态更新完成后再重置标志
    setTimeout(() => {
        isRestoringFromRoute = false
    }, 200)
}

// 监听 availableTaskTypes，当它加载完成后处理待处理的路由恢复
watch(availableTaskTypes, (newVal) => {
    if (newVal && newVal.length > 0 && pendingRouteRestore) {
        // availableTaskTypes 加载完成，处理待处理的路由恢复
        const { newQuery, oldQuery } = pendingRouteRestore
        pendingRouteRestore = null
        restoreFromRoute(newQuery, oldQuery)
    }
}, { immediate: true })

watch(() => route.query, (newQuery, oldQuery) => {
    restoreFromRoute(newQuery, oldQuery)
    // 注意：分享数据导入功能已移至 Share.vue 中的 createSimilar 函数
    // 这里不再需要处理分享数据导入
}, { immediate: true })

// 监听组件状态变化，同步到URL
watch([selectedTaskId, isCreationAreaExpanded, selectedModel], (newVals, oldVals) => {
    // 如果正在更新 URL 或正在从路由恢复状态，跳过处理，避免循环更新
    if (isUpdatingUrl || isRestoringFromRoute) {
        return
    }

    // 检查任务类型是否变化
    const taskTypeChanged = oldVals && oldVals[0] !== newVals[0]

    // 如果任务类型变化，检查当前模型是否属于新任务类型
    if (taskTypeChanged && selectedTaskId.value && selectedModel.value) {
        const isModelValid = availableModelClasses.value.includes(selectedModel.value)
        // 如果模型不属于新任务类型，延迟更新路由，等待模型更新完成
        if (!isModelValid) {
            setTimeout(() => {
                // 再次检查，确保模型已经更新
                if (!isUpdatingUrl && !isRestoringFromRoute) {
                    updateRouteFromState()
                }
            }, 150)
            return
        }
    }

    updateRouteFromState()
}, { deep: true })

// 更新路由的函数
const updateRouteFromState = () => {
    // 如果正在更新 URL 或正在从路由恢复状态，跳过处理，避免循环更新
    if (isUpdatingUrl || isRestoringFromRoute) {
        return
    }

    // 获取当前查询参数，保留其他参数（如分享相关的参数）
    const currentQuery = { ...route.query }
    const query = {}

    // 只更新我们关心的参数
    if (selectedTaskId.value) {
        query.taskType = selectedTaskId.value
    } else {
        // 如果任务类型被清除，也从 URL 中移除
        delete currentQuery.taskType
    }

    if (isCreationAreaExpanded.value) {
        query.expanded = 'true'
    } else {
        // 如果创作区域收缩，从 URL 中移除 expanded 参数
        delete currentQuery.expanded
    }

    if (selectedModel.value) {
        query.model = selectedModel.value
    } else {
        // 如果模型被清除，也从 URL 中移除
        delete currentQuery.model
    }

    // 合并查询参数，保留其他参数
    const finalQuery = { ...currentQuery, ...query }

    // 检查是否需要更新 URL（避免不必要的更新）
    const needsUpdate =
        finalQuery.taskType !== route.query.taskType ||
        finalQuery.expanded !== route.query.expanded ||
        finalQuery.model !== route.query.model

    if (needsUpdate) {
        isUpdatingUrl = true
        // 更新URL但不触发路由监听（使用 replace 而不是 push，避免历史记录堆积）
        router.replace({ query: finalQuery }).finally(() => {
            // 使用 nextTick 确保路由更新完成后再重置标志
            nextTick(() => {
                isUpdatingUrl = false
            })
        })
    }
}


// 组件挂载时初始化
onMounted(async () => {
    // 注意：watch route.query 已经使用 immediate: true 处理了 URL 参数的恢复
    // 这里不需要再次处理，避免重复执行
    // 如果需要额外的初始化逻辑，可以在这里添加

    // 初始化屏幕尺寸
    updateScreenSize()

    // 添加屏幕尺寸监听器
    resizeHandler = () => {
        updateScreenSize()
        scheduleMasonryUpdate()
    }
    window.addEventListener('resize', resizeHandler)

    // 添加全局鼠标事件监听用于拖拽边界框
    document.addEventListener('mousemove', dragBbox)
    document.addEventListener('mouseup', endDragBbox)

    // 加载精选模版数据
    await loadFeaturedTemplates(true)
    // 获取随机精选模版
    const randomTemplates = await getRandomFeaturedTemplates(10) // 获取10个模版
    currentFeaturedTemplates.value = randomTemplates
    await nextTick()
    scheduleMasonryUpdate()
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
        showAlert(t('imageDragSuccess'), 'success')
    } else {
        showAlert(t('pleaseDragImage'), 'warning')
    }
}

const handleLastFrameDrop = (e) => {
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

        handleLastFrameUpload(event)
        showAlert('尾帧图片拖拽上传成功', 'success')
    } else {
        showAlert('请拖拽图片文件', 'warning')
    }
}

const handleAudioDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    isDragOver.value = false

    const files = Array.from(e.dataTransfer.files)
    const audioFile = files.find(file => file.type.startsWith('audio/') || file.type.startsWith('video/'))

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
        showAlert(t('audioDragSuccess'), 'success')
    } else {
        showAlert(t('pleaseDragAudio'), 'warning')
    }
}

// 触发视频上传
const triggerVideoUpload = () => {
    // 使用 nextTick 确保 DOM 已更新
    nextTick(() => {
        const videoInput = document.querySelector('input[type="file"][data-role="video-input"]')
        if (videoInput) {
            videoInput.click()
        } else {
            console.warn('视频输入框未找到，请确保已选择 animate 任务类型')
        }
    })
}

// 处理视频拖拽上传
const handleVideoDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    isDragOver.value = false

    const files = Array.from(e.dataTransfer.files)
    const videoFile = files.find(file => file.type.startsWith('video/'))

    if (videoFile) {
        // 创建FileList对象来模拟input[type="file"]的change事件
        const dataTransfer = new DataTransfer()
        dataTransfer.items.add(videoFile)
        const fileList = dataTransfer.files

        // 创建模拟的change事件
        const event = {
            target: {
                files: fileList
            }
        }

        handleVideoUpload(event)
        showAlert(t('videoDragSuccess'), 'success')
    } else {
        showAlert(t('pleaseDragVideo'), 'warning')
    }
}

// 格式化音频预览时间
const formatAudioPreviewTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

// 切换音频预览播放/暂停
const toggleAudioPreviewPlayback = () => {
    if (!audioPreviewElement.value) return

    if (audioPreviewElement.value.paused) {
        audioPreviewElement.value.play().catch(error => {
            console.log('播放失败:', error)
        })
    } else {
        audioPreviewElement.value.pause()
    }
}

// 音频预览加载完成
const onAudioPreviewLoaded = () => {
    if (audioPreviewElement.value) {
        audioPreviewDuration.value = audioPreviewElement.value.duration || 0
    }
}

// 音频预览时间更新
const onAudioPreviewTimeUpdate = () => {
    if (audioPreviewElement.value && !audioPreviewIsDragging.value) {
        audioPreviewCurrentTime.value = audioPreviewElement.value.currentTime || 0
    }
}

// 音频预览进度条变化处理（点击或拖拽）
const onAudioPreviewProgressChange = (event) => {
    if (audioPreviewDuration.value > 0 && audioPreviewElement.value && event.target) {
        const newTime = parseFloat(event.target.value)
        audioPreviewCurrentTime.value = newTime
        // 立即更新音频位置
        audioPreviewElement.value.currentTime = newTime
    }
}

// 音频预览进度条拖拽结束处理
const onAudioPreviewProgressEnd = (event) => {
    if (audioPreviewElement.value && audioPreviewDuration.value > 0 && event.target) {
        const newTime = parseFloat(event.target.value)
        audioPreviewElement.value.currentTime = newTime
        audioPreviewCurrentTime.value = newTime
    }
    audioPreviewIsDragging.value = false
}

// 音频预览播放结束
const onAudioPreviewEnded = () => {
    audioPreviewIsPlaying.value = false
    audioPreviewCurrentTime.value = 0
}

// 监听音频预览变化，重置状态
watch(() => getCurrentAudioPreview(), (newPreview) => {
    // 停止当前播放
    if (audioPreviewElement.value) {
        audioPreviewElement.value.pause()
    }
    audioPreviewIsPlaying.value = false
    audioPreviewCurrentTime.value = 0
    audioPreviewDuration.value = 0

    if (newPreview) {
        // 等待 DOM 更新后加载新音频
        nextTick(() => {
            if (audioPreviewElement.value) {
                audioPreviewElement.value.load()
            }
        })
    }
})

// 监听分离后的音频变化，重置播放器状态
watch(() => s2vForm.value.separatedAudios, (newAudios, oldAudios) => {
    // 如果音频列表发生变化（重新分割），清理旧的音频元素和状态
    if (newAudios && newAudios.length > 0 && oldAudios && oldAudios.length > 0) {
        // 检查是否是重新分割（音频数量或内容发生变化）
        const isReseparation = newAudios.length !== oldAudios.length ||
            newAudios.some((audio, index) => {
                const oldAudio = oldAudios[index]
                return !oldAudio || audio.audioDataUrl !== oldAudio.audioDataUrl
            })

        if (isReseparation) {
            // 停止所有正在播放的音频
            separatedAudioElements.value.forEach((audioElement, index) => {
                if (audioElement) {
                    audioElement.pause()
                    separatedAudioElements.value[index] = null
                }
            })

            // 清理所有状态
            separatedAudioElements.value = []
            separatedAudioPlaying.value = {}
            separatedAudioDuration.value = {}
            separatedAudioCurrentTime.value = {}
            separatedAudioIsDragging.value = {}

            // 等待 DOM 更新后重新加载音频
            nextTick(() => {
                // 音频元素会在模板中自动重新创建和加载
            })
        }
    } else if (!newAudios || newAudios.length === 0) {
        // 如果音频列表被清空，清理所有状态
        separatedAudioElements.value.forEach((audioElement, index) => {
            if (audioElement) {
                audioElement.pause()
                separatedAudioElements.value[index] = null
            }
        })
        separatedAudioElements.value = []
        separatedAudioPlaying.value = {}
        separatedAudioDuration.value = {}
        separatedAudioCurrentTime.value = {}
        separatedAudioIsDragging.value = {}
    }
}, { deep: true })

// 分离后的音频播放器控制函数
const toggleSeparatedAudioPlayback = (index) => {
    const audioElement = separatedAudioElements.value[index]
    if (!audioElement) return

    if (audioElement.paused) {
        audioElement.play().catch(error => {
            console.log('播放失败:', error)
        })
    } else {
        audioElement.pause()
    }
}

const getSeparatedAudioPlaying = (index) => {
    return separatedAudioPlaying.value[index] || false
}

const getSeparatedAudioDuration = (index) => {
    return separatedAudioDuration.value[index] || 0
}

const getSeparatedAudioCurrentTime = (index) => {
    return separatedAudioCurrentTime.value[index] || 0
}

const onSeparatedAudioLoaded = (index) => {
    const audioElement = separatedAudioElements.value[index]
    if (audioElement) {
        separatedAudioDuration.value[index] = audioElement.duration || 0
    }
}

const onSeparatedAudioTimeUpdate = (index) => {
    const audioElement = separatedAudioElements.value[index]
    if (audioElement && !separatedAudioIsDragging.value[index]) {
        separatedAudioCurrentTime.value[index] = audioElement.currentTime || 0
    }
}

const onSeparatedAudioProgressChange = (index, event) => {
    if (separatedAudioDuration.value[index] > 0 && separatedAudioElements.value[index] && event.target) {
        const newTime = parseFloat(event.target.value)
        separatedAudioCurrentTime.value[index] = newTime
        separatedAudioElements.value[index].currentTime = newTime
    }
}

const onSeparatedAudioProgressEnd = (index, event) => {
    const audioElement = separatedAudioElements.value[index]
    if (audioElement && separatedAudioDuration.value[index] > 0 && event.target) {
        const newTime = parseFloat(event.target.value)
        audioElement.currentTime = newTime
        separatedAudioCurrentTime.value[index] = newTime
    }
    separatedAudioIsDragging.value[index] = false
}

const onSeparatedAudioEnded = (index) => {
    separatedAudioPlaying.value[index] = false
    separatedAudioCurrentTime.value[index] = 0
}

const handleSeparatedAudioError = (index) => {
    console.error(`分离后的音频 ${index} 加载失败`)
    separatedAudioPlaying.value[index] = false
}

// 拖拽排序函数 - 角色
const onRoleDragStart = (event, index) => {
    draggedRoleIndex.value = index
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/html', event.target.outerHTML)

    // 创建拖拽预览
    const target = event.currentTarget
    const rect = target.getBoundingClientRect()
    dragOffset.value = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    }

    // 创建拖拽预览图片
    const dragImage = target.cloneNode(true)
    // 设置固定尺寸，确保预览正确显示
    dragImage.style.width = `${rect.width}px`
    dragImage.style.height = `${rect.height}px`
    dragImage.style.position = 'fixed'
    dragImage.style.top = '-9999px'
    dragImage.style.left = '-9999px'
    dragImage.style.opacity = '0.9'
    dragImage.style.transform = 'rotate(2deg)'
    dragImage.style.pointerEvents = 'none'
    dragImage.style.zIndex = '10000'
    dragImage.style.boxShadow = '0 8px 24px rgba(0,0,0,0.3)'
    dragImage.style.backgroundColor = 'transparent'

    // 立即添加到 DOM
    document.body.appendChild(dragImage)

    // 强制重排，确保元素已渲染
    void dragImage.offsetHeight

    // 同步设置拖拽图片（必须在 dragstart 事件中同步调用）
    try {
        event.dataTransfer.setDragImage(dragImage, dragOffset.value.x, dragOffset.value.y)
    } catch (e) {
        console.warn('Failed to set drag image:', e)
    }

    // 延迟移除预览元素
    setTimeout(() => {
        if (dragImage.parentNode) {
            dragImage.parentNode.removeChild(dragImage)
        }
    }, 0)
}

const onRoleDragOver = (event, index) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
    if (draggedRoleIndex.value !== index) {
        dragOverRoleIndex.value = index
    }
}

const onRoleDragLeave = () => {
    dragOverRoleIndex.value = -1
}

const onRoleDrop = (event, targetIndex) => {
    event.preventDefault()
    if (draggedRoleIndex.value === -1 || draggedRoleIndex.value === targetIndex) {
        draggedRoleIndex.value = -1
        dragOverRoleIndex.value = -1
        return
    }

    const form = getCurrentForm()
    if (!form || !form.detectedFaces) return

    // 保存原始状态
    const originalFaces = [...form.detectedFaces]

    // 重新排序角色（只改变角色顺序，不影响音频顺序）
    const faces = [...form.detectedFaces]
    const draggedFace = faces[draggedRoleIndex.value]
    faces.splice(draggedRoleIndex.value, 1)
    faces.splice(targetIndex, 0, draggedFace)
    form.detectedFaces = faces

    // 更新音频的 roleIndex 和 roleName，以匹配新的角色位置
    // 但不改变音频的显示顺序
    if (s2vForm.value.separatedAudios && s2vForm.value.separatedAudios.length > 0) {
        s2vForm.value.separatedAudios.forEach((audio) => {
            // 找到这个音频原来对应的角色
            const originalRoleIndex = audio.roleIndex !== undefined ? audio.roleIndex : -1
            if (originalRoleIndex >= 0 && originalRoleIndex < originalFaces.length) {
                const originalFace = originalFaces[originalRoleIndex]
                // 找到这个角色在新列表中的位置
                const newRoleIndex = faces.findIndex(f => f === originalFace)
                if (newRoleIndex >= 0) {
                    audio.roleIndex = newRoleIndex
                    audio.roleName = faces[newRoleIndex].roleName || `角色${newRoleIndex + 1}`
                }
            }
        })
        // 触发响应式更新
        s2vForm.value.separatedAudios = [...s2vForm.value.separatedAudios]
    }

    draggedRoleIndex.value = -1
    dragOverRoleIndex.value = -1
}

// 拖拽排序函数 - 音频
const onAudioDragStart = (event, index) => {
    draggedAudioIndex.value = index
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/html', event.target.outerHTML)

    // 创建拖拽预览
    const target = event.currentTarget
    const rect = target.getBoundingClientRect()
    dragOffset.value = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    }

    // 创建拖拽预览图片
    const dragImage = target.cloneNode(true)
    // 设置固定尺寸，确保预览正确显示
    dragImage.style.width = `${rect.width}px`
    dragImage.style.height = `${rect.height}px`
    dragImage.style.position = 'fixed'
    dragImage.style.top = '-9999px'
    dragImage.style.left = '-9999px'
    dragImage.style.opacity = '0.9'
    dragImage.style.transform = 'rotate(2deg)'
    dragImage.style.pointerEvents = 'none'
    dragImage.style.zIndex = '10000'
    dragImage.style.boxShadow = '0 8px 24px rgba(0,0,0,0.3)'
    dragImage.style.backgroundColor = 'transparent'

    // 立即添加到 DOM
    document.body.appendChild(dragImage)

    // 强制重排，确保元素已渲染
    void dragImage.offsetHeight

    // 同步设置拖拽图片（必须在 dragstart 事件中同步调用）
    try {
        event.dataTransfer.setDragImage(dragImage, dragOffset.value.x, dragOffset.value.y)
    } catch (e) {
        console.warn('Failed to set drag image:', e)
    }

    // 延迟移除预览元素
    setTimeout(() => {
        if (dragImage.parentNode) {
            dragImage.parentNode.removeChild(dragImage)
        }
    }, 0)
}

const onAudioDragOver = (event, index) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
    if (draggedAudioIndex.value !== index) {
        dragOverAudioIndex.value = index
    }
}

const onAudioDragLeave = () => {
    dragOverAudioIndex.value = -1
}

const onAudioDrop = (event, targetIndex) => {
    event.preventDefault()
    if (draggedAudioIndex.value === -1 || draggedAudioIndex.value === targetIndex) {
        draggedAudioIndex.value = -1
        dragOverAudioIndex.value = -1
        return
    }

    if (!s2vForm.value.separatedAudios) return

    // 重新排序音频（只改变音频顺序，不影响角色顺序）
    const audios = [...s2vForm.value.separatedAudios]
    const draggedAudio = audios[draggedAudioIndex.value]
    audios.splice(draggedAudioIndex.value, 1)
    audios.splice(targetIndex, 0, draggedAudio)

    // 音频的 roleIndex 和 roleName 保持不变，因为它们仍然对应原来的角色
    // 不需要更新 roleIndex，因为角色顺序没有改变

    s2vForm.value.separatedAudios = audios

    draggedAudioIndex.value = -1
    dragOverAudioIndex.value = -1
}


// 尺寸设置相关状态
const showCustomSize = ref(false)  // 是否显示自定义尺寸输入
const customWidth = ref('')
const customHeight = ref('')

// 图片任务宽高比选项
const imageAspectRatios = [
    { label: '16:9', value: '16:9', key: 'aspectRatio16_9' },
    { label: '9:16', value: '9:16', key: 'aspectRatio9_16' },
    { label: '1:1', value: '1:1', key: 'aspectRatio1_1' },
    { label: '4:3', value: '4:3', key: 'aspectRatio4_3' },
    { label: '3:4', value: '3:4', key: 'aspectRatio3_4' }
]

// 宽高比到尺寸的映射（基于后端 ASPECT_RATIO_MAP）
// 格式: [width, height] -> 转换为 customShape 格式 [height, width]
const aspectRatioToSizeMap = {
    '16:9': [928, 1664],   // [height, width] 对应后端的 [1664, 928]
    '9:16': [1664, 928],   // [height, width] 对应后端的 [928, 1664]
    '1:1': [1328, 1328],   // [height, width] 对应后端的 [1328, 1328]
    '4:3': [1140, 1472],  // [height, width] 对应后端的 [1472, 1140]
    '3:4': [1024, 768]     // [height, width] 对应后端的 [768, 1024]
}

// 视频任务预设分辨率选项（仅用于 t2v 任务）
// value 格式: [height, width]
const videoResolutions = [
    { label: '480p_landscape', value: [480, 832], key: 'resolution480pLandscape' }, // 480p（横屏）480×832
    { label: '480p_portrait', value: [832, 480], key: 'resolution480pPortrait' }, // 480p（竖屏）832×480
    { label: '720p_landscape', value: [720, 1280], key: 'resolution720pLandscape' }, // 720p（横屏）720×1280
    { label: '720p_portrait', value: [1280, 720], key: 'resolution720pPortrait' }, // 720p（竖屏）1280×720
]


// 选择视频预设分辨率
const selectVideoResolution = (resolution) => {
    const form = getCurrentForm()
    if (form) {
        form.customShape = resolution.value
        showCustomSize.value = false
        customWidth.value = ''
        customHeight.value = ''
    }
}

// 选择图片宽高比（转换为 custom_shape）
const selectImageAspectRatio = (aspectRatio) => {
    const form = getCurrentForm()
    if (form) {
        form.aspectRatio = aspectRatio.value
        // 设置 customShape，格式为 [height, width]
        if (aspectRatioToSizeMap[aspectRatio.value]) {
            form.customShape = aspectRatioToSizeMap[aspectRatio.value]
        }
        showCustomSize.value = false
        customWidth.value = ''
        customHeight.value = ''
    }
}

// 使用默认尺寸
const useDefaultSize = () => {
    const form = getCurrentForm()
    if (form) {
        form.customShape = null
        form.aspectRatio = null
        showCustomSize.value = false
        customWidth.value = ''
        customHeight.value = ''
    }
}

// 应用自定义尺寸（自动应用，无需手动调用）
const applyCustomSize = () => {
    const width = parseInt(customWidth.value)
    const height = parseInt(customHeight.value)

    // 如果输入为空或无效，不应用
    if (!customWidth.value || !customHeight.value || isNaN(width) || isNaN(height) || width <= 0 || height <= 0) {
        return
    }

    const form = getCurrentForm()
    if (form) {
        form.customShape = [height, width]
    }
}

// 打开自定义尺寸输入
const openCustomSize = () => {
    showCustomSize.value = true
    const form = getCurrentForm()
    if (form && form.customShape && Array.isArray(form.customShape) && form.customShape.length === 2) {
        customHeight.value = form.customShape[0].toString()
        customWidth.value = form.customShape[1].toString()
    }
}

// 检查当前 customShape 是否匹配某个预设宽高比
const getCurrentAspectRatio = () => {
    const form = getCurrentForm()
    if (!form || !form.customShape || !Array.isArray(form.customShape) || form.customShape.length !== 2) {
        return null
    }
    const [height, width] = form.customShape
    // 检查是否匹配某个预设值
    // customShape 格式是 [height, width]
    for (const [ratioValue, sizeArray] of Object.entries(aspectRatioToSizeMap)) {
        const [presetHeight, presetWidth] = sizeArray
        if (height === presetHeight && width === presetWidth) {
            return ratioValue  // 返回 '16:9' 等格式
        }
    }
    return null
}

// 获取当前选中的尺寸值（用于下拉菜单）
const getCurrentSizeValue = () => {
    const form = getCurrentForm()

    // t2v 任务：检查是否匹配预设分辨率
    if (selectedTaskId.value === 't2v') {
        if (!form || !form.customShape) {
            return '720p_landscape'  // 默认 720p（横屏）
        }
        const [height, width] = form.customShape
        for (const resolution of videoResolutions) {
            if (resolution.value[0] === height && resolution.value[1] === width) {
                return resolution.label
            }
        }
        return '720p_landscape'  // 默认 720p（横屏）
    }

    // 图片任务：检查是否匹配预设宽高比
    if (selectedTaskId.value === 't2i' || selectedTaskId.value === 'i2i') {
        // 如果自定义尺寸输入框已打开，优先显示 "custom"
        if (showCustomSize.value) {
            return 'custom'
        }
        if (!form || !form.customShape) {
            return 'default'  // 默认 16:9 横屏
        }
        const aspectRatio = getCurrentAspectRatio()
        if (aspectRatio) {
            return aspectRatio
        }
        return 'custom'
    }

    // 其他任务：使用输入尺寸
    return 'default'
}

// 视频任务尺寸选项（仅用于 t2v 任务）
const videoSizeOptions = computed(() => {
    // 只有 t2v 任务才显示尺寸选项
    if (selectedTaskId.value !== 't2v') {
        return []
    }

    const options = []

    videoResolutions.forEach(resolution => {
        options.push({
            value: resolution.label,
            label: t(resolution.key),
            icon: 'fas fa-video'
        })
    })

    return options
})

// 图片任务尺寸选项
const imageSizeOptions = computed(() => {
    const options = []

    // t2i 任务不显示"使用默认尺寸"选项
    if (selectedTaskId.value !== 't2i') {
        options.push({
            value: 'default',
            label: t('useInputSize'),
            icon: 'fas fa-undo'
        })
    }

    imageAspectRatios.forEach(ratio => {
        options.push({
            value: ratio.value,
            label: t(ratio.key),
            icon: 'fas fa-image'
        })
    })

    options.push({
        value: 'custom',
        label: t('custom'),
        icon: 'fas fa-edit'
    })

    return options
})

// 处理尺寸下拉选择
const handleSizeSelect = (item) => {
    if (item.value === 'default') {
        useDefaultSize()
    } else if (item.value === 'custom') {
        openCustomSize()
    } else {
        // t2v 任务：选择横屏或竖屏
        if (selectedTaskId.value === 't2v') {
            const resolution = videoResolutions.find(r => r.label === item.value)
            if (resolution) {
                selectVideoResolution(resolution)
            }
        }
        // 图片任务：选择宽高比
        else if (selectedTaskId.value === 't2i' || selectedTaskId.value === 'i2i') {
            const ratio = imageAspectRatios.find(r => r.value === item.value)
            if (ratio) {
                selectImageAspectRatio(ratio)
            }
        }
    }
}

// 组件卸载时清理
onUnmounted(() => {
    if (resizeHandler) {
        window.removeEventListener('resize', resizeHandler)
    }
    // 停止音频预览播放
    if (audioPreviewElement.value) {
        audioPreviewElement.value.pause()
        audioPreviewElement.value = null
    }
    // 停止并清理所有分离后的音频播放器
    separatedAudioElements.value.forEach((audioElement, index) => {
        if (audioElement) {
            audioElement.pause()
            separatedAudioElements.value[index] = null
        }
    })
    separatedAudioElements.value = []
    separatedAudioPlaying.value = {}
    separatedAudioDuration.value = {}
    separatedAudioCurrentTime.value = {}
    separatedAudioIsDragging.value = {}
})

</script>
<template>
    <div
      v-if="templateLoading || downloadLoading"
      class="fixed right-6 top-24 sm:top-20 z-[9999] w-auto min-w-[260px] sm:min-w-[300px] max-w-[calc(100vw-2.5rem)] sm:max-w-md px-4 sm:px-5 transition-all duration-300 ease-out"
    >
      <div
        class="pointer-events-auto text-[#1d1d1f] bg-white/95 dark:text-white dark:bg-[#0d0d12]/90 backdrop-blur-[20px] backdrop-saturate-[180%] border border-black/8 dark:border-white/8 rounded-2xl shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),0_2px_4px_-1px_rgba(0,0,0,0.06),0_0_0_1px_rgba(0,0,0,0.05)] dark:shadow-[0_12px_32px_rgba(0,0,0,0.6),0_4px_12px_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.06)]"
      >
        <div class="flex items-center gap-3 px-5 py-3">
          <div class="flex items-center justify-center w-9 h-9 rounded-full bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]">
            <i class="fas fa-spinner fa-spin text-sm"></i>
          </div>
          <div class="flex-1 text-sm font-medium tracking-tight">
            {{ templateLoading ? (templateLoadingMessage || t('prefillLoadingDefault')) : (downloadLoadingMessage || t('downloadPreparing')) }}
          </div>
        </div>
      </div>
    </div>
                <!-- 主内容区域 - 响应式布局 -->
                <div class="flex-1 flex flex-col min-h-0 mobile-content">
                    <!-- 生成视频区域 -->
                    <div class="flex-1 flex flex-col">
                        <!-- 内容区域 -->
                        <div class="flex-1 p-6">


                        <!-- 任务创建面板 -->
                        <div class="max-w-4xl mx-auto min-h-[100vh] flex flex-col" id="task-creator">
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
                                            <span v-else-if="selectedTaskId === 'flf2v'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseUploadFirstAndLastFrameImagesAndTheMostDetailedVideoScript') }}</span>
                                            <span v-else-if="selectedTaskId === 'i2i'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseUploadAnImageAsTheReferenceImageAndThePromptForImageGeneration') }}</span>
                                            <span v-else-if="selectedTaskId === 't2i'"
                                                  class="inline-block animate-fade-in">{{ t('pleaseEnterThePromptForImageGeneration') || '请输入图片生成提示词，描述图片的内容、风格和细节要求' }}</span>
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

                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v' || selectedTaskId === 'animate' || selectedTaskId === 'flf2v' || selectedTaskId === 'i2i'" class="upload-section">
                                    <!-- 上传图片 - Apple 风格 -->
                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v' || selectedTaskId === 'animate' || selectedTaskId === 'flf2v' || selectedTaskId === 'i2i'">
                                        <!-- 图片标签 -->
                                        <div class="flex justify-between items-center mb-3">
                                            <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                {{ selectedTaskId === 'flf2v' ? t('firstFrameImage') : t('image') }}
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
                                                'p-8': !getCurrentImagePreview() && (selectedTaskId !== 'i2i' || getI2IImagePreviews().length === 0)
                                            }"
                                            >
                                            <!-- 默认上传界面 - Apple 风格 -->
                                            <div v-if="!getCurrentImagePreview() && (selectedTaskId !== 'i2i' || getI2IImagePreviews().length === 0)" class="flex flex-col items-center justify-center h-full">
                                            <p class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">
                                                {{ selectedTaskId === 'i2i' ? (t('uploadImages') || '上传图片（支持多张）') : t('uploadImage') }}
                                            </p>
                                            <p class="text-xs text-[#86868b] dark:text-[#98989d] mb-6 tracking-tight">
                                                {{ selectedTaskId === 'i2i' ? (t('supportedImageFormats') + ' ' + (t('multipleImagesSupported') || '（支持多张）')) : t('supportedImageFormats') }}
                                            </p>
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

                                            <!-- 图片预览区域 -->
                                            <!-- i2i 模式：多图预览 -->
                                            <div v-if="selectedTaskId === 'i2i' && getI2IImagePreviews().length > 0" class="flex items-center justify-center w-full min-h-[220px]">
                                                <div class="flex flex-wrap justify-center items-center gap-3 p-2 w-full">
                                                    <div v-for="(preview, index) in getI2IImagePreviews()" :key="index"
                                                        class="relative group flex items-center justify-center bg-white/40 dark:bg-[#2c2c2e]/40 rounded-xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all duration-200"
                                                        style="max-height: 220px; aspect-ratio: 1; width: calc((100% - 12px) / 2); min-width: 100px; max-width: 220px;">
                                                        <img :src="preview" :alt="`${t('previewImage')} ${index + 1}`"
                                                            class="max-w-full max-h-[220px] w-auto h-auto object-contain">
                                                        <!-- 删除按钮 -->
                                                        <button @click.stop="removeImage(index)"
                                                            class="absolute top-2 right-2 w-8 h-8 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100 opacity-0 group-hover:opacity-100"
                                                            :title="t('deleteImage')">
                                                            <i class="fas fa-times text-xs"></i>
                                                        </button>
                                                        <!-- 图片序号 -->
                                                        <div class="absolute bottom-2 left-2 px-2 py-1 bg-black/50 dark:bg-black/70 text-white text-xs rounded backdrop-blur-sm">
                                                            {{ index + 1 }}
                                                        </div>
                                                    </div>
                                                    <!-- 继续添加图片按钮（最多3张） -->
                                                    <div v-if="getI2IImagePreviews().length < 3" @click="triggerImageUpload"
                                                        class="relative group flex items-center justify-center bg-white/40 dark:bg-[#2c2c2e]/40 rounded-xl overflow-hidden border-2 border-dashed border-[color:var(--brand-primary)]/30 dark:border-[color:var(--brand-primary-light)]/30 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all duration-200 cursor-pointer"
                                                        style="max-height: 220px; aspect-ratio: 1; width: calc((100% - 12px) / 2); min-width: 100px; max-width: 220px;">
                                                        <div class="flex flex-col items-center gap-2">
                                                            <i class="fas fa-plus text-2xl text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                            <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('addMoreImages') || '添加更多' }}</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            <!-- 其他模式或 i2i 单图：单图预览 -->
                                            <div v-else-if="getCurrentImagePreview()" class="flex items-center justify-center w-full min-h-[220px]">
                                                <!-- 主图预览 - Apple 风格 -->
                                                <div class="relative w-auto max-w-full min-h-[220px] flex items-center justify-center group">
                                                    <img :src="getCurrentImagePreviewUrl()" alt="t('previewImage')"
                                                        class="max-w-full max-h-[220px] w-auto h-auto object-contain rounded-xl transition-all duration-200">

                                                    <!-- 删除按钮 - Apple 风格 -->
                                                    <div
                                                        class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-200">
                                                        <button @click.stop="handleRemoveImage"
                                                            class="w-11 h-11 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100"
                                                            :title="t('deleteImage')">
                                                            <i class="fas fa-trash text-base"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                                <input type="file" ref="imageInput" @change="handleImageUpload"
                                                :accept="selectedTaskId === 'i2i' ? 'image/*' : 'image/*'"
                                                :multiple="selectedTaskId === 'i2i'"
                                                style="display: none;">
                                            </div>

                                            <!-- 角色检测加载提示 -->
                                            <div v-if="faceDetecting" class="mt-3 flex items-center justify-center gap-2 text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                <span>{{ t('detectingCharacters') }}</span>
                                            </div>
                                    </div>

                                    <!-- 上传尾帧图片 - flf2v 任务 - Apple 风格 -->
                                    <div v-if="selectedTaskId === 'flf2v'" class="mt-6">
                                        <!-- 尾帧图片标签 -->
                                        <div class="flex justify-between items-center mb-3">
                                            <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                {{ t('lastFrameImage') }}
                                            </label>
                                        </div>
                                        <!-- 上传尾帧图片区域 - Apple 风格 -->
                                        <div class="relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl p-2 min-h-[220px] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_16px_rgba(0,0,0,0.3)]"
                                            @drop.prevent.stop="handleLastFrameDrop"
                                            @dragover.prevent.stop="handleDragOver"
                                            @dragenter.prevent.stop="handleDragEnter"
                                            @dragleave.prevent.stop="handleDragLeave"
                                            :class="{
                                                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': isDragOver,
                                                'p-8': !getCurrentLastFramePreview()
                                            }"
                                            >
                                            <!-- 默认上传界面 - Apple 风格 -->
                                            <div v-if="!getCurrentLastFramePreview()" class="flex flex-col items-center justify-center h-full">
                                                <p class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ t('uploadLastFrameImage') }}</p>
                                                <p class="text-xs text-[#86868b] dark:text-[#98989d] mb-6 tracking-tight">{{ t('supportedImageFormats') }}</p>
                                                <div class="flex items-center justify-center gap-4">
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button
                                                            class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.3)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100"
                                                            @click="triggerLastFrameUpload"
                                                            :title="t('uploadLastFrameImage')">
                                                            <i class="fas fa-upload text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('upload') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button
                                                            @click.stop="isSelectingLastFrame = true; showImageTemplates = true; mediaModalTab = 'history'; getImageHistory()"
                                                            class="w-12 h-12 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                            :title="t('templates')">
                                                            <i class="fas fa-history text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('templates') }}</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- 尾帧图片预览 - Apple 风格 -->
                                            <div v-if="getCurrentLastFramePreview()" class="relative w-full min-h-[220px] flex items-center justify-center group">
                                                <img :src="getCurrentLastFramePreviewUrl()" alt="t('previewLastFrameImage')"
                                                    class="max-w-full max-h-[220px] w-auto h-auto object-contain rounded-xl transition-all duration-200">

                                                <!-- 删除按钮 - Apple 风格 -->
                                                <div
                                                    class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-200">
                                                    <div class="flex gap-3">
                                                        <button @click.stop="removeLastFrame"
                                                            class="w-11 h-11 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100"
                                                            :title="t('deleteLastFrameImage')">
                                                            <i class="fas fa-trash text-base"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                            <input type="file" ref="lastFrameInput" @change="handleLastFrameUpload" accept="image/*" data-role="last-frame-input"
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
                                                        <button @click.stop="showVoiceTTSModal = true"
                                                            class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border border-black/8 dark:border-white/8 text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                            :title="t('textToSpeech')">
                                                            <i class="fi fi-bs-text text-lg"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('textToSpeech') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button @click.stop="router.push('/podcast_generate')"
                                                            class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border border-black/8 dark:border-white/8 text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100"
                                                            :title="t('podcast.dualPersonPodcast')">
                                                            <!-- 讲话的icon，用fa-microphone-alt如果有，否则fa-microphone -->
                                                            <i class="fi fi-bs-signal-stream text-xl"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('podcast.dualPersonPodcast') }}</span>
                                                    </div>
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
                                                        <i class="fas fa-microphone-alt text-base" :class="{ 'animate-pulse': isRecording }"></i>
                                                    </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ isRecording ? formatRecordingDuration(recordingDuration) : t('recordAudio') }}</span>
                                                    </div>

                                        </div>
                                            </div>

                                        <!-- 音频预览 - 原始音频播放器 -->
                                            <div v-if="getCurrentAudioPreview()" class="relative w-full min-h-[220px] flex items-center justify-center">
                                                <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)] w-full p-4">
                                                    <div class="relative flex items-center mb-3">
                                                        <!-- 头像容器 -->
                                                        <div class="relative mr-3 flex-shrink-0">
                                                            <!-- 透明白色头像 -->
                                                            <div class="w-12 h-12 rounded-full bg-white/40 dark:bg-white/20 border border-white/30 dark:border-white/20 transition-all duration-200"></div>
                                                            <!-- 播放/暂停按钮 -->
                                                            <button
                                                                @click="toggleAudioPreviewPlayback"
                                                                class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white cursor-pointer hover:scale-110 transition-all duration-200 z-20 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]"
                                                            >
                                                                <i :class="audioPreviewIsPlaying ? 'fas fa-pause' : 'fas fa-play'" class="text-xs ml-0.5"></i>
                                                            </button>
                                                        </div>

                                                        <!-- 音频信息 -->
                                                        <div class="flex-1 min-w-0">
                                                            <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate">
                                                                {{ t('audio') }}
                                                            </div>
                                                        </div>

                                                        <!-- 音频时长 -->
                                                        <div class="text-xs font-medium text-[#86868b] dark:text-[#98989d] tracking-tight flex-shrink-0 mr-3">
                                                            {{ formatAudioPreviewTime(audioPreviewCurrentTime) }} / {{ formatAudioPreviewTime(audioPreviewDuration) }}
                                                        </div>

                                                        <!-- 删除按钮 -->
                                                        <button @click.stop="removeAudio"
                                                            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100 flex-shrink-0"
                                                            :title="t('deleteAudio')">
                                                            <i class="fas fa-trash text-sm"></i>
                                                        </button>
                                                    </div>

                                                    <!-- 进度条 -->
                                                    <div class="flex items-center gap-2" v-if="audioPreviewDuration > 0">
                                                        <input
                                                            type="range"
                                                            :min="0"
                                                            :max="audioPreviewDuration"
                                                            :value="audioPreviewCurrentTime"
                                                            @input="onAudioPreviewProgressChange"
                                                            @change="onAudioPreviewProgressChange"
                                                            @mousedown="audioPreviewIsDragging = true"
                                                            @mouseup="onAudioPreviewProgressEnd"
                                                            @touchstart="audioPreviewIsDragging = true"
                                                            @touchend="onAudioPreviewProgressEnd"
                                                            class="flex-1 h-1 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                                                        />
                                                    </div>
                                                </div>

                                                <!-- 隐藏的音频元素 -->
                                                <audio
                                                    ref="audioPreviewElement"
                                                    :src="getCurrentAudioPreviewUrl()"
                                                    @loadedmetadata="onAudioPreviewLoaded"
                                                    @timeupdate="onAudioPreviewTimeUpdate"
                                                    @ended="onAudioPreviewEnded"
                                                    @play="audioPreviewIsPlaying = true"
                                                    @pause="audioPreviewIsPlaying = false"
                                                    @error="handleAudioError"
                                                    class="hidden"
                                                ></audio>
                                            </div>

                                            <input type="file" ref="audioInput" @change="handleAudioUpload" accept="audio/*,audio/mp4,audio/x-m4a,video/*" data-role="audio-input"
                                            style="display: none;">
                                        </div>

                                        <!-- 音频分割加载提示 -->
                                        <div v-if="audioSeparating" class="mt-3 flex items-center justify-center gap-2 text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">
                                            <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                            <span>{{t('splitingAudio')}}</span>
                                        </div>
                                    </div>

                                    <!-- 上传视频 - Apple 风格（用于 animate 任务类型） -->
                                    <div v-if="selectedTaskId === 'animate'">
                                        <!-- 视频标签 -->
                                        <div class="flex justify-between items-center mb-3">
                                            <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                {{ t('video') || '视频' }}
                                            </label>
                                        </div>
                                        <!-- 上传视频区域 - Apple 风格 -->
                                        <div class="relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl p-2 min-h-[220px] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_16px_rgba(0,0,0,0.3)]"
                                            @drop="handleVideoDrop"
                                            @dragover="handleDragOver"
                                            @dragenter="handleDragEnter"
                                            @dragleave="handleDragLeave"
                                            :class="{
                                                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': isDragOver,
                                                'p-8': !getCurrentVideoPreview()
                                            }"
                                            >
                                            <!-- 默认上传界面 - Apple 风格 -->
                                            <div v-if="!getCurrentVideoPreview()" class="flex flex-col items-center justify-center h-full">
                                                <p class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ t('uploadVideo')}}</p>
                                                <p class="text-xs text-[#86868b] dark:text-[#98989d] mb-6 tracking-tight">{{ t('supportedVideoFormats') }}</p>
                                                <div class="flex items-center justify-center gap-4">
                                                    <div class="flex flex-col items-center gap-2">
                                                        <button
                                                            class="w-12 h-12 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.3)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100"
                                                            @click="triggerVideoUpload"
                                                            :title="t('uploadVideo') || '上传视频'">
                                                            <i class="fas fa-upload text-base"></i>
                                                        </button>
                                                        <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('upload') }}</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- 视频预览区域 -->
                                            <div v-if="getCurrentVideoPreview()" class="relative w-full min-h-[220px] flex items-center justify-center">
                                                <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)] w-full p-4">
                                                    <div class="relative flex items-center mb-3">
                                                        <!-- 视频预览 -->
                                                        <div class="flex-1 min-w-0">
                                                            <video
                                                                :src="getCurrentVideoPreviewUrl()"
                                                                class="w-full max-h-[180px] rounded-lg object-contain"
                                                                controls
                                                                preload="metadata"
                                                            ></video>
                                                        </div>

                                                        <!-- 删除按钮 -->
                                                        <button @click.stop="removeVideo"
                                                            class="ml-3 w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full transition-all duration-200 hover:scale-110 hover:shadow-[0_4px_12px_rgba(239,68,68,0.2)] dark:hover:shadow-[0_4px_12px_rgba(248,113,113,0.3)] active:scale-100 flex-shrink-0"
                                                            :title="t('deleteVideo') || '删除视频'">
                                                            <i class="fas fa-trash text-sm"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>

                                            <input type="file" ref="videoInput" @change="handleVideoUpload" accept="video/*" data-role="video-input"
                                                style="display: none;">
                                        </div>
                                    </div>
                                </div>

                                <!-- 角色和音频配对区域 -->
                                <div v-if="selectedTaskId === 's2v'" class="mt-8">
                                    <!-- 模式切换开关 - 始终显示 -->
                                    <div class="flex justify-center items-center mb-4">
                                        <div class="flex items-center gap-3">

                                            <!-- 开关按钮 -->
                                            <button
                                                @click="toggleRoleMode"
                                                class="relative w-14 h-7 rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-primary)]/20 dark:focus:ring-[color:var(--brand-primary-light)]/20"
                                                :class="isMultiRoleMode ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)]' : 'bg-[#86868b]/30 dark:bg-[#98989d]/30'"
                                                :title="isMultiRoleMode ? '切换到单角色模式' : '切换到多角色模式'"
                                            >
                                                <!-- 滑动圆点 -->
                                                <span
                                                    class="absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow-md transition-transform duration-300 flex items-center justify-center"
                                                    :class="{ 'translate-x-7': isMultiRoleMode, 'translate-x-0': !isMultiRoleMode }"
                                                >
                                                    <i :class="isMultiRoleMode ? 'fas fa-users text-[8px] text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]' : 'fas fa-user text-[8px] text-[#86868b] dark:text-[#98989d]'"></i>
                                                </span>
                                            </button>

                                            <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight" :class="{ 'text-[#86868b] dark:text-[#98989d]': isMultiRoleMode }">{{ isMultiRoleMode ? '多角色模式' : '单角色模式' }}</span>

                                            <!-- Info 图标按钮 -->
                                            <button
                                                @click="showRoleModeInfo = true"
                                                class="w-5 h-5 flex items-center justify-center text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors duration-200 rounded-full hover:bg-[#86868b]/10 dark:hover:bg-[#98989d]/10"
                                                :title="t('roleModeInfo.title')"
                                            >
                                                <i class="fas fa-info-circle text-xs"></i>
                                            </button>
                                        </div>
                                    </div>

                                    <!-- 角色模式说明弹窗 - Apple 风格 -->
                                    <div v-if="showRoleModeInfo"
                                        class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[70] flex items-center justify-center p-4"
                                        @click="showRoleModeInfo = false">
                                        <div class="w-full max-w-md bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden"
                                            @click.stop>
                                            <!-- 弹窗头部 -->
                                            <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
                                                <h3 class="text-lg font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                                    {{ t('roleModeInfo.title') }}
                                                </h3>
                                                <button @click="showRoleModeInfo = false"
                                                    class="w-8 h-8 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-red-500 dark:hover:text-red-400 hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100"
                                                    :title="t('close')">
                                                    <i class="fas fa-times text-sm"></i>
                                                </button>
                                            </div>

                                            <!-- 弹窗内容 -->
                                            <div class="p-6 space-y-6">
                                                <!-- 单角色模式说明 -->
                                                <div class="space-y-3">
                                                    <h4 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight flex items-center gap-2">
                                                        <i class="fas fa-user text-sm text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        {{ t('roleModeInfo.singleMode.title') }}
                                                    </h4>
                                                    <ul class="space-y-2 pl-6">
                                                        <li v-for="(point, index) in tm('roleModeInfo.singleMode.points')" :key="index"
                                                            class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight leading-relaxed flex items-start gap-2">
                                                            <span class="text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] mt-1.5 flex-shrink-0">•</span>
                                                            <span>{{ point }}</span>
                                                        </li>
                                                    </ul>
                                                </div>

                                                <!-- 多角色模式说明 -->
                                                <div class="space-y-3">
                                                    <h4 class="text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight flex items-center gap-2">
                                                        <i class="fas fa-users text-sm text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        {{ t('roleModeInfo.multiMode.title') }}
                                                    </h4>
                                                    <ul class="space-y-2 pl-6">
                                                        <li v-for="(point, index) in tm('roleModeInfo.multiMode.points')" :key="index"
                                                            class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight leading-relaxed flex items-center gap-2">
                                                            <span class="text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] flex-shrink-0">•</span>
                                                            <span>{{ point }}</span>
                                                        </li>
                                                    </ul>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 保存角色加载提示 -->
                                    <div v-if="faceSaving" class="flex items-center justify-center gap-2 text-sm text-[#86868b] dark:text-[#98989d] tracking-tight mb-4">
                                        <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <span>正在保存角色并更新音频...</span>
                                    </div>

                                    <!-- 角色和音频配对区域 - 每行一个配对（仅在多角色模式且有角色时显示） -->
                                    <div v-if="isMultiRoleMode && currentDetectedFaces && currentDetectedFaces.length > 0" class="flex flex-col items-center space-y-3">
                                        <div
                                            v-for="(face, index) in currentDetectedFaces"
                                            :key="index"
                                            class="flex items-stretch gap-4"
                                            :class="{
                                                'border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/50': dragOverRoleIndex === index || dragOverAudioIndex === index
                                            }"
                                        >
                                            <!-- 左侧：角色卡片 -->
                                            <div
                                                class="w-85 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-3 transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
                                                :class="{
                                                    'border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/50 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': dragOverRoleIndex === index,
                                                    'opacity-40 scale-95 shadow-lg': draggedRoleIndex === index,
                                                    'transform translate-y-0': draggedRoleIndex !== index
                                                }"
                                                @dragover.prevent="onRoleDragOver($event, index)"
                                                @dragleave="onRoleDragLeave"
                                                @drop="onRoleDrop($event, index)"
                                            >
                                                <!-- 角色区域 - 可拖拽 -->
                                                <div
                                                    class="flex items-center justify-between gap-2 h-full w-full transition-all duration-200"
                                                    :class="{
                                                        'opacity-50 scale-95': draggedRoleIndex === index,
                                                        'opacity-100': draggedRoleIndex !== index
                                                    }"
                                                    :draggable="true"
                                                    @dragstart="onRoleDragStart($event, index)"
                                                    @dragend="draggedRoleIndex = -1; dragOverRoleIndex = -1"
                                                >
                                                    <!-- 左侧：拖拽手柄和角色名 -->
                                                    <div class="flex items-center gap-2 flex-1 min-w-0">
                                                        <!-- 拖拽手柄 -->
                                                        <div class="cursor-move text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors">
                                                            <i class="fas fa-grip-vertical text-sm"></i>
                                                        </div>

                                                        <!-- 角色名显示/编辑 -->
                                                        <div class="flex items-center">
                                                            <!-- 编辑模式 -->
                                                            <input
                                                                v-if="face.isEditing"
                                                                type="text"
                                                                :value="face.roleName"
                                                                :data-face-index="index"
                                                                @input="updateFaceRoleName(index, $event.target.value)"
                                                                @blur="saveFaceRoleName(index, $event.target.value)"
                                                                @keyup.enter="saveFaceRoleName(index, $event.target.value)"
                                                                @keyup.esc="toggleFaceEditing(index)"
                                                                :ref="(el) => { if (el && face.isEditing) { nextTick(() => el.focus()); } }"
                                                                class="w-24 px-2 py-1.5 text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] bg-white/80 dark:bg-[#2c2c2e]/80 border border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/60 rounded-lg focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-primary)]/20 dark:focus:ring-[color:var(--brand-primary-light)]/20 transition-all duration-200"
                                                                :placeholder="`角色${index + 1}`"
                                                                @click.stop>
                                                            <!-- 显示模式 - 可点击编辑 -->
                                                            <span
                                                                v-else
                                                                @click.stop="toggleFaceEditing(index)"
                                                                class="w-24 px-2 py-1.5 text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] truncate tracking-tight cursor-text hover:bg-[color:var(--brand-primary)]/10 dark:hover:bg-[color:var(--brand-primary-light)]/15 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] rounded transition-colors duration-200"
                                                            >
                                                                {{ face.roleName || `角色${index + 1}` }}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    <!-- 右侧：头像、编辑按钮和删除按钮 -->
                                                    <div class="flex items-center gap-2 flex-shrink-0">
                                                        <!-- 角色头像容器 - 相对定位，用于放置编辑按钮 -->
                                                        <div class="relative flex-shrink-0">
                                                            <!-- 角色头像 - 可点击 -->
                                                            <div
                                                                @click.stop="openFaceEditModal(index)"
                                                                class="flex-shrink-0 w-14 h-14 rounded-lg overflow-hidden border border-black/8 dark:border-white/8 bg-black/5 dark:bg-white/5 cursor-pointer hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all duration-200 hover:scale-105"
                                                            >
                                                                <img v-if="face.face_image"
                                                                    :src="'data:image/png;base64,' + face.face_image"
                                                                    alt="Face"
                                                                    class="w-full h-full object-cover"
                                                                    @error="(e) => { console.error('Face image load error:', index, e); e.target.style.display = 'none'; }">
                                                                <div v-else class="w-full h-full flex items-center justify-center text-[#86868b] dark:text-[#98989d] text-xs">
                                                                    <i class="fas fa-image"></i>
                                                                </div>
                                                            </div>

                                                            <!-- 编辑按钮 - 放在头像右上角 -->
                                                            <button
                                                                v-if="!face.isEditing"
                                                                @click.stop="openFaceEditModal(index)"
                                                                class="absolute -top-1 -right-1 w-5 h-5 flex items-center justify-center bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[10px] border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] rounded-full transition-all duration-200 hover:scale-110 shadow-sm"
                                                                :title="t('edit') || '编辑'">
                                                                <i class="fas fa-edit text-xs"></i>
                                                            </button>
                                                            <!-- 保存按钮 -->
                                                            <button
                                                                v-else
                                                                @click.stop="() => {
                                                                    const inputEl = document.querySelector(`input[data-face-index='${index}']`);
                                                                    const newRoleName = inputEl?.value || face.roleName;
                                                                    saveFaceRoleName(index, newRoleName);
                                                                }"
                                                                class="absolute -top-1 -right-1 w-5 h-5 flex items-center justify-center bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 text-white rounded-full transition-all duration-200 hover:scale-110 shadow-sm"
                                                                :title="t('save') || '保存'">
                                                                <i class="fas fa-check text-xs"></i>
                                                            </button>
                                                        </div>

                                                        <!-- 删除按钮 -->
                                                        <button
                                                            @click.stop="removeFace(index)"
                                                            class="flex-shrink-0 w-6 h-6 flex items-center justify-center text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-all duration-200"
                                                            :title="t('delete') || '删除'">
                                                            <i class="fas fa-trash text-xs"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- 中间：链接符号 -->
                                            <div class="flex items-center justify-center flex-shrink-0">
                                                <div class="w-8 h-8 flex items-center justify-center text-[#86868b] dark:text-[#98989d]">
                                                    <i class="fas fa-link text-lg"></i>
                                                </div>
                                            </div>

                                            <!-- 右侧：音频卡片 -->
                                            <div
                                                v-if="currentSeparatedAudios && currentSeparatedAudios.length > index"
                                                class="w-85 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-3 transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
                                                :class="{
                                                    'border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/50 bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10': dragOverAudioIndex === index,
                                                    'opacity-40 scale-95 shadow-lg': draggedAudioIndex === index,
                                                    'transform translate-y-0': draggedAudioIndex !== index
                                                }"
                                                @dragover.prevent="onAudioDragOver($event, index)"
                                                @dragleave="onAudioDragLeave"
                                                @drop="onAudioDrop($event, index)"
                                            >
                                                <!-- 音频区域 - 可拖拽 -->
                                                <div
                                                    class="flex items-center gap-2 h-full transition-all duration-200"
                                                    :class="{
                                                        'opacity-50 scale-95': draggedAudioIndex === index,
                                                        'opacity-100': draggedAudioIndex !== index
                                                    }"
                                                    :draggable="true"
                                                    @dragstart="onAudioDragStart($event, index)"
                                                    @dragend="draggedAudioIndex = -1; dragOverAudioIndex = -1"
                                                >
                                                    <!-- 拖拽手柄 -->
                                                    <div class="cursor-move text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors">
                                                        <i class="fas fa-grip-vertical text-sm"></i>
                                                    </div>

                                                    <!-- 音色名显示/编辑 -->
                                                    <div class="flex items-center">
                                                        <!-- 编辑模式 -->
                                                        <input
                                                            v-if="currentSeparatedAudios[index].isEditing"
                                                            type="text"
                                                            :value="currentSeparatedAudios[index].audioName"
                                                            @input="updateSeparatedAudioName(index, $event.target.value)"
                                                            @blur="saveSeparatedAudioName(index, $event.target.value)"
                                                            @keyup.enter="saveSeparatedAudioName(index, $event.target.value)"
                                                            @keyup.esc="toggleSeparatedAudioEditing(index)"
                                                            :ref="(el) => { if (el && currentSeparatedAudios[index].isEditing) { nextTick(() => el.focus()); } }"
                                                            class="w-24 px-2 py-1 text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] bg-white/80 dark:bg-[#2c2c2e]/80 border border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/60 rounded-lg focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-primary)]/20 dark:focus:ring-[color:var(--brand-primary-light)]/20 transition-all duration-200"
                                                            :placeholder="`音色${index + 1}`"
                                                            @click.stop>
                                                        <!-- 显示模式 - 可点击编辑 -->
                                                        <span
                                                            v-else
                                                            @click.stop="toggleSeparatedAudioEditing(index)"
                                                            class="w-24 px-2 py-1 text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate cursor-text hover:bg-[color:var(--brand-primary)]/10 dark:hover:bg-[color:var(--brand-primary-light)]/15 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] rounded transition-colors duration-200"
                                                        >
                                                            {{ currentSeparatedAudios[index].audioName || `音色${index + 1}` }}
                                                        </span>
                                                    </div>

                                                    <!-- 音频播放器 -->
                                                    <div class="flex items-center gap-2 justify-center flex-shrink-0">
                                                        <!-- 播放/暂停按钮 -->
                                                        <button
                                                            @click="toggleSeparatedAudioPlayback(index)"
                                                            class="flex-shrink-0 w-10 h-10 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white cursor-pointer hover:scale-110 transition-all duration-200 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]"
                                                        >
                                                            <i :class="getSeparatedAudioPlaying(index) ? 'fas fa-pause' : 'fas fa-play'" class="text-xs ml-0.5"></i>
                                                        </button>

                                                        <!-- 右侧：时长和进度条 -->
                                                        <div class="flex flex-col justify-center" style="gap: 2px;">
                                                            <!-- 音频时长 - 显示在进度条上方 -->
                                                            <div class="text-xs font-medium text-[#86868b] dark:text-[#98989d] tracking-tight text-center" style="width: 128px;">
                                                                {{ formatAudioPreviewTime(getSeparatedAudioCurrentTime(index)) }} / {{ formatAudioPreviewTime(getSeparatedAudioDuration(index)) }}
                                                            </div>

                                                            <!-- 进度条 -->
                                                            <div class="w-32" v-if="getSeparatedAudioDuration(index) > 0">
                                                                <input
                                                                    type="range"
                                                                    :min="0"
                                                                    :max="getSeparatedAudioDuration(index)"
                                                                    :value="getSeparatedAudioCurrentTime(index)"
                                                                    @input="(e) => onSeparatedAudioProgressChange(index, e)"
                                                                    @change="(e) => onSeparatedAudioProgressChange(index, e)"
                                                                    @mousedown="separatedAudioIsDragging[index] = true"
                                                                    @mouseup="(e) => onSeparatedAudioProgressEnd(index, e)"
                                                                    class="w-full h-1 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                                                                />
                                                            </div>
                                                        </div>

                                                        <!-- 隐藏的音频元素 -->
                                                        <audio
                                                            :ref="el => { if (el) separatedAudioElements[index] = el }"
                                                            :src="currentSeparatedAudios[index].audioDataUrl"
                                                            @loadedmetadata="() => onSeparatedAudioLoaded(index)"
                                                            @timeupdate="() => onSeparatedAudioTimeUpdate(index)"
                                                            @ended="() => onSeparatedAudioEnded(index)"
                                                            @play="() => separatedAudioPlaying[index] = true"
                                                            @pause="() => separatedAudioPlaying[index] = false"
                                                            @error="() => handleSeparatedAudioError(index)"
                                                            class="hidden"
                                                        ></audio>
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- 音频占位符（如果没有对应的分离音频） -->
                                            <div
                                                v-else
                                                class="w-85 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-3 flex items-center justify-center text-sm text-[#86868b] dark:text-[#98989d] tracking-tight"
                                            >
                                                <span>{{ t('waitingForMultipleRolesAudio') }}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 新增角色按钮 -->
                                    <div v-if="selectedTaskId === 's2v' && getCurrentImagePreview() && isMultiRoleMode" class="flex justify-center mt-4">
                                        <button
                                            @click="openFaceEditModal(-1)"
                                            class="w-8 h-8 flex items-center justify-center bg-[#86868b]/20 dark:bg-[#98989d]/20 text-[#86868b] dark:text-[#98989d] rounded-full transition-all duration-200 hover:bg-[#86868b]/30 dark:hover:bg-[#98989d]/30 hover:scale-110 active:scale-100"
                                            :title="t('addRole') || '新增角色'"
                                        >
                                            <i class="fas fa-plus text-sm"></i>
                                        </button>
                                    </div>
                                </div>

                                        <!-- 提示词输入区域 - Apple 风格（animate 任务类型不显示） -->
                                        <div v-if="selectedTaskId !== 'animate'">
                                            <div class="mt-8 space-y-3 flex justify-between items-center mb-3">
                                                <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center tracking-tight">
                                                        {{ selectedTaskId === 's2v' ? t('promptOptional') : t('prompt') }}
                                                        <button @click="showPromptModal = true; promptModalTab = 'templates'"
                                                            class="ml-2 text-xs text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] transition-colors"
                                                            :title="t('promptTemplates')">
                                                        <i class="fas fa-lightbulb text-lg"></i>
                                                    </button>
                                                </label>
                                                <div class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                    {{ getCurrentForm().prompt?.length || 0 }} / 1000
                                                </div>
                                            </div>
                                            <div class="relative">
                                                <textarea v-model="getCurrentForm().prompt"
                                                    class="relative w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] transition-all duration-200 resize-none main-scrollbar placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)]"
                                                    :placeholder="getPromptPlaceholder()"
                                                    rows="2"
                                                    maxlength="1000"
                                                    required></textarea>
                                            </div>

                                            <div class="flex justify-between items-center mt-3">
                                                <button @click="clearPrompt"
                                                    class="flex items-center text-sm rounded-lg px-3 py-1.5 transition-all duration-200 text-[#86868b] dark:text-[#98989d] hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)] hover:bg-black/4 dark:hover:bg-white/6 group tracking-tight">
                                                    <i class="fas fa-sync-alt text-sm mr-2 group-hover:rotate-180 transition-transform duration-300"></i>
                                                    {{ t('clear') }}
                                                </button>

                                            </div>
                                        </div>

                                <!-- 尺寸设置区域 - Apple 风格（仅 t2v 和图片任务显示） -->
                                <div v-if="selectedTaskId === 't2v' || selectedTaskId === 't2i' || selectedTaskId === 'i2i'" class="mt-6">
                                    <div class="flex justify-between items-center mb-3">
                                        <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
                                            {{ t('sizeSetting') }}
                                        </label>
                                    </div>

                                    <!-- t2v 任务：横屏/竖屏选择 -->
                                    <div v-if="selectedTaskId === 't2v'">
                                        <DropdownMenu
                                            :items="videoSizeOptions"
                                            :selected-value="getCurrentSizeValue()"
                                            :placeholder="t('sizeSettingDescription')"
                                            @select-item="handleSizeSelect"
                                            class="flex-1 max-w-50"
                                        />
                                    </div>

                                    <!-- 图片任务：下拉选择 -->
                                    <div v-else-if="selectedTaskId === 't2i' || selectedTaskId === 'i2i'">
                                        <div class="flex items-center gap-3">
                                            <DropdownMenu
                                                :items="imageSizeOptions"
                                                :selected-value="getCurrentSizeValue()"
                                                :placeholder="t('sizeSettingDescription')"
                                                @select-item="handleSizeSelect"
                                                class="flex-1 max-w-50"
                                            />

                                            <!-- 自定义尺寸输入 -->
                                            <div v-if="showCustomSize" class="flex items-center gap-1.5">
                                                <input
                                                    v-model="customWidth"
                                                    type="number"
                                                    min="1"
                                                    :placeholder="t('width')"
                                                    @input="applyCustomSize"
                                                    class="w-16 h-8 px-2 text-xs bg-white/95 dark:bg-[#1e1e1e]/95 border border-black/8 dark:border-white/8 rounded-lg text-[#1d1d1f] dark:text-[#f5f5f7] focus:outline-none focus:ring-1 focus:ring-[color:var(--brand-primary)]/30 dark:focus:ring-[color:var(--brand-primary-light)]/40 focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/50 transition-all duration-200 tracking-tight placeholder:text-[#86868b]/60 dark:placeholder:text-[#98989d]/60">
                                                <span class="text-[#86868b] dark:text-[#98989d] text-xs font-light">×</span>
                                                <input
                                                    v-model="customHeight"
                                                    type="number"
                                                    min="1"
                                                    :placeholder="t('height')"
                                                    @input="applyCustomSize"
                                                    class="w-16 h-8 px-2 text-xs bg-white/95 dark:bg-[#1e1e1e]/95 border border-black/8 dark:border-white/8 rounded-lg text-[#1d1d1f] dark:text-[#f5f5f7] focus:outline-none focus:ring-1 focus:ring-[color:var(--brand-primary)]/30 dark:focus:ring-[color:var(--brand-primary-light)]/40 focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/50 transition-all duration-200 tracking-tight placeholder:text-[#86868b]/60 dark:placeholder:text-[#98989d]/60">
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <!-- 提交按钮 - Apple 极简风格 -->
                                <div class="flex justify-center mt-8">
                                    <button @click="handleSubmitTask" :disabled="submitting || templateLoading"
                                            class="gap-3 cursor-pointer relative bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-full px-10 py-4 text-base font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_8px_24px_rgba(0,0,0,0.12)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.3)] active:scale-100 transition-all duration-200 ease-out min-w-[250px] max-w-[400px] tracking-tight disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
                                            :class="{ 'disabled': submitting || templateLoading }">

                                        <i v-if="submitting" class="fas fa-spinner fa-spin text-lg mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <i v-else-if="templateLoading" class="fas fa-spinner fa-spin text-lg mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                        <i v-else class="fi fi-sr-select text-lg text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] transition-all duration-200 pointer-events-none"></i>
                                        <span class="pl-2 text-base font-semibold transition-all duration-200 pointer-events-none">
                                            {{ submitting ? t('submitting') : templateLoading ? '模板加载中...' : (selectedTaskId === 't2i' || selectedTaskId === 'i2i') ? (t('generateImage') || '生成图片') : t('generateVideo') }}
                                        </span>
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
                            <div
                                ref="featuredMasonryRef"
                                class="relative"
                                :style="{ height: masonryHeight + 'px' }">
                            <!-- 随机列 -->
                            <div v-for="(column, columnIndex) in templatesWithRandomColumns.columns" :key="columnIndex"
                                    data-masonry-column
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
                                            :poster="item?.inputs?.input_image ? getTemplateFileUrl(item.inputs.input_image,'images') : undefined"
                                            class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                            preload="auto" playsinline webkit-playsinline
                                            @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                                            @loadeddata="handleMasonryVideoLoaded($event)"
                                            @ended="handleMasonryVideoEnded($event)"
                                            @error="handleMasonryVideoError($event)"></video>
                                    <!-- 图片缩略图 -->
                                        <img v-else-if="item?.inputs?.input_image"
                                        :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                                        :alt="item.params?.prompt || '模板图片'"
                                        class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                        @load="handleMasonryImageLoaded"
                                        @error="handleMasonryImageError" />
                                        <!-- 如果没有图片，显示占位符 -->
                                        <div v-else class="w-full h-[200px] flex items-center justify-center bg-[#f5f5f7] dark:bg-[#1c1c1e]">
                                            <i class="fas fa-image text-3xl text-[#86868b]/30 dark:text-[#98989d]/30"></i>
                                        </div>
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

            <!-- 脸部编辑模态框 - 显示放大图片和可拖拽的边界框 -->
            <div v-if="showFaceEditModal"
                class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 dark:bg-black/80 backdrop-blur-sm"
                @click="closeFaceEditModal">
                <div
                    @click.stop
                    class="relative bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] rounded-2xl p-6 max-w-4xl max-h-[90vh] overflow-auto shadow-[0_12px_32px_rgba(0,0,0,0.6)] dark:shadow-[0_12px_32px_rgba(0,0,0,0.8)]">
                    <!-- 关闭按钮 -->
                    <button
                        @click="closeFaceEditModal"
                        class="absolute top-4 right-4 w-10 h-10 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 z-10"
                        :title="t('close') || '关闭'">
                        <i class="fas fa-times text-sm"></i>
                    </button>

                    <!-- 标题 -->
                    <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-4 tracking-tight">
                        {{ isAddingNewFace ? (t('addNewRole') || '新增角色') : (t('adjustFaceBox') || '调整人脸边界框') }}
                    </h3>

                    <!-- 图片容器 -->
                    <div
                        ref="imageContainerRef"
                        class="relative inline-block max-w-full"
                        :style="imageNaturalSize.width > 0 && imageNaturalSize.height > 0 && !imageLoaded ? {
                            width: `${Math.min(imageNaturalSize.width, 800)}px`,
                            height: `${Math.min(imageNaturalSize.height, 600)}px`,
                            aspectRatio: `${imageNaturalSize.width} / ${imageNaturalSize.height}`
                        } : {}">
                        <!-- 占位符 - 图片加载前显示 -->
                        <div
                            v-show="!imageLoaded"
                            class="absolute inset-0 w-full h-full min-w-[400px] min-h-[300px] bg-[#f5f5f7] dark:bg-[#1e1e1e] rounded-xl flex items-center justify-center z-10">
                            <div class="flex flex-col items-center gap-3">
                                <i class="fas fa-spinner fa-spin text-2xl text-[#86868b] dark:text-[#98989d]"></i>
                                <span class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('loading') || '加载中...' }}</span>
                            </div>
                        </div>
                        <!-- 实际图片 - 始终渲染，但加载完成后才显示 -->
                        <img
                            :src="originalImageUrl"
                            alt="Face Edit"
                            class="max-w-full max-h-[70vh] h-auto object-contain rounded-xl"
                            :class="{ 'opacity-0': !imageLoaded }"
                            @load="handleFaceEditImageLoad"
                            @error="handleFaceEditImageError">

                        <!-- 遮罩层 - 框外区域变暗 -->
                        <svg
                            v-if="editingFaceBbox.length === 4 && getBboxStyle.left"
                            class="absolute inset-0 pointer-events-none z-[5]"
                            :style="{
                                left: 0,
                                top: 0,
                                width: '100%',
                                height: '100%'
                            }">
                            <defs>
                                <mask id="bbox-mask">
                                    <rect width="100%" height="100%" fill="white"/>
                                    <rect
                                        :x="getBboxStyle.left"
                                        :y="getBboxStyle.top"
                                        :width="getBboxStyle.width"
                                        :height="getBboxStyle.height"
                                        fill="black"/>
                                </mask>
                            </defs>
                            <rect
                                width="100%"
                                height="100%"
                                fill="rgba(0,0,0,0.5)"
                                mask="url(#bbox-mask)"/>
                        </svg>

                        <!-- 角色名字标签 - 显示在边界框上方 -->
                        <div
                            v-if="editingFaceBbox.length === 4 && getBboxStyle.left && getRoleNameLabelStyle.roleName"
                            :style="{
                                left: getRoleNameLabelStyle.left,
                                top: getRoleNameLabelStyle.top,
                                transform: getRoleNameLabelStyle.transform
                            }"
                            class="absolute px-2 py-1 text-xs font-medium text-white bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-md shadow-lg whitespace-nowrap pointer-events-none z-10">
                            {{ getRoleNameLabelStyle.roleName }}
                        </div>

                        <!-- 边界框 -->
                        <div
                            v-if="editingFaceBbox.length === 4 && getBboxStyle.left"
                            :style="getBboxStyle"
                            @mousedown="(e) => startDragBbox(e, 'move')"
                            class="absolute border-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] cursor-move bg-transparent hover:bg-[color:var(--brand-primary)]/5 dark:hover:bg-[color:var(--brand-primary-light)]/5 transition-colors duration-200"
                            :class="{ 'ring-2 ring-[color:var(--brand-primary)]/50 dark:ring-[color:var(--brand-primary-light)]/50 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/10': isDraggingBbox }"
                            style="box-sizing: border-box;">

                            <!-- 四个角的拖拽手柄 -->
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-nw')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    left: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    top: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full border-2 border-white dark:border-[#2c2c2e] shadow-lg cursor-nw-resize hover:scale-110 transition-transform z-20"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-ne')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    right: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    top: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full border-2 border-white dark:border-[#2c2c2e] shadow-lg cursor-ne-resize hover:scale-110 transition-transform z-20"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-sw')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    left: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    bottom: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full border-2 border-white dark:border-[#2c2c2e] shadow-lg cursor-sw-resize hover:scale-110 transition-transform z-20"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-se')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    right: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    bottom: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full border-2 border-white dark:border-[#2c2c2e] shadow-lg cursor-se-resize hover:scale-110 transition-transform z-20"></div>

                            <!-- 四个边的拖拽手柄 -->
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-n')"
                                :style="{
                                    width: 'calc(100% + 16px)',
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    left: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    top: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute cursor-n-resize hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/20 transition-colors rounded-t z-10"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-s')"
                                :style="{
                                    width: 'calc(100% + 16px)',
                                    height: `${getBboxStyle.indicatorSize || 16}px`,
                                    left: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    bottom: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute cursor-s-resize hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/20 transition-colors rounded-b z-10"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-w')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: 'calc(100% + 16px)',
                                    left: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    top: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute cursor-w-resize hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/20 transition-colors rounded-l z-10"></div>
                            <div
                                @mousedown.stop="(e) => startDragBbox(e, 'resize-e')"
                                :style="{
                                    width: `${getBboxStyle.indicatorSize || 16}px`,
                                    height: 'calc(100% + 16px)',
                                    right: `${-(getBboxStyle.indicatorSize || 16) / 2}px`,
                                    top: `${-(getBboxStyle.indicatorSize || 16) / 2}px`
                                }"
                                class="absolute cursor-e-resize hover:bg-[color:var(--brand-primary)]/20 dark:hover:bg-[color:var(--brand-primary-light)]/20 transition-colors rounded-r z-10"></div>
                        </div>
                    </div>

                    <!-- 操作按钮 -->
                    <div class="flex items-center justify-end gap-3 mt-6">
                        <button
                            @click="closeFaceEditModal"
                            class="px-4 py-2 text-sm font-medium text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-black/4 dark:hover:bg-white/6 rounded-lg transition-all duration-200 tracking-tight">
                            {{ t('cancel') || '取消' }}
                        </button>
                        <button
                            @click="saveFaceBbox"
                            class="px-4 py-2 text-sm font-medium text-white bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] hover:opacity-90 rounded-lg transition-all duration-200 tracking-tight">
                            {{ t('save') || '保存' }}
                        </button>
                    </div>
                </div>
            </div>

</template>

<style scoped>
/* Apple 风格极简设计 - 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
</style>
