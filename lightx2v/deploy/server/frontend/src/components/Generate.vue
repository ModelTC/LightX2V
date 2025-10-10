<script setup>
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { watch, onMounted, computed, ref, nextTick, onUnmounted } from 'vue'
import ModelDropdown from './ModelDropdown.vue'
import MediaTemplate from './MediaTemplate.vue'

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

import {
            submitting,
            templateLoading,
            showTaskTypeMenu,
            showModelMenu,
            isRecording,
            recordingDuration,
            startRecording,
            stopRecording,
            formatRecordingDuration,
            getCurrentForm,
            getCurrentImagePreview,
            getCurrentAudioPreview,
            availableTaskTypes,
            availableModelClasses,
            currentTaskHints,
            currentHintIndex,
            selectedTaskId,
            isCreationAreaExpanded,
            isContracting,
            expandCreationArea,
            contractCreationArea,
            handleImageUpload,
            selectTask,
            selectModel,
            triggerImageUpload,
            triggerAudioUpload,
            removeImage,
            removeAudio,
            handleAudioUpload,
            selectImageTemplate,
            selectAudioTemplate,
            previewAudioTemplate,
            imageTemplates,
            audioTemplates,
            showImageTemplates,
            showAudioTemplates,
            mediaModalTab,
            templatePaginationInfo,
            templateCurrentPage,
            templatePageInput,
            imageHistory,
            audioHistory,
            showPromptModal,
            promptModalTab,
            submitTask,
            goToTemplatePage,
            jumpToTemplatePage,
            getVisibleTemplatePages,
            getTemplateFileUrl,
            clearPrompt,
            getTaskTypeIcon,
            getTaskTypeName,
            getPromptPlaceholder,
            getHistoryImageUrl,
            getCurrentImagePreviewUrl,
            getCurrentAudioPreviewUrl,
            handleAudioError,
            getImageHistory,
            getAudioHistory,
            selectImageHistory,
            selectAudioHistory,
            previewAudioHistory,
            clearImageHistory,
            clearAudioHistory,
            getAudioMimeType,
            selectedModel,
            // 精选模版相关
            featuredTemplates,
            featuredTemplatesLoading,
            loadFeaturedTemplates,
            getRandomFeaturedTemplates,
            previewTemplateDetail,
            useTemplate,
            applyTemplateImage,
            applyTemplateAudio,
            playVideo,
            pauseVideo,
            toggleVideoPlay,
            onVideoLoaded,
            onVideoError,
            onVideoEnded,
            handleThumbnailError,
            switchToInspirationView,
        } from '../utils/other'

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
                <div class="flex-1 flex flex-col min-h-0 mobile-content main-scrollbar content-area">
                    <!-- 生成视频区域 -->
                    <div class="flex-1 flex flex-col">
                        <!-- 内容区域 -->
                        <div class="flex-1 p-6">


                        <!-- 任务创建面板 -->
                        <div class="max-w-4xl mx-auto" id="task-creator">
                            <!-- 合并的创作区域 -->
                            <div class="creation-area-container">

                                <div class="default-state-container">
                                    <!-- 两个并列的下拉菜单 -->
                                    <div class="flex justify-center gap-10 mb-6">
                                        <!-- 任务类型下拉菜单 -->
                                        <ModelDropdown
                                            :available-models="availableTaskTypes.map(taskType => getTaskTypeName(taskType))"
                                            :selected-model="getTaskTypeName(selectedTaskId)"
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

                                        <h2 class="text-4xl font-bold text-laser-purple mb-6">{{ t('whatDoYouWantToDo') }}</h2>

                                        <!-- 动态滚动提示 -->
                                        <div class="hint-container mb-8 pb-10">
                                            <div class="hint-text text-gray-400 text-lg min-h-[60px] flex items-center justify-center">
                                                <transition name="hint-fade" mode="out-in">
                                                    <p :key="currentHintIndex" class="text-center">
                                                        {{ currentTaskHints[currentHintIndex] }}
                                                    </p>
                                                </transition>
                                            </div>
                                            <!-- 提示指示器 -->
                                            <div class="flex justify-center mt-4 space-x-2">
                                                <div v-for="(hint, index) in currentTaskHints" :key="index"
                                                    class="w-2 h-2 rounded-full transition-all duration-300"
                                                    :class="index === currentHintIndex ? 'bg-laser-purple' : 'bg-gray-600'">
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 展开开关 -->
                                    <div class="relative group cursor-pointer max-w-3/5" @click="expandCreationArea">
                                        <button
                                            class="relative w-full bg-dark-light/80 border border-laser-purple rounded-full pl-10 pr-10 py-6 text-base hover:border-laser-purple transition-all duration-300 resize-none hover:shadow-2xl"
                                        >
                                        <i class="fi fi-sr-cursor-finger-click text-lg text-gradient-icon transition-all duration-300 pointer-events-none group-hover:drop-shadow-[0_0_8px_rgba(168,85,247,0.6)] group-hover:animate-pulse"></i>
                                        <span class="pl-2 text-base font-bold text-gradient-icon transition-all duration-300 pointer-events-none group-hover:drop-shadow-[0_0_8px_rgba(168,85,247,0.6)] group-hover:animate-pulse">{{ t('startCreatingVideo') }}</span>
                                        </button>

                                        <!-- 装饰性边框 -->
                                        <div class="absolute inset-0 rounded-full border border-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                                    </div>
                                </div>

                                <!-- 展开状态：素材区域 -->
                                <div v-if="isCreationAreaExpanded" class="mb-8 prompt-input-section">
                                    <!-- 中心文字 -->
                                    <div class="text-center">

                                        <h2 class="text-4xl font-bold text-laser-purple mb-6 animate-fade-in">{{ t('whatMaterialsDoYouNeed') }}</h2>

                                        <p class="text-gray-400 text-lg mb-8 transition-all duration-300">
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

                                 <!-- 收缩开关 -->

                                <div
                                    class="creation-area transition-all duration-500 ease-out max-w-10xl mx-auto"
                                    @click.stop>
                                    <!-- 收起按钮 -->
                                    <div class="flex justify-center mb-4">
                                        <button @click="contractCreationArea"
                                                class="flex items-center gap-2 px-4 py-2 text-gray-400 hover:text-laser-purple transition-all duration-300 hover:bg-laser-purple/10 rounded-lg group"
                                                :class="{ 'animate-pulse': isContracting }">
                                            <i class="fas fa-compress-alt text-sm transition-transform duration-200 group-hover:scale-110"
                                               :class="{ 'animate-spin': isContracting }"></i>
                                            <span class="text-sm font-medium">
                                                {{ t('collapseCreationArea') }}
                                            </span>
                                            <i class="fas fa-chevron-up text-xs transition-transform duration-200 group-hover:translate-y-[-2px]"
                                               :class="{ 'animate-bounce': isContracting }"></i>
                                        </button>
                                    </div>

                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v'" class="upload-section">
                                    <!-- 上传图片 -->
                                    <div v-if="selectedTaskId === 'i2v' || selectedTaskId === 's2v'">
                                        <!-- 图片历史和素材库 -->
                                        <div class="flex justify-between items-center mb-2">
                                                <label class="block text-sm text-gray-400 items-center">
                                                                {{ t('image') }}
                                                </label>
                                        </div>
                                        <!-- 上传图片 -->
                                        <div class="upload-area"
                                            @drop="handleImageDrop"
                                            @dragover="handleDragOver"
                                            @dragenter="handleDragEnter"
                                            @dragleave="handleDragLeave"
                                            :class="{ 'drag-over': isDragOver }"
                                            >
                                            <!-- 默认上传界面 -->
                                            <div v-if="!getCurrentImagePreview()" class="upload-content">
                                            <p class="text-base text-white font-bold mb-4">{{ t('uploadImage') }}</p>
                                            <p class="text-xs text-gray-400 mb-4">{{ t('supportedImageFormats') }}</p>
                                            <div class="flex items-center justify-center space-x-6">
                                                        <div class="flex flex-col items-center space-y-2">
                                                            <button
                                                                class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                                @click="triggerImageUpload" 
                                                                :title="t('uploadImage')">
                                                                <i class="fas fa-upload text-lg"></i>
                                                            </button>
                                                            <span class="text-xs text-gray-300">{{ t('upload') }}</span>
                                                        </div>
                                                        <div class="flex flex-col items-center space-y-2">
                                                            <button
                                                                @click.stop="showImageTemplates = true; mediaModalTab = 'history'; getImageHistory()"
                                                                class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                                :title="t('templates')">
                                                                <i class="fas fa-history text-lg"></i>
                                                            </button>
                                                            <span class="text-xs text-gray-300">{{ t('templates') }}</span>
                                                        </div>
                                            </div>
                                            </div>

                                            <!-- 图片预览 -->
                                            <div v-if="getCurrentImagePreview()" class="image-preview group">
                                                    <img :src="getCurrentImagePreviewUrl()" alt="t('previewImage')"
                                                        class="w-full h-full object-cover rounded-lg transition-all duration-300 group-hover:brightness-50">

                                                <!-- 悬停时显示的操作按钮，位置在中下方 -->
                                                    <div
                                                        class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-300">
                                                    <div class="flex space-x-3">
                                                            <button @click.stop="removeImage"
                                                            class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                                :title="t('deleteImage')">
                                                            <i class="fas fa-trash text-lg"></i>
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                                <input type="file" ref="imageInput" @change="handleImageUpload" accept="image/*"
                                                style="display: none;">
                                            </div>
                                    </div>

                                    <!-- 上传音频 -->
                                    <div v-if="selectedTaskId === 's2v'">
                                        <!-- 音频历史和素材库 -->
                                        <div class="flex justify-between items-center mb-2">
                                                        <label class="block text-sm text-gray-400 flex items-center">
                                                                        {{ t('audio') }}
                                                        </label>
                                        </div>
                                        <!-- 上传音频 -->
                                        <div class="upload-area"
                                            @drop="handleAudioDrop"
                                            @dragover="handleDragOver"
                                            @dragenter="handleDragEnter"
                                            @dragleave="handleDragLeave"
                                            :class="{ 'drag-over': isDragOver }"
                                            >
                                        <!-- 默认上传界面 -->
                                            <div v-if="!getCurrentAudioPreview()" class="upload-content"
                                                >
                                                <p class="text-base text-white font-bold mb-4">{{ t('uploadAudio') }}</p>
                                                <p class="text-xs text-gray-400 mb-4">{{ t('supportedAudioFormats') }}</p>
                                            <div class="flex items-center justify-center space-x-6">
                                                    <div class="flex flex-col items-center space-y-2">
                                                        <button
                                                            @click.stop="showAudioTemplates = true; mediaModalTab = 'history'; getAudioHistory()"
                                                            class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                            :title="t('templates')">
                                                            <i class="fas fa-history text-lg"></i>
                                                        </button>
                                                        <span class="text-xs text-gray-300">{{ t('templates') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center space-y-2">
                                                        <button
                                                            class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                            @click="triggerAudioUpload" 
                                                            :title="t('uploadAudio')">
                                                            <i class="fas fa-upload text-lg"></i>
                                                        </button>
                                                        <span class="text-xs text-gray-300">{{ t('upload') }}</span>
                                                    </div>
                                                    <div class="flex flex-col items-center space-y-2">
                                                        <button @click.stop="isRecording ? stopRecording() : startRecording()"
                                                        class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                            :title="isRecording ? t('stopRecording') : t('recordAudio')"
                                                            :class="{ 'bg-red-500/80': isRecording }">
                                                        <i class="fas fa-microphone text-lg" :class="{ 'animate-pulse': isRecording, 'text-red-500': isRecording }"></i>
                                                    </button>
                                                        <span class="text-xs text-gray-300">{{ isRecording ? formatRecordingDuration(recordingDuration) : t('recordAudio') }}</span>
                                                    </div>

                                        </div>
                                            </div>

                                        <!-- 音频预览 -->
                                            <div v-if="getCurrentAudioPreview()" class="audio-preview group">
                                            <audio controls class="w-full h-full" @error="handleAudioError" @loadstart="console.log('音频开始加载')" @canplay="console.log('音频可以播放')">
                                                    <source :src="getCurrentAudioPreviewUrl()" :type="getAudioMimeType()" preload="metadata">
                                            </audio>

                                            <!-- 悬停时显示的操作按钮，位置在中下方 -->
                                                <div
                                                    class="absolute inset-x-0 bottom-4 flex items-center justify-center opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity duration-300">
                                                <div class="flex space-x-3">
                                                        <button @click.stop="removeAudio"
                                                        class="w-12 h-12 flex items-center justify-center bg-white/15 text-white p-3 rounded-full transition-all duration-200 hover:scale-110 shadow-lg"
                                                            :title="t('deleteAudio')">
                                                        <i class="fas fa-trash text-lg"></i>
                                                    </button>
                                                </div>
                                            </div>
                                            </div>

                                            <input type="file" ref="audioInput" @change="handleAudioUpload" accept="audio/*"
                                            style="display: none;">
                                        </div>
                                    </div>
                                </div>

                                        <!-- 提示词输入区域 -->
                                        <div class="flex justify-between items-center mb-2">
                                            <label class="block text-sm text-gray-400 flex items-center">
                                                    {{ t('prompt') }}
                                                    <button @click="showPromptModal = true; promptModalTab = 'templates'"
                                                        class="text-xs text-gray-400 hover:text-gradient-primary transition-colors pl-3"
                                                        :title="t('promptTemplates')">
                                                    <i class="fas fa-lightbulb"></i>
                                                </button>
                                            </label>
                                            <div class="text-xs text-gray-400">
                                                {{ getCurrentForm().prompt?.length || 0 }} / 1000
                                            </div>
                                        </div>
                                        <div class="relative group cursor-pointer">
                                            <textarea v-model="getCurrentForm().prompt"
                                                class="relative w-full bg-dark-light/80 border border-laser-purple/30 rounded-xl p-6 text-base min-h-[100px] focus:ring-1 transition-all duration-300 resize-none main-scrollbar focus:shadow-2x placeholder-gray-500"
                                                :placeholder="getPromptPlaceholder()"
                                                rows="4"
                                                maxlength="500"
                                                required></textarea>

                                            <!-- 装饰性边框 -->
                                            <div class="absolute inset-0 rounded-xl border border-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                                        </div>

                                        <div class="flex justify-between items-center">
                                            <button @click="clearPrompt"
                                                class="flex items-center text-sm rounded-lg px-2 transition-all duration-200 text-gray-400 hover:text-laser-purple hover:bg-laser-purple/10 group">
                                                <i class="fas fa-sync-alt text-sm mr-2 group-hover:rotate-180 transition-transform duration-300"></i>
                                                {{ t('clear') }}
                                            </button>

                                        </div>
                                <!-- 提交按钮 -->
                                <div class="flex justify-center mt-6">
                                    <button @click="submitTask" :disabled="submitting || templateLoading"
                                        class="generate-button btn-primary"
                                        :class="{ 'disabled': submitting || templateLoading }">
                                        <i v-if="submitting" class="fas fa-spinner fa-spin text-xl mr-3"></i>
                                        <i v-else-if="templateLoading" class="fas fa-spinner fa-spin text-xl mr-3"></i>
                                        <i v-else class="fas fa-play text-xl mr-3"></i>
                                        {{ submitting ? t('submitting') : templateLoading ? '模板加载中...' : t('generateVideo') }}
                                    </button>
                                </div>

                                </div>

                                </div>

                            </div>
                </div>

                        </div>
                </div>
                        </div>
                    

                    <!-- 精选模版区域 -->
                    <div v-if="currentFeaturedTemplates.length > 0" class="flex-1 flex flex-col min-h-0 border-t lg:border-t-0 lg:border-l border-gray-700/30">
                        <div class="flex-1 p-4 lg:p-6">
                            <!-- 控制区域 -->
                            <div class="flex-col flex items-center justify-center mb-4 lg:mb-6 gap-3 lg:gap-4">
                                <!-- 左侧：发现文字和随机按钮 -->
                                <div class="flex items-center gap-3 lg:gap-4">
                                    <h2 class="text-2xl lg:text-3xl font-bold text-white">{{ t('discover') }}</h2>
                                    <!-- 随机图标按钮 -->
                                    <button @click="refreshRandomTemplates" 
                                            :disabled="featuredTemplatesLoading"
                                            class="w-8 h-8 lg:w-10 lg:h-10 flex items-center justify-center bg-laser-purple/20 hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                                            :title="t('refreshRandomTemplates')">
                                        <i class="fas fa-random text-sm lg:text-lg" 
                                           :class="{ 'animate-spin': featuredTemplatesLoading }"></i>
                                    </button>
                                </div>
                                <!-- 右侧：更多按钮 -->
                                <button @click="switchToInspirationView()" 
                                    class="flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-white hover:bg-gray-600/20 rounded transition-all duration-200"
                                    :title="t('viewMore')">
                                <span class="text-sm">{{ t('more') }}</span>
                                <i class="fas fa-arrow-right text-xs"></i>
                            </button>
                            </div>

                            <!-- 精选模版随机列布局 -->
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
                                <!-- 列内的模版卡片 -->
                                <div v-for="item in column.templates" :key="item.task_id"
                                        class="mb-3 group relative bg-dark-light rounded-xl overflow-hidden border border-gray-700/50 hover:border-laser-purple/40 transition-all duration-300 hover:shadow-laser/20">
                                <!-- 视频缩略图区域 -->
                                <div class="cursor-pointer bg-gray-800 relative flex flex-col"
                                @click="previewTemplateDetail(item)"
                                :title="t('viewTemplateDetail')">
                                        <!-- 视频预览 -->
                                        <video v-if="item?.outputs?.output_video"
                                            :src="getTemplateFileUrl(item.outputs.output_video,'videos')"
                                            :poster="getTemplateFileUrl(item.inputs.input_image,'images')"
                                            class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                                            preload="auto" playsinline webkit-playsinline
                                            @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                                            @loadeddata="onVideoLoaded($event)"
                                            @ended="onVideoEnded($event)"
                                            @error="onVideoError($event)"></video>
                                    <!-- 图片缩略图 -->
                                        <img v-else
                                        :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                                        :alt="item.params?.prompt || '模板图片'"
                                        class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                                        @error="handleThumbnailError" />
                                        <!-- 移动端播放按钮 -->
                                        <button v-if="item?.outputs?.output_video" 
                                            @click.stop="toggleVideoPlay($event)"
                                            class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/70 transition-colors z-20">
                                            <i class="fas fa-play text-sm"></i>
                                        </button>
                                    <!-- 悬浮操作按钮（下方居中，仅桌面端） -->
                                    <div
                                        class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
                                        <div class="flex space-x-3 pointer-events-auto">
                                            <button @click.stop="applyTemplateImage(item)"
                                                class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                                                :title="t('applyImage')">
                                                <i class="fas fa-image text-sm"></i>
                                            </button>
                                            <button @click.stop="applyTemplateAudio(item)"
                                                class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                                                :title="t('applyAudio')">
                                                <i class="fas fa-music text-sm"></i>
                                            </button>
                                            <button @click.stop="useTemplate(item)"
                                                class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
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
</template>

<style scoped>
/* 生成按钮样式 - 简约大气 */
.generate-button {
    padding: 1rem 3rem;
    background: #8b5cf6;
    border: none;
    border-radius: 0.5rem;
    color: white;
    font-size: 1.125rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 200px;
    letter-spacing: 0.025em;
    
    /* 简约阴影 */
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

.generate-button:hover:not(.disabled) {
    background: #7c3aed;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
}

.generate-button:active:not(.disabled) {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
}

.generate-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    background: #6b7280;
    box-shadow: 0 2px 8px rgba(107, 114, 128, 0.2);
}

.generate-button.disabled:hover {
    transform: none;
    box-shadow: 0 2px 8px rgba(107, 114, 128, 0.2);
}

/* 响应式设计 */
@media (max-width: 768px) {
    .generate-button {
        padding: 0.875rem 2.5rem;
        font-size: 1rem;
        min-width: 180px;
    }
}

/* 拖拽样式 */
.upload-area.drag-over {
    border-color: #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.1) !important;
    transform: scale(1.02);
    transition: all 0.3s ease;
}

.upload-area.drag-over .upload-content {
    opacity: 0.7;
}

.upload-area.drag-over::before {
    content: '拖拽文件到这里';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #8b5cf6;
    font-size: 1.125rem;
    font-weight: 600;
    z-index: 10;
    pointer-events: none;
}
</style>