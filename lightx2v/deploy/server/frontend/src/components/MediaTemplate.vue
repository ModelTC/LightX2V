<script setup>
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// 音频播放状态管理
const playingAudioId = ref(null)
const audioDurations = ref({})

import {
    getTemplateFileUrl,
    getHistoryImageUrl,
    goToTemplatePage,
    jumpToTemplatePage,
    getVisibleTemplatePages,
    selectImageHistory,
    selectImageTemplate,
    selectAudioHistory,
    selectAudioTemplate,
    previewAudioHistory,
    previewAudioTemplate,
    stopAudioPlayback,
    setAudioStopCallback,
    clearImageHistory,
    clearAudioHistory,
    templatePaginationInfo,
    templateCurrentPage,
    templatePageInput,
    showImageTemplates,
    showAudioTemplates,
    imageHistory,
    audioHistory,
    imageTemplates,
    audioTemplates,
    mergedTemplates,
    mediaModalTab,
    getImageHistory,
    getAudioHistory,
    isPageLoading
} from '../utils/other'

// 格式化音频时长
const formatDuration = (seconds) => {
    if (!seconds || isNaN(seconds)) return '--:--'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// 获取音频时长
const getAudioDuration = async (url, id) => {
    if (audioDurations.value[id]) return audioDurations.value[id]

    return new Promise((resolve) => {
        const audio = new Audio()
        audio.addEventListener('loadedmetadata', () => {
            audioDurations.value[id] = audio.duration
            resolve(audio.duration)
        })
        audio.addEventListener('error', () => {
            resolve(0)
        })
        audio.src = url
    })
}

// 处理音频预览播放/停止
const handleAudioPreview = async (item, isTemplate = false) => {
    const id = isTemplate ? `template_${item.filename}` : `history_${item.filename}`
    const url = isTemplate ? getTemplateFileUrl(item.filename, 'audios') : item.url

    // 如果当前正在播放这个音频，则停止
    if (playingAudioId.value === id) {
        playingAudioId.value = null
        stopAudioPlayback() // 调用停止音频播放函数
        return
    }

    // 停止其他正在播放的音频
    playingAudioId.value = null
    stopAudioPlayback() // 先停止当前播放的音频

    // 播放新音频
    try {
        // 设置停止回调，当音频停止时更新UI状态
        setAudioStopCallback(() => {
            playingAudioId.value = null
        })

        if (isTemplate) {
            previewAudioTemplate(item)
        } else {
            previewAudioHistory({ url })
        }
        playingAudioId.value = id

        // 获取音频时长
        await getAudioDuration(url, id)
    } catch (error) {
        console.error('音频播放失败:', error)
    }
}

// 检查是否正在播放
const isPlaying = (item, isTemplate = false) => {
    const id = isTemplate ? `template_${item.filename}` : `history_${item.filename}`
    return playingAudioId.value === id
}

// 获取音频时长显示
const getDurationDisplay = (item, isTemplate = false) => {
    const id = isTemplate ? `template_${item.filename}` : `history_${item.filename}`
    return formatDuration(audioDurations.value[id])
}

// 预加载音频时长
const preloadAudioDurations = (items, isTemplate = false) => {
    items.forEach(item => {
        const id = isTemplate ? `template_${item.filename}` : `history_${item.filename}`
        const url = isTemplate ? getTemplateFileUrl(item.filename, 'audios') : item.url

        // 如果已经有时长数据，跳过
        if (audioDurations.value[id] || !url) return

        // 异步加载时长
        getAudioDuration(url, id)
    })
}

// 监听音频历史和模板列表变化，预加载时长
watch(audioHistory, (newHistory) => {
    if (newHistory && newHistory.length > 0) {
        preloadAudioDurations(newHistory, false)
    }
}, { immediate: true, deep: true })

watch(audioTemplates, (newTemplates) => {
    if (newTemplates && newTemplates.length > 0) {
        preloadAudioDurations(newTemplates, true)
    }
}, { immediate: true, deep: true })
</script>

<template>

                        <!-- 模板选择浮窗 - Apple 极简风格 -->
                        <div v-cloak>
                            <div v-if="showImageTemplates || showAudioTemplates"
                                class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
                                @click="showImageTemplates = false; showAudioTemplates = false">
                                <div class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] backdrop-saturate-[180%] border border-black/8 dark:border-white/8 rounded-3xl px-8 py-8 max-w-4xl w-full mx-6 h-[90vh] overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.12)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.4)]"
                                    @click.stop>
                                    <!-- 浮窗头部 - Apple 风格 -->
                                    <div class="flex items-center justify-between mb-8">
                                        <h3 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
                                                <i v-if="showImageTemplates"
                                                    class="fas fa-image text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                <i v-if="showAudioTemplates"
                                                    class="fas fa-music text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                {{ showImageTemplates ? t('imageTemplates') : t('audioTemplates') }}
                                        </h3>
                                        <button @click="showImageTemplates = false; showAudioTemplates = false"
                                                class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
                                            <i class="fas fa-times text-base"></i>
                                        </button>
                                    </div>

                                    <!-- 标签页切换 - Apple 风格 -->
                                    <div class="flex gap-2 mb-8">
                                            <button
                                                @click="mediaModalTab = 'history'; showImageTemplates && getImageHistory(); showAudioTemplates && getAudioHistory()"
                                                class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight" :class="mediaModalTab === 'history'
                                                    ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                                    : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                            <i class="fas fa-history mr-2"></i>
                                                {{ t('history') }}
                                        </button>
                                        <button @click="mediaModalTab = 'templates'"
                                                class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight" :class="mediaModalTab === 'templates'
                                                    ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                                    : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                            <i class="fas fa-layer-group mr-2"></i>
                                                {{ t('templates') }}
                                        </button>
                                    </div>

                                    <!-- 图片历史记录 - Apple 风格 -->
                                          <div v-if="showImageTemplates && mediaModalTab === 'history'"
                                             class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar pr-6 pl-1">
                                            <div v-if="imageHistory.length === 0"
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-history text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                                            </div>
                                                <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium mb-2 tracking-tight">{{ t('noHistoryRecords') }}</p>
                                                <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('imageHistoryAutoSave') }}</p>
                                        </div>
                                        <div v-else class="space-y-4 pt-2">
                                            <div class="flex items-center justify-between mb-6 px-1">
                                                    <span class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('total') }} {{ imageHistory.length }}
                                                        {{ t('records') }}</span>
                                                <button @click="clearImageHistory"
                                                        class="text-xs text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 transition-colors flex items-center gap-1.5 tracking-tight"
                                                        :title="t('clearHistory')">
                                                    <i class="fas fa-trash"></i>
                                                        {{ t('clear') }}
                                                </button>
                                            </div>
                                            <div class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4 px-1">
                                                <div v-for="(history, index) in imageHistory" :key="index"
                                                    @click="selectImageHistory(history)"
                                                    class="break-inside-avoid mb-4 relative group cursor-pointer rounded-2xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)]">
                                                        <img :src="getHistoryImageUrl(history)" :alt="history.filename"
                                                            class="w-full h-auto object-contain">
                                                        <div
                                                            class="absolute inset-0 bg-black/50 dark:bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                        <i class="fas fa-check text-white text-xl"></i>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                     <!-- 图片模板网格 - Apple 风格 -->
                                         <div v-if="showImageTemplates && mediaModalTab === 'templates'" class="pr-6 pl-1">

                                            <!-- 图片模板分页组件 - Apple 风格 -->
                                            <div v-if="templatePaginationInfo" class="mt-6">
                                                <div class="flex items-center justify-between text-xs mb-4">
                                                    <div class="flex items-center space-x-1 text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                        <span>{{ templatePaginationInfo.total }} {{ t('records') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="templatePaginationInfo.total_pages > 1" class="flex justify-center">
                                                    <nav class="isolate inline-flex gap-1" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToTemplatePage(templateCurrentPage - 1)"
                                                            :disabled="templateCurrentPage <= 1"
                                                            class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                                            <i class="fas fa-chevron-left text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisibleTemplatePages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToTemplatePage(page)"
                                                                :class="[
                                                                    'relative inline-flex items-center justify-center min-w-[36px] h-9 px-3 text-sm font-medium rounded-lg transition-all duration-200',
                                                                    page === templateCurrentPage
                                                                        ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                                                        : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'
                                                                ]"
                                                                :aria-current="page === templateCurrentPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                                            <span v-else class="relative inline-flex items-center px-2 text-sm font-semibold text-[#86868b] dark:text-[#98989d]">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToTemplatePage(templateCurrentPage + 1)"
                                                            :disabled="templateCurrentPage >= templatePaginationInfo.total_pages"
                                                            class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage >= templatePaginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                                            <i class="fas fa-chevron-right text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                                </div>
                                            </div>
                                         <div class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar pr-2 pt-2">
                                            <div class="space-y-4">
                                                <div v-if="isPageLoading" class="flex items-center justify-center">
                                                    <div class="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/90 dark:bg-[#2c2c2e]/90 border border-black/8 dark:border-white/8 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] shadow-[0_4px_16px_rgba(0,0,0,0.08)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.35)]">
                                                        <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span>{{ t('loading') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="mergedTemplates.filter(t => t.image).length > 0" class="columns-2 sm:columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4 px-1">
                                                    <div v-for="template in mergedTemplates.filter(t => t.image)" :key="template.id"
                                                        @click="selectImageTemplate(template.image)"
                                                        class="break-inside-avoid mb-4 relative group cursor-pointer rounded-2xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)]">
                                                            <img :src="template.image.url" :alt="template.image.filename"
                                                            class="w-full h-auto object-contain" preload="metadata">
                                                            <div
                                                                class="absolute inset-0 bg-black/50 dark:bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                            <i class="fas fa-check text-white text-2xl"></i>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div v-else
                                                    class="flex flex-col items-center justify-center py-12 text-center">
                                                    <div
                                                        class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                                                    <i class="fas fa-image text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                                                </div>
                                                    <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium tracking-tight">{{ t('noImageTemplates') }}</p>
                                                </div>
                                            </div>
                                         </div>

                                    </div>

                                    <!-- 音频历史记录 - Apple 风格 -->
                                          <div v-if="showAudioTemplates && mediaModalTab === 'history'"
                                             class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar pr-6 pl-1">
                                            <div v-if="audioHistory.length === 0"
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-history text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                                            </div>
                                                <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium mb-2 tracking-tight">{{ t('noHistoryRecords') }}</p>
                                                <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('audioHistoryAutoSave') }}</p>
                                        </div>
                                        <div v-else class="space-y-3 pt-2">
                                            <div class="flex items-center justify-between mb-6 px-1">
                                                    <span class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('total') }} {{ audioHistory.length }}
                                                        {{ t('records') }}</span>
                                                <button @click="clearAudioHistory"
                                                        class="text-xs text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 transition-colors flex items-center gap-1.5 tracking-tight"
                                                        :title="t('clearHistory')">
                                                    <i class="fas fa-trash"></i>
                                                        {{ t('clear') }}
                                                </button>
                                            </div>
                                            <div class="space-y-3 px-1">
                                                <div v-for="(history, index) in audioHistory" :key="index"
                                                    @click="selectAudioHistory(history)"
                                                    class="flex items-center gap-4 p-4 rounded-2xl border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all cursor-pointer bg-white/80 dark:bg-[#2c2c2e]/80 hover:bg-white dark:hover:bg-[#3a3a3c] hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] group">
                                                    <div class="w-12 h-12 rounded-xl overflow-hidden flex-shrink-0 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                                                        <img v-if="history.imageUrl" :src="history.imageUrl" :alt="history.filename" class="w-full h-full object-cover" @error="history.imageUrl = null" />
                                                        <i v-else class="fas fa-music text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
                                                    </div>
                                                    <div class="flex-1 min-w-0">
                                                        <div class="text-[#86868b] dark:text-[#98989d] text-sm flex items-center gap-2 tracking-tight">
                                                            <span>{{ t('historyAudio') }}</span>
                                                            <span class="text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]">•</span>
                                                            <span>{{ getDurationDisplay(history, false) }}</span>
                                                        </div>
                                                    </div>
                                                    <button @click.stop="handleAudioPreview(history, false)"
                                                            class="px-4 py-2 rounded-lg transition-all cursor-pointer relative z-10 flex items-center gap-2 flex-shrink-0 tracking-tight"
                                                            :class="isPlaying(history, false)
                                                                ? 'text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300'
                                                                : 'text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:text-[color:var(--brand-primary)]/80 dark:hover:text-[color:var(--brand-primary-light)]/80'"
                                                            style="pointer-events: auto;">
                                                        <i :class="isPlaying(history, false) ? 'fas fa-stop' : 'fas fa-play'"></i>
                                                        <span class="text-sm font-medium">{{ isPlaying(history, false) ? t('stop') : t('preview') }}</span>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 音频模板列表 - Apple 风格 -->
                                          <div v-if="showAudioTemplates && mediaModalTab === 'templates'" class="pr-6 pl-1">
                                                                                    <!-- 音频模板分页组件 - Apple 风格 -->
                                                                                    <div v-if="templatePaginationInfo" class="mt-6">
                                                                                        <div class="flex items-center justify-between text-xs mb-4">
                                                                                            <div class="flex items-center space-x-1 text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                                                                <span>{{ templatePaginationInfo.total }} {{ t('records') }}</span>
                                                                                            </div>
                                                                                        </div>
                                                                                        <div v-if="templatePaginationInfo.total_pages > 1" class="flex justify-center">
                                                                                            <nav class="isolate inline-flex gap-1" aria-label="Pagination">
                                                                                                <!-- 上一页按钮 -->
                                                                                                <button @click="goToTemplatePage(templateCurrentPage - 1)"
                                                                                                    :disabled="templateCurrentPage <= 1"
                                                                                                    class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                                                                    :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage <= 1 }"
                                                                                                    :title="t('previousPage')">
                                                                                                    <span class="sr-only">{{ t('previousPage') }}</span>
                                                                                                    <i class="fas fa-chevron-left text-xs mx-auto" aria-hidden="true"></i>
                                                                                                </button>

                                                                                                <!-- 页码按钮 -->
                                                                                                <template v-for="page in getVisibleTemplatePages()" :key="page">
                                                                                                    <button v-if="page !== '...'" @click="goToTemplatePage(page)"
                                                                                                        :class="[
                                                                                                            'relative inline-flex items-center justify-center min-w-[36px] h-9 px-3 text-sm font-medium rounded-lg transition-all duration-200',
                                                                                                            page === templateCurrentPage
                                                                                                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                                                                                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'
                                                                                                        ]"
                                                                                                        :aria-current="page === templateCurrentPage ? 'page' : undefined">
                                                                                                        {{ page }}
                                                                                                    </button>
                                                                                                    <span v-else class="relative inline-flex items-center px-2 text-sm font-semibold text-[#86868b] dark:text-[#98989d]">...</span>
                                                                                                </template>

                                                                                                <!-- 下一页按钮 -->
                                                                                                <button @click="goToTemplatePage(templateCurrentPage + 1)"
                                                                                                    :disabled="templateCurrentPage >= templatePaginationInfo.total_pages"
                                                                                                    class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                                                                    :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage >= templatePaginationInfo.total_pages }"
                                                                                                    :title="t('nextPage')">
                                                                                                    <span class="sr-only">{{ t('nextPage') }}</span>
                                                                                                    <i class="fas fa-chevron-right text-xs mx-auto" aria-hidden="true"></i>
                                                                                                </button>
                                                                                            </nav>
                                                                                        </div>
                                                                                    </div>
                                        <div class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar pr-2 pt-2">
                                            <div class="space-y-4">
                                                <div v-if="isPageLoading" class="flex items-center justify-center">
                                                    <div class="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/90 dark:bg-[#2c2c2e]/90 border border-black/8 dark:border-white/8 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] shadow-[0_4px_16px_rgba(0,0,0,0.08)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.35)]">
                                                        <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                                                        <span>{{ t('loading') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="mergedTemplates.length > 0" class="space-y-3 px-1">
                                                    <div v-for="template in mergedTemplates" :key="template.id"
                                                    @click="selectAudioTemplate(template.audio)"
                                                        class="flex items-center gap-4 p-4 rounded-2xl border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/50 dark:hover:border-[color:var(--brand-primary-light)]/50 transition-all cursor-pointer bg-white/80 dark:bg-[#2c2c2e]/80 hover:bg-white dark:hover:bg-[#3a3a3c] hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] group">
                                                            <div
                                                        class="w-12 h-12 rounded-xl overflow-hidden flex-shrink-0 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                                                        <img v-if="template.image?.url" :src="template.image.url" :alt="t('audioTemplates')" class="w-full h-full object-cover" @error="template.image.url = null" />
                                                        <i v-else class="fas fa-music text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
                                                        </div>
                                                        <div class="flex-1 min-w-0">
                                                        <div class="text-[#86868b] dark:text-[#98989d] text-sm flex items-center gap-2 tracking-tight">
                                                            <span>{{ t('audioTemplates') }}</span>
                                                            <span v-if="template.audio" class="text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]">•</span>
                                                            <span v-if="template.audio">{{ getDurationDisplay(template.audio, true) }}</span>
                                                        </div>
                                                        </div>
                                                        <div class="flex items-center gap-2 flex-shrink-0">
                                                            <button v-if="template.image" @click.stop="selectImageTemplate(template.image)"
                                                                    class="px-4 py-2 rounded-lg transition-all cursor-pointer relative z-10 flex items-center gap-2 tracking-tight text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:text-[color:var(--brand-primary)]/80 dark:hover:text-[color:var(--brand-primary-light)]/80"
                                                                    style="pointer-events: auto;">
                                                                <i class="fas fa-image"></i>
                                                                <span class="text-sm font-medium">{{ t('useImage') }}</span>
                                                            </button>
                                                            <button v-if="template.audio" @click.stop="handleAudioPreview(template.audio, true)"
                                                                    class="px-4 py-2 rounded-lg transition-all cursor-pointer relative z-10 flex items-center gap-2 tracking-tight"
                                                                    :class="isPlaying(template.audio, true)
                                                                        ? 'text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300'
                                                                        : 'text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:text-[color:var(--brand-primary)]/80 dark:hover:text-[color:var(--brand-primary-light)]/80'"
                                                                    style="pointer-events: auto;">
                                                                <i :class="isPlaying(template.audio, true) ? 'fas fa-stop' : 'fas fa-play'"></i>
                                                                <span class="text-sm font-medium">{{ isPlaying(template.audio, true) ? t('stop')  : t('preview') }}</span>
                                                            </button>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div v-else
                                                    class="flex flex-col items-center justify-center py-12 text-center">
                                                    <div
                                                        class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                                                    <i class="fas fa-music text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                                                </div>
                                                <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium tracking-tight">目前暂无音频模板</p>
                                            </div>
                                            </div>
                                        </div>

                                        </div>
                                </div>
                        </div>
                    </div>
</template>
