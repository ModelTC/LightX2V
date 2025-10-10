<script setup>
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

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
    mediaModalTab,
    getImageHistory,
    getAudioHistory,
} from '../utils/other'
</script>

<template>

                        <!-- 模板选择浮窗 -->
                        <div v-cloak>
                            <div v-if="showImageTemplates || showAudioTemplates"
                                class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
                                @click="showImageTemplates = false; showAudioTemplates = false">
                                <div class="bg-secondary rounded-xl p-6 max-w-4xl w-full mx-4 h-[90vh] overflow-hidden"
                                    @click.stop>
                                    <!-- 浮窗头部 -->
                                    <div class="flex items-center justify-between mb-4">
                                        <h3 class="text-lg font-medium text-white">
                                                <i v-if="showImageTemplates"
                                                    class="fas fa-image text-gradient-primary mr-2"></i>
                                                <i v-if="showAudioTemplates"
                                                    class="fas fa-music text-gradient-primary mr-2"></i>
                                                {{ showImageTemplates ? t('imageTemplates') : t('audioTemplates') }}
                                        </h3>
                                        <button @click="showImageTemplates = false; showAudioTemplates = false"
                                                class="text-gray-400 hover:text-white transition-colors">
                                            <i class="fas fa-times text-xl"></i>
                                        </button>
                                    </div>

                                    <!-- 标签页切换 -->
                                    <div class="flex border-b border-gray-700 mb-6">
                                            <button
                                                @click="mediaModalTab = 'history'; showImageTemplates && getImageHistory(); showAudioTemplates && getAudioHistory()"
                                                class="px-4 py-2 text-sm font-medium transition-colors" :class="mediaModalTab === 'history'
                                                    ? 'text-gradient-primary border-b-2 border-laser-purple'
                                                    : 'text-gray-400 hover:text-gray-300'">
                                            <i class="fas fa-history mr-2"></i>
                                                {{ t('history') }}
                                        </button>
                                        <button @click="mediaModalTab = 'templates'"
                                                class="px-4 py-2 text-sm font-medium transition-colors" :class="mediaModalTab === 'templates'
                                                    ? 'text-gradient-primary border-b-2 border-laser-purple'
                                                    : 'text-gray-400 hover:text-gray-300'">
                                            <i class="fas fa-layer-group mr-2"></i>
                                                {{ t('templates') }}
                                        </button>
                                    </div>

                                    <!-- 图片历史记录 -->
                                          <div v-if="showImageTemplates && mediaModalTab === 'history'"
                                             class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar">
                                            <div v-if="imageHistory.length === 0"
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-auto bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-history text-gradient-primary text-2xl"></i>
                                            </div>
                                                <p class="text-gray-400 text-lg mb-2">{{ t('noHistoryRecords') }}</p>
                                                <p class="text-gray-500 text-sm">{{ t('imageHistoryAutoSave') }}</p>
                                        </div>
                                        <div v-else class="space-y-3">
                                            <div class="flex items-center justify-between mb-4">
                                                    <span class="text-sm text-gray-400">{{ t('total') }} {{ imageHistory.length }}
                                                        {{ t('records') }}</span>
                                                <button @click="clearImageHistory"
                                                        class="text-xs text-red-400 hover:text-red-300 transition-colors flex items-center gap-1"
                                                        :title="t('clearHistory')">
                                                    <i class="fas fa-trash"></i>
                                                        {{ t('clear') }}
                                                </button>
                                            </div>
                                            <div class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4">
                                                <div v-for="(history, index) in imageHistory" :key="index"
                                                    @click="selectImageHistory(history)"
                                                    class="break-inside-avoid mb-4 relative group cursor-pointer rounded-lg overflow-hidden border border-gray-700 hover:border-laser-purple/50 transition-all">
                                                        <img :src="getHistoryImageUrl(history)" :alt="history.filename"
                                                            class="w-full h-auto object-contain">
                                                        <div
                                                            class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                        <i class="fas fa-check text-white text-xl"></i>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                     <!-- 图片模板网格 -->
                                         <div v-if="showImageTemplates && mediaModalTab === 'templates'">

                                            <!-- 图片模板分页组件 -->
                                            <div v-if="templatePaginationInfo" class="mt-8">
                                                <div class="flex items-center justify-between text-xs text-gray-400 mb-4">
                                                    <div class="flex items-center space-x-1 text-gray-500">
                                                        <span>{{ templatePaginationInfo.total }} {{ t('records') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="templatePaginationInfo.total_pages > 1" class="flex justify-center">
                                                    <nav class="isolate inline-flex -space-x-px rounded-md" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToTemplatePage(templateCurrentPage - 1)"
                                                            :disabled="templateCurrentPage <= 1"
                                                            class="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                                            <i class="fas fa-chevron-left text-sm" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisibleTemplatePages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToTemplatePage(page)"
                                                                :class="[
                                                                    'relative inline-flex items-center px-4 py-2 text-sm font-semibold focus:z-20 focus:outline-offset-0',
                                                                    page === templateCurrentPage 
                                                                        ? 'z-10 text-white focus-visible:outline-2 focus-visible:outline-offset-2 bg-laser-purple focus-visible:outline-laser-purple'
                                                                        : 'text-gray-200 inset-ring inset-ring-gray-700 hover:bg-white/5'
                                                                ]"
                                                                :aria-current="page === templateCurrentPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                                            <span v-else class="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-400 inset-ring inset-ring-gray-700 focus:outline-offset-0">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToTemplatePage(templateCurrentPage + 1)"
                                                            :disabled="templateCurrentPage >= templatePaginationInfo.total_pages"
                                                            class="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage >= templatePaginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                                            <i class="fas fa-chevron-right text-sm" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                                </div>
                                            </div>
                                         <div class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar">
                                         <div v-if="imageTemplates.length > 0" class="columns-2 sm:columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4">
                                            <div v-for="template in imageTemplates" :key="template.filename"
                                                @click="selectImageTemplate(template)"
                                                class="break-inside-avoid mb-4 relative group cursor-pointer rounded-lg border border-gray-700 hover:border-laser-purple/50 transition-all">
                                                    <img :src="getTemplateFileUrl(template.filename,'images')" :alt="template.filename"
                                                    class="w-full h-auto object-contain" preload="metadata">
                                                    <div
                                                        class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                    <i class="fas fa-check text-white text-2xl"></i>
                                                </div>
                                            </div>
                                        </div>
                                            <div v-else
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-16 bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-image text-gradient-primary text-2xl"></i>
                                            </div>
                                                <p class="text-gray-400 text-lg mb-2">{{ t('noImageTemplates') }}</p>
                                        </div>
                                    </div>

                                    </div>

                                    <!-- 音频历史记录 -->
                                          <div v-if="showAudioTemplates && mediaModalTab === 'history'"
                                             class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar">
                                            <div v-if="audioHistory.length === 0"
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-16 bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-history text-gradient-primary text-2xl"></i>
                                            </div>
                                                <p class="text-gray-400 text-lg mb-2">{{ t('noHistoryRecords') }}</p>
                                                <p class="text-gray-500 text-sm">{{ t('audioHistoryAutoSave') }}</p>
                                        </div>
                                        <div v-else class="space-y-3">
                                            <div class="flex items-center justify-between mb-4">
                                                    <span class="text-sm text-gray-400">共 {{ audioHistory.length }}
                                                        {{ t('records') }}</span>
                                                <button @click="clearAudioHistory"
                                                        class="text-xs text-red-400 hover:text-red-300 transition-colors flex items-center gap-1"
                                                        :title="t('clearHistory')">
                                                    <i class="fas fa-trash"></i>
                                                        {{ t('clear') }}
                                                </button>
                                            </div>
                                            <div class="space-y-3">
                                                <div v-for="(history, index) in audioHistory" :key="index"
                                                    @click="selectAudioHistory(history)"
                                                    class="flex items-center gap-4 p-4 rounded-lg border border-gray-700 hover:border-laser-purple/50 transition-all cursor-pointer bg-dark-light/50 group">
                                                        <div
                                                            class="w-12 h-12 bg-laser-purple/20 rounded-lg flex items-center justify-center">
                                                        <i class="fas fa-music text-gradient-primary text-xl"></i>
                                                    </div>
                                                    <div class="flex-1">
                                                            <div
                                                                class="text-white font-medium group-hover:text-gradient-primary transition-colors">
                                                                {{ history.filename }}</div>
                                                            <div class="text-gray-400 text-sm">{{ t('audioFile') }}</div>
                                                    </div>
                                                    <button @click.stop="previewAudioHistory(history)"
                                                            class="px-3 py-2 bg-laser-purple/20 hover:bg-laser-purple/30 text-gradient-primary rounded-lg transition-all cursor-pointer relative z-10"
                                                            style="pointer-events: auto;">
                                                        <i class="fas fa-play mr-2"></i>
                                                            {{ t('preview') }}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- 音频模板列表 -->
                                          <div v-if="showAudioTemplates && mediaModalTab === 'templates'">
                                                                                    <!-- 音频模板分页组件 -->
                                                                                    <div v-if="templatePaginationInfo" class="mt-8">
                                                                                        <div class="flex items-center justify-between text-xs text-gray-400 mb-4">
                                                                                            <div class="flex items-center space-x-1 text-gray-500">
                                                                                                <span>{{ templatePaginationInfo.total }} {{ t('records') }}</span>
                                                                                            </div>
                                                                                        </div>
                                                                                        <div v-if="templatePaginationInfo.total_pages > 1" class="flex justify-center">
                                                                                            <nav class="isolate inline-flex -space-x-px rounded-md" aria-label="Pagination">
                                                                                                <!-- 上一页按钮 -->
                                                                                                <button @click="goToTemplatePage(templateCurrentPage - 1)"
                                                                                                    :disabled="templateCurrentPage <= 1"
                                                                                                    class="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                                                                    :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage <= 1 }"
                                                                                                    :title="t('previousPage')">
                                                                                                    <span class="sr-only">{{ t('previousPage') }}</span>
                                                                                                    <i class="fas fa-chevron-left text-sm" aria-hidden="true"></i>
                                                                                                </button>

                                                                                                <!-- 页码按钮 -->
                                                                                                <template v-for="page in getVisibleTemplatePages()" :key="page">
                                                                                                    <button v-if="page !== '...'" @click="goToTemplatePage(page)"
                                                                                                        :class="[
                                                                                                            'relative inline-flex items-center px-4 py-2 text-sm font-semibold focus:z-20 focus:outline-offset-0',
                                                                                                            page === templateCurrentPage 
                                                                                                                ? 'z-10 text-white focus-visible:outline-2 focus-visible:outline-offset-2 bg-laser-purple focus-visible:outline-laser-purple'
                                                                                                                : 'text-gray-200 inset-ring inset-ring-gray-700 hover:bg-white/5'
                                                                                                        ]"
                                                                                                        :aria-current="page === templateCurrentPage ? 'page' : undefined">
                                                                                                        {{ page }}
                                                                                                    </button>
                                                                                                    <span v-else class="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-400 inset-ring inset-ring-gray-700 focus:outline-offset-0">...</span>
                                                                                                </template>

                                                                                                <!-- 下一页按钮 -->
                                                                                                <button @click="goToTemplatePage(templateCurrentPage + 1)"
                                                                                                    :disabled="templateCurrentPage >= templatePaginationInfo.total_pages"
                                                                                                    class="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                                                                    :class="{ 'opacity-50 cursor-not-allowed': templateCurrentPage >= templatePaginationInfo.total_pages }"
                                                                                                    :title="t('nextPage')">
                                                                                                    <span class="sr-only">{{ t('nextPage') }}</span>
                                                                                                    <i class="fas fa-chevron-right text-sm" aria-hidden="true"></i>
                                                                                                </button>
                                                                                            </nav>
                                                                                        </div>
                                                                                    </div>
                                        <div class="overflow-y-auto flex-1 max-h-[60vh] main-scrollbar">
                                        <div v-if="audioTemplates.length > 0" class="space-y-3">
                                            <div v-for="template in audioTemplates" :key="template.filename"
                                                @click="selectAudioTemplate(template)"
                                                class="flex items-center gap-4 p-4 rounded-lg border border-gray-700 hover:border-laser-purple/50 transition-all cursor-pointer bg-dark-light/50">
                                                    <div
                                                        class="w-12 h-12 bg-laser-purple/20 rounded-lg flex items-center justify-center">
                                                    <i class="fas fa-music text-gradient-primary text-xl"></i>
                                                </div>
                                                <div class="flex-1">
                                                        <div class="text-white font-medium">{{ template.filename }}
                                                        </div>
                                                        <div class="text-gray-400 text-sm">{{ t('audioTemplates') }}</div>
                                                </div>
                                                <button @click.stop="previewAudioTemplate(template)"
                                                        class="px-3 py-2 bg-laser-purple/20 hover:bg-laser-purple/30 text-gradient-primary rounded-lg transition-all cursor-pointer relative z-10"
                                                        style="pointer-events: auto;">
                                                    <i class="fas fa-play mr-2"></i>
                                                        {{ t('preview') }}
                                                </button>
                                            </div>
                                        </div>
                                            <div v-else
                                                class="flex flex-col items-center justify-center py-12 text-center">
                                                <div
                                                    class="w-16 h-16 bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                                <i class="fas fa-music text-gradient-primary text-2xl"></i>
                                            </div>
                                            <p class="text-gray-400 text-lg mb-2">目前暂无音频模板</p>
                                        </div>
                                        </div>

                                        </div>
                                </div>
                        </div>
                    </div>
</template>
