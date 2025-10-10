<script setup>
import { showPromptModal, 
        promptModalTab, 
        getPromptTemplates, 
        selectPromptTemplate,
        promptHistory,
        selectPromptHistory } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
        <!-- 提示词模板和历史记录弹窗 -->
        <div v-cloak>
                <div v-if="showPromptModal" class="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
                @click="showPromptModal = false">
                <div class="bg-secondary rounded-xl p-6 max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden"
                    @click.stop>
                    <!-- 浮窗头部 -->
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-medium text-white">
                            <i class="fas fa-lightbulb text-gradient-primary mr-2"></i>
                                {{ t('promptTemplates') }}
                        </h3>
                        <button @click="showPromptModal = false"
                                class="text-gray-400 hover:text-white transition-colors">
                            <i class="fas fa-times text-xl"></i>
                        </button>
                    </div>

                    <!-- 标签页切换 -->
                    <div class="flex border-b border-gray-700 mb-6">
                        <button @click="promptModalTab = 'templates'"
                                class="px-4 py-2 text-sm font-medium transition-colors" :class="promptModalTab === 'templates'
                                    ? 'text-gradient-primary border-b-2 border-laser-purple'
                                    : 'text-gray-400 hover:text-gray-300'">
                            <i class="fas fa-layer-group mr-2"></i>
                                {{ t('templates') }}
                        </button>
                        <button @click="promptModalTab = 'history'"
                                class="px-4 py-2 text-sm font-medium transition-colors" :class="promptModalTab === 'history'
                                    ? 'text-gradient-primary border-b-2 border-laser-purple'
                                    : 'text-gray-400 hover:text-gray-300'">
                            <i class="fas fa-history mr-2"></i>
                                {{ t('history') }}
                        </button>
                    </div>

                    <!-- 模板内容 -->
                    <div v-if="promptModalTab === 'templates'" class="overflow-y-auto max-h-[50vh]">
                            <div v-if="getPromptTemplates(selectedTaskId).length > 0"
                                class="grid grid-cols-1 md:grid-cols-2 gap-4 overflow-y-auto max-h-[50vh] main-scrollbar">
                                <button v-for="template in getPromptTemplates(selectedTaskId)" :key="template.id"
                                @click="selectPromptTemplate(template)"
                                    class="break-inside-avoid mb-4 p-4 text-left bg-dark-light rounded-lg hover:bg-laser-purple/20 transition-all border border-transparent hover:border-laser-purple/40 group">
                                    <div
                                        class="font-medium text-sm mb-2 text-white group-hover:text-gradient-primary transition-colors">
                                    {{ template.title }}
                                </div>
                                <div class="text-xs text-gray-400 line-clamp-3 leading-relaxed">
                                    {{ template.prompt }}
                                </div>
                                <div class="mt-3 flex items-center justify-between">
                                        <span class="text-xs text-laser-purple/60">{{ t('clickApply') }}</span>
                                        <i
                                            class="fas fa-arrow-right text-xs text-laser-purple/60 group-hover:translate-x-1 transition-transform"></i>
                                </div>
                            </button>
                        </div>
                        <div v-else class="flex flex-col items-center justify-center py-12 text-center">
                                <div
                                    class="w-16 h-16 bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                <i class="fas fa-layer-group text-gradient-primary text-2xl"></i>
                            </div>
                                <p class="text-gray-400 text-lg mb-2">{{ t('noAvailableTemplates') }}</p>
                                <p class="text-gray-500 text-sm">{{ t('pleaseSelectTaskType') }}</p>
                        </div>
                    </div>

                    <!-- 历史记录内容 -->
                    <div v-if="promptModalTab === 'history'" class="overflow-y-auto max-h-[50vh]">
                            <div v-if="promptHistory.length === 0"
                                class="flex flex-col items-center justify-center py-12 text-center">
                                <div
                                    class="w-16 h-16 bg-laser-purple/20 rounded-full flex items-center justify-center mb-4">
                                <i class="fas fa-history text-gradient-primary text-2xl"></i>
                            </div>
                                <p class="text-gray-400 text-lg mb-2">{{ t('noHistoryRecords') }}</p>
                                <p class="text-gray-500 text-sm">{{ t('promptHistoryAutoSave') }}</p>
                        </div>
                        <div v-else class="space-y-3">
                            <div class="flex items-center justify-between mb-4">
                                    <span class="text-sm text-gray-400">{{ promptHistory.length }} {{ t('records') }}</span>
                                <button @click="clearPromptHistory"
                                        class="text-xs text-red-400 hover:text-red-300 transition-colors flex items-center gap-1"
                                        title="{{ t('clearHistory') }}">
                                    <i class="fas fa-trash"></i>
                                        {{ t('clear') }}
                                </button>
                            </div>
                                <button v-for="(history, index) in promptHistory" :key="index"
                                @click="selectPromptHistory(history)"
                                    class="w-full p-4 text-left bg-dark-light rounded-lg hover:bg-laser-purple/20 transition-all border border-transparent hover:border-laser-purple/40 group">
                                    <div
                                        class="text-sm text-gray-300 line-clamp-3 leading-relaxed group-hover:text-white transition-colors">
                                    {{ history }}
                                </div>
                                <div class="mt-2 flex items-center justify-between">
                                        <span class="text-xs text-laser-purple/60">{{ t('clickApply') }}</span>
                                        <i
                                            class="fas fa-arrow-right text-xs text-laser-purple/60 group-hover:translate-x-1 transition-transform"></i>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
</template>