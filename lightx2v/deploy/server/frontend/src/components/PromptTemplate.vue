<script setup>
import { showPromptModal,
        promptModalTab,
        getPromptTemplates,
        selectPromptTemplate,
        promptHistory,
        selectPromptHistory,
        clearPromptHistory,
        selectedTaskId } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
</script>

<template>
    <!-- 提示词模板和历史记录弹窗 - Apple 极简风格 -->
    <div v-cloak>
        <div v-if="showPromptModal"
             class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
             @click="showPromptModal = false">
            <div class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] backdrop-saturate-[180%] border border-black/8 dark:border-white/8 rounded-3xl p-8 max-w-4xl w-full mx-4 max-h-[80vh] overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.12)] dark:shadow-[0_8px_32px_rgba(0,0,0,0.4)]"
                 @click.stop>
                <!-- 浮窗头部 - Apple 风格 -->
                <div class="flex items-center justify-between mb-6">
                    <h3 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
                        <i class="fas fa-lightbulb text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                        {{ t('promptTemplates') }}
                    </h3>
                    <button @click="showPromptModal = false"
                            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
                        <i class="fas fa-times text-base"></i>
                    </button>
                </div>

                <!-- 标签页切换 - Apple 风格 -->
                <div class="flex gap-2 mb-6">
                    <button @click="promptModalTab = 'templates'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                            :class="promptModalTab === 'templates'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                        <i class="fas fa-layer-group mr-2"></i>
                        {{ t('templates') }}
                    </button>
                    <button @click="promptModalTab = 'history'"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                            :class="promptModalTab === 'history'
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                        <i class="fas fa-history mr-2"></i>
                        {{ t('history') }}
                    </button>
                </div>

                <!-- 模板内容 - Apple 风格 -->
                <div v-if="promptModalTab === 'templates'" class="overflow-y-auto max-h-[50vh] main-scrollbar">
                    <div v-if="getPromptTemplates(selectedTaskId).length > 0"
                         class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <button v-for="template in getPromptTemplates(selectedTaskId)" :key="template.id"
                                @click="selectPromptTemplate(template)"
                                class="p-5 text-left bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 transition-all duration-200 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] group active:scale-[0.98]">
                            <div class="font-semibold text-[15px] mb-3 text-[#1d1d1f] dark:text-[#f5f5f7] group-hover:text-[color:var(--brand-primary)] dark:group-hover:text-[color:var(--brand-primary-light)] transition-colors tracking-tight">
                                {{ template.title }}
                            </div>
                            <div class="text-[13px] text-[#86868b] dark:text-[#98989d] line-clamp-3 leading-relaxed tracking-tight">
                                {{ template.prompt }}
                            </div>
                            <div class="mt-4 flex items-center justify-between">
                                <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('clickApply') }}</span>
                                <i class="fas fa-arrow-right text-xs text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] group-hover:translate-x-1 transition-transform"></i>
                            </div>
                        </button>
                    </div>
                    <div v-else class="flex flex-col items-center justify-center py-12 text-center">
                        <div class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                            <i class="fas fa-layer-group text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                        </div>
                        <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium mb-2 tracking-tight">{{ t('noAvailableTemplates') }}</p>
                        <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('pleaseSelectTaskType') }}</p>
                    </div>
                </div>

                <!-- 历史记录内容 - Apple 风格 -->
                <div v-if="promptModalTab === 'history'" class="overflow-y-auto max-h-[50vh] main-scrollbar">
                    <div v-if="promptHistory.length === 0"
                         class="flex flex-col items-center justify-center py-12 text-center">
                        <div class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
                            <i class="fas fa-history text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
                        </div>
                        <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium mb-2 tracking-tight">{{ t('noHistoryRecords') }}</p>
                        <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('promptHistoryAutoSave') }}</p>
                    </div>
                    <div v-else class="space-y-3">
                        <div class="flex items-center justify-between mb-4">
                            <span class="text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ promptHistory.length }} {{ t('records') }}</span>
                            <button @click="clearPromptHistory"
                                    class="text-xs text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 transition-colors flex items-center gap-1.5 tracking-tight"
                                    :title="t('clearHistory')">
                                <i class="fas fa-trash"></i>
                                {{ t('clear') }}
                            </button>
                        </div>
                        <button v-for="(history, index) in promptHistory" :key="index"
                                @click="selectPromptHistory(history)"
                                class="w-full p-5 text-left bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 transition-all duration-200 hover:shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)] group active:scale-[0.98]">
                            <div class="text-[13px] text-[#1d1d1f] dark:text-[#f5f5f7] line-clamp-3 leading-relaxed group-hover:text-[color:var(--brand-primary)] dark:group-hover:text-[color:var(--brand-primary-light)] transition-colors tracking-tight">
                                {{ history }}
                            </div>
                            <div class="mt-3 flex items-center justify-between">
                                <span class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('clickApply') }}</span>
                                <i class="fas fa-arrow-right text-xs text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] group-hover:translate-x-1 transition-transform"></i>
                            </div>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* Apple 风格极简黑白设计 */
</style>
