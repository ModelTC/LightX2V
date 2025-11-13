<script setup>
import {  confirmDialog,
            showConfirmDialog} from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
        <!-- 自定义确认对话框 - Apple 极简风格 -->
        <div v-cloak>
            <div v-if="confirmDialog.show" class="fixed inset-0 z-[70] flex items-center justify-center p-4">
                <!-- 背景遮罩 - Apple 风格 -->
                <div class="absolute inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm" @click="confirmDialog.cancel()">
                </div>

                <!-- 对话框内容 - Apple 风格 -->
                <div class="relative bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] max-w-md w-full mx-4 overflow-hidden transform transition-all duration-300 ease-out"
                     :class="confirmDialog.show ? 'scale-100 opacity-100' : 'scale-95 opacity-0'">
                    <!-- 头部 - Apple 风格 -->
                    <div class="flex items-center justify-between px-6 py-5 border-b border-black/8 dark:border-white/8">
                        <div class="flex items-center gap-3">
                            <div class="w-9 h-9 bg-red-500/10 dark:bg-red-400/10 rounded-full flex items-center justify-center">
                                <i class="fas fa-exclamation-triangle text-red-500 dark:text-red-400 text-base"></i>
                            </div>
                            <h3 class="text-lg font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ confirmDialog.title }}</h3>
                        </div>
                        <button @click="confirmDialog.cancel()"
                                class="w-8 h-8 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
                            <i class="fas fa-times text-xs"></i>
                        </button>
                    </div>

                    <!-- 内容 - Apple 风格 -->
                    <div class="px-6 py-5">
                        <p class="text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] leading-relaxed tracking-tight">{{ confirmDialog.message }}</p>

                        <!-- 警告信息 - Apple 风格 -->
                        <div v-if="confirmDialog.warning"
                             class="mt-4 bg-red-500/5 dark:bg-red-400/5 border border-red-500/20 dark:border-red-400/20 rounded-2xl p-4">
                            <div class="flex items-start gap-3">
                                <i class="fas fa-info-circle text-red-500 dark:text-red-400 text-sm mt-0.5"></i>
                                <div class="flex-1">
                                    <p class="text-sm font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2 tracking-tight">{{ confirmDialog.warning.title }}</p>
                                    <ul class="space-y-1.5 text-xs text-[#86868b] dark:text-[#98989d]">
                                        <li v-for="item in confirmDialog.warning.items" :key="item"
                                            class="flex items-start gap-2 tracking-tight">
                                            <i class="fas fa-circle text-[6px] text-red-500 dark:text-red-400 mt-1"></i>
                                            <span class="flex-1">{{ item }}</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 底部按钮 - Apple 风格 -->
                    <div class="flex gap-3 px-6 pb-6">
                        <button @click="confirmDialog.cancel()"
                                class="flex-1 px-5 py-3 bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 font-medium text-[15px] tracking-tight hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98]">
                            {{ t('cancel') }}
                        </button>
                        <button @click="confirmDialog.confirm()"
                                class="flex-1 px-5 py-3 bg-red-500 dark:bg-red-500 text-white rounded-full transition-all duration-200 font-semibold text-[15px] tracking-tight hover:bg-red-600 dark:hover:bg-red-600 hover:shadow-[0_8px_24px_rgba(239,68,68,0.35)] dark:hover:shadow-[0_8px_24px_rgba(239,68,68,0.4)] active:scale-[0.98] flex items-center justify-center gap-2">
                            <i class="fas fa-check text-sm"></i>
                            {{ confirmDialog.confirmText }}
                        </button>
                    </div>
                </div>
            </div>
        </div>
</template>
