<script setup>
import {  confirmDialog,
            showConfirmDialog} from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
        <!-- 自定义确认对话框 -->
         <div v-cloak>
        <div v-if="confirmDialog.show" class="fixed inset-0 z-50 flex items-center justify-center p-4">
            <!-- 背景遮罩 -->
                    <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="confirmDialog.show = false">
                    </div>

            <!-- 对话框内容 -->
            <div class="relative bg-dark-light border border-laser-purple/30 rounded-xl shadow-2xl max-w-md w-full mx-4 transform transition-all duration-300 ease-out"
                 :class="confirmDialog.show ? 'scale-100 opacity-100' : 'scale-95 opacity-0'">
                <!-- 头部 -->
                <div class="flex items-center justify-between p-6 border-b border-gray-700">
                    <div class="flex items-center gap-3">
                        <div class="w-10 h-10 bg-red-500/20 rounded-full flex items-center justify-center">
                            <i class="fas fa-exclamation-triangle text-red-400 text-lg"></i>
                                            </div>
                        <h3 class="text-lg font-semibold text-white">{{ confirmDialog.title }}</h3>
                    </div>
                    <button @click="confirmDialog.show = false"
                            class="text-gray-400 hover:text-gray-300 transition-colors">
                        <i class="fas fa-times text-lg"></i>
                                                </button>
                                                </div>

                <!-- 内容 -->
                <div class="p-6">
                    <p class="text-gray-300 leading-relaxed mb-6">{{ confirmDialog.message }}</p>

                    <!-- 警告信息 -->
                            <div v-if="confirmDialog.warning"
                                class="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-6">
                        <div class="flex items-start gap-3">
                            <i class="fas fa-info-circle text-red-400 mt-0.5"></i>
                            <div class="text-sm text-red-300">
                                <p class="font-medium mb-2">{{ confirmDialog.warning.title }}</p>
                                <ul class="space-y-1 text-xs">
                                            <li v-for="item in confirmDialog.warning.items" :key="item"
                                                class="flex items-center gap-2">
                                        <i class="fas fa-minus text-red-400 text-xs"></i>
                                        {{ item }}
                                    </li>
                                </ul>
                                            </div>
                                        </div>
                                    </div>
                </div>

                <!-- 底部按钮 -->
                <div class="flex gap-3 p-6 pt-0">
                    <button @click="confirmDialog.cancel()"
                            class="flex-1 px-4 py-2.5 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-all duration-200 font-medium">
                                {{ t('cancel') }}
                    </button>
                    <button @click="confirmDialog.confirm()"
                            class="flex-1 px-4 py-2.5 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-all duration-200 font-medium flex items-center justify-center gap-2">
                        <i class="fas fa-trash text-sm"></i>
                        {{ confirmDialog.confirmText }}
                    </button>
                </div>
            </div>
        </div>
        </div>
</template>