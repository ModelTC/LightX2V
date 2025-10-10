<script setup>
import { showImageZoomModal, closeImageZoomModal, zoomedImageUrl } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
            <!-- 图片放大弹窗 -->
             <div v-cloak>
            <div v-if="showImageZoomModal" class="fixed inset-0 bg-black/80 z-60 flex items-center justify-center p-4"
                @click="closeImageZoomModal">
                <div class="relative max-w-4xl max-h-[90vh] bg-secondary rounded-xl overflow-hidden" @click.stop>
                    <div class="flex items-center justify-between p-4 border-b border-gray-700">
                        <h3 class="text-lg font-medium text-white">{{ t('imagePreview') }}</h3>
                        <button @click="closeImageZoomModal"
                            class="text-gray-400 hover:text-white transition-colors">
                            <i class="fas fa-times text-xl"></i>
                        </button>
                    </div>
                    <div class="p-4">
                        <img :src="zoomedImageUrl" :alt="'图片预览'"
                            class="w-full h-auto max-h-[70vh] object-contain rounded-lg" />
                    </div>
                </div>
            </div>
            </div>
</template>
