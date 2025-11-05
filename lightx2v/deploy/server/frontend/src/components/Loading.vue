<script setup>
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
        <!-- 加载状态 - Apple 极简风格 -->
        <div class="flex flex-col items-center justify-center">
                <div class="relative w-12 h-12 mb-6">
                        <!-- Apple 风格加载环 -->
                        <div class="absolute inset-0 rounded-full border-2 border-black/8 dark:border-white/8"></div>
                        <div class="absolute inset-0 rounded-full border-2 border-transparent border-t-[color:var(--brand-primary)] dark:border-t-[color:var(--brand-primary-light)] animate-spin"></div>
                </div>
                <p class="text-sm font-medium text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loading') }}</p>
        </div>
</template>
