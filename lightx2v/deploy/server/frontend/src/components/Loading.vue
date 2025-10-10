<script setup>
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
        <!-- 加载状态 -->
        <div class="text-center">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-laser-purple mx-auto mb-4"></div>
                <p class="text-gray-400">{{ t('loading') }}</p>
        </div>
</template>
