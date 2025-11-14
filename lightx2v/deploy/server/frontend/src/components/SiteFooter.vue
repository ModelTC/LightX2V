<script setup>
import { useI18n } from 'vue-i18n'
const currentYear = new Date().getFullYear()
const { t } = useI18n()
</script>

<template>
  <div class="bg-transparent text-neutral-600 dark:text-neutral-300">
    <div class="mx-auto w-full max-w-4xl pt-16 pb-16 text-center">
      <div class="flex flex-col gap-10 items-center justify-center">
        <div class="max-w-sm space-y-6">
          <div class="inline-flex items-center justify-center gap-3 text-xl font-semibold tracking-tight text-neutral-900 dark:text-white">
            <img src="../../public/logo.svg" alt="LightX2V" class="h-8 w-8" loading="lazy" />
            <span>LightX2V</span>
          </div>
          <p class="leading-relaxed text-neutral-500 dark:text-neutral-400">
            {{ t('footer.tagline') }}
          </p>
          <div class="flex flex-wrap items-center justify-center gap-5 text-sm text-neutral-500 dark:text-neutral-400">
            <a href="https://www.light-ai.top/" target="_blank" rel="noopener" class="transition hover:text-neutral-900 dark:hover:text-white">{{ t('footer.links.home') }}</a>
            <a href="https://github.com/ModelTC/LightX2V" target="_blank" rel="noopener"
              class="inline-flex items-center gap-2 transition hover:text-neutral-900 dark:hover:text-white">
              <img src="https://github.githubassets.com/favicons/favicon.svg" :alt="t('footer.alt.github')" class="h-4 w-4 transition dark:invert" loading="lazy" />
              {{ t('footer.links.github') }}
            </a>
            <a href="https://xhslink.com/m/45NsEK8minq" class="inline-flex items-center gap-2 transition hover:text-neutral-900 dark:hover:text-white">
              <img src="https://www.xiaohongshu.com/favicon.ico" :alt="t('footer.alt.xiaohongshu')" class="h-4 w-4" loading="lazy" />
              {{ t('footer.links.xiaohongshu') }}
            </a>
          </div>
        </div>
      </div>
      <div class="mt-12 border-t border-black/10 dark:border-white/10 pt-8 text-xs text-neutral-400 dark:text-neutral-500 flex items-center justify-center">
        <p>{{ t('footer.copyright', { year: currentYear }) }}</p>
      </div>
    </div>
  </div>
</template>
