<script setup>
import {  switchToCreateView, switchToProjectsView, switchToInspirationView } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
  <!-- 左侧功能区 - 响应式悬浮按钮 -->
  <div class="fixed top-20 sm:top-1/2 sm:-translate-y-1/2 right-3 sm:right-auto sm:left-5 z-[10] w-auto">
    <!-- 功能导航-->
    <div class="p-2 flex flex-col justify-center">
      <!-- 统一的圆角矩形容器-->
      <nav class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.4)] overflow-hidden flex flex-col w-12 sm:w-14">

        <!-- 生成视频功能 -->
        <div
          @click="switchToCreateView"
          class="flex items-center justify-center h-16 cursor-pointer transition-all duration-200 ease-out group"
          :class="$route.path === '/generate'
            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
            : 'text-[#86868b] dark:text-[#98989d] hover:bg-black/4 dark:hover:bg-white/6 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)]'"
          :title="t('generateVideo')">
          <i class="fas fa-plus text-xl transition-all duration-200 group-hover:scale-110"></i>
        </div>

        <!-- 分割线 - Apple 风格 -->
        <div class="h-px bg-black/8 dark:bg-white/8 mx-3"></div>

        <!-- 我的项目功能 -->
        <div
          @click="switchToProjectsView"
          class="flex items-center justify-center h-16 cursor-pointer transition-all duration-200 ease-out group"
          :class="$route.path === '/projects'
            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
            : 'text-[#86868b] dark:text-[#98989d] hover:bg-black/4 dark:hover:bg-white/6 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)]'"
          :title="t('myProjects')">
          <i class="fas fa-folder-open text-lg transition-all duration-200 group-hover:scale-110"></i>
        </div>

        <!-- 分割线 - Apple 风格 -->
        <div class="h-px bg-black/8 dark:bg-white/8 mx-3"></div>

        <!-- 灵感广场功能 -->
        <div
          @click="switchToInspirationView"
          class="flex items-center justify-center h-16 cursor-pointer transition-all duration-200 ease-out group"
          :class="$route.path === '/inspirations'
            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
            : 'text-[#86868b] dark:text-[#98989d] hover:bg-black/4 dark:hover:bg-white/6 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)]'"
          :title="t('inspiration')">
          <i class="fas fa-lightbulb text-lg transition-all duration-200 group-hover:scale-110"></i>
        </div>

      </nav>
    </div>
  </div>
</template>
