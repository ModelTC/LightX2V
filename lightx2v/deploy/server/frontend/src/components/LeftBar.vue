<script setup>
import {  switchToCreateView, switchToProjectsView, switchToInspirationView } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
  <!-- 左侧功能区 - Apple 极简风格 - 响应式布局 -->
  <div class="relative flex flex-col z-10 pl-0 sm:pl-5 w-full sm:w-24">
    <!-- 功能导航 - Apple 风格统一容器 -->
    <div class="p-2 flex flex-col justify-center h-full mobile-nav-buttons sm:mt-[-10vh]">
      <!-- 统一的圆角矩形容器 - Apple 风格 - 响应式方向 -->
      <nav class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.4)] overflow-hidden flex flex-row sm:flex-col w-full sm:w-16">

        <!-- 生成视频功能 -->
        <div
          @click="switchToCreateView"
          class="flex items-center justify-center flex-1 sm:flex-none h-14 sm:h-16 cursor-pointer transition-all duration-200 ease-out mobile-nav-btn group"
          :class="$route.path === '/generate'
            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
            : 'text-[#86868b] dark:text-[#98989d] hover:bg-black/4 dark:hover:bg-white/6 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)]'"
          :title="t('generateVideo')">
          <i class="fas fa-plus text-xl transition-all duration-200 group-hover:scale-110"></i>
        </div>

        <!-- 分割线 - Apple 风格 - 响应式方向 -->
        <div class="w-px sm:w-auto sm:h-px bg-black/8 dark:bg-white/8 my-3 sm:my-0 sm:mx-3"></div>

        <!-- 我的项目功能 -->
        <div
          @click="switchToProjectsView"
          class="flex items-center justify-center flex-1 sm:flex-none h-14 sm:h-16 cursor-pointer transition-all duration-200 ease-out mobile-nav-btn group"
          :class="$route.path === '/projects'
            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
            : 'text-[#86868b] dark:text-[#98989d] hover:bg-black/4 dark:hover:bg-white/6 hover:text-[color:var(--brand-primary)] dark:hover:text-[color:var(--brand-primary-light)]'"
          :title="t('myProjects')">
          <i class="fas fa-folder-open text-lg transition-all duration-200 group-hover:scale-110"></i>
        </div>

        <!-- 分割线 - Apple 风格 - 响应式方向 -->
        <div class="w-px sm:w-auto sm:h-px bg-black/8 dark:bg-white/8 my-3 sm:my-0 sm:mx-3"></div>

        <!-- 灵感广场功能 -->
        <div
          @click="switchToInspirationView"
          class="flex items-center justify-center flex-1 sm:flex-none h-14 sm:h-16 cursor-pointer transition-all duration-200 ease-out mobile-nav-btn group"
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
