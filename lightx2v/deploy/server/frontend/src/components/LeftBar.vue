<script setup>
import {  switchToCreateView, switchToProjectsView, switchToInspirationView } from '../utils/other'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
</script>
<template>
  <!-- 左侧功能区 -->
  <div class="relative w-20 pl-5 flex flex-col z-10">
    <!-- 功能导航 -->
    <div class="p-2 flex flex-col justify-center h-full mobile-nav-buttons" style="margin-top: -10vh;">
      <nav class="lg:space-y-3 md:space-y-3 sm:space-x-3 flex flex-col">
        
        <!-- 生成视频功能 -->
        <div
          @click="switchToCreateView"
          class="w-18 h-18 flex items-center justify-center rounded-lg transition-all duration-300
                 font-medium text-sm border border-transparent hover:scale-105 hover:shadow-lg
                 hover:shadow-laser-purple/20 mobile-nav-btn"
          :class="$route.path === '/generate'
            ? 'bg-laser-purple/40 text-white border-laser-purple/40 shadow-lg shadow-laser-purple/20'
            : 'text-gray-400 hover:text-white hover:bg-dark-light hover:border-laser-purple/30'"
          :title="t('generateVideo')">
          <i class="fi fi-sr-add text-3xl transition-transform duration-300 group-hover:animate-pulse"></i>
        </div>

        <!-- 我的项目功能 -->
        <div
          @click="switchToProjectsView"
          class="w-18 h-18 flex items-center justify-center rounded-lg transition-all duration-300
                 font-medium text-sm border border-transparent hover:scale-105 hover:shadow-lg
                 hover:shadow-laser-purple/20 mobile-nav-btn"
          :class="$route.path === '/projects'
            ? 'bg-laser-purple/40 text-white border-laser-purple/40 shadow-lg shadow-laser-purple/20'
            : 'text-gray-400 hover:text-white hover:bg-dark-light hover:border-laser-purple/30'"
          :title="t('myProjects')">
          <i class="fi fi-br-house-chimney-heart text-3xl transition-transform duration-300 group-hover:animate-pulse"></i>
        </div>

        <!-- 灵感广场功能 -->
        <div
          @click="switchToInspirationView"
          class="w-18 h-18 flex items-center justify-center rounded-lg transition-all duration-300
                 font-medium text-sm border border-transparent hover:scale-105 hover:shadow-lg
                 hover:shadow-laser-purple/40 mobile-nav-btn"
          :class="$route.path === '/inspirations'
            ? 'bg-laser-purple/40 text-white border-laser-purple/40 shadow-lg shadow-laser-purple/20'
            : 'text-gray-400 hover:text-white hover:bg-dark-light hover:border-laser-purple/30'"
          :title="t('inspiration')">
          
          <i class="fi fi-sr-sparkles text-3xl transition-transform duration-300 group-hover:animate-pulse"></i>
        </div>
      </nav>
    </div>
  </div>
</template>

