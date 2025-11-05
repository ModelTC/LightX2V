<script setup>
import TopBar from '../components/TopBar.vue'
import LeftBar from '../components/LeftBar.vue'
import Alert from '../components/Alert.vue'
import Confirm from '../components/Confirm.vue'
import TaskDetails from '../components/TaskDetails.vue'
import TemplateDetails from '../components/TemplateDetails.vue'
import PromptTemplate from '../components/PromptTemplate.vue'
import Loading from '../components/Loading.vue'
import { useI18n } from 'vue-i18n'
import { isLoading } from '../utils/other'

const { t } = useI18n()
</script>

<template>
  <!-- 主容器 - Apple 极简风格 - 配合80%缩放铺满屏幕 -->
  <div class="bg-[#f5f5f7] dark:bg-[#000000] transition-colors duration-300 w-full h-full">
    <!-- 主内容区域 -->
    <div class="flex flex-col w-full h-full">
      <!-- 顶部导航栏 -->
      <TopBar />

      <!-- 内容区域 - 响应式布局 -->
      <div class="flex flex-col sm:flex-row flex-1 h-full">
        <!-- 左侧/底部导航栏 - 响应式 -->
        <LeftBar />

        <!-- 路由视图内容 -->
        <div class="flex-1 overflow-y-auto main-scrollbar">
          <router-view></router-view>
        </div>
      </div>
    </div>

    <!-- 全局组件 -->
    <Alert />
    <Confirm />
    <TaskDetails />
    <TemplateDetails />
    <PromptTemplate />

    <!-- 全局加载覆盖层 - Apple 风格 -->
    <div v-show="isLoading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-[9999] transition-opacity duration-300">
      <Loading />
    </div>
  </div>
</template>

<style scoped>
/* Apple 风格极简设计 - 所有样式已通过 Tailwind CSS 在 template 中定义 */
</style>
