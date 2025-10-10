<script setup>
import FloatingParticles from '../components/FloatingParticles.vue'
import LoginCard from '../components/LoginCard.vue'
import Alert from '../components/Alert.vue'
import Loading from '../components/Loading.vue'
import TemplateDisplay from '../components/TemplateDisplay.vue'
import { isLoading, featuredTemplates, loadFeaturedTemplates, getRandomFeaturedTemplates } from '../utils/other'
import { ref, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
import { loadLanguageAsync, switchLang } from '../utils/i18n'

// 当前显示的精选模版
const currentFeaturedTemplates = ref([])

// 获取随机精选模版
const refreshRandomTemplates = async () => {
    try {
        const randomTemplates = await getRandomFeaturedTemplates(5) // 获取5个模版
        currentFeaturedTemplates.value = randomTemplates
    } catch (error) {
        console.error('刷新随机模版失败:', error)
    }
}

// 组件挂载时初始化
onMounted(async () => {
    // 加载精选模版数据
    isLoading.value = true
    await loadFeaturedTemplates(true)
    // 获取随机精选模版
    const randomTemplates = await getRandomFeaturedTemplates(5) // 获取5个模版
    currentFeaturedTemplates.value = randomTemplates
    isLoading.value = false
})
</script>

<template>
  <div class="login-container w-full min-h-screen flex items-center justify-center p-4">
    <FloatingParticles />

    <!-- 主卡片容器 -->
    <div class="w-full max-w-7xl mx-auto">
      <div class="bg-dark-light/80 backdrop-blur-sm rounded-2xl border border-gray-700/50 shadow-2xl overflow-hidden h-auto lg:h-[100vh]" style="box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 30px rgba(154, 114, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);">
        <div class="grid grid-cols-1 lg:grid-cols-2 h-auto lg:h-full">

          <!-- 左侧：登录区域 -->
          <div class="flex flex-col items-center justify-center">
            <LoginCard />
          </div>

          <!-- 右侧：模版展示区域 -->
          <div v-if="currentFeaturedTemplates.length > 0"
               class="flex flex-col lg:border-t-0 lg:border-l border-gray-700/50">

            <!-- 区域头部 -->
            <div class="pt-4 border-gray-700/50">
              <div class="flex items-center justify-center">
                <p class="text-gray-400 text-sm">{{ t('templatesGeneratedByLightX2V') }}</p>
                <button @click="refreshRandomTemplates"
                        class="w-8 h-8 ml-2 flex items-center justify-center hover:bg-laser-purple/40 text-laser-purple rounded-full transition-all duration-300 hover:scale-110"
                        :title="t('refreshRandomTemplates')">
                  <i class="fas fa-random text-sm"></i>
                </button>
              </div>
            </div>

            <!-- 可滚动的模版展示区域 -->
            <div class="flex-1 overflow-y-auto main-scrollbar">
              <TemplateDisplay
                :templates="currentFeaturedTemplates"
                :show-actions="false"
                layout="grid"
                :max-templates="4"
              />
            </div>
          </div>

          <!-- 如果没有模版数据，显示占位区域 -->
          <div v-else class="flex items-center justify-center border-t lg:border-t-0 lg:border-l border-gray-700/50">
            <div class="text-center">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-laser-purple mx-auto mb-4"></div>
                <p class="text-gray-400">{{ t('loading') }}</p>
        </div>
          </div>

        </div>
      </div>
    </div>
  </div>

  <Alert />
  <!-- 全局路由跳转Loading覆盖层 -->
  <div v-show="isLoading" class="bg-gradient-main flex items-center justify-center">
      <Loading />
  </div>
</template>
