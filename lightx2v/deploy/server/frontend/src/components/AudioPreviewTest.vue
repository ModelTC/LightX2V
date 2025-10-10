<template>
  <div class="p-6">
    <h2 class="text-xl font-bold text-white mb-4">音频预览测试</h2>
    
    <!-- 测试音频模板预览 -->
    <div class="mb-6">
      <h3 class="text-lg text-white mb-2">音频模板预览测试</h3>
      <div class="space-y-2">
        <div v-for="template in audioTemplates" :key="template.filename" 
             class="flex items-center gap-4 p-3 bg-dark-light rounded-lg">
          <span class="text-white">{{ template.filename }}</span>
          <button @click="previewAudioTemplate(template)"
                  class="px-3 py-1 bg-laser-purple text-white rounded hover:bg-laser-purple/80">
            预览
          </button>
        </div>
        <div v-if="audioTemplates.length === 0" class="text-gray-400">
          暂无音频模板
        </div>
      </div>
    </div>

    <!-- 测试音频历史预览 -->
    <div class="mb-6">
      <h3 class="text-lg text-white mb-2">音频历史预览测试</h3>
      <div class="space-y-2">
        <div v-for="history in audioHistory" :key="history.filename" 
             class="flex items-center gap-4 p-3 bg-dark-light rounded-lg">
          <span class="text-white">{{ history.filename }}</span>
          <button @click="previewAudioHistory(history)"
                  class="px-3 py-1 bg-laser-purple text-white rounded hover:bg-laser-purple/80">
            预览
          </button>
        </div>
        <div v-if="audioHistory.length === 0" class="text-gray-400">
          暂无音频历史
        </div>
      </div>
    </div>

    <!-- 调试信息 -->
    <div class="mt-6 p-4 bg-gray-800 rounded-lg">
      <h3 class="text-lg text-white mb-2">调试信息</h3>
      <div class="text-sm text-gray-300">
        <p>音频模板数量: {{ audioTemplates.length }}</p>
        <p>音频历史数量: {{ audioHistory.length }}</p>
        <div v-if="audioTemplates.length > 0">
          <p>第一个音频模板:</p>
          <pre class="text-xs bg-gray-900 p-2 rounded mt-1">{{ JSON.stringify(audioTemplates[0], null, 2) }}</pre>
        </div>
        <div v-if="audioHistory.length > 0">
          <p>第一个音频历史:</p>
          <pre class="text-xs bg-gray-900 p-2 rounded mt-1">{{ JSON.stringify(audioHistory[0], null, 2) }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { 
  audioTemplates,
  audioHistory,
  previewAudioTemplate,
  previewAudioHistory
} from '../utils/other'
</script>
