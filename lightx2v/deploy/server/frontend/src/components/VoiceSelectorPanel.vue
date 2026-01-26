<template>
  <div>
    <!-- 标签切换 -->
    <div class="flex items-center gap-2 mb-4">
      <button
        @click="localVoiceTab = 'ai'"
        class="px-4 py-2 rounded-lg text-sm font-medium transition-all"
        :class="localVoiceTab === 'ai'
          ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
          : 'bg-white/80 dark:bg-[#2c2c2e]/80 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c]'"
      >
        {{ t('aiVoice') }}
      </button>
      <button
        @click="localVoiceTab = 'clone'"
        class="px-4 py-2 rounded-lg text-sm font-medium transition-all"
        :class="localVoiceTab === 'clone'
          ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
          : 'bg-white/80 dark:bg-[#2c2c2e]/80 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c]'"
      >
        {{ t('clonedVoice') }}
      </button>
    </div>

    <!-- AI音色区域 -->
    <div v-if="localVoiceTab === 'ai'">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-2">
          <i class="fas fa-microphone-alt text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
          <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('selectVoice') }}</span>

          <button
            v-if="showHistoryButton"
            @click="$emit('open-history')"
            class="w-8 h-8 flex items-center justify-center rounded-full bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200"
            :title="t('ttsHistoryTabVoice')"
          >
            <i class="fas fa-history text-xs"></i>
          </button>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <!-- 搜索框 - Apple 风格 -->
        <div class="relative" :class="searchBoxWidth">
          <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-[#86868b] dark:text-[#98989d] text-xs pointer-events-none z-10"></i>
          <input
            v-model="localSearchQuery"
            :placeholder="t('searchVoice')"
            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-lg py-2 pl-9 pr-3 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 transition-all duration-200"
            type="text"
          />
        </div>

        <!-- 筛选按钮 - Apple 风格 -->
        <button @click="$emit('toggle-filter')"
          class="flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-lg transition-all duration-200 text-sm font-medium tracking-tight">
          <i class="fas fa-filter text-xs"></i>
          <span>{{ t('filter') }}</span>
        </button>
      </div>

      <!-- 音色列表容器 - Apple 风格 -->
      <div class="bg-white/50 dark:bg-[#2c2c2e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-2xl p-5 min-h-[500px] max-h-[500px] overflow-y-auto main-scrollbar pr-3 mt-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <label
            v-for="(voice, index) in filteredVoices"
            :key="index"
            class="relative m-0 p-0 cursor-pointer"
          >
            <input
              type="radio"
              :value="voice.voice_type"
              :checked="selectedVoice === voice.voice_type"
              @change="$emit('select-voice', voice)"
              class="sr-only"
            />
            <div
              class="relative flex items-center p-4 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
              :class="{
                'border-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/12 dark:bg-[color:var(--brand-primary-light)]/20 shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.35)] ring-2 ring-[color:var(--brand-primary)]/20 dark:ring-[color:var(--brand-primary-light)]/30': selectedVoice === voice.voice_type
              }"
            >
              <!-- 选中指示器 - Apple 风格 -->
              <div v-if="selectedVoice === voice.voice_type" class="absolute top-2 left-2 w-5 h-5 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full flex items-center justify-center z-10 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]">
                <i class="fas fa-check text-white text-[10px]"></i>
              </div>
              <!-- V2 标签 - Apple 风格 -->
              <div v-if="voice.version === '2.0'" class="absolute top-2 right-2 px-2 py-1 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white text-[10px] font-semibold rounded-md z-10">
                v2.0
              </div>

              <!-- 头像容器 -->
              <div class="relative mr-3 flex-shrink-0">
                <!-- Female Avatar -->
                <img
                  v-if="isFemaleVoice(voice.voice_type)"
                  src="../../public/female.svg"
                  alt="Female Avatar"
                  class="w-12 h-12 rounded-full object-cover bg-white transition-all duration-200"
                />
                <!-- Male Avatar -->
                <img
                  v-else
                  src="../../public/male.svg"
                  alt="Male Avatar"
                  class="w-12 h-12 rounded-full object-cover bg-white transition-all duration-200"
                />
                <!-- Loading 指示器 - Apple 风格 -->
                <div v-if="isGenerating && selectedVoice === voice.voice_type" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white z-20">
                  <i class="fas fa-spinner fa-spin text-xs"></i>
                </div>
              </div>

              <!-- 音色信息 -->
              <div class="flex-1 min-w-0">
                <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-1 tracking-tight truncate">
                  {{ voice.name }}
                </div>
                <div class="flex flex-wrap gap-1.5">
                  <span v-if="voice.scene" class="inline-block px-2 py-0.5 bg-black/5 dark:bg-white/5 text-[#86868b] dark:text-[#98989d] rounded text-[11px] font-medium">
                    {{ voice.scene }}
                  </span>
                  <span
                    v-for="langCode in voice.language"
                    :key="langCode"
                    class="inline-block px-2 py-0.5 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded text-[11px] font-medium"
                  >
                    {{ getLanguageDisplayName(langCode) }}
                  </span>
                </div>
              </div>
            </div>
          </label>
        </div>
      </div>
    </div>

    <!-- 克隆音色区域 -->
    <div v-else class="space-y-4">
      <div class="bg-white/50 dark:bg-[#2c2c2e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-2xl p-5 min-h-[500px] max-h-[500px] overflow-y-auto main-scrollbar pr-3 mt-4">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <!-- 添加音色按钮 -->
          <button
            @click="$emit('open-clone-modal')"
            class="relative flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border-2 border-dashed border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/50 rounded-xl transition-all duration-200 hover:bg-[color:var(--brand-primary)]/10 dark:hover:bg-[color:var(--brand-primary-light)]/10 hover:border-[color:var(--brand-primary)] dark:hover:border-[color:var(--brand-primary-light)]"
          >
            <div class="flex flex-row items-center gap-2">
              <div class="w-12 h-12 rounded-full bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                <i class="fas fa-plus text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
              </div>
              <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('addClonedVoice') }}</span>
            </div>
          </button>

          <!-- 克隆音色列表 -->
          <label
            v-for="(voice, index) in clonedVoices"
            :key="index"
            class="relative m-0 p-0 cursor-pointer"
          >
            <input
              type="radio"
              :value="`clone_${voice.speaker_id}`"
              :checked="selectedVoice === `clone_${voice.speaker_id}`"
              @change="$emit('select-clone-voice', voice)"
              class="sr-only"
            />
            <div
              class="relative flex items-center p-4 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
              :class="{
                'border-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/12 dark:bg-[color:var(--brand-primary-light)]/20 shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.35)] ring-2 ring-[color:var(--brand-primary)]/20 dark:ring-[color:var(--brand-primary-light)]/30': selectedVoice === `clone_${voice.speaker_id}`
              }"
            >
              <!-- 选中指示器 -->
              <div v-if="selectedVoice === `clone_${voice.speaker_id}`" class="absolute top-2 left-2 w-5 h-5 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full flex items-center justify-center z-10 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]">
                <i class="fas fa-check text-white text-[10px]"></i>
              </div>

              <!-- 头像容器 -->
              <div class="relative mr-3 flex-shrink-0">
                <div class="w-12 h-12 rounded-full bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                  <i class="fas fa-user text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
                </div>
                <!-- Loading 指示器 -->
                <div v-if="isGenerating && selectedVoice === `clone_${voice.speaker_id}`" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white z-20">
                  <i class="fas fa-spinner fa-spin text-xs"></i>
                </div>
              </div>

              <!-- 音色信息 -->
              <div class="flex-1 min-w-0">
                <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-1 tracking-tight truncate">
                  {{ voice.name || t('unnamedVoice') }}
                </div>
                <div class="text-xs text-[#86868b] dark:text-[#98989d]">
                  {{ formatDate(voice.create_t) }}
                </div>
              </div>

              <!-- 删除按钮 -->
              <button
                v-if="showDeleteButton"
                @click.stop="$emit('delete-clone-voice', voice)"
                class="ml-2 w-8 h-8 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 hover:border-red-200 dark:hover:border-red-500/20 rounded-full transition-all duration-200 hover:scale-110 active:scale-100 flex-shrink-0"
                :title="t('delete')"
              >
                <i class="fas fa-trash-alt text-xs"></i>
              </button>
            </div>
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  // 音色数据
  filteredVoices: {
    type: Array,
    required: true,
    default: () => []
  },
  clonedVoices: {
    type: Array,
    required: true,
    default: () => []
  },
  // 当前选中的音色
  selectedVoice: {
    type: String,
    default: ''
  },
  // 是否正在生成
  isGenerating: {
    type: Boolean,
    default: false
  },
  // 搜索查询（可选，如果提供则使用外部控制，否则内部管理）
  searchQuery: {
    type: String,
    default: ''
  },
  // 初始标签页
  initialTab: {
    type: String,
    default: 'ai',
    validator: (value) => ['ai', 'clone'].includes(value)
  },
  // 是否显示历史按钮
  showHistoryButton: {
    type: Boolean,
    default: false
  },
  // 是否显示删除按钮（克隆音色）
  showDeleteButton: {
    type: Boolean,
    default: true
  },
  // 搜索框宽度
  searchBoxWidth: {
    type: String,
    default: 'w-52'
  },
  // 方法
  isFemaleVoice: {
    type: Function,
    required: true
  },
  getLanguageDisplayName: {
    type: Function,
    required: true
  },
  formatDate: {
    type: Function,
    required: true
  },
  t: {
    type: Function,
    required: true
  },
})

const emit = defineEmits([
  'select-voice',
  'select-clone-voice',
  'open-clone-modal',
  'delete-clone-voice',
  'open-history',
  'toggle-filter',
  'update:searchQuery',
  'update:tab'
])

// 内部状态
const localVoiceTab = ref(props.initialTab)
const localSearchQuery = ref(props.searchQuery || '')

// 监听外部 initialTab 变化
watch(() => props.initialTab, (newVal) => {
  if (newVal && newVal !== localVoiceTab.value) {
    localVoiceTab.value = newVal
  }
})

// 监听外部 searchQuery 变化
watch(() => props.searchQuery, (newVal) => {
  if (newVal !== undefined && newVal !== localSearchQuery.value) {
    localSearchQuery.value = newVal
  }
})

// 监听内部 searchQuery 变化，同步到外部
watch(localSearchQuery, (newVal) => {
  emit('update:searchQuery', newVal)
})

// 监听标签页变化
watch(localVoiceTab, (newVal) => {
  emit('update:tab', newVal)
})

// 暴露方法供父组件调用
defineExpose({
  reset: () => {
    localVoiceTab.value = props.initialTab
    localSearchQuery.value = ''
  }
})
</script>

<style scoped>
/* 隐藏 radio input */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
</style>
