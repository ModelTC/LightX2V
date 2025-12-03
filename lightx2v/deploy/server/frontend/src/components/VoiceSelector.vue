<template>
  <div class="voice-selector voice-selector-component w-full" :class="{ 'dropdown-mode': mode === 'dropdown' }">
    <!-- 完整模式：包含搜索和筛选 -->
    <template v-if="mode === 'full'">
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

      <div v-if="showSearch || showFilter" class="flex items-center gap-3 mb-4">
        <!-- 搜索框 - Apple 风格 -->
        <div v-if="showSearch" class="relative w-52">
          <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-[#86868b] dark:text-[#98989d] text-xs pointer-events-none z-10"></i>
          <input
            :value="searchQuery"
            @input="$emit('update-search', $event.target.value)"
            :placeholder="t('searchVoice')"
            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-lg py-2 pl-9 pr-3 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 transition-all duration-200"
            type="text"
          />
        </div>

        <!-- 筛选按钮 - Apple 风格 -->
        <button
          v-if="showFilter"
          @click="$emit('toggle-filter')"
          class="flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-lg transition-all duration-200 text-sm font-medium tracking-tight"
        >
          <i class="fas fa-filter text-xs"></i>
          <span>{{ t('filter') }}</span>
        </button>
      </div>
    </template>

    <!-- 音色列表容器 - Apple 风格 -->
    <div
      :class="{
        'bg-white/50 dark:bg-[#2c2c2e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-2xl p-5 max-h-[500px] overflow-y-auto main-scrollbar pr-3': mode === 'full',
        'p-3 max-h-96 overflow-y-auto main-scrollbar': mode === 'dropdown'
      }"
      ref="voiceListContainer"
    >
      <div :class="{ 'grid grid-cols-1 md:grid-cols-2 gap-3': mode === 'full', 'space-y-2': mode === 'dropdown' }">
        <label
          v-for="(voice, index) in filteredVoices"
          :key="index"
          :class="{
            'relative m-0 p-0 cursor-pointer': mode === 'full',
            'relative m-0 p-0 cursor-pointer': mode === 'dropdown'
          }"
        >
          <input
            type="radio"
            :value="voice.voice_type"
            :checked="selectedVoice === voice.voice_type"
            @change="$emit('select-voice', voice)"
            class="sr-only"
          />
          <div
            class="relative flex items-center bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
            :class="{
              'border-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/12 dark:bg-[color:var(--brand-primary-light)]/20 shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.35)] ring-2 ring-[color:var(--brand-primary)]/20 dark:ring-[color:var(--brand-primary-light)]/30': selectedVoice === voice.voice_type,
              'p-4': mode === 'full',
              'p-3': mode === 'dropdown'
            }"
            @click="mode === 'dropdown' && $emit('select-voice', voice)"
          >
            <!-- 选中指示器 - Apple 风格 -->
            <div v-if="selectedVoice === voice.voice_type"
              class="absolute w-5 h-5 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full flex items-center justify-center z-10 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]"
              :class="'top-2 left-2'"
            >
              <i class="fas fa-check text-white text-[10px]"></i>
            </div>
            <!-- V2 标签 - Apple 风格（在 dropdown 模式下选中时隐藏） -->
            <div v-if="voice.version === '2.0'" class="absolute top-2 right-2 px-2 py-1 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white text-[10px] font-semibold rounded-md z-10">
              v2.0
            </div>

            <!-- 头像容器 -->
            <div class="relative flex-shrink-0 mr-3"
            >
              <!-- Female Avatar -->
              <img
                v-if="isFemaleVoice(voice.voice_type)"
                src="../../public/female.svg"
                alt="Female Avatar"
                :class="{
                  'w-12 h-12': mode === 'full',
                  'w-10 h-10': mode === 'dropdown'
                }"
                class="rounded-full object-cover bg-white transition-all duration-200"
              />
              <!-- Male Avatar -->
              <img
                v-else
                src="../../public/male.svg"
                alt="Male Avatar"
                :class="{
                  'w-12 h-12': mode === 'full',
                  'w-10 h-10': mode === 'dropdown'
                }"
                class="rounded-full object-cover bg-white transition-all duration-200"
              />
              <!-- Loading 指示器 - Apple 风格 -->
              <div v-if="isGenerating && selectedVoice === voice.voice_type" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white z-20">
                <i class="fas fa-spinner fa-spin text-xs"></i>
              </div>
            </div>

            <!-- 音色信息 -->
            <div class="flex-1 min-w-0">
              <div class="font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate"
                :class="{
                  'text-sm mb-1': mode === 'full' || (mode === 'dropdown' && selectedVoice !== voice.voice_type),
                  'text-xs': mode === 'dropdown' && selectedVoice === voice.voice_type
                }"
              >
                {{ voice.name }}
              </div>
              <!-- 场景和语言标签 - 在 full 模式下或 dropdown 模式下未选中时显示 -->
              <div
                class="flex flex-wrap gap-1.5"
              >
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
</template>

<script setup>
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Props
const props = defineProps({
  voices: {
    type: Array,
    default: () => []
  },
  filteredVoices: {
    type: Array,
    required: true
  },
  selectedVoice: {
    type: String,
    default: ''
  },
  searchQuery: {
    type: String,
    default: ''
  },
  isGenerating: {
    type: Boolean,
    default: false
  },
  mode: {
    type: String,
    default: 'full', // 'full' | 'dropdown'
    validator: (value) => ['full', 'dropdown'].includes(value)
  },
  showSearch: {
    type: Boolean,
    default: true
  },
  showFilter: {
    type: Boolean,
    default: true
  },
  showHistoryButton: {
    type: Boolean,
    default: true
  }
})

// Emits
const emit = defineEmits(['select-voice', 'update-search', 'toggle-filter', 'open-history'])

// Refs
const voiceListContainer = ref(null)

// 检查是否为女性音色
const isFemaleVoice = (name) => {
  return name.toLowerCase().includes('female')
}

// 语言代码转显示名称
const getLanguageDisplayName = (langCode) => {
  const languageMap = {
    'chinese': '中文',
    'en_us': '美式英语',
    'en_gb': '英式英语',
    'en_au': '澳洲英语',
    'es': '西语',
    'ja': '日语'
  }
  return languageMap[langCode] || langCode
}

// 暴露给父组件的方法
defineExpose({
  voiceListContainer
})
</script>

<style scoped>
/* 所有样式已通过 Tailwind CSS 在 template 中定义 */
</style>
