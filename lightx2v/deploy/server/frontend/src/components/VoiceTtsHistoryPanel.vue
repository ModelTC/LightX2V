<script setup>
import { ref, watch, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  history: {
    type: Array,
    default: () => []
  },
  mode: {
    type: String,
    default: 'combined'
  },
  getVoiceName: {
    type: Function,
    default: () => ''
  }
})

const emit = defineEmits(['close', 'apply', 'delete'])

const { t } = useI18n()

const normalizedMode = computed(() => {
  const modes = ['combined', 'text', 'instruction', 'voice']
  return modes.includes(props.mode) ? props.mode : 'combined'
})

const makeTextEntries = () => {
  const seen = new Set()
  const list = []
  for (const entry of props.history || []) {
    const value = (entry?.text || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    list.push({ id: value, value, label: value })
  }
  return list
}

const makeInstructionEntries = () => {
  const seen = new Set()
  const list = []
  for (const entry of props.history || []) {
    const value = (entry?.instruction || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    list.push({ id: value, value, label: value })
  }
  return list
}

const makeVoiceEntries = () => {
  const seen = new Set()
  const list = []
  for (const entry of props.history || []) {
    const value = (entry?.voiceType || '').trim()
    const label = props.getVoiceName(entry) || entry?.voiceName || value
    if (!value || seen.has(value)) continue
    seen.add(value)
    list.push({ id: value, value, label })
  }
  return list
}

const filteredHistory = computed(() => {
  switch (normalizedMode.value) {
    case 'text':
      return makeTextEntries()
    case 'instruction':
      return makeInstructionEntries()
    case 'voice':
      return makeVoiceEntries()
    case 'combined':
    default:
      return props.history || []
  }
})

const totalCount = computed(() => filteredHistory.value.length)

const selectedKey = ref(null)

const panelTitle = computed(() => {
  const map = {
    combined: t('ttsHistoryTitleCombined'),
    text: t('ttsHistoryTitleText'),
    instruction: t('ttsHistoryTitleInstruction'),
    voice: t('ttsHistoryTitleVoice')
  }
  return map[normalizedMode.value] || t('ttsHistoryTitle')
})

const isFemaleVoice = (entry) => {
  const value = (entry?.voiceType || entry?.voiceName || entry?.label || '').toLowerCase()
  return value.includes('female') || value.includes('女')
}

const getEntryKey = (entry) => {
  if (normalizedMode.value === 'combined') {
    return entry?.id ?? null
  }
  return entry?.value ?? null
}

const ensureSelection = () => {
  if (!props.visible) {
    selectedKey.value = null
    return
  }

  const list = filteredHistory.value
  if (!list.length) {
    selectedKey.value = null
    return
  }

  const currentKey = selectedKey.value
  if (list.some((entry) => getEntryKey(entry) === currentKey)) {
    return
  }

  selectedKey.value = getEntryKey(list[0])
}

watch(() => props.visible, ensureSelection)
watch(filteredHistory, ensureSelection)
watch(normalizedMode, ensureSelection)

const isCombinedMode = computed(() => normalizedMode.value === 'combined')
const isApplyDisabled = computed(() => !props.visible || !selectedKey.value)

const handleOverlayClick = () => {
  emit('close')
}

const handlePanelClick = (event) => {
  event.stopPropagation()
}

const handleEntryClick = (entry) => {
  selectedKey.value = getEntryKey(entry)
}

const handleApplyClick = () => {
  if (isApplyDisabled.value) return

  if (normalizedMode.value === 'combined') {
    const entry = filteredHistory.value.find((item) => getEntryKey(item) === selectedKey.value)
    if (entry) {
      emit('apply', entry)
    }
  } else {
    emit('apply', selectedKey.value)
  }
}

const handleDeleteClick = (event, entry) => {
  if (!isCombinedMode.value) return
  event.stopPropagation()
  emit('delete', entry)
}

const getEntryVoiceLabel = (entry) => {
  return props.getVoiceName(entry) || entry.voiceType || t('ttsHistoryVoiceEmpty')
}
</script>

<template>
  <div
    v-if="visible"
    class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4"
    @click="handleOverlayClick"
  >
    <div
      class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl w-full max-w-2xl max-h-[85vh] overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex flex-col"
      @click.stop="handlePanelClick"
    >
      <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
        <div class="flex items-center gap-3">
          <h3 class="text-lg font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight flex items-center gap-2">
            <i class="fas fa-history text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
            <span>{{ panelTitle }}</span>
          </h3>
          <span
            v-if="totalCount > 0"
            class="px-2 py-0.5 rounded-full text-xs font-medium bg-black/5 dark:bg-white/10 text-[#86868b] dark:text-[#98989d]"
          >
            {{ totalCount }}
          </span>
        </div>
        <div class="flex items-center gap-2">
          <button
            @click.stop="handleApplyClick"
            :disabled="isApplyDisabled"
            class="w-9 h-9 flex items-center justify-center rounded-full transition-all duration-200 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 active:scale-100"
            :title="t('ttsHistoryApplySelected')"
          >
            <i class="fas fa-check text-sm"></i>
          </button>
          <button
            @click.stop="emit('close')"
            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100"
          >
            <i class="fas fa-times text-sm"></i>
          </button>
        </div>
      </div>

      <div class="flex-1 min-h-[50vh] overflow-y-auto p-6 main-scrollbar">
        <div v-if="!filteredHistory.length" class="flex flex-col items-center justify-center py-12 text-center">
          <div class="w-16 h-16 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 rounded-full flex items-center justify-center mb-4">
            <i class="fas fa-book text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-2xl"></i>
          </div>
          <p class="text-[#1d1d1f] dark:text-[#f5f5f7] text-lg font-medium mb-2 tracking-tight">{{ t('ttsHistoryEmpty') }}</p>
          <p class="text-[#86868b] dark:text-[#98989d] text-sm tracking-tight">{{ t('ttsHistoryEmptyHint') }}</p>
        </div>

        <template v-else>
          <div v-if="normalizedMode === 'voice'" class="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div
              v-for="entry in filteredHistory"
              :key="getEntryKey(entry)"
              @click="handleEntryClick(entry)"
              class="p-4 border border-black/8 dark:border-white/8 rounded-2xl bg-white/80 dark:bg-[#2c2c2e]/80 hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 cursor-pointer flex items-center gap-3"
              :class="{
                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] shadow-[0_0_0_2px_rgba(var(--brand-primary-rgb),0.2)] dark:shadow-[0_0_0_2px_rgba(var(--brand-primary-light-rgb),0.25)] ring-2 ring-[color:var(--brand-primary)]/20 dark:ring-[color:var(--brand-primary-light)]/25': getEntryKey(entry) === selectedKey
              }"
            >
              <div class="relative flex-shrink-0">
                <img
                      v-if="isFemaleVoice(entry)"
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
              </div>
              <div class="flex-1 min-w-0 space-y-1">
                <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate">
                  {{ entry.label }}
                </div>
                <div class="text-xs text-[#86868b] dark:text-[#98989d] tracking-tight truncate">
                  {{ entry.voiceType }}
                </div>
              </div>
            </div>
          </div>
          <div v-else class="space-y-3">
            <div
              v-for="entry in filteredHistory"
              :key="getEntryKey(entry)"
              @click="handleEntryClick(entry)"
              class="p-4 border border-black/8 dark:border-white/8 rounded-2xl bg-white/80 dark:bg-[#2c2c2e]/80 hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 cursor-pointer"
              :class="{
                'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] shadow-[0_0_0_2px_rgba(var(--brand-primary-rgb),0.2)] dark:shadow-[0_0_0_2px_rgba(var(--brand-primary-light-rgb),0.25)]': getEntryKey(entry) === selectedKey
              }"
            >
              <div class="flex flex-col gap-3">
                <div class="flex items-start justify-between gap-3">
                  <div class="flex-1 min-w-0 space-y-2">
                    <template v-if="normalizedMode === 'combined'">
                      <div class="text-sm font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight break-words whitespace-pre-line">
                        <span class="text-xs uppercase text-[#86868b] dark:text-[#98989d] mr-2">{{ t('ttsHistoryTextLabel') }}:</span>
                        <span>{{ entry.text || t('ttsHistoryTextEmpty') }}</span>
                      </div>
                      <div class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight break-words whitespace-pre-line">
                        <span class="text-xs uppercase text-[#86868b] dark:text-[#98989d] mr-2">{{ t('ttsHistoryInstructionLabel') }}:</span>
                        <span>{{ entry.instruction || t('ttsHistoryInstructionEmpty') }}</span>
                      </div>
                      <div class="text-sm text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight break-words">
                        <span class="text-xs uppercase text-[#86868b] dark:text-[#98989d] mr-2">{{ t('ttsHistoryVoiceLabel') }}:</span>
                        <span>{{ getEntryVoiceLabel(entry) }}</span>
                      </div>
                    </template>
                    <template v-else>
                      <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight break-words whitespace-pre-line">
                        <span>{{ entry.label }}</span>
                      </div>
                    </template>
                  </div>
                  <button
                    v-if="isCombinedMode"
                    @click="handleDeleteClick($event, entry)"
                    class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 hover:text-red-600 dark:hover:text-red-300 hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200"
                    :title="t('ttsHistoryDeleteEntry')"
                  >
                    <i class="fas fa-trash text-sm"></i>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>
