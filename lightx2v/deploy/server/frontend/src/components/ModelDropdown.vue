<template>
  <DropdownMenu
    :items="modelItems"
    :selected-value="selectedModel"
    :placeholder="t('selectModel')"
    :empty-message="t('selectTaskTypeFirst')"
    @select-item="handleSelectModel"
  />
</template>

<script setup>
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import DropdownMenu from './DropdownMenu.vue'

const { t } = useI18n()

// Props
const props = defineProps({
  availableModels: {
    type: Array,
    default: () => []
  },
  selectedModel: {
    type: String,
    default: ''
  }
})

// Emits
const emit = defineEmits(['select-model'])

// Computed
const modelItems = computed(() => {
  return props.availableModels.map(model => {
    // 如果 model 已经是对象格式（包含 value 和 label），直接使用
    if (typeof model === 'object' && model !== null && model.value !== undefined) {
      return {
        value: model.value,
        label: model.label,
        icon: model.icon || 'fas fa-cog'
      }
    }
    // 否则，将字符串转换为对象格式
    return {
    value: model,
    label: model,
    icon: 'fas fa-cog'
    }
  })
})

// Methods
const handleSelectModel = (item) => {
  emit('select-model', item.value)
}
</script>
