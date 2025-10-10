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
  return props.availableModels.map(model => ({
    value: model,
    label: model,
    icon: 'fas fa-cog'
  }))
})

// Methods
const handleSelectModel = (item) => {
  emit('select-model', item.value)
}
</script>
