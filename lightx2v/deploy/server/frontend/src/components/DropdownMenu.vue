<template>
  <div class="relative inline-block text-left">
    <Menu as="div">
      <div>
        <!-- Apple 风格菜单按钮 -->
        <MenuButton class="inline-flex w-full sm:min-w-[120px] md:min-w-[160px] lg:min-w-[200px] justify-between items-center px-4 py-2.5 sm:px-3.5 sm:py-2 bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] backdrop-saturate-[180%] border border-black/8 dark:border-white/8 rounded-xl text-sm sm:text-[13px] font-medium text-[#1d1d1f] dark:text-[#f5f5f7] cursor-pointer transition-all duration-200 ease-out shadow-[0_1px_3px_0_rgba(0,0,0,0.08),0_0_0_1px_rgba(0,0,0,0.02)] dark:shadow-[0_1px_3px_0_rgba(0,0,0,0.3),0_0_0_1px_rgba(255,255,255,0.04)] tracking-tight hover:bg-white dark:hover:bg-[#282828]/98 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_2px_6px_0_rgba(0,0,0,0.12),0_0_0_1px_rgba(0,0,0,0.04)] dark:hover:shadow-[0_2px_6px_0_rgba(0,0,0,0.4),0_0_0_1px_rgba(255,255,255,0.06)] focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_2px_6px_0_rgba(var(--brand-primary-rgb),0.15),0_0_0_3px_rgba(var(--brand-primary-rgb),0.1)] dark:focus:shadow-[0_2px_6px_0_rgba(var(--brand-primary-light-rgb),0.3),0_0_0_3px_rgba(var(--brand-primary-light-rgb),0.2)]">
          <span class="flex-1 text-left">{{ selectedLabel || placeholder }}</span>
          <ChevronDownIcon class="w-4 h-4 ml-2 flex-shrink-0 text-[#86868b] dark:text-[#98989d] transition-all duration-200" aria-hidden="true" />
        </MenuButton>
      </div>

      <transition
        enter-active-class="transition-all duration-200 ease-[cubic-bezier(0.34,1.56,0.64,1)]"
        enter-from-class="opacity-0 scale-95 -translate-y-2"
        enter-to-class="opacity-100 scale-100 translate-y-0"
        leave-active-class="transition-all duration-150 ease-[cubic-bezier(0.4,0,1,1)]"
        leave-from-class="opacity-100 scale-100 translate-y-0"
        leave-to-class="opacity-0 scale-95 -translate-y-1"
      >
        <!-- Apple 风格下拉菜单 -->
        <MenuItems class="absolute right-0 mt-2 min-w-[200px] w-max max-w-[320px] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[20px] backdrop-saturate-[180%] border border-black/8 dark:border-white/8 rounded-xl shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),0_10px_15px_-3px_rgba(0,0,0,0.1),0_0_0_1px_rgba(0,0,0,0.05)] dark:shadow-[0_4px_6px_-1px_rgba(0,0,0,0.3),0_10px_15px_-3px_rgba(0,0,0,0.2),0_0_0_1px_rgba(255,255,255,0.06)] overflow-hidden z-50 outline-none">
          <div class="p-1.5">
            <MenuItem v-for="item in items" :key="item.value" v-slot="{ active }">
              <button
                @click="selectItem(item)"
                :class="[
                  'flex w-full items-center px-3 py-2 sm:px-2.5 sm:py-1.5 border-0 bg-transparent rounded-lg text-sm sm:text-[13px] text-[#1d1d1f] dark:text-[#f5f5f7] text-left cursor-pointer transition-all duration-150 ease-out tracking-tight',
                  active ? 'bg-black/4 dark:bg-white/8' : '',
                  selectedValue === item.value ? 'font-medium bg-black/6 dark:bg-white/12' : 'font-normal'
                ]"
              >
                <i v-if="item.icon" :class="[item.icon, 'w-4 h-4 mr-2.5 sm:mr-2 flex-shrink-0 text-[#86868b] dark:text-[#98989d] transition-colors', active ? 'text-[#1d1d1f] dark:text-[#f5f5f7]' : '']" aria-hidden="true"></i>
                <span class="flex-1">{{ item.label }}</span>
                <i v-if="selectedValue === item.value" class="fas fa-check w-3.5 h-3.5 ml-3 flex-shrink-0 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
              </button>
            </MenuItem>
          </div>

          <div v-if="items.length === 0" class="p-1.5">
            <div class="px-3 py-3 text-center text-[13px] text-[#86868b] dark:text-[#98989d]">
              {{ emptyMessage }}
            </div>
          </div>
        </MenuItems>
      </transition>
    </Menu>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { Menu, MenuButton, MenuItems, MenuItem } from '@headlessui/vue'
import { ChevronDownIcon } from '@heroicons/vue/20/solid'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Props
const props = defineProps({
  items: {
    type: Array,
    default: () => []
  },
  selectedValue: {
    type: [String, Number],
    default: ''
  },
  placeholder: {
    type: String,
    default: ''
  },
  emptyMessage: {
    type: String,
    default: ''
  }
})

// Emits
const emit = defineEmits(['select-item'])

// Computed
const selectedLabel = computed(() => {
  const selectedItem = props.items.find(item => item.value === props.selectedValue)
  return selectedItem ? selectedItem.label : ''
})

// Methods
const selectItem = (item) => {
  emit('select-item', item)
}
</script>

<style scoped>
/* 所有样式已通过 Tailwind CSS 的 dark: 前缀在 template 中定义 */
/* Apple 风格的极简黑白设计 */
</style>
