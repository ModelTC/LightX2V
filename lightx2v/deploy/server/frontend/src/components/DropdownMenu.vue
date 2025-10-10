<template>
  <div class="relative inline-block text-left">
    <Menu as="div">
      <div>
        <MenuButton
          class="inline-flex w-full justify-center rounded-md bg-dark-light border border-laser-purple/30 px-4 py-3 text-sm font-medium text-white hover:bg-dark-light/80 hover:border-laser-purple/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-laser-purple/50 transition-all duration-200"
        >
          {{ selectedLabel || placeholder }}
          <ChevronDownIcon
            class="-mr-1 ml-2 h-5 w-5 text-laser-purple hover:text-gradient-primary transition-colors"
            aria-hidden="true"
          />
        </MenuButton>
      </div>

      <transition
        enter-active-class="transition duration-100 ease-out"
        enter-from-class="transform scale-95 opacity-0"
        enter-to-class="transform scale-100 opacity-100"
        leave-active-class="transition duration-75 ease-in"
        leave-from-class="transform scale-100 opacity-100"
        leave-to-class="transform scale-95 opacity-0"
      >
        <MenuItems
          class="absolute right-0 mt-2 w-56 origin-top-right divide-y divide-gray-700/50 rounded-md bg-dark-light border border-laser-purple/30 shadow-lg ring-1 ring-black/5 focus:outline-none z-50"
        >
          <div class="px-1 py-1">
            <MenuItem v-for="item in items" :key="item.value" v-slot="{ active }">
              <button
                @click="selectItem(item)"
                :class="[
                  active ? 'bg-laser-purple/20 text-white' : 'text-gray-300',
                  'group flex w-full items-center rounded-md px-3 py-2 text-sm transition-colors duration-200',
                ]"
              >
                <i v-if="item.icon" :class="[item.icon, 'mr-3 h-4 w-4 text-laser-purple']" aria-hidden="true"></i>
                {{ item.label }}
                <i v-if="selectedValue === item.value" class="fas fa-check ml-auto text-laser-purple"></i>
              </button>
            </MenuItem>
          </div>

          <div v-if="items.length === 0" class="px-1 py-1">
            <div class="px-3 py-2 text-sm text-gray-500 text-center">
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
