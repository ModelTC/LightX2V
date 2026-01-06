<script setup>
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps({
  imageUrl: {
    type: String,
    required: true
  },
  initialBbox: {
    type: Array,
    default: () => [0, 0, 0, 0] // [x1, y1, x2, y2]
  },
  characterLabel: {
    type: String,
    default: 'New Character'
  },
  isAddingNew: {
    type: Boolean,
    default: false
  },
  existingCharacters: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['save', 'cancel'])

// 状态
const imageContainerRef = ref(null)
const imageLoaded = ref(false)
const editingFaceBbox = ref([...props.initialBbox])
const isDraggingBbox = ref(false)
const dragType = ref('move')
const dragStartPos = ref({ x: 0, y: 0 })
const dragStartBbox = ref([0, 0, 0, 0])

// 监听 initialBbox 变化
watch(() => props.initialBbox, (newBbox) => {
  if (newBbox && newBbox.length === 4) {
    editingFaceBbox.value = [...newBbox]
  }
}, { immediate: true })

// 计算边界框样式
const getBboxStyle = computed(() => {
  if (!imageContainerRef.value || editingFaceBbox.value.length !== 4 || !imageLoaded.value) {
    return {}
  }

  const container = imageContainerRef.value
  const img = container.querySelector('img')
  if (!img || !img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
    return {}
  }

  const imgRect = img.getBoundingClientRect()
  const containerRect = container.getBoundingClientRect()
  const displayWidth = imgRect.width
  const displayHeight = imgRect.height
  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight

  if (naturalWidth === 0 || naturalHeight === 0) {
    return {}
  }

  // 检查 #app 是否有 transform: scale
  const appElement = document.getElementById('app')
  let appScale = 1
  if (appElement) {
    const appStyle = window.getComputedStyle(appElement)
    const transform = appStyle.transform
    if (transform && transform !== 'none') {
      const matrix = transform.match(/matrix\(([^)]+)\)/)
      if (matrix) {
        const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
        appScale = values[0] || 1
      } else {
        const scaleMatch = transform.match(/scale\(([^)]+)\)/)
        if (scaleMatch) {
          appScale = parseFloat(scaleMatch[1])
        }
      }
    }
  }

  const scaleX = displayWidth / (naturalWidth * appScale)
  const scaleY = displayHeight / (naturalHeight * appScale)

  let offsetX = imgRect.left - containerRect.left
  let offsetY = imgRect.top - containerRect.top

  if (Math.abs(offsetX) < 1 && Math.abs(offsetY) < 1) {
    offsetX = 0
    offsetY = 0
  }

  const [x1, y1, x2, y2] = editingFaceBbox.value
  const left = offsetX + x1 * scaleX
  const top = offsetY + y1 * scaleY
  const width = (x2 - x1) * scaleX
  const height = (y2 - y1) * scaleY

  return {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`
  }
})

// 计算 clip-path
const clipPath = computed(() => {
  if (!getBboxStyle.value.left) return ''

  const left = parseFloat(getBboxStyle.value.left) || 0
  const top = parseFloat(getBboxStyle.value.top) || 0
  const width = parseFloat(getBboxStyle.value.width) || 0
  const height = parseFloat(getBboxStyle.value.height) || 0
  const right = left + width
  const bottom = top + height

  return `polygon(
    0% 0%, 0% 100%,
    ${left}px 100%, ${left}px ${top}px,
    ${right}px ${top}px, ${right}px ${bottom}px,
    ${left}px ${bottom}px, ${left}px 100%,
    100% 100%, 100% 0%
  )`
})

// 计算现有角色的显示样式
const getExistingCharacterStyle = (char) => {
  if (!char.bbox || !Array.isArray(char.bbox) || char.bbox.length !== 4) {
    return { display: 'none' }
  }

  if (!imageContainerRef.value || !imageLoaded.value) {
    return { display: 'none' }
  }

  const container = imageContainerRef.value
  const img = container.querySelector('img')
  if (!img || !img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
    return { display: 'none' }
  }

  const imgRect = img.getBoundingClientRect()
  const containerRect = container.getBoundingClientRect()
  const displayWidth = imgRect.width
  const displayHeight = imgRect.height
  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight

  if (naturalWidth === 0 || naturalHeight === 0) {
    return { display: 'none' }
  }

  const appElement = document.getElementById('app')
  let appScale = 1
  if (appElement) {
    const appStyle = window.getComputedStyle(appElement)
    const transform = appStyle.transform
    if (transform && transform !== 'none') {
      const matrix = transform.match(/matrix\(([^)]+)\)/)
      if (matrix) {
        const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
        appScale = values[0] || 1
      } else {
        const scaleMatch = transform.match(/scale\(([^)]+)\)/)
        if (scaleMatch) {
          appScale = parseFloat(scaleMatch[1])
        }
      }
    }
  }

  const scaleX = displayWidth / (naturalWidth * appScale)
  const scaleY = displayHeight / (naturalHeight * appScale)
  const offsetX = imgRect.left - containerRect.left
  const offsetY = imgRect.top - containerRect.top

  const [x1, y1, x2, y2] = char.bbox
  const left = offsetX + x1 * scaleX
  const top = offsetY + y1 * scaleY
  const width = (x2 - x1) * scaleX
  const height = (y2 - y1) * scaleY

  return {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`
  }
}

// 图片加载处理
const handleImageLoad = () => {
  const img = imageContainerRef.value?.querySelector('img')
  if (img && img.naturalWidth > 0 && img.naturalHeight > 0) {
    imageLoaded.value = true

    // 如果是新增模式且没有初始 bbox，设置默认值
    if (props.isAddingNew && (!props.initialBbox || props.initialBbox.every(v => v === 0))) {
      const imgNaturalWidth = img.naturalWidth
      const imgNaturalHeight = img.naturalHeight
      // 增大初始框大小：使用图片尺寸的 50%，最大不超过 400px
      const bboxSize = Math.min(imgNaturalWidth, imgNaturalHeight, 400) * 0.5
      const centerX = imgNaturalWidth / 2
      const centerY = imgNaturalHeight / 2

      editingFaceBbox.value = [
        centerX - bboxSize / 2,
        centerY - bboxSize / 2,
        centerX + bboxSize / 2,
        centerY + bboxSize / 2
      ]
    }
  }
}

const handleImageError = () => {
  imageLoaded.value = true
}

// 拖拽处理
const startDragBbox = (event, type = 'move') => {
  event.preventDefault()
  event.stopPropagation()

  const container = imageContainerRef.value
  if (!container) return

  const img = container.querySelector('img')
  if (!img) return

  const imgRect = img.getBoundingClientRect()
  const containerRect = container.getBoundingClientRect()
  const displayWidth = imgRect.width
  const displayHeight = imgRect.height
  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight

  if (naturalWidth === 0 || naturalHeight === 0 || displayWidth === 0 || displayHeight === 0) {
    return
  }

  const appElement = document.getElementById('app')
  let appScale = 1
  if (appElement) {
    const appStyle = window.getComputedStyle(appElement)
    const transform = appStyle.transform
    if (transform && transform !== 'none') {
      const matrix = transform.match(/matrix\(([^)]+)\)/)
      if (matrix) {
        const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
        appScale = values[0] || 1
      } else {
        const scaleMatch = transform.match(/scale\(([^)]+)\)/)
        if (scaleMatch) {
          appScale = parseFloat(scaleMatch[1])
        }
      }
    }
  }

  const scaleX = displayWidth / (naturalWidth * appScale)
  const scaleY = displayHeight / (naturalHeight * appScale)
  const offsetX = imgRect.left - containerRect.left
  const offsetY = imgRect.top - containerRect.top

  const clickX = event.clientX - containerRect.left
  const clickY = event.clientY - containerRect.top

  if (type !== 'move') {
    isDraggingBbox.value = true
    dragType.value = type
    dragStartPos.value = { x: clickX, y: clickY }
    dragStartBbox.value = [...editingFaceBbox.value]
    return
  }

  const [x1, y1, x2, y2] = editingFaceBbox.value
  const bboxRect = {
    left: offsetX + x1 * scaleX,
    top: offsetY + y1 * scaleY,
    right: offsetX + x2 * scaleX,
    bottom: offsetY + y2 * scaleY
  }

  if (clickX < bboxRect.left || clickX > bboxRect.right ||
      clickY < bboxRect.top || clickY > bboxRect.bottom) {
    return
  }

  isDraggingBbox.value = true
  dragType.value = 'move'
  dragStartPos.value = { x: clickX, y: clickY }
  dragStartBbox.value = [...editingFaceBbox.value]
}

const dragBbox = (event) => {
  if (!isDraggingBbox.value) return

  const container = imageContainerRef.value
  if (!container) return

  const img = container.querySelector('img')
  if (!img || !img.complete) return

  const imgRect = img.getBoundingClientRect()
  const containerRect = container.getBoundingClientRect()
  const displayWidth = imgRect.width
  const displayHeight = imgRect.height
  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight

  if (naturalWidth === 0 || naturalHeight === 0) return

  const appElement = document.getElementById('app')
  let appScale = 1
  if (appElement) {
    const appStyle = window.getComputedStyle(appElement)
    const transform = appStyle.transform
    if (transform && transform !== 'none') {
      const matrix = transform.match(/matrix\(([^)]+)\)/)
      if (matrix) {
        const values = matrix[1].split(',').map(v => parseFloat(v.trim()))
        appScale = values[0] || 1
      } else {
        const scaleMatch = transform.match(/scale\(([^)]+)\)/)
        if (scaleMatch) {
          appScale = parseFloat(scaleMatch[1])
        }
      }
    }
  }

  const scaleX = displayWidth / (naturalWidth * appScale)
  const scaleY = displayHeight / (naturalHeight * appScale)
  const offsetX = imgRect.left - containerRect.left
  const offsetY = imgRect.top - containerRect.top

  const currentX = event.clientX - containerRect.left
  const currentY = event.clientY - containerRect.top

  const imgCurrentX = currentX - offsetX
  const imgCurrentY = currentY - offsetY
  const imgStartX = dragStartPos.value.x - offsetX
  const imgStartY = dragStartPos.value.y - offsetY

  const deltaX = (imgCurrentX - imgStartX) / scaleX
  const deltaY = (imgCurrentY - imgStartY) / scaleY

  const [startX1, startY1, startX2, startY2] = dragStartBbox.value
  const startWidth = startX2 - startX1
  const startHeight = startY2 - startY1

  let newX1 = startX1
  let newY1 = startY1
  let newX2 = startX2
  let newY2 = startY2

  const type = dragType.value
  const minSize = 10

  if (type === 'move') {
    newX1 = startX1 + deltaX
    newY1 = startY1 + deltaY
    newX2 = startX2 + deltaX
    newY2 = startY2 + deltaY
  } else if (type === 'resize-n') {
    newY1 = Math.min(startY1 + deltaY, startY2 - minSize)
    newX1 = startX1
    newX2 = startX2
    newY2 = startY2
  } else if (type === 'resize-s') {
    newY2 = Math.max(startY2 + deltaY, startY1 + minSize)
    newX1 = startX1
    newY1 = startY1
    newX2 = startX2
  } else if (type === 'resize-w') {
    newX1 = Math.min(startX1 + deltaX, startX2 - minSize)
    newY1 = startY1
    newX2 = startX2
    newY2 = startY2
  } else if (type === 'resize-e') {
    newX2 = Math.max(startX2 + deltaX, startX1 + minSize)
    newX1 = startX1
    newY1 = startY1
    newY2 = startY2
  } else if (type === 'resize-nw') {
    newX1 = Math.min(startX1 + deltaX, startX2 - minSize)
    newY1 = Math.min(startY1 + deltaY, startY2 - minSize)
    newX2 = startX2
    newY2 = startY2
  } else if (type === 'resize-ne') {
    newX2 = Math.max(startX2 + deltaX, startX1 + minSize)
    newY1 = Math.min(startY1 + deltaY, startY2 - minSize)
    newX1 = startX1
    newY2 = startY2
  } else if (type === 'resize-sw') {
    newX1 = Math.min(startX1 + deltaX, startX2 - minSize)
    newY2 = Math.max(startY2 + deltaY, startY1 + minSize)
    newX2 = startX2
    newY1 = startY1
  } else if (type === 'resize-se') {
    newX2 = Math.max(startX2 + deltaX, startX1 + minSize)
    newY2 = Math.max(startY2 + deltaY, startY1 + minSize)
    newX1 = startX1
    newY1 = startY1
  }

  // 边界限制
  if (newX1 < 0) {
    newX1 = 0
    if (type.includes('w') || type === 'resize-nw' || type === 'resize-sw') {
      newX2 = Math.max(newX2, minSize)
    }
  }
  if (newX2 > naturalWidth) {
    newX2 = naturalWidth
    if (type.includes('e') || type === 'resize-ne' || type === 'resize-se') {
      newX1 = Math.min(newX1, naturalWidth - minSize)
    }
  }
  if (newY1 < 0) {
    newY1 = 0
    if (type.includes('n') || type === 'resize-nw' || type === 'resize-ne') {
      newY2 = Math.max(newY2, minSize)
    }
  }
  if (newY2 > naturalHeight) {
    newY2 = naturalHeight
    if (type.includes('s') || type === 'resize-sw' || type === 'resize-se') {
      newY1 = Math.min(newY1, naturalHeight - minSize)
    }
  }

  if (newX2 - newX1 < minSize) {
    if (type.includes('w') || type === 'resize-nw' || type === 'resize-sw') {
      newX1 = newX2 - minSize
    } else {
      newX2 = newX1 + minSize
    }
  }
  if (newY2 - newY1 < minSize) {
    if (type.includes('n') || type === 'resize-nw' || type === 'resize-ne') {
      newY1 = newY2 - minSize
    } else {
      newY2 = newY1 + minSize
    }
  }

  editingFaceBbox.value = [newX1, newY1, newX2, newY2]
}

const endDragBbox = () => {
  isDraggingBbox.value = false
  dragType.value = 'move'
}

// 保存
const handleSave = () => {
  emit('save', [...editingFaceBbox.value])
}

// 取消
const handleCancel = () => {
  emit('cancel')
}

// 监听拖拽事件
onMounted(() => {
  document.addEventListener('mousemove', dragBbox)
  document.addEventListener('mouseup', endDragBbox)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', dragBbox)
  document.removeEventListener('mouseup', endDragBbox)
})
</script>

<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8 animate-in fade-in duration-300">
    <!-- Backdrop -->
    <div class="absolute inset-0 bg-black/80 backdrop-blur-sm" @click="handleCancel" />

    <!-- Modal Container -->
    <div class="relative w-full max-w-5xl max-h-[90vh] flex flex-col bg-[#0d0d0d] rounded-[32px] overflow-hidden shadow-[0_0_100px_rgba(0,0,0,0.5)] border border-white/10">

      <!-- Header -->
      <div class="flex items-center justify-between px-8 py-5 bg-[#141414] border-b border-white/5">
        <div class="flex items-center gap-4">
          <div class="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center border border-white/10">
            <i class="fas fa-expand text-white/40 text-lg"></i>
          </div>
          <div>
            <h2 class="text-lg font-bold text-white tracking-tight">{{ isAddingNew ? (t('addNewRole') || '新增角色') : (t('adjustFaceBox') || '调整人脸边界框') }}</h2>
            <p class="text-[10px] text-white/20 uppercase tracking-widest font-bold">Crop character region</p>
          </div>
        </div>
        <button
          @click="handleCancel"
          class="p-2 hover:bg-white/5 rounded-full transition-all group active:scale-95"
        >
          <i class="fas fa-times text-white/20 group-hover:text-white transition-colors text-lg"></i>
        </button>
      </div>

      <!-- Main Cropping Area -->
      <div class="flex-1 relative flex items-center justify-center p-8 bg-black/40 overflow-hidden min-h-[400px]">
        <div class="relative inline-block max-w-full max-h-full transition-all" ref="imageContainerRef">
          <img
            :src="imageUrl"
            alt="Source"
            class="max-w-full max-h-[60vh] object-contain select-none pointer-events-none rounded-lg border border-white/5"
            :class="{ 'opacity-0': !imageLoaded }"
            @load="handleImageLoad"
            @error="handleImageError"
          />

          <!-- Loading Placeholder -->
          <div
            v-show="!imageLoaded"
            class="absolute inset-0 w-full h-full min-w-[400px] min-h-[300px] flex items-center justify-center z-10">
            <div class="flex flex-col items-center gap-3">
              <i class="fas fa-spinner fa-spin text-2xl text-white/40"></i>
              <span class="text-sm text-white/40">{{ t('loading') || '加载中...' }}</span>
            </div>
          </div>

          <!-- Dark Overlay (Scrim) -->
          <div
            v-if="imageLoaded && getBboxStyle.left"
            class="absolute inset-0 bg-black/60 pointer-events-none transition-opacity"
            :style="{ clipPath: clipPath }"
          />

          <!-- Guidelines for Existing Characters -->
          <div
            v-for="(char, index) in existingCharacters"
            :key="index"
            class="absolute border border-white/50 border-dashed pointer-events-none z-10"
            :style="getExistingCharacterStyle(char)"
          >
            <div class="absolute -top-7 left-0 bg-white/10 backdrop-blur-md border border-white/20 px-2 py-0.5 rounded text-[10px] font-bold text-white/90 uppercase tracking-wider whitespace-nowrap shadow-xl">
              {{ char.roleName || char.name || `角色${index + 1}` }}
            </div>
          </div>

          <!-- Draggable Selection Box -->
          <div
            v-if="imageLoaded && getBboxStyle.left"
            :style="{
              left: getBboxStyle.left,
              top: getBboxStyle.top,
              width: getBboxStyle.width,
              height: getBboxStyle.height,
              boxShadow: '0 0 40px rgba(0,0,0,0.8)'
            }"
            @mousedown="(e) => startDragBbox(e, 'move')"
            class="absolute border border-[#4FD1C5] cursor-move transition-colors duration-200 group z-20"
          >
            <!-- Active Label -->
            <div class="absolute -top-10 left-0 flex items-center gap-2 bg-[#1a1a1a] border border-white/10 text-[#4FD1C5] text-[11px] font-bold px-3 py-1.5 rounded-lg shadow-2xl whitespace-nowrap uppercase tracking-widest select-none pointer-events-none">
              <i class="fas fa-arrows-alt text-xs"></i>
              {{ characterLabel }}
            </div>

            <!-- Dimension Specs -->
            <div class="absolute -bottom-8 right-0 text-[10px] font-medium text-white/30 tracking-tight">
              {{ Math.round(parseFloat(getBboxStyle.width || 0)) }} × {{ Math.round(parseFloat(getBboxStyle.height || 0)) }}
            </div>

            <!-- Edge Handles -->
            <div
              class="absolute -top-1.5 left-4 right-4 h-3 cursor-ns-resize z-20 hover:bg-[#4FD1C5]/10 flex items-center justify-center transition-colors group/edge"
              @mousedown.stop="(e) => startDragBbox(e, 'resize-n')"
            >
              <div class="w-6 h-1 bg-[#4FD1C5]/40 rounded-full group-hover/edge:bg-[#4FD1C5]/80 transition-all" />
            </div>
            <div
              class="absolute -bottom-1.5 left-4 right-4 h-3 cursor-ns-resize z-20 hover:bg-[#4FD1C5]/10 flex items-center justify-center transition-colors group/edge"
              @mousedown.stop="(e) => startDragBbox(e, 'resize-s')"
            >
              <div class="w-6 h-1 bg-[#4FD1C5]/40 rounded-full group-hover/edge:bg-[#4FD1C5]/80 transition-all" />
            </div>
            <div
              class="absolute top-4 bottom-4 -left-1.5 w-3 cursor-ew-resize z-20 hover:bg-[#4FD1C5]/10 flex items-center justify-center transition-colors group/edge"
              @mousedown.stop="(e) => startDragBbox(e, 'resize-w')"
            >
              <div class="h-6 w-1 bg-[#4FD1C5]/40 rounded-full group-hover/edge:bg-[#4FD1C5]/80 transition-all" />
            </div>
            <div
              class="absolute top-4 bottom-4 -right-1.5 w-3 cursor-ew-resize z-20 hover:bg-[#4FD1C5]/10 flex items-center justify-center transition-colors group/edge"
              @mousedown.stop="(e) => startDragBbox(e, 'resize-e')"
            >
              <div class="h-6 w-1 bg-[#4FD1C5]/40 rounded-full group-hover/edge:bg-[#4FD1C5]/80 transition-all" />
            </div>

            <!-- Corner Knobs -->
            <div
              @mousedown.stop="(e) => startDragBbox(e, 'resize-nw')"
              class="absolute -top-2 -left-2 w-4 h-4 rounded-full flex items-center justify-center cursor-nwse-resize z-30 transition-all group/corner hover:bg-[#4FD1C5]/10"
            >
              <div class="w-2.5 h-2.5 bg-white rounded-full border border-black/20 shadow-lg group-hover/corner:bg-[#4FD1C5] group-hover/corner:scale-110 transition-all" />
            </div>
            <div
              @mousedown.stop="(e) => startDragBbox(e, 'resize-ne')"
              class="absolute -top-2 -right-2 w-4 h-4 rounded-full flex items-center justify-center cursor-nesw-resize z-30 transition-all group/corner hover:bg-[#4FD1C5]/10"
            >
              <div class="w-2.5 h-2.5 bg-white rounded-full border border-black/20 shadow-lg group-hover/corner:bg-[#4FD1C5] group-hover/corner:scale-110 transition-all" />
            </div>
            <div
              @mousedown.stop="(e) => startDragBbox(e, 'resize-sw')"
              class="absolute -bottom-2 -left-2 w-4 h-4 rounded-full flex items-center justify-center cursor-nesw-resize z-30 transition-all group/corner hover:bg-[#4FD1C5]/10"
            >
              <div class="w-2.5 h-2.5 bg-white rounded-full border border-black/20 shadow-lg group-hover/corner:bg-[#4FD1C5] group-hover/corner:scale-110 transition-all" />
            </div>
            <div
              @mousedown.stop="(e) => startDragBbox(e, 'resize-se')"
              class="absolute -bottom-2 -right-2 w-4 h-4 rounded-full flex items-center justify-center cursor-nwse-resize z-30 transition-all group/corner hover:bg-[#4FD1C5]/10"
            >
              <div class="w-2.5 h-2.5 bg-white rounded-full border border-black/20 shadow-lg group-hover/corner:bg-[#4FD1C5] group-hover/corner:scale-110 transition-all" />
            </div>

            <!-- Grid Lines -->
            <div class="absolute inset-0 pointer-events-none opacity-5">
              <div class="absolute inset-0 grid grid-cols-3 grid-rows-3">
                <div v-for="i in 8" :key="i"
                  :class="{
                    'border-b border-white': i < 6,
                    'border-r border-white': [0,1,3,4,6,7].includes(i-1)
                  }"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="flex items-center justify-end px-10 py-6 bg-[#141414] border-t border-white/5 gap-4">
        <button
          @click="handleCancel"
          class="px-6 py-2.5 rounded-xl text-white/40 font-semibold hover:text-white transition-colors"
        >
          {{ t('cancel') || '取消' }}
        </button>
        <button
          @click="handleSave"
          class="px-10 py-2.5 rounded-xl bg-[#4FD1C5] text-black font-bold hover:brightness-110 transition-all active:scale-95 shadow-lg shadow-[#4FD1C5]/10"
        >
          {{ t('save') || '保存' }} Changes
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.animate-in {
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
