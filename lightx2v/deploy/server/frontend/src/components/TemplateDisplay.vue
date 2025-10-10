<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Props
const props = defineProps({
  templates: {
    type: Array,
    default: () => []
  },
  showActions: {
    type: Boolean,
    default: true
  },
  layout: {
    type: String,
    default: 'grid', // 'grid' 或 'waterfall'
    validator: (value) => ['grid', 'waterfall'].includes(value)
  },
  columns: {
    type: Number,
    default: 2
  },
  maxTemplates: {
    type: Number,
    default: 10
  }
})

// 从 utils/other 导入需要的函数
import {
  getTemplateFileUrl,
  handleThumbnailError,
  playVideo,
  pauseVideo,
  toggleVideoPlay,
  onVideoLoaded,
  onVideoError,
  onVideoEnded,
  previewTemplateDetail,
  useTemplate,
  applyTemplateImage,
  applyTemplateAudio,
  copyShareLink
} from '../utils/other'

// 屏幕尺寸响应式状态
const screenSize = ref('large')

// 更新屏幕尺寸
const updateScreenSize = () => {
  screenSize.value = window.innerWidth >= 1024 ? 'large' : 'small'
}

// 随机列布局相关函数（用于网格布局）
const generateRandomColumnLayout = (templates) => {
  if (!templates || templates.length === 0) return { columns: [], templates: [] }
  
  const numColumns = props.columns
  
  // 生成随机列宽（总和为100%）
  const columnWidths = []
  let remainingWidth = 100
  
  for (let i = 0; i < numColumns; i++) {
    if (i === numColumns - 1) {
      columnWidths.push(remainingWidth)
    } else {
      const minWidth = 20
      const maxWidth = Math.min(50, remainingWidth - (numColumns - i - 1) * minWidth)
      const width = Math.random() * (maxWidth - minWidth) + minWidth
      columnWidths.push(Math.round(width))
      remainingWidth -= Math.round(width)
    }
  }
  
  // 生成每列的起始位置
  const columnStartPositions = []
  for (let i = 0; i < numColumns; i++) {
    const startPosition = Math.random() * 20
    columnStartPositions.push(Math.round(startPosition))
  }
  
  // 计算每列的起始left位置
  const columnLeftPositions = []
  let currentLeft = 0
  for (let i = 0; i < numColumns; i++) {
    columnLeftPositions.push(currentLeft)
    currentLeft += columnWidths[i]
  }
  
  // 将模版分配到各列
  const columnTemplates = Array.from({ length: numColumns }, () => [])
  templates.forEach((template, index) => {
    const columnIndex = index % numColumns
    columnTemplates[columnIndex].push(template)
  })
  
  // 生成列配置
  const columns = columnWidths.map((width, index) => ({
    width: `${width}%`,
    left: `${columnLeftPositions[index]}%`,
    top: `${columnStartPositions[index]}%`,
    templates: columnTemplates[index]
  }))
  
  return { columns, templates }
}

// 计算属性：带随机列布局的模版
const templatesWithRandomColumns = computed(() => {
  if (props.layout === 'waterfall') {
    return { columns: [], templates: props.templates }
  }
  return generateRandomColumnLayout(props.templates)
})

// 组件挂载时初始化
onMounted(() => {
  updateScreenSize()
  window.addEventListener('resize', updateScreenSize)
})
</script>

<template>
  <div class="template-display">
    <!-- 瀑布流布局 -->
    <div v-if="layout === 'waterfall'" class="waterfall-layout">
      <div class="columns-2 md:columns-2 lg:columns-3 xl:columns-3 gap-4">
        <div v-for="item in templates" :key="item.task_id"
             class="break-inside-avoid mb-4 group relative bg-dark-light/50 rounded-xl overflow-hidden border border-gray-700/30 hover:border-laser-purple/50 transition-all duration-300 hover:shadow-laser/30 hover:bg-dark-light/70">
          <!-- 视频缩略图区域 -->
          <div class="cursor-pointer bg-gray-800 relative flex flex-col"
               @click="showActions ? previewTemplateDetail(item) : null"
               :title="showActions ? t('viewTemplateDetail') : ''">
            <!-- 视频预览 -->
            <video v-if="item?.outputs?.output_video"
                   :src="getTemplateFileUrl(item.outputs.output_video,'videos')"
                   :poster="getTemplateFileUrl(item.inputs.input_image,'images')"
                   class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                   preload="auto" playsinline webkit-playsinline
                   @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                   @loadeddata="onVideoLoaded($event)"
                   @ended="onVideoEnded($event)"
                   @error="onVideoError($event)"></video>
            <!-- 图片缩略图 -->
            <img v-else
                 :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                 :alt="item.params?.prompt || '模板图片'"
                 class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                 @error="handleThumbnailError" />
            <!-- 移动端播放按钮 -->
            <button v-if="item?.outputs?.output_video" 
                    @click.stop="toggleVideoPlay($event)"
                    class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/70 transition-colors z-20">
              <i class="fas fa-play text-sm"></i>
            </button>
            <!-- 悬浮操作按钮（仅当 showActions 为 true 时显示） -->
            <div v-if="showActions"
                 class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
              <div class="flex space-x-2 pointer-events-auto">
                <button @click.stop="applyTemplateImage(item)"
                        class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                        :title="t('applyImage')">
                  <i class="fas fa-image text-sm"></i>
                </button>
                <button @click.stop="applyTemplateAudio(item)"
                        class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                        :title="t('applyAudio')">
                  <i class="fas fa-music text-sm"></i>
                </button>
                <button @click.stop="useTemplate(item)"
                        class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                        :title="t('useTemplate')">
                  <i class="fas fa-clone text-sm"></i>
                </button>
                <button @click.stop="copyShareLink(item.task_id, 'template')"
                        class="w-10 h-10 rounded-full bg-blue-500 backdrop-blur-sm flex items-center justify-center text-white hover:bg-blue-600 transition-colors"
                        :title="t('shareTemplate')">
                  <i class="fas fa-share-alt text-sm"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 网格布局 -->
    <div v-else class="grid-layout">
      <div class="relative min-h-[400px] lg:min-h-[600px]">
        <!-- 随机列 -->
        <div v-for="(column, columnIndex) in templatesWithRandomColumns.columns" :key="columnIndex"
             class="absolute transition-all duration-500 animate-fade-in"
             :style="{
               width: column.width,
               left: column.left,
               top: column.top,
               animationDelay: `${columnIndex * 0.2}s`
             }">
          <!-- 列内的模版卡片 -->
          <div v-for="item in column.templates" :key="item.task_id"
               class="mb-3 group relative bg-dark-light rounded-xl overflow-hidden border border-gray-700/50 hover:border-laser-purple/40 transition-all duration-300 hover:shadow-laser/20">
            <!-- 视频缩略图区域 -->
            <div class="cursor-pointer bg-gray-800 relative flex flex-col"
                 @click="showActions ? previewTemplateDetail(item) : null"
                 :title="showActions ? t('viewTemplateDetail') : ''">
              <!-- 视频预览 -->
              <video v-if="item?.outputs?.output_video"
                     :src="getTemplateFileUrl(item.outputs.output_video,'videos')"
                     :poster="getTemplateFileUrl(item.inputs.input_image,'images')"
                     class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                     preload="auto" playsinline webkit-playsinline
                     @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                     @loadeddata="onVideoLoaded($event)"
                     @ended="onVideoEnded($event)"
                     @error="onVideoError($event)"></video>
              <!-- 图片缩略图 -->
              <img v-else
                   :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                   :alt="item.params?.prompt || '模板图片'"
                   class="w-full h-auto object-contain group-hover:scale-105 transition-transform duration-300"
                   @error="handleThumbnailError" />
              <!-- 移动端播放按钮 -->
              <button v-if="item?.outputs?.output_video" 
                      @click.stop="toggleVideoPlay($event)"
                      class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center text-white hover:bg-black/70 transition-colors z-20">
                <i class="fas fa-play text-sm"></i>
              </button>
              <!-- 悬浮操作按钮（仅当 showActions 为 true 时显示） -->
              <div v-if="showActions"
                   class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
                <div class="flex space-x-3 pointer-events-auto">
                  <button @click.stop="applyTemplateImage(item)"
                          class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                          :title="t('applyImage')">
                    <i class="fas fa-image text-sm"></i>
                  </button>
                  <button @click.stop="applyTemplateAudio(item)"
                          class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                          :title="t('applyAudio')">
                    <i class="fas fa-music text-sm"></i>
                  </button>
                  <button @click.stop="useTemplate(item)"
                          class="w-10 h-10 rounded-full bg-laser-purple backdrop-blur-sm flex items-center justify-center text-white hover:bg-laser-purple transition-colors"
                          :title="t('useTemplate')">
                    <i class="fas fa-clone text-sm"></i>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.template-display {
  width: 100%;
}

.waterfall-layout {
  width: 100%;
}

.grid-layout {
  width: 100%;
}

/* 动画效果 */
@keyframes fade-in {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-fade-in {
  animation: fade-in 0.6s ease-out forwards;
}
</style>
