<script setup>
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { watch, onMounted } from 'vue'

// Props
const props = defineProps({
  query: {
    type: Object,
    default: () => ({})
  },
  templateId: {
    type: String,
    default: null
  }
})

const { t, locale } = useI18n()
const route = useRoute()
const router = useRouter()
import {
            goToInspirationPage,
            getVisibleInspirationPages,
            getTemplateFileUrl,
            handleThumbnailError,
            inspirationSearchQuery,
            selectedInspirationCategory,
            inspirationItems,
            InspirationCategories,
            selectInspirationCategory,
            handleInspirationSearch,
            inspirationPaginationInfo,
            inspirationCurrentPage,
            previewTemplateDetail,
            useTemplate,
            applyTemplateImage,
            applyTemplateAudio,
            playVideo,
            pauseVideo,
            toggleVideoPlay,
            onVideoLoaded,
            onVideoError,
            onVideoEnded,
            openTemplateFromRoute,
            copyShareLink
        } from '../utils/other'

// 监听模板详情路由
watch(() => route.params.templateId, (newTemplateId) => {
    if (newTemplateId && route.name === 'TemplateDetail') {
        openTemplateFromRoute(newTemplateId)
    }
}, { immediate: true })

// 路由监听和URL同步
watch(() => route.query, (newQuery) => {
    // 同步URL参数到组件状态
    if (newQuery.search) {
        inspirationSearchQuery.value = newQuery.search
    }
    if (newQuery.category) {
        selectedInspirationCategory.value = newQuery.category
    }
    if (newQuery.page) {
        const page = parseInt(newQuery.page)
        if (page > 0 && page !== inspirationCurrentPage.value) {
            goToInspirationPage(page)
        }
    }
}, { immediate: true })

// 监听组件状态变化，同步到URL
watch([inspirationSearchQuery, selectedInspirationCategory, inspirationCurrentPage], () => {
    const query = {}
    if (inspirationSearchQuery.value) {
        query.search = inspirationSearchQuery.value
    }
    if (selectedInspirationCategory.value && selectedInspirationCategory.value !== 'all') {
        query.category = selectedInspirationCategory.value
    }
    if (inspirationCurrentPage.value > 1) {
        query.page = inspirationCurrentPage.value.toString()
    }
    
    // 更新URL但不触发路由监听
    router.replace({ query })
})

// 组件挂载时初始化
onMounted(() => {
    // 确保URL参数正确同步
    const query = route.query
    if (query.search) {
        inspirationSearchQuery.value = query.search
    }
    if (query.category) {
        selectedInspirationCategory.value = query.category
    }
    if (query.page) {
        const page = parseInt(query.page)
        if (page > 0) {
            goToInspirationPage(page)
        }
    }
})

</script>
<template>
                        <!-- 灵感广场区域 -->
                        <div class="flex-1 flex flex-col min-h-0 mobile-content">
                            <!-- 内容区域 -->
                            <div class="flex-1 overflow-y-auto p-6 content-area main-scrollbar">
                                <!-- 灵感广场功能区 -->
                                <div class="max-w-4xl mx-auto" id="inspiration-gallery">
                                    <!-- 固定功能区 -->
                                    <div class="flex-shrink-0 p-1">
                                        <!-- 标题区域 -->
                                        <div class="text-center mb-8">
                                                <h1 class="text-3xl font-bold text-white mb-2">{{ t('inspirationGallery') }}</h1>
                                                <p class="text-gray-400">{{ t('discoverCreativity') }}</p>
                                        </div>

                                        <!-- 搜索和筛选区域 -->
                                        <div class="flex flex-col md:flex-row gap-4 mb-6">
                                            <!-- 搜索框 -->
                                            <div class="relative flex-1">
                                                    <i
                                                        class="fas fa-search absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none z-10"></i>
                                                    <input v-model="inspirationSearchQuery"
                                                    @keyup.enter="handleInspirationSearch"
                                                    @input="handleInspirationSearch"
                                                    class="w-full bg-dark-light border border-laser-purple/30 rounded-lg py-3 pl-10 pr-4 text-sm focus:outline-none focus:ring-2 focus:ring-laser-purple/50 transition-all focus:border-laser focus:shadow-laser"
                                                        :placeholder="t('searchInspiration')" type="text" />
                                            </div>

                                            <!-- 分类筛选 -->
                                                <div class="flex gap-2 flex-wrap">
                                                    <button v-for="category in InspirationCategories" :key="category"
                                                    @click="selectInspirationCategory(category)"
                                                    :class="selectedInspirationCategory === category
                                                        ? 'bg-laser-purple/20 text-white border-laser-purple/40'
                                                        : 'bg-dark-light text-gray-400 hover:text-white hover:bg-dark-light/80'"
                                                        class="px-4 py-2 rounded-lg border border-transparent transition-all duration-200 text-sm font-medium">
                                                    {{ category }}
                                                </button>
                                            </div>
                                        </div>
                                    <!-- 灵感广场分页组件 -->
                                        <div v-if="inspirationPaginationInfo">
                                                <div class="flex items-center justify-between text-xs text-gray-400">
                                                    <div class="flex items-center space-x-1 text-gray-500">
                                                        <span>{{ inspirationPaginationInfo.total }} {{ t('records') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="inspirationPaginationInfo.total_pages > 1" class="flex justify-center">
                                                    <nav class="isolate inline-flex -space-x-px rounded-md" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToInspirationPage(inspirationCurrentPage - 1)"
                                                            :disabled="inspirationCurrentPage <= 1"
                                                            class="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': inspirationCurrentPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                                            <i class="fas fa-chevron-left text-sm" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisibleInspirationPages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToInspirationPage(page)"
                                                                :class="[
                                                                    'relative inline-flex items-center px-4 py-2 text-sm font-semibold focus:z-20 focus:outline-offset-0',
                                                                    page === inspirationCurrentPage 
                                                                        ? 'z-10 text-white focus-visible:outline-2 focus-visible:outline-offset-2 bg-laser-purple focus-visible:outline-laser-purple'
                                                                        : 'text-gray-200 inset-ring inset-ring-gray-700 hover:bg-white/5'
                                                                ]"
                                                                :aria-current="page === inspirationCurrentPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                                            <span v-else class="relative inline-flex items-center px-4 py-2 text-sm font-semibold text-gray-400 inset-ring inset-ring-gray-700 focus:outline-offset-0">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToInspirationPage(inspirationCurrentPage + 1)"
                                                            :disabled="inspirationCurrentPage >= inspirationPaginationInfo.total_pages"
                                                            class="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 inset-ring inset-ring-gray-700 hover:bg-white/5 focus:z-20 focus:outline-offset-0"
                                                            :class="{ 'opacity-50 cursor-not-allowed': inspirationCurrentPage >= inspirationPaginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                                            <i class="fas fa-chevron-right text-sm" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                                </div>
                                        </div>
                                        <!-- 灵感内容网格 -->
                                        <div class="flex-1 overflow-y-auto main-scrollbar min-h-0 p-4"
                                        style="overflow-x: visible;">
                                        <div class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-3">
                                            <!-- 灵感卡片 -->
                                            <div v-for="item in inspirationItems" :key="item.task_id"
                                                    class="break-inside-avoid mb-3 group relative bg-dark-light rounded-xl overflow-hidden border border-gray-700/50 hover:border-laser-purple/40 transition-all duration-300 hover:shadow-laser/20">
                                                <!-- 视频缩略图区域 -->
                                                <div class="cursor-pointer bg-gray-800 relative flex flex-col"
                                                @click="previewTemplateDetail(item)"
                                                :title="t('viewTemplateDetail')">
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
                                                    <!-- 悬浮操作按钮（下方居中，仅桌面端） -->
                                                    <div
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
                        </div>
                        </div>
                </div>
        </div>
</template>