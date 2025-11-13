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
            copyShareLink,
            isPageLoading
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
    <!-- 灵感广场区域 - Apple 极简风格 -->
                        <div class="flex-1 flex flex-col min-h-0 mobile-content">
                            <!-- 内容区域 -->
                            <div class="flex-1 overflow-y-auto p-6 content-area main-scrollbar">
                                <!-- 灵感广场功能区 -->
            <div class="max-w-7xl mx-auto" id="inspiration-gallery">
                <!-- 标题区域 - Apple 风格 -->
                <div class="text-center mb-10">
                    <h1 class="text-4xl sm:text-5xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">{{ t('inspirationGallery') }}</h1>
                    <p class="text-base text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('discoverCreativity') }}</p>
                                        </div>

                <!-- 搜索和筛选区域 - Apple 风格 -->
                <div class="flex flex-col md:flex-row gap-4 mb-8">
                    <!-- 搜索框 - Apple 风格 -->
                                            <div class="relative flex-1">
                        <i class="fas fa-search absolute left-4 top-1/2 -translate-y-1/2 text-[#86868b] dark:text-[#98989d] pointer-events-none z-10"></i>
                                                    <input v-model="inspirationSearchQuery"
                                                    @keyup.enter="handleInspirationSearch"
                                                    @input="handleInspirationSearch"
                            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl py-3 pl-11 pr-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200"
                            :placeholder="t('searchInspiration')"
                            type="text" />
                                            </div>

                    <!-- 分类筛选 - Apple 风格 -->
                                                <div class="flex gap-2 flex-wrap">
                        <!-- "全部"按钮 -->
                        <button @click="selectInspirationCategory('')"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                            :class="selectedInspirationCategory === '' || !selectedInspirationCategory
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                            {{ t('all') }}
                        </button>
                        <!-- 其他分类按钮 -->
                                                    <button v-for="category in InspirationCategories" :key="category"
                                                    @click="selectInspirationCategory(category)"
                            class="px-5 py-2.5 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                                                    :class="selectedInspirationCategory === category
                                ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'">
                                                    {{ category }}
                                                </button>
                                            </div>
                                        </div>

                <!-- 灵感广场分页组件 - Apple 风格 -->
                <div v-if="inspirationPaginationInfo" class="mb-6">
                    <div class="flex items-center justify-between text-xs mb-4">
                        <div class="flex items-center space-x-1 text-[#86868b] dark:text-[#98989d] tracking-tight">
                                                        <span>{{ inspirationPaginationInfo.total }} {{ t('records') }}</span>
                                                    </div>
                                                </div>
                                                <div v-if="inspirationPaginationInfo.total_pages > 1" class="flex justify-center">
                        <nav class="isolate inline-flex gap-1" aria-label="Pagination">
                                                        <!-- 上一页按钮 -->
                                                        <button @click="goToInspirationPage(inspirationCurrentPage - 1)"
                                                            :disabled="inspirationCurrentPage <= 1"
                                class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': inspirationCurrentPage <= 1 }"
                                                            :title="t('previousPage')">
                                                            <span class="sr-only">{{ t('previousPage') }}</span>
                                <i class="fas fa-chevron-left text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>

                                                        <!-- 页码按钮 -->
                                                        <template v-for="page in getVisibleInspirationPages()" :key="page">
                                                            <button v-if="page !== '...'" @click="goToInspirationPage(page)"
                                                                :class="[
                                        'relative inline-flex items-center justify-center min-w-[36px] h-9 px-3 text-sm font-medium rounded-lg transition-all duration-200',
                                                                    page === inspirationCurrentPage
                                            ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                                            : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'
                                                                ]"
                                                                :aria-current="page === inspirationCurrentPage ? 'page' : undefined">
                                                                {{ page }}
                                                            </button>
                                <span v-else class="relative inline-flex items-center px-2 text-sm font-semibold text-[#86868b] dark:text-[#98989d]">...</span>
                                                        </template>

                                                        <!-- 下一页按钮 -->
                                                        <button @click="goToInspirationPage(inspirationCurrentPage + 1)"
                                                            :disabled="inspirationCurrentPage >= inspirationPaginationInfo.total_pages"
                                class="relative inline-flex items-center w-9 h-9 rounded-lg bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-all duration-200"
                                                            :class="{ 'opacity-50 cursor-not-allowed': inspirationCurrentPage >= inspirationPaginationInfo.total_pages }"
                                                            :title="t('nextPage')">
                                                            <span class="sr-only">{{ t('nextPage') }}</span>
                                <i class="fas fa-chevron-right text-xs mx-auto" aria-hidden="true"></i>
                                                        </button>
                                                    </nav>
                                                </div>
                                        </div>

                <!-- 灵感内容网格 - Apple 风格 -->
                <div class="space-y-4">
                    <div v-if="isPageLoading" class="flex items-center justify-center">
                        <div class="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/90 dark:bg-[#2c2c2e]/90 border border-black/8 dark:border-white/8 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] shadow-[0_4px_16px_rgba(0,0,0,0.08)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.35)]">
                            <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                            <span>{{ t('loading') }}</span>
                        </div>
                    </div>
                    <div class="columns-2 md:columns-3 lg:columns-4 xl:columns-5 gap-4">
                    <!-- 灵感卡片 - Apple 风格 -->
                                            <div v-for="item in inspirationItems" :key="item.task_id"
                        class="break-inside-avoid mb-4 group relative bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] rounded-2xl overflow-hidden border border-black/8 dark:border-white/8 hover:border-[color:var(--brand-primary)]/30 dark:hover:border-[color:var(--brand-primary-light)]/30 hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.15)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.2)]">
                                                <!-- 视频缩略图区域 -->
                        <div class="cursor-pointer bg-black/2 dark:bg-white/2 relative flex flex-col"
                                                @click="previewTemplateDetail(item)"
                                                :title="t('viewTemplateDetail')">
                                                        <!-- 视频预览 -->
                                                        <video v-if="item?.outputs?.output_video"
                                                            :src="getTemplateFileUrl(item.outputs.output_video,'videos')"
                                                            :poster="getTemplateFileUrl(item.inputs.input_image,'images')"
                                class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                                            preload="auto" playsinline webkit-playsinline
                                                            @mouseenter="playVideo($event)" @mouseleave="pauseVideo($event)"
                                                            @loadeddata="onVideoLoaded($event)"
                                                            @ended="onVideoEnded($event)"
                                                            @error="onVideoError($event)"></video>
                                                    <!-- 图片缩略图 -->
                                                        <img v-else
                                                        :src="getTemplateFileUrl(item.inputs.input_image,'images')"
                                                        :alt="item.params?.prompt || '模板图片'"
                                class="w-full h-auto object-contain group-hover:scale-[1.02] transition-transform duration-200"
                                                        @error="handleThumbnailError" />

                            <!-- 移动端播放按钮 - Apple 风格 -->
                                                        <button v-if="item?.outputs?.output_video"
                                                            @click.stop="toggleVideoPlay($event)"
                                class="md:hidden absolute bottom-3 left-1/2 transform -translate-x-1/2 w-10 h-10 rounded-full bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.2)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-105 transition-all duration-200 z-20">
                                                            <i class="fas fa-play text-sm"></i>
                                                        </button>

                            <!-- 悬浮操作按钮（下方居中，仅桌面端）- Apple 风格 -->
                            <div class="hidden md:flex absolute bottom-3 left-1/2 transform -translate-x-1/2 items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10 w-full">
                                <div class="flex gap-2 pointer-events-auto">
                                                            <button @click.stop="applyTemplateImage(item)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                :title="t('applyImage')">
                                                                <i class="fas fa-image text-sm"></i>
                                                            </button>
                                                            <button @click.stop="applyTemplateAudio(item)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                :title="t('applyAudio')">
                                                                <i class="fas fa-music text-sm"></i>
                                                            </button>
                                                            <button @click.stop="useTemplate(item)"
                                        class="w-10 h-10 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)] flex items-center justify-center text-white hover:scale-110 active:scale-100 transition-all duration-200"
                                                                :title="t('useTemplate')">
                                                                <i class="fas fa-clone text-sm"></i>
                                                            </button>
                                                            <button @click.stop="copyShareLink(item.task_id, 'template')"
                                        class="w-10 h-10 rounded-full bg-white dark:bg-[#3a3a3c] backdrop-blur-[20px] shadow-[0_2px_8px_rgba(0,0,0,0.12)] dark:shadow-[0_2px_8px_rgba(0,0,0,0.4)] flex items-center justify-center text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-110 active:scale-100 transition-all duration-200"
                                                                :title="t('shareTemplate')">
                                                                <i class="fas fa-share-alt text-sm"></i>
                                                            </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    </div>
                </div>

                <!-- GitHub 仓库链接 - Apple 极简风格 -->
                <div class="fixed bottom-6 right-6 z-50">
                    <a href="https://github.com/ModelTC/LightX2V"
                       target="_blank"
                       rel="noopener noreferrer"
                       class="flex items-center gap-2.5 px-4 py-2.5 bg-white/85 dark:bg-[#1e1e1e]/85 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-full shadow-[0_4px_16px_rgba(0,0,0,0.1)] dark:shadow-[0_4px_16px_rgba(0,0,0,0.3)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.15)] dark:hover:shadow-[0_8px_24px_rgba(0,0,0,0.4)] hover:scale-105 active:scale-100 transition-all duration-200 group"
                       title="Star us on GitHub">
                        <i class="fab fa-github text-lg text-[#1d1d1f] dark:text-[#f5f5f7] transition-transform duration-200 group-hover:rotate-12"></i>
                        <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">LightX2V</span>
                        <i class="fas fa-external-link-alt text-xs text-[#86868b] dark:text-[#98989d] transition-all duration-200 group-hover:translate-x-0.5 group-hover:-translate-y-0.5"></i>
                    </a>
                        </div>
                        </div>
                </div>
        </div>
</template>
