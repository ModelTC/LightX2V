<script setup>
import { alert, getAlertClass, getAlertIconBgClass, getAlertIcon } from '../utils/other'
import { useI18n } from 'vue-i18n'
import { ref, onMounted, onUnmounted } from 'vue'

const { t, locale } = useI18n()

// 响应式变量控制Alert位置
const alertPosition = ref({ top: '1rem' })

// 防抖函数
let scrollTimeout = null

// 监听滚动事件，动态调整Alert位置
const handleScroll = () => {
    // 清除之前的定时器
    if (scrollTimeout) {
        clearTimeout(scrollTimeout)
    }
    
    // 设置新的定时器，防抖处理
    scrollTimeout = setTimeout(() => {
        const scrollY = window.scrollY
        const viewportHeight = window.innerHeight
        
        // 如果用户滚动了超过50px，将Alert显示在视口内
        if (scrollY > 50) {
            // 计算Alert应该显示的位置，确保在视口内可见
            // 距离滚动位置20px，但不超过视口底部200px
            const alertTop = Math.min(scrollY + 20, scrollY + viewportHeight - 200)
            alertPosition.value = { top: `${alertTop}px` }
        } else {
            // 在页面顶部时，显示在固定位置
            alertPosition.value = { top: '1rem' }
        }
    }, 10) // 10ms防抖延迟
}

onMounted(() => {
    window.addEventListener('scroll', handleScroll, { passive: true })
    // 初始化时也调用一次，确保位置正确
    handleScroll()
})

onUnmounted(() => {
    window.removeEventListener('scroll', handleScroll)
    if (scrollTimeout) {
        clearTimeout(scrollTimeout)
    }
})
</script>
<template>
            <!-- 增强的提示消息系统 -->
            <div v-cloak>
                <div v-if="alert.show"
                    class="fixed left-1/2 transform -translate-x-1/2 z-[9999] max-w-xs w-full px-4 transition-all duration-300 ease-out"
                    :style="alertPosition"
                    :class="getAlertClass(alert.type)">
                        <div
                            class="bg-gray-800/95 backdrop-blur-sm border border-gray-700/50 rounded-lg shadow-lg transition-all duration-300 ease-out">
                        <div class="flex items-center p-3">
                            <div class="flex-shrink-0 mr-3">
                                <div class="w-6 h-6 rounded-full flex items-center justify-center"
                                    :class="getAlertIconBgClass(alert.type)">
                                    <i :class="getAlertIcon(alert.type)" class="text-xs"></i>
                                </div>
                            </div>
                            <div class="flex-1">
                                <p class="text-xs font-medium text-gray-200">
                                    {{ alert.message }}
                                </p>
                            </div>
                            <div class="flex-shrink-0 ml-2">
                                <button @click="alert.show = false"
                                        class="w-5 h-5 rounded-full flex items-center justify-center text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 transition-all duration-200">
                                    <i class="fas fa-times text-xs"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
</template>