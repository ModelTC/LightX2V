<script setup>
import { alert, getAlertClass, getAlertIconBgClass, getAlertIcon } from '../utils/other'
import { useI18n } from 'vue-i18n'
import { ref, onMounted, onUnmounted } from 'vue'

const { t, locale } = useI18n()

// 处理操作按钮点击
const handleActionClick = () => {
    if (alert.value.action && alert.value.action.onClick) {
        alert.value.action.onClick()
        alert.value.show = false
    }
}

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
            <!-- Apple 风格极简提示消息 -->
            <div v-cloak>
                <transition
                    enter-active-class="alert-enter-active"
                    leave-active-class="alert-leave-active"
                    enter-from-class="alert-enter-from"
                    leave-to-class="alert-leave-to">
                    <div v-if="alert.show"
                        class="fixed left-1/2 transform -translate-x-1/2 z-[9999] w-auto min-w-[280px] sm:min-w-[320px] max-w-[calc(100vw-3rem)] sm:max-w-xl px-6 sm:px-6 transition-all duration-500 ease-out"
                        :style="alertPosition">
                        <div class="alert-container">
                            <div class="alert-content">
                                <!-- 图标 -->
                                <div class="alert-icon-wrapper">
                                    <i :class="getAlertIcon(alert.type)" class="alert-icon"></i>
                                </div>
                                <!-- 消息文本和操作按钮（一行显示） -->
                                <div class="alert-message">
                                    <span>{{ alert.message }}</span>
                                    <!-- 操作链接 - Apple 风格 -->
                                    <button v-if="alert.action" @click="handleActionClick" class="alert-action-link">
                                        {{ alert.action.label }}
                                    </button>
                                </div>
                                <!-- 关闭按钮 -->
                                <button @click="alert.show = false" class="alert-close-btn" aria-label="Close">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </transition>
            </div>
</template>

<style scoped>
/* Apple 风格 Alert 容器 */
.alert-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-radius: 16px;
    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.1),
        0 2px 4px -1px rgba(0, 0, 0, 0.06),
        0 0 0 1px rgba(0, 0, 0, 0.05);
    overflow: hidden;
}

/* 深色模式下的容器样式 */
:global(.dark) .alert-container {
    background: rgba(30, 30, 30, 0.95);
    box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.3),
        0 2px 4px -1px rgba(0, 0, 0, 0.2),
        0 0 0 1px rgba(255, 255, 255, 0.08);
}

/* Alert 内容 */
.alert-content {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
}

/* 图标包装器 */
.alert-icon-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* 图标样式 */
.alert-icon {
    font-size: 18px;
    color: #1d1d1f;
}

:global(.dark) .alert-icon {
    color: #f5f5f7;
}

/* 消息文本 */
.alert-message {
    flex: 1;
    font-size: 14px;
    font-weight: 500;
    line-height: 1.5;
    color: #1d1d1f;
    letter-spacing: -0.01em;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 15px;
}

:global(.dark) .alert-message {
    color: #f5f5f7;
}

/* 关闭按钮 */
.alert-close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: none;
    background: transparent;
    color: #86868b;
    cursor: pointer;
    flex-shrink: 0;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.alert-close-btn:hover {
    background: rgba(0, 0, 0, 0.05);
    color: #1d1d1f;
    transform: scale(1.05);
}

.alert-close-btn:active {
    transform: scale(0.95);
}

:global(.dark) .alert-close-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #f5f5f7;
}

.alert-close-btn i {
    font-size: 12px;
}

/* 操作链接 - Apple 风格下划线文本 */
.alert-action-link {
    display: inline;
    padding: 0;
    border: none;
    background: transparent;
    color: var(--brand-primary);
    font-size: inherit;
    font-weight: 600;
    text-decoration: underline;
    text-underline-offset: 2px;
    text-decoration-thickness: 1px;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
}

.alert-action-link:hover {
    color: var(--brand-primary);
    opacity: 0.8;
    text-decoration-thickness: 2px;
}

.alert-action-link:active {
    opacity: 0.6;
}

:global(.dark) .alert-action-link {
    color: var(--brand-primary-light);
}

:global(.dark) .alert-action-link:hover {
    color: var(--brand-primary-light);
}

/* 进入动画 */
.alert-enter-active {
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.alert-leave-active {
    transition: all 0.3s cubic-bezier(0.4, 0, 1, 1);
}

.alert-enter-from {
    opacity: 0;
    transform: translate(-50%, -20px) scale(0.95);
}

.alert-leave-to {
    opacity: 0;
    transform: translate(-50%, -10px) scale(0.98);
}

/* 响应式设计 */
@media (max-width: 640px) {
    .alert-content {
        padding: 12px 16px;
        gap: 8px;
    }

    .alert-message {
        font-size: 13px;
    }

    .alert-icon {
        font-size: 16px;
    }

    .alert-action-link {
        font-size: 13px;
    }
}
</style>
