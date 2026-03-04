<script setup>
import { ref, computed, onMounted, onUnmounted, onBeforeUnmount, nextTick, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
    handleAudioUpload,
    showAlert,
    apiCall,
    loadPodcastAudioFromCache,
    getPodcastAudioFromCache,
    setPodcastAudioToCache,
    getPodcastAudioUrlFromApi,
    switchToCreateView,
    getCurrentForm,
    selectedTaskId,
    isCreationAreaExpanded,
    setCurrentAudioPreview,
    isLoading,
} from '../utils/other'

import { useI18n } from 'vue-i18n'
import topMenu from '../components/TopBar.vue'
import Loading from '../components/Loading.vue'
import SiteFooter from '../components/SiteFooter.vue'
import Alert from '../components/Alert.vue'
import Confirm from '../components/Confirm.vue'

const router = useRouter()
const route = useRoute()
const { t, locale, tm } = useI18n()

// 当前 session_id（从路由参数获取）
const currentSessionId = ref(null)
const isDetailMode = ref(false) // 是否为详情模式

// 模板引用
const inputField = ref(null)
const playerSection = ref(null)
const statusText = ref(null)
const statusMessage = ref(null)
const stopBtn = ref(null)
const downloadBtn = ref(null)
const applyBtn = ref(null)
const subtitleSection = ref(null)
const playBtn = ref(null)
const progressBar = ref(null)
const currentTimeEl = ref(null)
const durationEl = ref(null)
const waveform = ref(null)
const progressContainer = ref(null)
const sidebar = ref(null)
const sidebarToggle = ref(null)
const toggleSubtitlesBtn = ref(null)
const audioUserInputEl = ref(null)
const audioElement = ref(null)  // template 中的 audio 元素引用

// 响应式状态
const input = ref('')
const showPlayer = ref(false)
const showStatus = ref(false)
const statusMsg = ref('')
const statusClass = ref('')
const showStopBtn = ref(false)
const showDownloadBtn = ref(false)
const showSubtitles = ref(false)
const isPlaying = ref(false)
const currentTime = ref(0)
const duration = ref(0)
const progress = ref(0)
const audioUserInput = ref('')
const sidebarCollapsed = ref(false)
const historyItems = ref([])
const loadingHistory = ref(false)
const loadingSessionDetail = ref(false)
// 音频 URL（响应式，用于 template 中的 audio 元素）
const audioUrl = ref('')
// 波形图数据（响应式数组，用于 template 渲染）
const waveformBars = ref([])

// 示例输入列表（使用计算属性支持语言切换）
const exampleInputs = computed(() => {
    try {
        // 使用 tm 函数直接获取翻译对象（数组）
        const messages = tm('podcast.exampleInputs')
        if (Array.isArray(messages)) {
            return messages
        }
        // 如果 tm 返回的不是数组，尝试使用 t 函数
        const result = t('podcast.exampleInputs', { returnObjects: true })
        if (Array.isArray(result)) {
            return result
        }
        // 如果都不行，返回默认值
        console.warn('exampleInputs not found in translations, using defaults')
        return locale.value === 'zh' ? [
            'https://github.com/ModelTC/LightX2V',
            'LLM大模型的原理',
            '什么是深度学习？',
            '如何平衡工作和生活？',
            '如何科学减肥'
        ] : [
            'https://github.com/ModelTC/LightX2V',
            'Principles of LLM Large Models',
            'What is Deep Learning?',
            'How to Balance Work and Life?',
            'How to Lose Weight Scientifically'
        ]
    } catch (error) {
        console.warn('Failed to load exampleInputs:', error)
        // 返回默认值
        return locale.value === 'zh' ? [
            'https://github.com/ModelTC/LightX2V',
            'LLM大模型的原理',
            '什么是深度学习？',
            '如何平衡工作和生活？',
            '如何科学减肥'
        ] : [
            'https://github.com/ModelTC/LightX2V',
            'Principles of LLM Large Models',
            'What is Deep Learning?',
            'How to Balance Work and Life?',
            'How to Lose Weight Scientifically'
        ]
    }
})

// 音频相关状态（非响应式，用于内部逻辑）
let audio = null
let isDragging = false
let audioContext = null
let analyser = null  // 用于 audio 元素的分析器
let webAudioAnalyser = null  // 用于 WebAudio 的分析器（独立）
let animationFrameId = null
// 字幕数据改为响应式，用于 template 渲染
const subtitles = ref([])
const subtitleTimestamps = ref([])
let wsConnection = null
let mergedAudioUrl = null
let currentAudioUrl = null
let sessionAudioUrl = null  // 当前会话的音频 URL（用于 applyToDigitalHuman）
let lastAudioDuration = 0
let audioUpdateChecker = null
let isSwitching = false
let autoFollowSubtitles = true
let userScrollTimeout = null
let mediaSource = null
let sourceBuffer = null
let audioQueue = []
let lastBytePosition = 0
let totalAudioSize = 0
let shouldResumePlayback = false
let isGenerating = false  // 是否正在生成播客
let isInitialAudioLoadComplete = false  // 初始音频加载是否完成
let pendingAppendSize = 0  // 正在追加的数据大小（用于更新 lastBytePosition）

// WebAudio 流式播放相关
let webAudioContext = null
let webAudioQueue = []
let webAudioPlaying = false
let webAudioCurrentTime = 0
let webAudioStartTime = 0
let webAudioTotalDuration = 0
let webAudioSourceNodes = []
let webAudioTimeUpdateFrame = null

// 设备检测
function isIOSSafari() {
    const ua = window.navigator.userAgent
    const isIOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
    const isSafari = /^((?!chrome|android).)*safari/i.test(ua)
    return isIOS && isSafari
}

// 为 URL 添加缓存破坏参数（仅对 API URL，不对 CDN 预签名 URL）
function addCacheBustingParam(url) {
    if (!url) return url
    // 如果是 CDN URL（http/https），不添加参数，避免破坏预签名 URL
    if (url.startsWith('http://') || url.startsWith('https://')) {
        return url
    }
    // 对于 API URL，添加缓存破坏参数
    const separator = url.includes('?') ? '&' : '?'
    let newUrl = `${url}${separator}t=${Date.now()}`;
    const token = localStorage.getItem('accessToken')
    if (token) {
        newUrl = `${newUrl}&token=${token}`
    }
    return newUrl;
}

// 检测是否为移动端
function isMobileDevice() {
    return window.matchMedia && window.matchMedia('(max-width: 768px)').matches
}

// 初始化波形图（使用响应式数据）
function initWaveform() {
    try {
        // 根据设备类型初始化波形条数组：移动端50个，桌面端100个
        const barCount = isMobileDevice() ? 50 : 100
        const bars = []
        for (let i = 0; i < barCount; i++) {
            bars.push({
                id: i,
                height: 30, // 当前高度（用于平滑过渡）
                targetHeight: 30, // 目标高度
                intensity: 50 // 初始强度（0-100）
            })
        }

        waveformBars.value = bars
        // 立即渲染一次波形（即使没有播放）
        nextTick(() => {
            renderSimulatedWaveform()
        })
    } catch (error) {
        console.error('Error initializing waveform:', error)
    }
}

// 初始化 WebAudio Context（用于流式播放）
function initWebAudioContext() {
    if (!webAudioContext || webAudioContext.state === 'closed') {
        webAudioContext = new (window.AudioContext || window.webkitAudioContext)()
    }
    return webAudioContext
}

// 处理接收到的 WAV chunk
async function handleWebAudioChunk(arrayBuffer) {
    try {
        const context = initWebAudioContext()

        // 解码 WAV chunk
        const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0))

        // 添加到队列
        webAudioQueue.push(audioBuffer)
        webAudioTotalDuration += audioBuffer.duration

        // 更新总时长显示
        duration.value = webAudioTotalDuration

        // 显示播放器（如果还没显示）
        if (!showPlayer.value) {
            showPlayer.value = true
            await nextTick()
            // 确保波形图容器存在后再初始化
            if (waveform.value) {
                initWaveform()
            }
            statusMsg.value = t('podcast.ready')
        }

        // 确保波形图已初始化（如果播放器已显示但波形图未初始化）
        if (showPlayer.value && waveformBars.value.length === 0) {
            initWaveform()
        }

        console.log(`✅ Received WAV chunk: ${audioBuffer.duration.toFixed(2)}s, queue: ${webAudioQueue.length}, total duration: ${webAudioTotalDuration.toFixed(2)}s`)
    } catch (error) {
        console.error('Error handling WebAudio chunk:', error)
        statusMsg.value = t('podcast.audioDecodeFailed', { error: error.message })
    }
}

// 播放下一个 WebAudio chunk
async function playNextWebAudioChunk() {
    if (webAudioQueue.length === 0) {
        webAudioPlaying = false
        if (isPlaying.value) {
            // 所有音频播放完成
            isPlaying.value = false
            webAudioCurrentTime = 0
        }
        return
    }

    if (!webAudioPlaying && !isPlaying.value) {
        // 用户没有点击播放，不自动播放
        return
    }

    webAudioPlaying = true
    const context = initWebAudioContext()

    // 确保 AudioContext 已恢复（移动端需要用户交互）
    if (context.state === 'suspended') {
        await context.resume()
    }

    const audioBuffer = webAudioQueue.shift()

    // 创建 AudioBufferSourceNode
    const source = context.createBufferSource()
    source.buffer = audioBuffer

    // 创建分析器用于波形显示（WebAudio 模式，使用独立的 analyser）
    if (!webAudioAnalyser || webAudioAnalyser.context !== context) {
        webAudioAnalyser = context.createAnalyser()
        webAudioAnalyser.fftSize = 256
        // 降低平滑系数，让波形变化更敏感、更明显（0.3 比 0.8 更敏感）
        webAudioAnalyser.smoothingTimeConstant = 0.3
        webAudioAnalyser.connect(context.destination)
    }
    // 确保 source 连接到 webAudioAnalyser，webAudioAnalyser 连接到 destination
    source.connect(webAudioAnalyser)

    // 记录开始时间（只在第一次播放时设置）
    if (webAudioStartTime === 0) {
        webAudioStartTime = context.currentTime
    }

    // 计算当前音频块的开始时间（按顺序播放）
    const chunkStartTime = webAudioStartTime + webAudioCurrentTime

    // 播放（使用计算好的开始时间，确保按顺序播放）
    source.start(chunkStartTime)

    console.log(`🎵 Playing chunk: start=${chunkStartTime.toFixed(3)}, duration=${audioBuffer.duration.toFixed(3)}, currentTime=${webAudioCurrentTime.toFixed(3)}`)

    // 保存 source node（用于停止）
    webAudioSourceNodes.push(source)

    // 监听播放结束
    source.onended = () => {
        webAudioCurrentTime += audioBuffer.duration
        currentTime.value = webAudioCurrentTime
        // 更新进度条
        if (webAudioTotalDuration > 0) {
            progress.value = (webAudioCurrentTime / webAudioTotalDuration) * 100
        }
        // 更新字幕高亮
        updateActiveSubtitleForStreaming(webAudioCurrentTime)
        // 继续播放下一个
        playNextWebAudioChunk()
    }

    // 启动时间更新（如果还没启动）
    if (!webAudioTimeUpdateFrame) {
        startWebAudioTimeUpdate()
    }
}

// 更新 WebAudio 当前时间（与参考 HTML 一致，自动跟随字幕）
function startWebAudioTimeUpdate() {
    function updateWebAudioTime() {
        if (webAudioPlaying && isPlaying.value && webAudioContext) {
            const elapsed = webAudioContext.currentTime - webAudioStartTime
            webAudioCurrentTime = Math.min(elapsed, webAudioTotalDuration)
            currentTime.value = webAudioCurrentTime
            if (webAudioTotalDuration > 0) {
                progress.value = (webAudioCurrentTime / webAudioTotalDuration) * 100
            }
            // 自动更新字幕高亮并跟随（与参考 HTML 一致）
            updateActiveSubtitleForStreaming(webAudioCurrentTime)
            webAudioTimeUpdateFrame = requestAnimationFrame(updateWebAudioTime)
        } else {
            webAudioTimeUpdateFrame = null
        }
    }
    webAudioTimeUpdateFrame = requestAnimationFrame(updateWebAudioTime)
}

// 开始/暂停 WebAudio 播放
function toggleWebAudioPlayback() {
    if (!webAudioContext) {
        initWebAudioContext()
    }

    // 确保 AudioContext 已恢复（移动端需要用户交互）
    if (webAudioContext.state === 'suspended') {
        webAudioContext.resume().catch(err => {
            console.error('Failed to resume AudioContext:', err)
        })
    }

    if (webAudioPlaying || isPlaying.value) {
        // 停止所有正在播放的 source nodes
        webAudioSourceNodes.forEach(node => {
            try {
                node.stop()
            } catch (e) {
                // 可能已经停止
            }
        })
        webAudioSourceNodes = []
        webAudioPlaying = false
        isPlaying.value = false
        if (webAudioTimeUpdateFrame) {
            cancelAnimationFrame(webAudioTimeUpdateFrame)
            webAudioTimeUpdateFrame = null
        }
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }
    } else {
        // 开始播放
        if (webAudioQueue.length === 0) {
            statusMsg.value = t('podcast.noAudioAvailable')
            return
        }

        isPlaying.value = true
        const context = initWebAudioContext()
        if (webAudioStartTime === 0) {
            webAudioStartTime = context.currentTime
        }
        playNextWebAudioChunk()
        startWebAudioTimeUpdate()
        // 启动波形可视化
        if (!animationFrameId) {
            visualize()
        }
    }
}

// 存储 MediaElementSource，避免重复创建
let mediaElementSource = null
// 跨域音频的预分析波形数据（用于波形显示）
let crossOriginWaveformData = null
let crossOriginWaveformDataLoaded = false
let crossOriginWaveformMin = 0 // 波形数据的最小值（用于归一化）
let crossOriginWaveformMax = 0 // 波形数据的最大值（用于归一化）

// 检查音频 URL 是否跨域（可能导致 CORS 问题）
function isCrossOriginAudio(audioElement) {
    if (!audioElement || !audioElement.src) return false

    try {
        const audioUrl = new URL(audioElement.src, window.location.href)
        const currentOrigin = window.location.origin
        return audioUrl.origin !== currentOrigin
    } catch (e) {
        // 如果 URL 解析失败，可能是 blob: 或 data: URL，不算跨域
        return false
    }
}

// 为跨域音频预分析波形数据（用于波形显示）
let lastAnalyzedAudioUrl = null

async function initCrossOriginAudioAnalyzer() {
    if (!audio || !audio.src || !isCrossOriginAudio(audio)) return

    // 检查音频 URL 是否变化，如果变化了需要重新分析
    const currentAudioUrl = audio.src.split('?')[0] // 移除查询参数
    if (crossOriginWaveformDataLoaded && crossOriginWaveformData && lastAnalyzedAudioUrl === currentAudioUrl) {
        return // 已经分析过且 URL 没变化
    }

    // 如果 URL 变化了，清理旧数据
    if (lastAnalyzedAudioUrl !== currentAudioUrl) {
        crossOriginWaveformData = null
        crossOriginWaveformDataLoaded = false
        crossOriginWaveformMin = 0
        crossOriginWaveformMax = 0
    }

    try {
        console.log('🎵 Pre-analyzing cross-origin audio for waveform visualization...')

        // 创建临时的 AudioContext（仅用于解码）
        const tempContext = new (window.AudioContext || window.webkitAudioContext)()

        // 使用 fetch 获取音频数据（支持 CORS）
        // 注意：对于 CDN 预签名 URL，查询参数可能包含认证信息，应该保留
        const fullAudioUrl = audio.src // 保留完整 URL（包括查询参数）
        const audioUrlWithoutParams = audio.src.split('?')[0] // 移除查询参数的版本（备用）

        let response
        let lastError = null

        // 先尝试完整 URL（保留查询参数，可能包含认证信息）
        try {
            response = await fetch(fullAudioUrl, {
                mode: 'cors',
                credentials: 'omit'
            })

            // 如果 fetch 返回错误状态码（如 403），尝试使用 apiCall（支持认证）
            if (!response.ok) {
                console.warn(`Direct fetch returned ${response.status}, trying apiCall with authentication...`)
                lastError = new Error(`Direct fetch failed: ${response.status} ${response.statusText}`)
                try {
                    // 使用完整 URL 尝试 apiCall
                    response = await apiCall(fullAudioUrl)
                    if (!response || !response.ok) {
                        throw new Error(`apiCall failed: ${response ? response.status : 'no response'}`)
                    }
                } catch (apiError) {
                    // 如果完整 URL 失败，尝试移除查询参数的版本
                    console.warn('apiCall with full URL failed, trying without query params...')
                    try {
                        response = await apiCall(audioUrlWithoutParams)
                        if (!response || !response.ok) {
                            throw new Error(`apiCall without params failed: ${response ? response.status : 'no response'}`)
                        }
                    } catch (apiError2) {
                        throw new Error(`Both fetch and apiCall failed. Fetch: ${lastError.message}, apiCall (full): ${apiError.message || 'unknown'}, apiCall (no params): ${apiError2.message || 'unknown'}`)
                    }
                }
            }
        } catch (fetchError) {
            // 如果 fetch 失败（可能是 CORS 限制或网络错误），尝试使用 apiCall（支持认证）
            console.warn('Direct fetch failed, trying apiCall:', fetchError)
            lastError = fetchError
            try {
                // 先尝试完整 URL
                response = await apiCall(fullAudioUrl)
                if (!response) {
                    throw new Error('apiCall returned no response')
                }
                if (!response.ok) {
                    throw new Error(`apiCall failed: ${response.status} ${response.statusText}`)
                }
            } catch (apiError) {
                // 如果完整 URL 失败，尝试移除查询参数的版本
                console.warn('apiCall with full URL failed, trying without query params...')
                try {
                    response = await apiCall(audioUrlWithoutParams)
                    if (!response) {
                        throw new Error('apiCall (no params) returned no response')
                    }
                    if (!response.ok) {
                        throw new Error(`apiCall (no params) failed: ${response.status} ${response.statusText}`)
                    }
                } catch (apiError2) {
                    const errorMsg = `Failed to fetch audio. Fetch error: ${fetchError.message || 'unknown'}, apiCall (full) error: ${apiError.message || 'unknown'}, apiCall (no params) error: ${apiError2.message || 'unknown'}`
                    console.error('❌', errorMsg)
                    throw new Error(errorMsg)
                }
            }
        }

        if (!response || !response.ok) {
            const errorMsg = `Failed to fetch audio: ${response ? response.status : 'no response'}`
            console.error('❌', errorMsg)
            throw new Error(errorMsg)
        }

        const arrayBuffer = await response.arrayBuffer()

        // 解码音频数据
        const audioBuffer = await tempContext.decodeAudioData(arrayBuffer)

        // 关闭临时 context
        await tempContext.close()

        // 预分析音频数据，生成一个非常长的波形数据数组
        // 根据音频时长生成数据点：每 0.01 秒一个数据点（100Hz 采样率）
        const audioDuration = audioBuffer.duration // 音频时长（秒）
        const sampleRate = audioBuffer.sampleRate // 采样率
        const channelData = audioBuffer.getChannelData(0) // 使用第一个声道

        // 计算数据点数量：每 0.01 秒一个数据点
        const dataPointInterval = 0.05 // 秒
        const totalDataPoints = Math.ceil(audioDuration / dataPointInterval)
        const samplesPerDataPoint = Math.floor(sampleRate * dataPointInterval) // 每个数据点对应的样本数

        console.log(`📊 Generating long waveform: ${totalDataPoints} data points for ${audioDuration.toFixed(2)}s audio (${dataPointInterval}s per point)`)

        crossOriginWaveformData = []
        let minAmplitude = Infinity
        let maxAmplitude = -Infinity

        for (let i = 0; i < totalDataPoints; i++) {
            // 计算该数据点对应的样本范围
            const startSample = Math.floor(i * samplesPerDataPoint)
            const endSample = Math.min(startSample + samplesPerDataPoint, channelData.length)

            // 如果超出范围，跳过
            if (startSample >= channelData.length) {
                crossOriginWaveformData.push(0)
                continue
            }

            // 计算该段的 RMS（均方根值）和峰值
            let sumSquares = 0
            let count = 0
            let maxSampleAmplitude = 0

            for (let j = startSample; j < endSample; j++) {
                const sample = channelData[j]
                sumSquares += sample * sample
                count++
                if (Math.abs(sample) > maxSampleAmplitude) {
                    maxSampleAmplitude = Math.abs(sample)
                }
            }

            const rms = count > 0 ? Math.sqrt(sumSquares / count) : 0
            // 混合 RMS 和峰值（50% RMS + 50% 峰值）
            const amplitude = rms * 0.5 + maxSampleAmplitude * 0.5

            crossOriginWaveformData.push(amplitude)

            // 更新最小值和最大值
            if (amplitude < minAmplitude) {
                minAmplitude = amplitude
            }
            if (amplitude > maxAmplitude) {
                maxAmplitude = amplitude
            }
        }

        // 保存最小值和最大值，用于归一化
        crossOriginWaveformMin = minAmplitude
        crossOriginWaveformMax = maxAmplitude

        console.log(`✅ Generated ${crossOriginWaveformData.length} data points for waveform visualization`)
        console.log(`📊 Amplitude range: ${minAmplitude.toFixed(6)} to ${maxAmplitude.toFixed(6)}`)

        crossOriginWaveformDataLoaded = true
        lastAnalyzedAudioUrl = currentAudioUrl // 保存已分析的 URL
        console.log('✅ Cross-origin audio waveform data pre-analyzed successfully')
    } catch (error) {
        console.warn('⚠️ Failed to pre-analyze cross-origin audio:', error)
        console.warn('Will fall back to simulated waveform')
        // 清理失败的状态
        crossOriginWaveformData = null
        crossOriginWaveformDataLoaded = false
        crossOriginWaveformMin = 0
        crossOriginWaveformMax = 0
        lastAnalyzedAudioUrl = null
    }
}

// 初始化音频分析器
async function initAudioAnalyzer() {
    if (!audio) return

    // 检查音频是否跨域（跨域音频使用 MediaElementSource 会导致 CORS 限制，音频会无声）
    if (isCrossOriginAudio(audio)) {
        console.warn('⚠️ Audio is cross-origin, skipping MediaElementSource creation to avoid CORS restrictions')
        console.warn('Will try to use pre-analyzed waveform data instead')
        analyser = null
        mediaElementSource = null
        // 尝试预分析音频数据用于波形显示
        await initCrossOriginAudioAnalyzer()
        return
    }

    // 如果已经有有效的 analyser 和 mediaElementSource，直接返回
    if (analyser && audioContext && audioContext.state !== 'closed' && mediaElementSource) {
        return
    }

    // 如果 audioContext 已关闭，需要重新创建
    if (audioContext && audioContext.state === 'closed') {
        audioContext = null
        analyser = null
        mediaElementSource = null
    }

    // 如果 audioContext 存在但 analyser 不存在，需要重新创建
    if (audioContext && audioContext.state !== 'closed' && !analyser) {
        // 关闭旧的 context 并重新创建
        try {
            await audioContext.close()
        } catch (e) {
            console.log('Error closing old context:', e)
        }
        audioContext = null
        analyser = null
        mediaElementSource = null
    }

    // 创建新的 AudioContext
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)()
    }

    // 确保 AudioContext 是 running 状态（重要：如果使用 MediaElementSource，音频必须通过 Web Audio API 播放）
    // 必须在创建 MediaElementSource 之前恢复，否则音频会无声
    if (audioContext.state === 'suspended') {
        try {
            await audioContext.resume()
            console.log('AudioContext resumed in initAudioAnalyzer, state:', audioContext.state)
        } catch (e) {
            console.warn('Error resuming AudioContext in initAudioAnalyzer:', e)
            // 如果恢复失败，不要创建 MediaElementSource，让音频直接播放
            console.warn('⚠️ AudioContext resume failed, skipping MediaElementSource creation to allow direct audio playback')
            return
        }
    }

    // 再次确认 AudioContext 是 running 状态（双重检查）
    if (audioContext.state !== 'running') {
        console.warn('⚠️ AudioContext is not running before creating MediaElementSource:', audioContext.state)
        console.warn('Skipping MediaElementSource creation to allow direct audio playback')
        return
    }

    try {
        analyser = audioContext.createAnalyser()
        analyser.fftSize = 256
        // 降低平滑系数，让波形变化更敏感、更明显（0.3 比 0.8 更敏感）
        analyser.smoothingTimeConstant = 0.3

        // 创建 MediaElementSource（只能调用一次）
        // 如果已经创建过，会抛出 InvalidStateError
        try {
            mediaElementSource = audioContext.createMediaElementSource(audio)
            // 连接：source -> analyser -> destination
            mediaElementSource.connect(analyser)
            analyser.connect(audioContext.destination)
            console.log('Audio analyzer initialized successfully, AudioContext state:', audioContext.state)

            // 再次确保 AudioContext 是 running 状态（创建 MediaElementSource 后）
            if (audioContext.state === 'suspended') {
                try {
                    await audioContext.resume()
                    console.log('AudioContext resumed after creating MediaElementSource, state:', audioContext.state)
                } catch (e) {
                    console.warn('Error resuming AudioContext after creating MediaElementSource:', e)
                    console.error('⚠️ AudioContext is suspended after creating MediaElementSource - audio may be silent!')
                }
            }

            // 最终验证 AudioContext 状态
            if (audioContext.state !== 'running') {
                console.error('⚠️ AudioContext is not running after creating MediaElementSource:', audioContext.state)
                console.error('Audio may be silent!')
            }
        } catch (error) {
            // 如果已经创建过 MediaElementSource，说明音频元素已经连接到另一个 context
            if (error.name === 'InvalidStateError' || error.message.includes('already been created') || error.message.includes('InvalidStateError')) {
                console.warn('MediaElementSource already exists for this audio element, cannot create analyzer')
                // 无法创建分析器，但不影响音频播放
                // 音频仍然可以通过 audio 元素直接播放（如果它还没有被连接到 Web Audio API）
                // 或者如果已经被连接，它会通过现有的连接播放
                analyser = null
                mediaElementSource = null
                // 不抛出错误，让音频继续播放
                return
            } else {
                throw error
            }
        }
    } catch (error) {
        console.error('Error creating audio analyzer:', error)
        analyser = null
        mediaElementSource = null
        // 即使分析器创建失败，也不应该阻止音频播放
        // 音频仍然可以通过 audio 元素直接播放
    }
}

// 波形可视化（使用真实音频数据）
function visualize() {
    // 检查是否有音频源（audio 元素或 WebAudio）
    const hasAudio = audio || (webAudioPlaying && webAudioContext)

    if (!hasAudio || waveformBars.value.length === 0) {
        if (waveformBars.value.length === 0) {
        initWaveform()
        }
        // 如果正在生成或正在播放，继续动画
        if (isGenerating || (audio && !audio.paused) || (webAudioPlaying && isPlaying.value)) {
            // 生成时或播放时，渲染模拟波形图
            if (waveformBars.value.length > 0) {
                renderSimulatedWaveform()
            }
            animationFrameId = requestAnimationFrame(visualize)
        } else {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId)
                animationFrameId = null
            }
        }
        return
    }

    // 优先使用真实音频数据，如果没有则尝试使用预分析的波形数据（跨域音频），最后回退到模拟波形
    if (analyser && audioContext && audioContext.state !== 'closed') {
        renderRealWaveform(analyser)
    } else if (webAudioAnalyser && webAudioContext && webAudioContext.state !== 'closed') {
        renderRealWaveform(webAudioAnalyser)
    } else if (crossOriginWaveformData && crossOriginWaveformDataLoaded && isCrossOriginAudio(audio)) {
        // 跨域音频：使用预分析的波形数据
        renderPreAnalyzedWaveform()
    } else {
        // 回退到模拟波形
        renderSimulatedWaveform()
    }

    // 检查是否正在播放或正在生成
    const isCurrentlyPlaying = (audio && !audio.paused) || (webAudioPlaying && isPlaying.value)

    if ((isCurrentlyPlaying || isGenerating) && !isDragging) {
        animationFrameId = requestAnimationFrame(visualize)
    } else {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }
    }
}

// 使用真实音频数据渲染波形（心电图样式，反映实时音量变化）
function renderRealWaveform(analyserNode) {
    if (!analyserNode || waveformBars.value.length === 0) return

    // 使用时域数据（心电图样式），更好地反映音量变化
    const bufferLength = analyserNode.frequencyBinCount
    const timeDataArray = new Uint8Array(bufferLength)
    analyserNode.getByteTimeDomainData(timeDataArray)

    // 计算整体音量（RMS - 均方根值）
    let sumSquares = 0
    for (let i = 0; i < timeDataArray.length; i++) {
        // 将 0-255 转换为 -128 到 127
        const sample = (timeDataArray[i] - 128) / 128
        sumSquares += sample * sample
    }
    const rms = Math.sqrt(sumSquares / timeDataArray.length) // RMS 值 (0-1)

    // 计算每个 bar 对应的时域数据范围
    const barsCount = waveformBars.value.length
    const samplesPerBar = Math.floor(bufferLength / barsCount)

    // 根据主题切换波形颜色
    const isDark = document.documentElement.classList.contains('dark')

    // 平滑过渡参数：控制过渡速度（值越小，过渡越慢）
    const smoothingFactor = 0.05 // 每次更新移动 5% 的距离（约 20 帧完成过渡，更平滑）

    waveformBars.value.forEach((bar, i) => {
        // 计算该 bar 对应的时域数据范围
        let sumAmplitude = 0
        let maxAmplitudeInBar = 0
        let count = 0
        const start = i * samplesPerBar
        const end = Math.min(start + samplesPerBar, bufferLength)

        for (let j = start; j < end; j++) {
            // 将 0-255 转换为 -1 到 1
            const sample = (timeDataArray[j] - 128) / 128
            const amplitude = Math.abs(sample) // 振幅（绝对值）
            sumAmplitude += amplitude
            count++
            if (amplitude > maxAmplitudeInBar) {
                maxAmplitudeInBar = amplitude
            }
        }

        // 计算平均振幅
        const avgAmplitude = count > 0 ? sumAmplitude / count : 0

        // 混合平均振幅和峰值振幅（50% 平均 + 50% 峰值），让峰值更明显
        let normalizedAmplitude = avgAmplitude * 0.5 + maxAmplitudeInBar * 0.5

        // 结合整体音量（RMS）进行动态调整（让波形更敏感地反映音量变化）
        normalizedAmplitude = normalizedAmplitude * 0.6 + rms * 0.4

        // 动态范围压缩：增强小信号，让低音量也能看到明显变化
        // 使用更激进的压缩（0.4 次方），让波形变化更明显
        const compressed = Math.pow(normalizedAmplitude, 0.4)

        // 高度范围：4px 到 76px（容器高度 80px，从底部向上延伸，像心电图一样）
        // 音量越大，高度越高
        const minHeight = 4
        const maxHeight = 76
        const heightRange = maxHeight - minHeight
        const targetHeight = minHeight + compressed * heightRange

        // 平滑过渡：从当前高度向目标高度移动
        // 使用线性插值（Lerp）实现平滑过渡
        const currentHeight = bar.height || bar.targetHeight || minHeight
        const heightDiff = targetHeight - currentHeight
        const newHeight = currentHeight + heightDiff * smoothingFactor

        // 更新目标高度（用于下次计算）
        bar.targetHeight = targetHeight

        // 更新当前高度（平滑过渡后的值）
        bar.height = newHeight

        // 强度用于渐变效果（基于振幅，音量越大强度越高）
        const intensity = Math.min(100, normalizedAmplitude * 180) // 增强强度显示，让变化更明显

        // 更新响应式数据
        try {
            bar.intensity = intensity
            bar.isDark = isDark
        } catch (e) {
            console.warn('Error updating waveform bar:', e)
        }
    })
}

// 使用预分析的波形数据渲染波形（跨域音频）
// 根据播放进度遍历长波形数据，显示对应位置的波形段
// 使用平滑过渡，让波形像心电图一样流动
function renderPreAnalyzedWaveform() {
    if (!crossOriginWaveformData || waveformBars.value.length === 0 || !audio) return

    // 获取当前播放进度
    const currentTime = audio.currentTime || 0
    const duration = audio.duration || 1

    // 波形数据是按 0.01 秒间隔生成的，计算当前时间对应的数据点索引
    const dataPointInterval = 0.07 // 秒（与生成时一致）
    const currentDataIndex = Math.floor(currentTime / dataPointInterval)

    const barsCount = waveformBars.value.length
    const dataLength = crossOriginWaveformData.length

    // 计算当前显示的波形数据范围
    // 从当前播放位置开始，显示后续的波形条（符合进度条的逻辑）
    // 如果接近结尾，则显示当前位置之前的波形条，确保始终显示满 100 个波形条
    let startDataIndex = currentDataIndex
    if (startDataIndex + barsCount > dataLength) {
        // 如果从当前位置开始会超出范围，则从末尾往前推
        startDataIndex = Math.max(0, dataLength - barsCount)
    }
    const endDataIndex = Math.min(startDataIndex + barsCount, dataLength)

    // 根据主题切换波形颜色
    const isDark = document.documentElement.classList.contains('dark')

    // 平滑过渡参数：控制过渡速度（值越小，过渡越慢）
    const smoothingFactor = 0.2 // 每次更新移动 15% 的距离（约 6-7 帧完成过渡）

    waveformBars.value.forEach((bar, i) => {
        // 计算对应的数据索引
        const dataIndex = startDataIndex + i

        // 如果超出数据范围，使用最后一个数据或最小值
        let amplitude = 0
        if (dataIndex >= 0 && dataIndex < dataLength) {
            amplitude = crossOriginWaveformData[dataIndex]
        } else if (dataLength > 0 && dataIndex >= dataLength) {
            // 超出范围，使用最后一个数据
            amplitude = crossOriginWaveformData[dataLength - 1]
        } else if (dataIndex < 0) {
            // 在开始之前，使用第一个数据
            amplitude = dataLength > 0 ? crossOriginWaveformData[0] : 0
        }

        // 归一化振幅：将 [min, max] 映射到 [0, 1]
        // 这样即使振幅差异很小，也能充分利用整个高度范围
        let normalized = 0
        const amplitudeRange = crossOriginWaveformMax - crossOriginWaveformMin
        if (amplitudeRange > 0) {
            // 线性映射：将 [min, max] 映射到 [0, 1]
            normalized = (amplitude - crossOriginWaveformMin) / amplitudeRange
            normalized = Math.max(0, Math.min(1, normalized)) // 确保在 [0, 1] 范围内
        } else {
            // 如果所有值都相同，使用 0.5（中间值）
            normalized = 0.5
        }

        // 动态范围压缩：增强小信号（可选，如果希望更平滑的过渡）
        // const compressed = Math.pow(normalized, 0.4)
        const compressed = normalized

        // 高度范围：4px 到 76px（从底部向上延伸，像心电图一样）
        const minHeight = 1
        const maxHeight = 150
        const heightRange = maxHeight - minHeight
        const targetHeight = minHeight + compressed * heightRange

        // 平滑过渡：从当前高度向目标高度移动
        // 使用线性插值（Lerp）实现平滑过渡
        const currentHeight = bar.height || bar.targetHeight || minHeight
        const heightDiff = targetHeight - currentHeight
        const newHeight = currentHeight + heightDiff * smoothingFactor

        // 更新目标高度（用于下次计算）
        bar.targetHeight = targetHeight

        // 更新当前高度（平滑过渡后的值）
        bar.height = newHeight

        // 强度用于渐变效果（使用归一化后的值）
        const intensity = Math.min(100, normalized * 50 + 50)
        bar.intensity = bar.height * 0.8
        bar.isDark = isDark
    })
}

// 模拟波形渲染（回退方案）
let simulatedWaveformStartTime = null  // 模拟波形动画开始时间

function renderSimulatedWaveform() {
    if (waveformBars.value.length === 0) return

    // 获取当前时间：优先使用 WebAudio 时间，否则使用 audio 元素时间，最后使用生成时的模拟时间
    let currentTime = 0
    let duration = 1

    if (webAudioPlaying && webAudioContext) {
        // WebAudio 流式播放模式
        currentTime = webAudioCurrentTime
        duration = webAudioTotalDuration || 1
    } else if (audio) {
        // 传统 audio 元素播放模式
        currentTime = audio.currentTime || 0
        duration = audio.buffered && audio.buffered.length > 0
            ? audio.buffered.end(audio.buffered.length - 1)
            : (audio.duration || 1)
    } else if (isGenerating) {
        // 生成时：使用基于时间的动画
        if (!simulatedWaveformStartTime) {
            simulatedWaveformStartTime = Date.now()
        }
        // 使用经过的时间作为进度（循环动画）
        const elapsed = (Date.now() - simulatedWaveformStartTime) / 1000  // 秒
        currentTime = elapsed % 10  // 10秒循环
        duration = 10
    }

    const progress = duration > 0 ? currentTime / duration : 0

    // 根据主题切换波形颜色
    const isDark = document.documentElement.classList.contains('dark')

    waveformBars.value.forEach((bar, i) => {
        const position = i / waveformBars.value.length
        const wave1 = Math.sin(position * Math.PI * 4 + progress * Math.PI * 2) * 0.5
        const wave2 = Math.sin(position * Math.PI * 8 + progress * Math.PI * 4) * 0.3
        const wave3 = Math.sin(position * Math.PI * 2 + progress * Math.PI * 1.5) * 0.2
        const wave4 = Math.sin(position * Math.PI * 5 + progress * Math.PI * 2.3) * 0.12
        const wave5 = Math.sin(position * Math.PI * 7 + progress * Math.PI * 0.9) * 0.09
        const wave6 = Math.sin(position * Math.PI * 3.3 + progress * Math.PI * 2.7) * 0.13
        const wave7 = Math.sin(position * Math.PI * 1.5 + progress * Math.PI * 6.2) * 0.08
        const wave8 = Math.cos(position * Math.PI * 6 + progress * Math.PI * 1.5) * 0.11
        const wave9 = Math.sin(position * Math.PI * 9 + progress * Math.PI * 3.5) * 0.07
        const wave10 = Math.cos(position * Math.PI * 4 + progress * Math.PI * 2.8) * 0.15
        const wave11 = Math.sin(position * Math.PI * 8.5 + progress * Math.PI * 1.1) * 0.06
        const wave12 = Math.cos(position * Math.PI * 2.7 + progress * Math.PI * 4.2) * 0.1
        const combined = (wave1 + wave2 + wave3 + wave4 + wave5 + wave6 + wave7 + wave8 + wave9 + wave10 + wave11 + wave12 + 1) / 2
        // 高度范围：10px 到 50px（容器高度60px，padding 10px，实际可用40px）
        const height = 1 + combined * 49
        const intensity = combined * 100

        // 更新响应式数据
        try {
            bar.height = height
            bar.intensity = intensity
            bar.isDark = isDark
        } catch (e) {
            console.warn('Error updating waveform bar:', e)
        }
    })
}

// 设置音频事件监听器
function setupAudioEventListeners() {
    if (!audio || !audioElement.value) return

    // 移除旧的事件监听器（如果存在）
    audio.removeEventListener('loadedmetadata', onAudioLoadedMetadata)
    audio.removeEventListener('canplay', onAudioCanPlay)
    audio.removeEventListener('timeupdate', onAudioTimeUpdate)
    audio.removeEventListener('ended', onAudioEnded)
    audio.removeEventListener('play', onAudioPlay)
    audio.removeEventListener('pause', onAudioPause)
    audio.removeEventListener('error', onAudioError)

    // 添加新的事件监听器
    audio.addEventListener('loadedmetadata', onAudioLoadedMetadata)
    audio.addEventListener('canplay', onAudioCanPlay)
    audio.addEventListener('timeupdate', onAudioTimeUpdate)
    audio.addEventListener('ended', onAudioEnded)
    audio.addEventListener('play', onAudioPlay)
    audio.addEventListener('pause', onAudioPause)
    audio.addEventListener('error', onAudioError)
}

let hasLoadedMetadata = false
let analyzerInitialized = false

function onAudioLoadedMetadata() {
    if (hasLoadedMetadata || !audio) return
    hasLoadedMetadata = true

    try {
        console.log('Audio loadedmetadata:', {
            duration: audio.duration,
            readyState: audio.readyState,
            src: audio.src.substring(0, 100)
        })

        // 在详情模式下显示播放器
        if (isDetailMode.value) {
            showPlayer.value = true
        }

        const total = getDisplayedDuration()
        if (total > 0) {
            duration.value = total
            // 初始化音频分析器（用于波形图）
            if (!analyzerInitialized) {
                initAudioAnalyzer().then(() => {
                    analyzerInitialized = true
                    console.log('Audio analyzer initialized successfully')
                }).catch(error => {
                    console.error('Error initializing audio analyzer:', error)
                })
            }
        }
    } catch (e) {
        console.warn('Error in onAudioLoadedMetadata:', e)
    }
}

function onAudioCanPlay() {
    if (!audio) return

    try {
        console.log('Audio canplay event:', {
            readyState: audio.readyState,
            paused: audio.paused,
            volume: audio.volume
        })
        // 音频可以播放时，确保分析器已初始化
        if (!analyzerInitialized && audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
            initAudioAnalyzer().then(() => {
                analyzerInitialized = true
                console.log('Audio analyzer initialized in canplay event')
            }).catch(error => {
                console.error('Error initializing audio analyzer:', error)
            })
        }
    } catch (e) {
        console.warn('Error in onAudioCanPlay:', e)
    }
}

function onAudioTimeUpdate() {
    if (!isDragging && audio) {
        try {
            currentTime.value = audio.currentTime
            const total = getDisplayedDuration()
            if (total > 0) {
                progress.value = (audio.currentTime / total) * 100
                duration.value = total
            }
            updateActiveSubtitleForStreaming(audio.currentTime)
        } catch (e) {
            console.warn('Error in onAudioTimeUpdate:', e)
        }
    }
}

function onAudioEnded() {
    try {
        isPlaying.value = false
    } catch (e) {
        console.warn('Error in onAudioEnded:', e)
    }
}

function onAudioPlay() {
    try {
        isPlaying.value = true
        // 启动波形图可视化
        if (!animationFrameId) {
            visualize()
        }
    } catch (e) {
        console.warn('Error in onAudioPlay:', e)
    }
}

function onAudioPause() {
    try {
        isPlaying.value = false
    } catch (e) {
        console.warn('Error in onAudioPause:', e)
    }
}

function onAudioError(e) {
    try {
        console.error('Audio error:', e, audio?.error)

        // 检查音频状态，如果还在加载中，不立即显示错误
        if (audio && audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
            // 音频还在加载中，可能是网络延迟，等待一段时间再判断
            console.log('Audio still loading, waiting before showing error...')
            setTimeout(() => {
                // 再次检查，如果仍然出错且没有加载任何数据，才显示错误
                if (audio && audio.readyState === HTMLMediaElement.HAVE_NOTHING && audio.error) {
                    const errorCode = audio.error.code
                    // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误
                    if (errorCode !== 1) {
                        showAlert(t('podcast.audioLoadFailedNetwork'), 'error')
                    }
                }
            }, 3000) // 等待 3 秒
            return
        }

        // 如果已经有元数据或可以播放，说明不是加载问题
        if (audio && audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
            console.log('Audio has metadata, error might be non-critical')
            return
        }

        // 检查错误代码
        if (audio?.error) {
            const errorCode = audio.error.code
            // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误
            if (errorCode === 1) {
                console.log('Audio error is ABORTED, likely user action, not showing error')
                return
            }
        }
    } catch (err) {
        console.warn('Error in onAudioError:', err)
    }
}

// 播放/暂停
async function togglePlayback() {
    // 使用 template 中的 audio 元素
    if (!audio || !audioElement.value) {
        // 如果 audio 变量未初始化，尝试初始化
        if (audioElement.value) {
            audio = audioElement.value
            setupAudioEventListeners()
        } else {
            return
        }
    }

    if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
        statusMsg.value = t('podcast.audioLoading')
        return
    }

    if (audio.paused) {
        try {
            // 确保音量设置为 1.0（重要：避免无声）
            audio.volume = 1.0
            // 确保音频元素未被静音（重要：避免无声）
            audio.muted = false

            // 初始化波形图（在播放前初始化，避免延迟）
            if (waveformBars.value.length === 0) {
                initWaveform()
            }

            // 检查音频是否跨域，如果是跨域且已有 MediaElementSource，需要清理（避免 CORS 限制导致无声）
            if (isCrossOriginAudio(audio) && mediaElementSource) {
                console.warn('⚠️ Audio is cross-origin but MediaElementSource exists, cleaning up to avoid CORS restrictions')
                // 断开连接
                try {
                    if (mediaElementSource) {
                        mediaElementSource.disconnect()
                    }
                    if (analyser) {
                        analyser.disconnect()
                    }
                } catch (e) {
                    console.warn('Error disconnecting MediaElementSource:', e)
                }
                mediaElementSource = null
                analyser = null
            }

            // 确保分析器已初始化（在用户交互时初始化，确保 AudioContext 可以恢复）
            // 注意：如果 MediaElementSource 已存在且音频不是跨域，不需要重新创建
            if (!mediaElementSource) {
                try {
                    await initAudioAnalyzer()
                    console.log('Audio analyzer initialized in togglePlayback')
                } catch (e) {
                    console.log('Analyzer init on play failed (non-critical):', e)
                    // 分析器初始化失败不影响音频播放
                }
            }

            // 确保 AudioContext 已恢复（如果被暂停）
            // 注意：如果使用了 MediaElementSource，音频必须通过 Web Audio API 播放，所以 AudioContext 必须是 running 状态
            // 但跨域音频不使用 MediaElementSource，所以不需要 AudioContext
            if (audioContext && !isCrossOriginAudio(audio)) {
                if (audioContext.state === 'suspended') {
                    try {
                        await audioContext.resume()
                        console.log('AudioContext resumed before play, state:', audioContext.state)
                    } catch (e) {
                        console.warn('Error resuming AudioContext:', e)
                        // AudioContext 恢复失败可能影响播放（如果使用了 MediaElementSource）
                        if (mediaElementSource) {
                            console.error('⚠️ AudioContext resume failed but MediaElementSource exists - audio may be silent!')
                            showAlert(t('podcast.audioMayBeSilent'), 'warning')
                        }
                    }
                }
                // 如果 AudioContext 已关闭，需要重新创建（如果使用了 MediaElementSource）
                if (audioContext.state === 'closed' && mediaElementSource) {
                    console.warn('AudioContext closed but MediaElementSource exists, reinitializing analyzer')
                    try {
                        // 清理旧的连接
                        mediaElementSource = null
                        analyser = null
                        await initAudioAnalyzer()
                    } catch (e) {
                        console.warn('Error reinitializing analyzer:', e)
                    }
                }
            }

            // 播放音频
            console.log('Attempting to play audio:', {
                readyState: audio.readyState,
                src: audio.src ? audio.src.substring(0, 100) : 'no src',
                currentSrc: audio.currentSrc ? audio.currentSrc.substring(0, 100) : 'no currentSrc',
                volume: audio.volume,
                muted: audio.muted,
                paused: audio.paused,
                audioContextState: audioContext ? audioContext.state : 'no context',
                hasMediaElementSource: !!mediaElementSource,
                hasAnalyser: !!analyser
            })

            await audio.play()
            isPlaying.value = true
            console.log('Audio playing successfully, AudioContext state:', audioContext ? audioContext.state : 'no context')

            // 如果使用了 MediaElementSource，再次确保 AudioContext 是 running 状态
            // 注意：跨域音频不应该使用 MediaElementSource（已在前面清理）
            if (mediaElementSource && audioContext && !isCrossOriginAudio(audio)) {
                if (audioContext.state === 'suspended') {
                    try {
                        await audioContext.resume()
                        console.log('AudioContext resumed after play, state:', audioContext.state)
                    } catch (e) {
                        console.warn('Error resuming AudioContext after play:', e)
                        console.error('⚠️ AudioContext is suspended after play - audio may be silent!')
                    }
                }
                // 验证连接是否正常
                if (audioContext.state !== 'running') {
                    console.error('⚠️ AudioContext is not running after play:', audioContext.state)
                    console.error('This may cause audio to be silent when using MediaElementSource')
                }
            } else if (isCrossOriginAudio(audio)) {
                // 跨域音频直接播放，不使用 MediaElementSource
                console.log('✅ Cross-origin audio playing directly (no MediaElementSource)')
            }

            // 启动可视化
            if (!animationFrameId) {
                visualize()
            }
        } catch (error) {
            console.error('Error playing audio:', error)
            isPlaying.value = false
            statusMsg.value = t('podcast.playbackFailed')
            showAlert(t('podcast.playbackFailedWithError', { error: error.message }), 'error')
        }
    } else {
        audio.pause()
        isPlaying.value = false
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }
    }
}

// 获取音频时长
function getAudioDuration() {
    if (!audio) return 0
    if (audio.buffered && audio.buffered.length > 0) {
        return audio.buffered.end(audio.buffered.length - 1)
    }
    if (audio.seekable && audio.seekable.length > 0) {
        return audio.seekable.end(audio.seekable.length - 1)
    }
    if (Number.isFinite(audio.duration)) {
        return audio.duration || 0
    }
    return 0
}

function getDisplayedDuration() {
    const d = getAudioDuration()
    if (!Number.isFinite(d) || d <= 0) return 0
    return d
}

// 格式化时间
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
}

// 进度条更新（与原始 HTML 完全一致）
function updateProgress(e) {
    if (!audio || !progressContainer.value) return

    const duration = getAudioDuration()
    if (duration === 0) return

    const rect = progressContainer.value.getBoundingClientRect()
    const clientX = (e && e.touches && e.touches[0]) ? e.touches[0].clientX : e.clientX
    const percent = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const newTime = percent * duration

    // 设置播放位置
    audio.currentTime = newTime

    // 更新进度条显示
    progress.value = percent * 100
    currentTime.value = newTime

    // 点击进度条时，确保自动跟随字幕
    autoFollowSubtitles = true

    // 清除之前的滚动定时器（避免与立即滚动冲突）
    if (scrollThrottleTimer) {
        clearTimeout(scrollThrottleTimer)
        scrollThrottleTimer = null
    }

    // 更新字幕高亮
    updateActiveSubtitleForStreaming(newTime)

    // 如果字幕区域已显示，立即滚动到对应字幕（不等待节流）
    if (showSubtitles.value && subtitleSection.value && subtitleTimestamps.value.length > 0) {
        // 找到对应时间点的字幕索引
        let targetIndex = -1
        for (let i = 0; i < subtitleTimestamps.value.length; i++) {
            if (newTime >= subtitleTimestamps.value[i].start && newTime <= subtitleTimestamps.value[i].end) {
                targetIndex = i
                break
            }
        }

        // 如果没有找到，找到最近的
        if (targetIndex === -1 && subtitleTimestamps.value.length > 0) {
            if (newTime >= subtitleTimestamps.value[subtitleTimestamps.value.length - 1].end) {
                targetIndex = subtitleTimestamps.value.length - 1
            } else {
                for (let i = 0; i < subtitleTimestamps.value.length; i++) {
                    if (newTime < subtitleTimestamps.value[i].start) {
                        targetIndex = Math.max(0, i - 1)
                        break
                    }
                }
            }
        }

        // 立即滚动到对应字幕
        if (targetIndex >= 0) {
            nextTick().then(() => {
                const targetSubtitleEl = subtitleSection.value?.querySelector(`#subtitle-${targetIndex}`)
                if (targetSubtitleEl && subtitleSection.value) {
                    const container = subtitleSection.value
                    const containerRect = container.getBoundingClientRect()
                    const elementRect = targetSubtitleEl.getBoundingClientRect()

                    // 计算元素相对于容器的位置
                    const elementTop = elementRect.top - containerRect.top + container.scrollTop
                    const elementHeight = elementRect.height
                    const containerHeight = container.clientHeight

                    // 计算滚动位置，使元素在容器中间
                    const scrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2)

                    // 平滑滚动到目标位置
                    container.scrollTo({
                        top: Math.max(0, scrollTop),
                        behavior: 'smooth'
                    })
                }
            })
        }
    }
}

// 进度条点击
function onProgressClick(e) {
    updateProgress(e)
}

// 进度条拖拽（鼠标）
function onProgressMouseDown(e) {
    if (!audio) return
    const duration = getAudioDuration()
    if (duration === 0) return
    isDragging = true
    updateProgress(e)
}

function onProgressMouseMove(e) {
    if (isDragging) {
        updateProgress(e)
    }
}

function onProgressMouseUp() {
    if (isDragging) {
        isDragging = false
    }
    // 拖拽结束后，如果正在播放且没有动画帧，恢复可视化
    if (audio && !audio.paused && !animationFrameId) {
        visualize()
    }
}

// 进度条拖拽（触摸）
function onProgressTouchStart(e) {
    if (!audio) return
    const duration = getAudioDuration()
    if (duration === 0) return
    isDragging = true
    updateProgress(e)
}

function onProgressTouchMove(e) {
    if (isDragging) {
        updateProgress(e)
    }
}

function onProgressTouchEnd() {
    if (isDragging) {
        isDragging = false
    }
    if (audio && !audio.paused && !animationFrameId) {
        visualize()
    }
}

// 当前激活的字幕索引（响应式）
const activeSubtitleIndex = ref(-1)
// 上次更新的索引，用于避免不必要的更新
let lastActiveSubtitleIndex = -1
// 滚动节流定时器
let scrollThrottleTimer = null

// 更新字幕高亮（与参考 HTML 完全一致，支持 WebAudio 和传统 audio 模式）
async function updateActiveSubtitleForStreaming(currentTime) {
    // 检查是否有字幕数据
    if (subtitles.value.length === 0 || subtitleTimestamps.value.length === 0) {
        if (activeSubtitleIndex.value !== -1) {
            activeSubtitleIndex.value = -1
            lastActiveSubtitleIndex = -1
        }
        return
    }

    // 检查是否有音频源（audio 元素或 WebAudio）
    const hasAudio = audio || (webAudioPlaying && webAudioContext)
    if (!hasAudio) return

    // 对于传统 audio 元素，检查 duration
    if (audio && audio.duration === 0) return

    // 根据时间戳找到当前播放的字幕
    let currentIndex = -1
    for (let i = 0; i < subtitleTimestamps.value.length; i++) {
        if (currentTime >= subtitleTimestamps.value[i].start && currentTime <= subtitleTimestamps.value[i].end) {
            currentIndex = i
            break
        }
    }

    // 如果没有找到，尝试找到最近的
    if (currentIndex === -1 && subtitleTimestamps.value.length > 0) {
        // 如果时间超过了最后一个字幕，显示最后一个
        if (currentTime >= subtitleTimestamps.value[subtitleTimestamps.value.length - 1].end) {
            currentIndex = subtitleTimestamps.value.length - 1
        } else {
            // 找到最近的字幕
            for (let i = 0; i < subtitleTimestamps.value.length; i++) {
                if (currentTime < subtitleTimestamps.value[i].start) {
                    currentIndex = Math.max(0, i - 1)
                    break
                }
            }
        }
    }

    // 只在索引真正变化时才更新，避免不必要的 DOM 操作
    if (currentIndex !== lastActiveSubtitleIndex) {
        activeSubtitleIndex.value = currentIndex
        lastActiveSubtitleIndex = currentIndex

        // 自动滚动到当前字幕（仅当开启自动跟随时，使用节流避免频繁滚动）
        if (autoFollowSubtitles && currentIndex >= 0 && subtitleSection.value && subtitleSection.value.parentElement) {
            // 清除之前的滚动定时器
            if (scrollThrottleTimer) {
                clearTimeout(scrollThrottleTimer)
            }

            // 使用节流，避免频繁滚动导致卡顿
            scrollThrottleTimer = setTimeout(async () => {
                try {
                    await nextTick()
                    if (!subtitleSection.value || !subtitleSection.value.parentElement) {
                        return
                    }
                    const currentSubtitleEl = subtitleSection.value.querySelector(`#subtitle-${currentIndex}`)
                    if (currentSubtitleEl && currentSubtitleEl.parentElement) {
                        // 只在字幕容器内滚动，不影响外部组件
                        const container = subtitleSection.value
                        const containerRect = container.getBoundingClientRect()
                        const elementRect = currentSubtitleEl.getBoundingClientRect()

                        // 计算元素相对于容器的位置
                        const elementTop = elementRect.top - containerRect.top + container.scrollTop
                        const elementHeight = elementRect.height
                        const containerHeight = container.clientHeight

                        // 计算滚动位置，使元素在容器中间
                        const scrollTop = elementTop - (containerHeight / 2) + (elementHeight / 2)

                        // 平滑滚动到目标位置
                        container.scrollTo({
                            top: Math.max(0, scrollTop),
                            behavior: 'smooth'
                        })
                    }
                } catch (e) {
                    // 忽略 DOM 操作错误（可能组件已卸载）
                    console.warn('Error scrolling to subtitle:', e)
                }
                scrollThrottleTimer = null
            }, 100) // 100ms 节流，减少滚动频率
        }
    }
}

// 字幕点击跳转
function onSubtitleClick(index) {
    if (!audio || !subtitleTimestamps.value[index]) return

    const jumpTime = subtitleTimestamps.value[index].start ?? 0
    const duration = getAudioDuration() || audio.duration || 0
    const targetTime = Math.max(0, Math.min(duration, jumpTime))

    try {
        audio.currentTime = targetTime
        // 点击视为"跟随当前字幕"的意图
        autoFollowSubtitles = true
        // 立即更新高亮
        updateActiveSubtitleForStreaming(targetTime)
    } catch (error) {
        console.error('Error jumping to subtitle:', error)
    }
}

// 切换字幕显示
function toggleSubtitles() {
    showSubtitles.value = !showSubtitles.value
}

// MediaSource 方式初始化音频（推荐，支持无缝流式更新）
async function initMediaSourceAudio(autoPlay = false) {
    if (isIOSSafari()) {
        // iOS Safari 对 MSE 支持有限，直接回退
        return loadAudio(autoPlay)
    }
    if (!mergedAudioUrl) return

    console.log('🎵 Initializing MediaSource audio...')

    try {
        // 优先使用 template 中的 audioElement
        if (!audioElement.value) {
            console.warn('audioElement not available, waiting...')
            await nextTick()
            if (!audioElement.value) {
                console.error('audioElement still not available, falling back to loadAudio')
                return loadAudio(autoPlay)
            }
        }

        // 使用 template 中的 audio 元素
        audio = audioElement.value
        setupAudioEventListeners()

        // 创建 MediaSource
        mediaSource = new MediaSource()
        const url = URL.createObjectURL(mediaSource)

        // 设置 MediaSource blob URL 到 audio 元素
        audio.src = url
        audio.volume = 1.0

        // 等待 sourceopen
        mediaSource.addEventListener('sourceopen', async () => {
            console.log('📂 MediaSource sourceopen')

            try {
                // 添加 SourceBuffer（使用 MP3 MIME type）
                sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg')

                // 监听更新结束事件（处理队列）
                sourceBuffer.addEventListener('updateend', () => {
                    console.log('📦 SourceBuffer updateend')

                    // 更新 lastBytePosition（关键：确保下次 Range Request 从正确位置开始）
                    if (pendingAppendSize > 0) {
                        lastBytePosition += pendingAppendSize
                        console.log(`📊 Updated lastBytePosition to ${lastBytePosition} bytes (added ${pendingAppendSize} bytes)`)
                        pendingAppendSize = 0
                    }

                    // 更新 MediaSource duration（关键：避免重复播放）
                    if (audio.buffered.length > 0) {
                        const bufferedEnd = audio.buffered.end(audio.buffered.length - 1)
                        // 确保 MediaSource duration 大于等于 buffered 的结束位置
                        // 如果 duration 小于 buffered 长度，会导致重复播放
                        if (mediaSource.duration === Infinity || mediaSource.duration < bufferedEnd) {
                            try {
                                mediaSource.duration = bufferedEnd
                                console.log(`📊 Updated MediaSource duration to ${bufferedEnd.toFixed(2)}s`)
                            } catch (e) {
                                // duration 可能已经设置，忽略错误
                                console.warn('Could not update MediaSource duration:', e)
                            }
                        }
                        // 更新显示
                        duration.value = bufferedEnd
                    }

                    // 自动刷新队列
                    flushQueue()
                })

                sourceBuffer.addEventListener('error', (e) => {
                    console.error('SourceBuffer error:', e)
                })

                // 加载初始音频（完整文件，因为是首次加载）
                console.log('📥 Fetching initial audio...')
                const audioUrlWithCache = addCacheBustingParam(mergedAudioUrl)
                const response = await apiCall(audioUrlWithCache)
                const blob = await response.blob()
                const arrayBuffer = await blob.arrayBuffer()

                console.log(`✅ Received ${arrayBuffer.byteLength} bytes`)

                // 记录总大小（用于显示）
                totalAudioSize = arrayBuffer.byteLength

                // 使用队列安全的追加方法
                // 注意：lastBytePosition 会在 updateend 事件中更新
                appendToBuffer(arrayBuffer)

                // 初始化波形图（等待音频可以播放）
                audio.addEventListener('canplay', async () => {
                    console.log('🎵 Audio can play, initializing waveform...')

                    // 确保波形图已初始化
                    if (waveformBars.value.length === 0) {
                        initWaveform()
                    }

                    // 更新 lastAudioDuration 为实际加载的音频时长，避免重复追加
                    if (audio.buffered.length > 0) {
                        const bufferedEnd = audio.buffered.end(audio.buffered.length - 1)
                        lastAudioDuration = bufferedEnd
                        console.log(`📊 Updated lastAudioDuration to ${bufferedEnd.toFixed(2)}s after initial load`)
                        // 标记初始加载已完成
                        isInitialAudioLoadComplete = true
                    }

                    // 尝试初始化音频分析器
                    try {
                        await initAudioAnalyzer()
                    } catch (e) {
                        console.log('Analyzer init failed, fallback to simulated waveform')
                    }
                })

                // 禁用音频更新检查器：直接使用 WebSocket 的 WAV chunk，不需要轮询 API
                // startAudioUpdateChecker()  // 禁用：WebSocket 已经在推送 WAV chunk

            } catch (err) {
                console.error('Error initializing source buffer:', err)
            }
        })

        mediaSource.addEventListener('error', (e) => {
            console.error('MediaSource error:', e)
        })

        // 绑定事件
        audio.addEventListener('timeupdate', () => {
            if (!isDragging && audio) {
                currentTime.value = audio.currentTime

                // 使用 buffered 计算进度（MediaSource 的 duration 是动态的）
                if (audio.buffered.length > 0) {
                    const bufferedEnd = audio.buffered.end(audio.buffered.length - 1)
                    progress.value = (audio.currentTime / bufferedEnd) * 100
                    duration.value = bufferedEnd
                }

                updateActiveSubtitleForStreaming(audio.currentTime)
            }
        })

        // 监听 buffered 更新（MediaSource 的 duration 是动态的）
        audio.addEventListener('progress', () => {
            if (audio.buffered.length > 0) {
                const bufferedEnd = audio.buffered.end(audio.buffered.length - 1)
                duration.value = bufferedEnd
            }
        })

        audio.addEventListener('ended', () => {
            isPlaying.value = false
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId)
                animationFrameId = null
            }
            stopAudioUpdateChecker()
        })

        audio.addEventListener('error', (e) => {
            console.error('MediaSource audio error:', e, audio?.error)
            if (audio?.error) {
                const errorCode = audio.error.code
                const errorMessages = {
                    1: 'MEDIA_ERR_ABORTED - 用户中止',
                    2: 'MEDIA_ERR_NETWORK - 网络错误',
                    3: 'MEDIA_ERR_DECODE - 解码错误',
                    4: 'MEDIA_ERR_SRC_NOT_SUPPORTED - 格式不支持'
                }
                console.error('MediaSource audio error code:', errorCode, errorMessages[errorCode] || t('podcast.unknownError'))

                // 检查音频状态，如果还在加载中，延迟显示错误
                if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
                    console.log('MediaSource audio still loading, waiting before showing error...')
                    setTimeout(() => {
                        // 再次检查，如果仍然出错且没有加载任何数据，才显示错误并回退
                        if (audio && audio.readyState === HTMLMediaElement.HAVE_NOTHING && audio.error) {
                            const currentErrorCode = audio.error.code
                            // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误
                            if (currentErrorCode !== 1) {
                                showAlert(t('podcast.audioLoadFailedWithError', { error: errorMessages[currentErrorCode] || t('podcast.unknownError') }), 'error')
                                // MediaSource 失败时回退到普通方式
                                loadAudio(autoPlay)
                            }
                        }
                    }, 3000) // 等待 3 秒
                    return
                }

                // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误
                if (errorCode === 1) {
                    console.log('MediaSource audio error is ABORTED, likely user action, not showing error')
                    return
                }

                // 其他情况才显示错误并回退
                showAlert(t('podcast.audioLoadFailedWithError', { error: errorMessages[errorCode] || t('podcast.unknownError') }), 'error')
                // MediaSource 失败时回退到普通方式
                loadAudio(autoPlay)
            }
        })

        // 如果是自动播放，等待 canplay 事件
        if (autoPlay) {
            audio.addEventListener('canplay', async () => {
                try {
                    await audio.play()
                    isPlaying.value = true
                    visualize()
                } catch (error) {
                    console.error('Error auto playing:', error)
                }
            })
        }

    } catch (err) {
        console.error('Error initializing MediaSource audio:', err)
        // 如果 MediaSource 失败，回退到普通方式
        loadAudio(autoPlay)
    }
}

// 加载音频（传统方式，作为后备）
async function loadAudio(autoPlay = false, retryCount = 0) {
    if (!mergedAudioUrl) return

    const maxRetries = 3
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
        animationFrameId = null
    }

    // 优先使用 template 中的 audioElement
    if (audioElement.value) {
        audio = audioElement.value
        // 确保音量设置为 1.0（重要：避免无声）
        audio.volume = 1.0
        // 设置 audioUrl，让 template 中的 audio 元素自动加载
        if (mergedAudioUrl.startsWith('http://') || mergedAudioUrl.startsWith('https://')) {
            audioUrl.value = mergedAudioUrl
        } else {
            audioUrl.value = addCacheBustingParam(mergedAudioUrl)
        }
        // 确保音频元素已加载
        await nextTick()
        if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
            audio.load()
        }
    } else {
        // 如果 audioElement 不存在，创建新的 Audio 对象（兼容旧逻辑）
        if (audio) {
            audio.pause()
            // 清理之前的 blob URL
            if (audio.src && audio.src.startsWith('blob:')) {
                URL.revokeObjectURL(audio.src)
            }
        }

        // 使用 fetch 获取音频（支持认证）
        try {
            const audioUrlWithCache = addCacheBustingParam(mergedAudioUrl)
            const response = await apiCall(audioUrlWithCache)
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }
            const blob = await response.blob()
            const blobUrl = URL.createObjectURL(blob)

            audio = new Audio(blobUrl)
            audio.volume = 1.0
            audio.preload = 'auto'
        } catch (error) {
            console.error('Error loading audio:', error)
            setTimeout(() => {
                loadAudio(autoPlay, retryCount + 1)
            }, 1000)
            return
        }
    }

    let hasLoadedMetadata = false
    let analyzerInitialized = false

    audio.addEventListener('loadedmetadata', async () => {
        if (hasLoadedMetadata) return
        hasLoadedMetadata = true

        const total = getDisplayedDuration()
        if (total > 0) {
            duration.value = total
            // 更新 lastAudioDuration 为实际加载的音频时长，避免重复追加
            lastAudioDuration = total
            console.log(`📊 Updated lastAudioDuration to ${total.toFixed(2)}s after initial load (loadAudio)`)
            // 标记初始加载已完成
            isInitialAudioLoadComplete = true
            if (!analyzerInitialized) {
                await initAudioAnalyzer()
                analyzerInitialized = true
            }
            // 禁用音频更新检查器：直接使用 WebSocket 的 WAV chunk，不需要轮询 API
            // startAudioUpdateChecker()  // 禁用：WebSocket 已经在推送 WAV chunk
        }
    })

    audio.addEventListener('canplay', async () => {
        if (shouldResumePlayback) {
            shouldResumePlayback = false
            try {
                await audio.play()
                isPlaying.value = true
                if (!animationFrameId) {
                    visualize()
                }
            } catch (error) {
                console.error('Error resuming audio:', error)
                isPlaying.value = false
            }
                } else if (!autoPlay) {
            isPlaying.value = false
            statusMsg.value = t('podcast.readyWithCount', { count: subtitles.value.length })
        }
    })

    audio.addEventListener('timeupdate', () => {
        if (!isDragging && audio) {
            currentTime.value = audio.currentTime
            const total = getDisplayedDuration()
            if (total > 0) {
                progress.value = (audio.currentTime / total) * 100
                duration.value = total
            }
            updateActiveSubtitleForStreaming(audio.currentTime)
        }
    })

    audio.addEventListener('ended', () => {
        isPlaying.value = false
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId)
            animationFrameId = null
        }
        stopAudioUpdateChecker()
    })

    audio.addEventListener('error', (e) => {
        console.error('Audio error:', e, audio?.error)
        // 输出更详细的错误信息
        if (audio?.error) {
            const errorCode = audio.error.code
            const errorMessages = {
                1: 'MEDIA_ERR_ABORTED - 用户中止',
                2: 'MEDIA_ERR_NETWORK - 网络错误',
                3: 'MEDIA_ERR_DECODE - 解码错误',
                4: 'MEDIA_ERR_SRC_NOT_SUPPORTED - 格式不支持'
            }
            console.error('Audio error code:', errorCode, errorMessages[errorCode] || t('podcast.unknownError'))
            console.error('Audio src:', audio.src?.substring(0, 100))

            // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误，也不重试
            if (errorCode === 1) {
                console.log('Audio error is ABORTED, likely user action, not retrying')
                return
            }
        }

        // 检查音频状态，如果还在加载中，延迟判断
        if (audio && audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
            console.log('Audio still loading, waiting before retrying...')
            setTimeout(() => {
                // 再次检查，如果仍然出错且没有加载任何数据，才重试或显示错误
                if (audio && audio.readyState === HTMLMediaElement.HAVE_NOTHING && audio.error) {
                    const currentErrorCode = audio.error.code
                    // MEDIA_ERR_ABORTED (1) 通常是用户操作导致的，不显示错误
                    if (currentErrorCode !== 1) {
                        if (!hasLoadedMetadata && retryCount < maxRetries) {
                            setTimeout(() => {
                                loadAudio(autoPlay, retryCount + 1)
                            }, 1000)
                        } else if (retryCount >= maxRetries) {
                            statusMsg.value = t('podcast.audioLoadFailedNetwork')
                            showAlert(t('podcast.audioLoadFailedFormat'), 'error')
                        }
                    }
                }
            }, 3000) // 等待 3 秒
            return
        }

        // 如果已经有元数据，说明不是加载问题，可能是其他错误
        if (audio && audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
            console.log('Audio has metadata, error might be non-critical')
            return
        }

        // 其他情况才重试或显示错误
        if (!hasLoadedMetadata && retryCount < maxRetries) {
            setTimeout(() => {
                loadAudio(autoPlay, retryCount + 1)
            }, 1000)
        } else if (retryCount >= maxRetries) {
            statusMsg.value = t('podcast.audioLoadFailedNetwork')
            showAlert(t('podcast.audioLoadFailedFormat'), 'error')
        }
    })
}

// 启动音频更新检查器
function startAudioUpdateChecker() {
    if (audioUpdateChecker) {
        clearInterval(audioUpdateChecker)
    }
    audioUpdateChecker = setInterval(() => {
        checkAndUpdateAudio()
    }, 5000)
}

// 停止音频更新检查器
function stopAudioUpdateChecker() {
    if (audioUpdateChecker) {
        clearInterval(audioUpdateChecker)
        audioUpdateChecker = null
    }
}

// 无缝切换音频（使用 Range Request 只拉新增部分）
async function switchAudioSeamlessly() {
    if (isSwitching || !mergedAudioUrl || !audio || !sourceBuffer || !mediaSource) return

    isSwitching = true

    const currentTime = audio.currentTime
    const wasPlaying = !audio.paused

    console.log(`📥 Fetching audio update from: ${mergedAudioUrl}, starting at byte ${lastBytePosition}`)

    try {
        // ✅ Range Request: 只拉新增部分
        // 注意：Range Request 使用 fetch 而不是 apiCall，因为 Range Request 需要直接控制 headers
        const audioUrlWithCache = addCacheBustingParam(mergedAudioUrl)

        // 对于 API URL，需要手动添加认证头
        let headers = {
            'Range': `bytes=${lastBytePosition}-`
        }

        // 如果是 API URL（不是 CDN URL），添加认证头
        if (!audioUrlWithCache.startsWith('http://') && !audioUrlWithCache.startsWith('https://')) {
            const token = localStorage.getItem('accessToken')
            if (token) {
                headers['Authorization'] = `Bearer ${token}`
            }
        }

        const response = await fetch(audioUrlWithCache, {
            headers: headers
        })

        if (response.status === 206) {
            // Range 请求成功（206 Partial Content）
            const blob = await response.blob()
            const arrayBuffer = await blob.arrayBuffer()

            if (arrayBuffer.byteLength > 0) {
                console.log(`✅ Received ${arrayBuffer.byteLength} bytes (total loaded: ${lastBytePosition + arrayBuffer.byteLength} bytes)`)

                // 添加到队列，由 updateend 事件处理
                audioQueue.push(arrayBuffer)

                // 尝试刷新队列
                flushQueue()

                // 注意：lastBytePosition 在 updateend 事件中更新，避免重复追加
            } else {
                console.log('No new data available')
            }
        } else if (response.ok && response.status === 200) {
            // Range 不支持，但返回了完整文件
            // 这种情况不应该在追加时发生，如果发生说明服务器不支持 Range Request
            console.warn('⚠️ Range request returned 200 (full file), this should not happen during append')
            // 检查 Content-Range 头，如果有，说明实际返回的是部分内容
            const contentRange = response.headers.get('Content-Range')
            if (contentRange) {
                // 解析 Content-Range: bytes 0-999/2000
                const match = contentRange.match(/bytes (\d+)-(\d+)\/(\d+)/)
                if (match) {
                    const start = parseInt(match[1])
                    const end = parseInt(match[2])
                    const total = parseInt(match[3])
                    console.log(`📊 Content-Range: ${start}-${end}/${total}`)
                    // 如果返回的是从 lastBytePosition 开始的内容，可以追加
                    if (start === lastBytePosition) {
                        const blob = await response.blob()
                        const arrayBuffer = await blob.arrayBuffer()
                        audioQueue.push(arrayBuffer)
                        flushQueue()
                    } else {
                        console.warn('⚠️ Content-Range start does not match lastBytePosition, skipping to avoid duplicate')
                    }
                }
            } else {
                // 没有 Content-Range，说明返回的是完整文件，不应该追加
                console.warn('⚠️ No Content-Range header, skipping to avoid duplicate')
            }
        } else {
            console.warn('Range request failed:', response.status, response.statusText)
            // 如果 Range 请求失败，尝试完整 fetch
            try {
                const fullResponse = await fetch(audioUrlWithCache, {
                    headers: headers
                })
                if (fullResponse.ok) {
                    const blob = await fullResponse.blob()
                    const arrayBuffer = await blob.arrayBuffer()
                    lastBytePosition = arrayBuffer.byteLength
                    totalAudioSize = arrayBuffer.byteLength
                    appendToBuffer(arrayBuffer)
                }
            } catch (e) {
                console.error('Fallback fetch also failed:', e)
            }
        }

    } catch (err) {
        console.error('❌ Error fetching audio:', err)
    } finally {
        isSwitching = false
    }
}

// 追加数据到 buffer（队列安全）
function appendToBuffer(arrayBuffer) {
    if (!sourceBuffer || sourceBuffer.updating || mediaSource.readyState !== 'open') {
        audioQueue.push(arrayBuffer)
        return
    }

    try {
        // 记录正在追加的数据大小
        pendingAppendSize = arrayBuffer.byteLength
        sourceBuffer.appendBuffer(arrayBuffer)
        console.log(`✅ Audio chunk appended to source buffer (${arrayBuffer.byteLength} bytes)`)
    } catch (e) {
        console.error('Error appending buffer:', e)
        pendingAppendSize = 0
        audioQueue.push(arrayBuffer)
    }
}

// 处理音频队列（在 updateend 时调用）
function flushQueue() {
    if (!sourceBuffer || sourceBuffer.updating || mediaSource.readyState !== 'open') {
        return
    }

    if (audioQueue.length === 0) {
        return
    }

    const chunk = audioQueue.shift()
    try {
        // 记录正在追加的数据大小
        pendingAppendSize = chunk.byteLength
        sourceBuffer.appendBuffer(chunk)
        console.log(`📦 Appended queued chunk (${chunk.byteLength} bytes)`)
    } catch (e) {
        console.error('Error appending queued buffer:', e)
        pendingAppendSize = 0
        audioQueue.unshift(chunk) // 失败时放回队列
    }
}

// 检查音频是否有更新并自动切换
async function checkAndUpdateAudio() {
    if (!mergedAudioUrl || !audio || isSwitching) return

    try {
        // 创建临时音频对象检查新长度
        const audioUrlWithCache = addCacheBustingParam(mergedAudioUrl)
        const response = await apiCall(audioUrlWithCache)
        if (!response.ok) return

        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)
        const checkAudio = new Audio(blobUrl)

        checkAudio.addEventListener('loadedmetadata', () => {
            const newDuration = checkAudio.duration
            URL.revokeObjectURL(blobUrl)
            checkAudio.remove()

            // 检查是否有新内容（新的duration大于旧的duration+2秒容差）
            if (newDuration > audio.duration + 2) {
                console.log(`Detected audio update: ${audio.duration}s -> ${newDuration}s`)
                if (mediaSource && sourceBuffer) {
                    // 桌面等支持 MSE：无缝追加
                    switchAudioSeamlessly()
                } else {
                    // iOS 等不支持 MSE：重新加载音频并恢复位置
                    reloadAudioForIOS()
                }
            }
        })

        checkAudio.addEventListener('error', () => {
            URL.revokeObjectURL(blobUrl)
            checkAudio.remove()
        })

        // 加载元数据
        checkAudio.load()
    } catch (error) {
        console.error('Error checking audio update:', error)
    }
}

// iOS 重新加载音频（流式更新）
async function reloadAudioForIOS() {
    if (!audio || !mergedAudioUrl) return

    const wasPlaying = !audio.paused
    const prevTime = audio.currentTime || 0
    const newSrc = addCacheBustingParam(mergedAudioUrl)

    try {
        // 使用 fetch 获取音频（支持认证）
        const response = await apiCall(newSrc)
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
        }
        const blob = await response.blob()
        const blobUrl = URL.createObjectURL(blob)

        // 检查音频时长是否有显著增加
        const checkAudio = new Audio(blobUrl)
        checkAudio.addEventListener('loadedmetadata', () => {
            const newDuration = checkAudio.duration
            const currentDuration = audio.duration || 0

            // 如果新时长显著大于当前时长（至少增加 1 秒），才重新加载
            if (newDuration > currentDuration + 1) {
                console.log(`📊 Audio duration updated: ${currentDuration.toFixed(2)}s -> ${newDuration.toFixed(2)}s`)

                // 清理旧的 blob URL
                if (audio.src && audio.src.startsWith('blob:')) {
                    URL.revokeObjectURL(audio.src)
                }

                try { audio.pause() } catch (_) {}
                audio.src = blobUrl

                const onLoaded = async () => {
                    try {
                        const durationVal = getDisplayedDuration()
                        if (durationVal) {
                            duration.value = durationVal
                        }
                        // 保持播放位置
                        audio.currentTime = Math.min(prevTime, getDisplayedDuration() || prevTime)
                        if (wasPlaying) {
                            await audio.play()
                            isPlaying.value = true
                            if (!animationFrameId) visualize()
                        } else {
                            isPlaying.value = false
                        }
                    } catch (e) {
                        console.error('Error reloading audio:', e)
                        isPlaying.value = false
                    }
                }

                audio.addEventListener('loadedmetadata', onLoaded, { once: true })
                audio.addEventListener('error', () => {
                    console.error('Error reloading audio')
                    URL.revokeObjectURL(blobUrl)
                    isPlaying.value = false
                }, { once: true })
                audio.load()
            } else {
                // 时长没有显著变化，只更新显示
                URL.revokeObjectURL(blobUrl)
                const durationVal = getDisplayedDuration()
                if (durationVal) {
                    duration.value = durationVal
                }
            }
            checkAudio.remove()
        })
        checkAudio.addEventListener('error', () => {
            console.warn('Error checking audio duration')
            URL.revokeObjectURL(blobUrl)
            checkAudio.remove()
        })
        checkAudio.load()
    } catch (error) {
        console.error('Error reloading audio:', error)
    }
}

// 生成播客
async function generatePodcast() {
    if (!input.value.trim()) {
        showAlert(t('podcast.enterLinkOrTopic'), 'warning')
        return
    }

    showStatus.value = true
    statusMsg.value = t('podcast.generating')
    statusClass.value = 'generating'
    showStopBtn.value = true
    showDownloadBtn.value = false

    // 设置生成状态
    isGenerating = true
    simulatedWaveformStartTime = null  // 重置模拟波形动画时间

    // 等待 DOM 更新
    await nextTick()

    // 重置状态
    subtitles.value = []
    subtitleTimestamps.value = []
    activeSubtitleIndex.value = -1
    audioUrl.value = ''
    mergedAudioUrl = null
    lastAudioDuration = 0
    isPlaying.value = false
    isSwitching = false
    stopAudioUpdateChecker()

    lastBytePosition = 0
    totalAudioSize = 0
    audioQueue = []
    hasLoadedMetadata = false
    analyzerInitialized = false
    // 清理跨域音频的预分析数据
    crossOriginWaveformData = null
    crossOriginWaveformDataLoaded = false
    crossOriginWaveformMin = 0
    crossOriginWaveformMax = 0
    lastAnalyzedAudioUrl = null
    if (audio) {
        audio.pause()
    }
    if (mediaSource) {
        mediaSource = null
    }
    if (sourceBuffer) {
        sourceBuffer = null
    }

    // 重置 WebAudio 状态（不再使用，但保留清理代码）
    webAudioQueue = []
    webAudioPlaying = false
    webAudioCurrentTime = 0
    webAudioStartTime = 0
    webAudioTotalDuration = 0
    webAudioSourceNodes.forEach(node => {
        try {
            node.stop()
        } catch (e) {
            // 可能已经停止
        }
    })
    webAudioSourceNodes = []
    if (webAudioTimeUpdateFrame) {
        cancelAnimationFrame(webAudioTimeUpdateFrame)
        webAudioTimeUpdateFrame = null
    }

    // 重置 MediaSource 相关状态（与 index.html 一致）
    lastBytePosition = 0
    totalAudioSize = 0
    audioQueue = []
    isInitialAudioLoadComplete = false  // 重置初始加载完成标志

    showSubtitles.value = false
    audioUserInput.value = input.value

    // 清空字幕（不再需要，因为使用响应式数据）

    // 显示播放器并初始化波形图（生成时显示模拟波形图）
    showPlayer.value = true
    await nextTick()

    // 初始化波形图
    if (waveformBars.value.length === 0) {
        initWaveform()
    }

    // 启动波形图动画（生成时显示模拟波形图）
    if (!animationFrameId) {
        visualize()
    }

    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
        // 获取 token 用于 WebSocket 认证
        const token = localStorage.getItem('accessToken')
        const wsUrl = token
            ? `${protocol}//${window.location.host}/api/v1/podcast/generate?token=${encodeURIComponent(token)}`
            : `${protocol}//${window.location.host}/api/v1/podcast/generate`
        wsConnection = new WebSocket(wsUrl)

        wsConnection.onopen = () => {
            wsConnection.send(JSON.stringify({ input: input.value }))
        }

        // 设置 WebSocket 接收二进制数据
        wsConnection.binaryType = 'arraybuffer'

        wsConnection.onmessage = async (event) => {
            // 忽略二进制数据（WAV chunk），改用 Range Request 方式
            // 这样可以保持音频连续性，避免分段播放

            // JSON 消息处理
            let message
            try {
                // 如果是二进制数据，跳过（不再使用 WebAudio chunk 方式）
                if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
                    return
                }
                message = JSON.parse(event.data)
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e, event.data)
                return
            }

            if (message.type === 'audio_update') {
                const audioData = message.data
                const currentDuration = audioData.duration || 0
                const durationChanged = currentDuration > lastAudioDuration
                lastAudioDuration = currentDuration

                if (isIOSSafari() && audioData.hls_url) {
                    window.__lastProgressiveUrl = audioData.url
                    mergedAudioUrl = audioData.hls_url
                } else {
                    mergedAudioUrl = audioData.url
                }

                if (audioData.text) {
                    const subtitleItem = {
                        text: audioData.text,
                        speaker: audioData.speaker
                    }
                    subtitles.value.push(subtitleItem)

                    if (audioData.duration !== undefined) {
                        const previousDuration = subtitleTimestamps.value.length > 0
                            ? subtitleTimestamps.value[subtitleTimestamps.value.length - 1].end
                            : 0
                        subtitleTimestamps.value.push({
                            start: previousDuration,
                            end: audioData.duration,
                            text: audioData.text,
                            speaker: audioData.speaker
                        })
                    }

                    // 如果字幕区域未显示，自动显示
                    if (!showSubtitles.value && subtitles.value.length > 0) {
                        showSubtitles.value = true
                        await nextTick()
                    }
                }

                // 使用 MediaSource 或传统 Audio 方式，通过 Range Request 追加新内容
                // 这样可以保持音频连续性，与 index.html 一致
                if (subtitles.value.length === 1 && !audio) {
                    statusMsg.value = t('podcast.preparingFirstAudio')
                    // 延迟2秒确保merged音频文件已完全写入（与 index.html 一致）
                    setTimeout(async () => {
                        if (isIOSSafari()) {
                            await loadAudio(false)
                        } else {
                            await initMediaSourceAudio(false)
                        }
                        // 显示播放器
                        showPlayer.value = true
                        await nextTick()

                        // 确保 audioElement 已绑定
                        if (audioElement.value && !audio) {
                            audio = audioElement.value
                            setupAudioEventListeners()
                        }

                        // 初始化波形图
                        if (waveformBars.value.length === 0) {
                            initWaveform()
                        }
                        statusMsg.value = t('podcast.readyWithCount', { count: subtitles.value.length })
                    }, 2000)
                } else if (subtitles.value.length > 1 && !audio) {
                    // 如果第一段错过了，第二段立即显示
                    statusMsg.value = t('podcast.preparingAudio')
                    setTimeout(async () => {
                        if (isIOSSafari()) {
                            await loadAudio(false)
                        } else {
                            await initMediaSourceAudio(false)
                        }
                        showPlayer.value = true
                        await nextTick()

                        // 确保 audioElement 已绑定
                        if (audioElement.value && !audio) {
                            audio = audioElement.value
                            setupAudioEventListeners()
                        }

                        // 初始化波形图
                        if (waveformBars.value.length === 0) {
                            initWaveform()
                        }
                        statusMsg.value = t('podcast.readyWithCount', { count: subtitles.value.length })
                    }, 2000)
                } else if (audio && durationChanged) {
                    // 音频时长已更新，立即追加新内容（无缝）
                    // 确保初始加载已完成，避免在初始加载时重复追加第一段
                    if (!isInitialAudioLoadComplete) {
                        console.log('⏸️ Skipping audio update: initial load not complete yet')
                        return
                    }

                    // 确保音频已经加载完成
                    const audioReady = audio.readyState >= HTMLMediaElement.HAVE_METADATA
                    const audioDuration = getDisplayedDuration() || audio.duration || 0
                    // 只有当音频已加载且新时长确实大于当前音频时长时才追加
                    if (audioReady && currentDuration > audioDuration + 0.5) {
                        console.log(`📊 Audio duration updated from ${audioDuration.toFixed(2)}s to ${currentDuration.toFixed(2)}s`)
                        if (mediaSource && sourceBuffer) {
                            // 桌面等支持 MSE：无缝追加
                            // 确保 lastBytePosition 已正确设置，避免重复请求
                            if (lastBytePosition > 0) {
                                switchAudioSeamlessly()
                            } else {
                                console.warn('⚠️ lastBytePosition is 0, skipping seamless switch to avoid duplicate')
                            }
                        } else {
                            // iOS 等不支持 MSE：重新加载音频并恢复位置
                            reloadAudioForIOS()
                        }
                        statusMsg.value = t('podcast.generatingStatusWithCount', { count: subtitles.value.length })
                        await nextTick()
                    } else {
                        console.log(`⏸️ Skipping audio update: audioReady=${audioReady}, currentDuration=${currentDuration.toFixed(2)}s, audioDuration=${audioDuration.toFixed(2)}s`)
                    }
                } else {
                    // 只更新状态
                    statusMsg.value = t('podcast.generatingStatusWithCount', { count: subtitles.value.length })
                    await nextTick()
                }
            } else if (message.type === 'complete') {
                statusClass.value = 'complete'
                statusMsg.value = t('podcast.completed', { count: subtitles.value.length })
                isGenerating = false  // 生成完成，停止模拟波形图动画
                simulatedWaveformStartTime = null
                await nextTick()
                wsConnection.close()
                stopAudioUpdateChecker()
                showStopBtn.value = false
                showDownloadBtn.value = true
                await nextTick()

                if (message.data && message.data.audio_url) {
                    currentAudioUrl = message.data.audio_url
                    if (message.data.timestamps_url) {
                        // 使用 apiCall 自动添加认证头
                        // 移除查询参数（如 ?t=timestamp），因为 apiCall 会处理 URL
                        const cleanTimestampsUrl = message.data.timestamps_url.split('?')[0]
                        apiCall(cleanTimestampsUrl)
                            .then(response => {
                                if (!response || !response.ok) {
                                    throw new Error(`HTTP error! status: ${response ? response.status : 'unknown'}`)
                                }
                                return response.json()
                            })
                            .then(timestamps => {
                                subtitleTimestamps.value = timestamps || []
                            })
                            .catch(err => {
                                console.warn('Failed to load subtitle timestamps:', err)
                            })
                    }
                    switchToFinalAudio(currentAudioUrl)
                }

                loadHistory()
            } else if (message.type === 'stopped') {
                statusMsg.value = t('podcast.stopped')
                isGenerating = false  // 停止生成，停止模拟波形图动画
                simulatedWaveformStartTime = null
                await nextTick()
                stopAudioUpdateChecker()
                showStopBtn.value = false
                showDownloadBtn.value = false
                statusClass.value = ''
                await nextTick()
                // 收到停止确认后，关闭 WebSocket 连接
                if (wsConnection && wsConnection.readyState !== WebSocket.CLOSED) {
                    wsConnection.close()
                }
            } else if (message.type === 'error' || message.error) {
                // 处理错误消息
                const errorMessage = message.error || message.message || t('podcast.generationFailed')
                showAlert(errorMessage, 'error')
                statusMsg.value = t('podcast.generationFailed')
                statusClass.value = ''
                isGenerating = false  // 生成失败，停止模拟波形图动画
                simulatedWaveformStartTime = null
                stopAudioUpdateChecker()
                showStopBtn.value = false
                showDownloadBtn.value = false
                await nextTick()
                // 关闭 WebSocket 连接
                if (wsConnection && wsConnection.readyState !== WebSocket.CLOSED) {
                    wsConnection.close()
                }
                return  // 提前返回，不继续处理
            }
        }

        wsConnection.onerror = (error) => {
            throw new Error('WebSocket连接错误')
        }
    } catch (error) {
        showAlert(t('podcast.generationFailed') + ': ' + error.message, 'error')
        statusMsg.value = t('podcast.generationFailed')
        statusClass.value = ''
        isGenerating = false  // 生成失败，停止模拟波形图动画
        simulatedWaveformStartTime = null
        stopAudioUpdateChecker()
        showStopBtn.value = false
        showDownloadBtn.value = false
    }
}

// 切换到最终音频
async function switchToFinalAudio(finalUrl) {
    try {
        const wasPlaying = audio && !audio.paused
        const prevTime = audio ? audio.currentTime : 0
        mergedAudioUrl = finalUrl
        if (mediaSource) { mediaSource = null }
        if (sourceBuffer) { sourceBuffer = null }
        shouldResumePlayback = !!wasPlaying
        // 优先使用 MediaSource 方式（支持无缝流式更新）
        await initMediaSourceAudio(false)
        if (!audio) return
        const onCanPlay = async () => {
            try {
                const total = getDisplayedDuration()
                if (total > 0) {
                    duration.value = total
                }
                if (prevTime > 0) {
                    audio.currentTime = Math.min(prevTime, total || prevTime)
                }
                if (wasPlaying) {
                    await audio.play()
                    isPlaying.value = true
                    if (!animationFrameId) visualize()
                } else {
                    isPlaying.value = false
                }
            } finally {
                audio.removeEventListener('canplay', onCanPlay)
            }
        }
        audio.addEventListener('canplay', onCanPlay)
    } catch (e) {
        console.error('Error switching to final audio:', e)
    }
}

// 停止生成
function stopGeneration() {
    if (wsConnection) {
        if (wsConnection.readyState === WebSocket.OPEN) {
            // 发送停止信号，但不立即关闭连接，等待后端确认
            try {
                wsConnection.send(JSON.stringify({ type: 'stop' }))
                // 设置一个超时，如果3秒内没有收到确认，则强制关闭
                const stopTimeout = setTimeout(() => {
                    if (wsConnection && wsConnection.readyState !== WebSocket.CLOSED) {
                        wsConnection.close()
                    }
                    clearTimeout(stopTimeout)
                }, 3000)
            } catch (error) {
                console.error('Error sending stop signal:', error)
                // 如果发送失败，直接关闭连接
                wsConnection.close()
            }
        } else if (wsConnection.readyState === WebSocket.CONNECTING) {
            // 如果还在连接中，直接关闭
            wsConnection.close()
        }
    }
    // 立即更新UI状态
    statusMsg.value = t('podcast.generating')
    stopAudioUpdateChecker()
    showStopBtn.value = false
    showDownloadBtn.value = false
    // 注意：statusClass 保持 'generating' 直到收到 'stopped' 消息
}

// 下载音频
function downloadAudio() {
    // 优先使用 sessionAudioUrl（详情模式），然后是 currentAudioUrl（生成完成），最后是 mergedAudioUrl（生成中）
    // 如果都没有，尝试使用 audioUrl.value（响应式音频 URL）
    const urlToDownload = sessionAudioUrl || currentAudioUrl || mergedAudioUrl || audioUrl.value
    if (urlToDownload) {
        const link = document.createElement('a')
        link.href = addCacheBustingParam(urlToDownload);
        link.download = 'podcast.mp3'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    } else {
        showAlert(t('podcast.noAudioToDownload'), 'warning')
    }
}

// 应用到数字人（参考 useTemplate 的实现方式）
async function applyToDigitalHuman() {
    // 优先使用当前会话的音频 URL，否则使用生成过程中的音频 URL
    const audioUrl = sessionAudioUrl || currentAudioUrl || mergedAudioUrl
    console.log('Applying to digital human, audioUrl:', audioUrl)

    if (!audioUrl) {
        showAlert(t('podcast.pleaseGenerateFirst'), 'warning')
        return
    }

    try {
        // 先设置任务类型为 s2v（语音驱动）
        selectedTaskId.value = 's2v'

        // 获取当前表单
        const currentForm = getCurrentForm()

        // 立即切换到创建视图并展开创作区域（参考 useTemplate）
        isCreationAreaExpanded.value = true
        switchToCreateView()

        // 异步加载音频文件
        try {
            // 使用 apiCall 获取音频（支持认证）
            const response = await apiCall(audioUrl)
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }
            const blob = await response.blob()

            // 根据文件扩展名确定正确的MIME类型
            let mimeType = blob.type
            if (!mimeType || mimeType === 'application/octet-stream') {
                // 从 URL 中提取扩展名
                const urlPath = audioUrl.split('?')[0]
                const ext = urlPath.toLowerCase().split('.').pop()
                const mimeTypes = {
                    'mp3': 'audio/mpeg',
                    'wav': 'audio/wav',
                    'mp4': 'audio/mp4',
                    'aac': 'audio/aac',
                    'ogg': 'audio/ogg',
                    'm4a': 'audio/mp4'
                }
                mimeType = mimeTypes[ext] || 'audio/mpeg'
            }

            const filename = `podcast_${Date.now()}.${mimeType.split('/')[1] || 'mp3'}`
            const file = new File([blob], filename, { type: mimeType })

            // 设置音频文件到表单
            currentForm.audioFile = file

            // 使用 FileReader 生成 data URL 作为预览（参考 useTemplate）
            const reader = new FileReader()
            reader.onload = (e) => {
                const audioDataUrl = e.target.result
                setCurrentAudioPreview(audioDataUrl)
                console.log('播客音频预览已设置')

                // 触发音频上传处理（用于音频分离等）
                const fileList = new DataTransfer()
                fileList.items.add(file)
                const event = {
                    target: {
                        files: fileList.files
                    }
                }
                handleAudioUpload(event)
            }
            reader.readAsDataURL(file)

            showAlert(t('podcast.applySuccess'), 'success')
        } catch (error) {
            console.error('加载播客音频失败:', error)
            showAlert(t('podcast.loadAudioFailed'), 'error')
        }
    } catch (error) {
        console.error('应用到数字人失败:', error)
        showAlert(t('podcast.applyFailed') + ': ' + error.message, 'error')
    }
}

// 存储历史会话的详细信息（字幕、时间戳等）
const historySessionData = new Map()

// 加载历史记录（只获取并存储历史数据）
async function loadHistory() {
    try {
        try {
            loadingHistory.value = true
        } catch (e) {
            console.warn('Error setting loadingHistory:', e)
            return
        }

        // 使用 apiCall 自动添加认证头，调用任务接口获取历史列表
        const response = await apiCall('/api/v1/podcast/history')
        if (!response || !response.ok) {
            const errorText = response ? `HTTP error! status: ${response.status}` : 'Network error'
            throw new Error(errorText)
        }
        const data = await response.json()

        // 清空历史数据
        historySessionData.clear()

        // 存储每个会话的完整数据到 Map（用于详情页加载）
        // 同时准备用于 template 渲染的列表数据
        const items = []
        if (data.sessions && Array.isArray(data.sessions)) {
            data.sessions.forEach((session, index) => {
                if (session.session_id) {
                    // 存储到 Map（用于详情页）
                    historySessionData.set(session.session_id, {
                        rounds: session.rounds || [],
                        subtitles: session.subtitles || [],
                        timestamps: session.timestamps || [],
                        user_input: session.user_input || '',
                        outputs: session.outputs || session.extra_info?.outputs || null,
                        has_audio: session.has_audio || false
                    })

                    // 添加到列表（用于 template 渲染）
                    items.push({
                        session_id: session.session_id,
                        user_input: session.user_input || '',
                        has_audio: session.has_audio || false,
                        displayText: (session.user_input || `会话 ${index + 1}`).length > 40
                            ? (session.user_input || `会话 ${index + 1}`).substring(0, 40) + '...'
                            : (session.user_input || `会话 ${index + 1}`)
                    })
                }
            })
        }

        // 更新响应式变量，template 会自动渲染
        try {
            historyItems.value = items
            loadingHistory.value = false
        } catch (e) {
            console.warn('Error setting historyItems:', e)
            try {
                loadingHistory.value = false
            } catch (e2) {
                console.warn('Error setting loadingHistory:', e2)
            }
        }
    } catch (error) {
        console.error('Error loading history:', error)
        try {
            historyItems.value = []
            loadingHistory.value = false
        } catch (e) {
            console.warn('Error setting historyItems:', e)
        }
    }
}

// 加载特定 session 的详细信息（从历史数据中获取，然后通过 outputs 获取音频 URL）
async function loadSessionDetail(sessionId) {
    try {
        try {
            loadingSessionDetail.value = true
        } catch (e) {
            console.warn('Error setting loadingSessionDetail:', e)
            return
        }

        // 停止当前播放
        if (audio) {
            try {
            audio.pause()
            } catch (e) {
                console.warn('Error pausing audio:', e)
            }
        }

        // 清空和重置状态
        try {
            audioUrl.value = ''
            subtitles.value = []
            subtitleTimestamps.value = []
            activeSubtitleIndex.value = -1
        showPlayer.value = false
        isPlaying.value = false
        currentTime.value = 0
        duration.value = 0
        progress.value = 0
        audioUserInput.value = ''
        } catch (e) {
            console.warn('Error resetting state:', e)
        }

        sessionAudioUrl = null
        hasLoadedMetadata = false
        analyzerInitialized = false
        // 重置音频分析器相关状态
        if (mediaElementSource) {
            mediaElementSource = null
        }
        if (analyser) {
            analyser = null
        }
        if (audioContext && audioContext.state !== 'closed') {
            try {
                audioContext.close()
            } catch (e) {
                console.warn('Error closing audio context:', e)
            }
            audioContext = null
        }
        // 清理跨域音频的预分析数据
        crossOriginWaveformData = null
        crossOriginWaveformDataLoaded = false
        crossOriginWaveformMin = 0
        crossOriginWaveformMax = 0
        lastAnalyzedAudioUrl = null

        // 重置波形图
        waveformBars.value = []

        await nextTick()

        // 从历史数据中获取会话信息
        const sessionData = historySessionData.get(sessionId)
        if (!sessionData) {
            // 如果历史数据中没有，显示错误
            try {
                showAlert(t('podcast.sessionDataNotFound'), 'error')
                loadingSessionDetail.value = false
            } catch (e) {
                console.warn('Error showing alert:', e)
            }
            return
        }

        // 设置用户输入和字幕数据
        try {
            audioUserInput.value = sessionData.user_input || ''

            // 使用 rounds 数据（优先级最高）
            if (sessionData.rounds && sessionData.rounds.length > 0) {
                subtitles.value = []
                subtitleTimestamps.value = []
                sessionData.rounds.forEach((round) => {
                    subtitles.value.push({
                        text: round.text || '',
                        speaker: round.speaker || ''
                    })
                    subtitleTimestamps.value.push({
                        start: round.start || 0.0,
                        end: round.end || 0.0,
                        text: round.text || '',
                        speaker: round.speaker || ''
                    })
                })
            } else if (sessionData.subtitles && sessionData.subtitles.length > 0) {
                subtitles.value = [...sessionData.subtitles]
            }

            if (sessionData.timestamps && sessionData.timestamps.length > 0) {
                subtitleTimestamps.value = [...sessionData.timestamps]
            }
        } catch (e) {
            console.warn('Error setting session data:', e)
            try {
                loadingSessionDetail.value = false
            } catch (e2) {
                console.warn('Error setting loadingSessionDetail:', e2)
            }
            return
        }

        // 在详情模式下，显示播放器区域（只要有数据就显示）
        if (isDetailMode.value) {
            try {
                console.log('Setting up player and subtitles:', {
                    isDetailMode: isDetailMode.value,
                    subtitlesCount: subtitles.value.length
                })

                // 如果有字幕数据，显示播放器和字幕
                if (subtitles.value.length > 0) {
                    showPlayer.value = true
                    await nextTick()

                    showSubtitles.value = true
                    await nextTick()
                    console.log('Player and subtitles shown:', {
                        showPlayer: showPlayer.value,
                        showSubtitles: showSubtitles.value
                    })
                } else {
                    // 即使没有字幕，也先显示播放器（音频 URL 会在后面设置）
                    showPlayer.value = true
                    await nextTick()
                    console.log('Player shown (no subtitles):', {
                        showPlayer: showPlayer.value
                    })
                }
            } catch (e) {
                console.warn('Error showing player/subtitles:', e)
            }
        } else {
            console.log('Not showing player:', {
                isDetailMode: isDetailMode.value
            })
        }

        try {
            // 调用 API 获取音频 URL（API 会从 outputs 或数据库获取路径并转换为 CDN URL）
            const audioUrlResponse = await apiCall(`/api/v1/podcast/session/${sessionId}/audio_url`)
            if (!audioUrlResponse || !audioUrlResponse.ok) {
                throw new Error(`Failed to get audio URL: ${audioUrlResponse ? audioUrlResponse.status : 'unknown'}`)
            }

            const audioUrlData = await audioUrlResponse.json()
            sessionAudioUrl = audioUrlData.audio_url

            if (!sessionAudioUrl) {
                throw new Error('Audio URL not found')
            }

            // 设置音频 URL
            try {
                if (sessionAudioUrl.startsWith('http://') || sessionAudioUrl.startsWith('https://')) {
                    audioUrl.value = sessionAudioUrl
                } else {
                    audioUrl.value = addCacheBustingParam(sessionAudioUrl)
                }
            } catch (e) {
                console.warn('Error setting audioUrl:', e)
            }

            await nextTick()

            // 确保播放器已显示（在详情模式下，只要有音频 URL 就应该显示）
            if (isDetailMode.value && !showPlayer.value) {
                try {
                    showPlayer.value = true
                    await nextTick()
                } catch (e) {
                    console.warn('Error showing player:', e)
                }
            }

            // 如果有字幕但字幕区域未显示，显示字幕区域
            if (isDetailMode.value && subtitles.value.length > 0 && !showSubtitles.value) {
                try {
                    showSubtitles.value = true
                    await nextTick()
                } catch (e) {
                    console.warn('Error showing subtitles:', e)
                }
            }

            // 初始化 audio 元素和事件监听器（详情页需要）
            // 等待 audioUrl 设置完成后再初始化
            await nextTick()

            // 确保 audioElement 已经渲染
            if (!audioElement.value) {
                console.warn('audioElement not available, waiting...')
                await nextTick()
            }

            if (audioElement.value) {
                // 重新绑定 audio 变量（确保使用最新的元素）
                audio = audioElement.value
                // 确保音量设置为 1.0（重要：避免无声）
                audio.volume = 1.0
                setupAudioEventListeners()

                // 确保音频元素已加载
                // 如果 audioUrl 已设置但音频还没有开始加载，触发加载
                if (audioUrl.value) {
                    try {
                        // 如果 src 已经设置但 readyState 还是 HAVE_NOTHING，强制重新加载
                        if (audio.src !== audioUrl.value) {
                            audio.src = audioUrl.value
                        }
                        if (audio.readyState === HTMLMediaElement.HAVE_NOTHING) {
                            audio.load()
                            console.log('Audio element load() called, src:', audioUrl.value.substring(0, 100))
                        } else {
                            console.log('Audio already loaded, readyState:', audio.readyState)
                            // 如果已经加载，手动触发元数据加载事件
                            if (audio.readyState >= HTMLMediaElement.HAVE_METADATA) {
                                onAudioLoadedMetadata()
                            }
                        }
                    } catch (e) {
                        console.warn('Error calling audio.load():', e)
                    }
                } else {
                    console.warn('audioUrl.value is empty')
                }
            } else {
                console.error('audioElement.value is still null after nextTick')
            }

            // 初始化音频上下文恢复（详情页需要，避免重复添加）
            if (!window.__podcastResumeContextOnGesture) {
                window.__podcastResumeContextOnGesture = function resumeContextOnGesture() {
                    if (audioContext && audioContext.state === 'suspended') {
                        audioContext.resume().catch(() => {})
                    }
                    document.removeEventListener('touchend', window.__podcastResumeContextOnGesture)
                    document.removeEventListener('click', window.__podcastResumeContextOnGesture)
                }
                document.addEventListener('touchend', window.__podcastResumeContextOnGesture, { passive: true })
                document.addEventListener('click', window.__podcastResumeContextOnGesture)
            }

            // 添加全局鼠标事件监听器（进度条拖拽，详情页需要，避免重复添加）
            if (!window.__podcastProgressListenersAdded) {
                document.addEventListener('mousemove', onProgressMouseMove)
                document.addEventListener('mouseup', onProgressMouseUp)
                document.addEventListener('touchmove', onProgressTouchMove, { passive: true })
                document.addEventListener('touchend', onProgressTouchEnd, { passive: true })
                window.__podcastProgressListenersAdded = true
            }

            // 初始化波形图（确保在音频元素准备好后初始化）
            await nextTick()
            if (waveformBars.value.length === 0) {
                initWaveform()
                console.log('Waveform initialized, bars count:', waveformBars.value.length)
            }

            // 如果音频已经可以播放，立即初始化分析器
            if (audio && audio.readyState >= HTMLMediaElement.HAVE_METADATA && !analyzerInitialized) {
                try {
                    await initAudioAnalyzer()
                    analyzerInitialized = true
                    console.log('Audio analyzer initialized in loadSessionDetail')
                } catch (e) {
                    console.warn('Error initializing audio analyzer in loadSessionDetail:', e)
                }
            }
            } catch (error) {
            try {
                console.error('Error loading audio:', error)
                showAlert(t('podcast.loadAudioFailedDetail'), 'error')
            } catch (e) {
                console.warn('Error in error handler:', e)
            }
        }
        } catch (error) {
        try {
            console.error('Error loading session detail:', error)
            showAlert(t('podcast.loadSessionFailed'), 'error')
        } catch (e) {
            console.warn('Error in error handler:', e)
        }
        } finally {
        try {
            loadingSessionDetail.value = false
        } catch (e) {
            console.warn('Error setting loadingSessionDetail in finally:', e)
        }
        }
    }

// 切换侧边栏
function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
}


// 用户滚动字幕
function handleUserScroll() {
    autoFollowSubtitles = false
    if (userScrollTimeout) {
        clearTimeout(userScrollTimeout)
    }
    userScrollTimeout = setTimeout(() => {
        autoFollowSubtitles = true
        userScrollTimeout = null
    }, 5000)
}

// 回车键生成
function onInputKeyPress(e) {
    if (e.key === 'Enter') {
        generatePodcast()
    }
}

// 点击示例输入
async function onExampleClick(example) {
    // 先设置输入框的值
    input.value = example
    // 等待 Vue 响应式更新完成
    await nextTick()
    // 然后开始生成
    generatePodcast()
}

// 监听路由变化
watch(() => route.params.session_id, async (newSessionId, oldSessionId) => {
    // 如果是 immediate 调用且组件还未完全挂载，等待一下
    if (!subtitleSection.value) {
        // 等待 DOM 更新
        await nextTick()
    }

    if (newSessionId) {
        // 详情模式
        try {
            currentSessionId.value = newSessionId
            isDetailMode.value = true
        } catch (e) {
            console.warn('Error setting detail mode:', e)
            return
        }

        // 如果历史数据为空，先加载历史数据（刷新页面时的情况）
        if (historySessionData.size === 0) {
            await loadHistory()
            // 等待历史数据加载完成
            await nextTick()
        }

        // 加载会话详情
        await loadSessionDetail(newSessionId)

        // 重新加载历史记录以更新高亮状态
        try {
            await loadHistory()
        } catch (e) {
            console.warn('Error loading history:', e)
        }
    } else {
        // 列表模式
        try {
        currentSessionId.value = null
        isDetailMode.value = false
        } catch (e) {
            console.warn('Error setting list mode:', e)
            return
        }

        // 停止播放并重置状态
        if (audio) {
            try {
            audio.pause()
            } catch (e) {
                console.warn('Error pausing audio:', e)
            }
            audio = null
        }

        // 移除音频相关的事件监听器（列表模式不需要）
        if (window.__podcastResumeContextOnGesture) {
            try {
                document.removeEventListener('touchend', window.__podcastResumeContextOnGesture)
                document.removeEventListener('click', window.__podcastResumeContextOnGesture)
            } catch (e) {
                console.warn('Error removing resume context listeners:', e)
            }
            window.__podcastResumeContextOnGesture = null
        }

        if (window.__podcastProgressListenersAdded) {
            try {
                document.removeEventListener('mousemove', onProgressMouseMove)
                document.removeEventListener('mouseup', onProgressMouseUp)
                document.removeEventListener('touchmove', onProgressTouchMove)
                document.removeEventListener('touchend', onProgressTouchEnd)
            } catch (e) {
                console.warn('Error removing progress listeners:', e)
            }
            window.__podcastProgressListenersAdded = false
        }

        // 安全地重置所有响应式状态
        try {
        showPlayer.value = false
        isPlaying.value = false
        currentTime.value = 0
        duration.value = 0
        progress.value = 0
        audioUserInput.value = ''
                audioUrl.value = ''  // 清空音频 URL
                subtitles.value = []  // 清空字幕
                subtitleTimestamps.value = []  // 清空时间戳
                activeSubtitleIndex.value = -1  // 重置激活字幕索引
                showSubtitles.value = false  // 隐藏字幕
        } catch (e) {
            console.warn('Error resetting state:', e)
        }

        sessionAudioUrl = null  // 清空会话音频 URL
    }
}, { immediate: true })  // 改为 true，确保在组件挂载时也处理路由参数

// 监听路由路径变化，管理历史记录刷新定时器
watch(() => route.path, (newPath, oldPath) => {
    const isPodcastGenerateRoute = () => {
        return route.path.startsWith('/podcast_generate')
    }

    // 如果进入 podcast_generate 路由，启动定时器
    if (isPodcastGenerateRoute() && !window.__podcastHistoryInterval) {
        const historyInterval = setInterval(() => {
            // 再次检查路由，如果不在 podcast_generate 路由下，清除定时器
            if (isPodcastGenerateRoute()) {
                loadHistory()
            } else {
                if (window.__podcastHistoryInterval) {
                    clearInterval(window.__podcastHistoryInterval)
                    window.__podcastHistoryInterval = null
                }
            }
        }, 60000)

        window.__podcastHistoryInterval = historyInterval
    }
    // 如果离开 podcast_generate 路由，清除定时器
    else if (!isPodcastGenerateRoute() && window.__podcastHistoryInterval) {
        clearInterval(window.__podcastHistoryInterval)
        window.__podcastHistoryInterval = null
    }
}, { immediate: true })

onMounted(async () => {
    // 等待 DOM 完全挂载
    await nextTick()

    // 小屏幕默认折叠侧边栏
    if (window.matchMedia && window.matchMedia('(max-width: 768px)').matches) {
        sidebarCollapsed.value = true
    }

    // 加载历史记录（主页侧边栏需要，详情页也需要历史数据）
    await loadHistory()

    // 如果 URL 中有 session_id，watch 回调会处理（immediate: true）
    // 但为了确保在刷新时能正确加载，这里也检查一下
    const sessionId = route.params.session_id
    if (sessionId && !isDetailMode.value) {
        // 如果 watch 没有处理（可能因为时机问题），手动处理
        await nextTick()
        if (route.params.session_id === sessionId && !isDetailMode.value) {
            currentSessionId.value = sessionId
            isDetailMode.value = true
            // 历史数据已经加载，直接加载详情
            await loadSessionDetail(sessionId)
            // 重新加载历史记录以更新高亮状态
            await loadHistory()
        }
    }

    // 注意：历史记录刷新定时器由路由监听器管理（watch route.path）
    // 只有在 podcast_generate 路由下时才会启动定时器
})

// 清理函数 - 必须在 onMounted 外部定义
onBeforeUnmount(() => {
    // 组件即将卸载，清理工作由 onUnmounted 处理
})

    onUnmounted(() => {
    // 清理定时器
    if (window.__podcastHistoryInterval) {
        clearInterval(window.__podcastHistoryInterval)
        window.__podcastHistoryInterval = null
    }

    // 清理滚动节流定时器
    if (scrollThrottleTimer) {
        clearTimeout(scrollThrottleTimer)
        scrollThrottleTimer = null
    }

        stopAudioUpdateChecker()

        // 清理进度条拖拽事件监听器
    try {
        document.removeEventListener('mousemove', onProgressMouseMove)
        document.removeEventListener('mouseup', onProgressMouseUp)
        document.removeEventListener('touchmove', onProgressTouchMove)
        document.removeEventListener('touchend', onProgressTouchEnd)
    } catch (e) {
        console.warn('Error removing progress event listeners:', e)
    }

    // 字幕事件监听器现在通过 template 绑定，Vue 会自动清理

    // 清理音频相关资源
        if (audio) {
        try {
            audio.pause()
            if (audio.src && audio.src.startsWith('blob:')) {
                URL.revokeObjectURL(audio.src)
            }
        } catch (e) {
            console.warn('Error cleaning up audio:', e)
        }
            audio = null
        }

    if (wsConnection && wsConnection.readyState !== WebSocket.CLOSED) {
        try {
            wsConnection.close()
        } catch (e) {
            console.warn('Error closing WebSocket:', e)
        }
    }

        if (animationFrameId) {
        try {
            cancelAnimationFrame(animationFrameId)
        } catch (e) {
            console.warn('Error canceling animation frame:', e)
        }
        animationFrameId = null
    }

        if (audioContext && audioContext.state !== 'closed') {
        try {
            audioContext.close()
        } catch (e) {
            console.warn('Error closing audio context:', e)
        }
    }

    // 清理 WebAudio 相关资源
    if (webAudioContext && webAudioContext.state !== 'closed') {
        try {
            webAudioContext.close()
        } catch (e) {
            console.warn('Error closing webAudio context:', e)
        }
    }

    if (webAudioTimeUpdateFrame) {
        try {
            cancelAnimationFrame(webAudioTimeUpdateFrame)
        } catch (e) {
            console.warn('Error canceling webAudio time update frame:', e)
        }
        webAudioTimeUpdateFrame = null
    }
})
</script>

<template>
    <div class="bg-[#f5f5f7] dark:bg-[#000000] transition-colors duration-300 w-full h-full flex flex-col overflow-hidden">
        <!-- TopBar -->
        <topMenu />

        <div class="flex flex-col sm:flex-row flex-1 w-full overflow-hidden">
            <!-- 侧边栏历史记录 -->
            <div
                class="w-full max-h-[300px] order-[-1] bg-white dark:bg-[#111] rounded-xl p-5 overflow-hidden flex flex-col transition-all duration-300 ease-in-out relative flex-shrink-0 sm:w-[300px] sm:max-h-none sm:order-none border border-black/6 dark:border-white/8"
                :class="{
                    'h-16 p-5 w-full sm:w-[50px] sm:p-5 sm:pl-2.5 sm:pr-2.5': sidebarCollapsed
                }"
                ref="sidebar">
                <!-- 加号按钮（生成播客） -->
                <button
                    class="absolute top-[15px] right-[50px] bg-black/6 dark:bg-white/10 border border-black/12 dark:border-white/20 rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 text-[#1d1d1f] dark:text-white text-sm z-[3] hover:bg-black/10 dark:hover:bg-white/20 sm:top-[15px] sm:right-[50px]"
                    :class="{
                        'right-[50px] sm:right-[50px]': sidebarCollapsed
                    }"
                    @click="router.push('/podcast_generate')"
                    :title="t('podcast.generatePodcast')"
                    :aria-label="t('podcast.generatePodcast')">
                    <svg
                        class="w-4 h-4 fill-current"
                        viewBox="0 0 24 24"
                        aria-hidden="true">
                        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
                    </svg>
                </button>
                <!-- 折叠按钮 -->
                <button
                    class="absolute top-[15px] right-[15px] bg-black/6 dark:bg-white/10 border border-black/12 dark:border-white/20 rounded-md w-7 h-7 flex items-center justify-center cursor-pointer transition-all duration-200 text-[#1d1d1f] dark:text-white text-sm z-[3] hover:bg-black/10 dark:hover:bg-white/20 sm:top-[15px] sm:right-[15px]"
                    :class="{
                        'right-2.5 sm:right-2.5': sidebarCollapsed
                    }"
                    @click="toggleSidebar"
                    :title="t('podcast.toggleSidebar')"
                    :aria-label="t('podcast.toggleSidebar')">
                    <svg
                        class="w-4 h-4 transition-transform duration-200 ease-in-out fill-current"
                        :class="{
                            'rotate-180': sidebarCollapsed
                        }"
                        viewBox="0 0 24 24"
                        aria-hidden="true">
                        <path d="M7.41 15.41L12 10.83l4.59 4.58L18 14l-6-6-6 6z"></path>
                    </svg>
                </button>
                <h3
                    class="text-base font-semibold mb-4 text-[#1d1d1f] dark:text-white pr-20 sticky top-0 z-[2] block">{{ t('podcast.historyTitle') }}</h3>
                <div
                    class="overflow-y-auto flex-1 [-webkit-overflow-scrolling:touch] [scrollbar-width:thin] [scrollbar-color:#d1d1d6_#f5f5f7] dark:[scrollbar-color:#333_#0a0a0a] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-[#f5f5f7] dark:[&::-webkit-scrollbar-track]:bg-[#0a0a0a] [&::-webkit-scrollbar-track]:rounded-sm [&::-webkit-scrollbar-thumb]:bg-[#d1d1d6] dark:[&::-webkit-scrollbar-thumb]:bg-[#333] [&::-webkit-scrollbar-thumb]:rounded-sm relative"
                    :class="{
                        'hidden sm:hidden': sidebarCollapsed
                    }">
                    <!-- Loading 覆盖层（加载历史任务时）- 在侧边栏内 -->
                    <div
                        v-if="loadingHistory"
                        class="absolute inset-0 flex items-center justify-center z-50 bg-white/80 dark:bg-[#111]/80 backdrop-blur-sm">
                        <Loading />
                    </div>

                    <!-- 历史记录列表 -->
                    <template v-if="!loadingHistory">
                        <div v-if="historyItems.length === 0" class="text-[#86868b] dark:text-[#666] text-[13px] text-center p-5">
                            {{ t('podcast.noHistory') }}
                        </div>
                        <div
                            v-for="(item, index) in historyItems"
                            :key="item.session_id || index"
                            @click="router.push(`/podcast_generate/${item.session_id}`)"
                            :class="[
                                'p-3 bg-white dark:bg-[#1a1a1a] rounded-lg mb-2 cursor-pointer transition-all duration-200 border',
                                currentSessionId === item.session_id
                                    ? 'border-[#1d1d1f] dark:border-white'
                                    : 'border-black/6 dark:border-transparent',
                                'hover:bg-[#f5f5f7] dark:hover:bg-[#222] hover:border-black/12 dark:hover:border-[#444]'
                            ]">
                            <div class="text-[13px] text-[#1d1d1f] dark:text-white mb-1 overflow-hidden text-ellipsis whitespace-nowrap">
                                {{ item.displayText }}
                            </div>
                            <div class="text-[11px] text-[#86868b] dark:text-[#666]">
                                {{ item.has_audio ? t('podcast.completedStatus') : t('podcast.generatingStatus') }}
                            </div>
                        </div>
                    </template>
                </div>
            </div>
            <!-- 内容区域包装器 -->
            <div class="flex-1 flex flex-col min-w-0 overflow-y-auto main-scrollbar w-full h-full">
                <!-- 主内容区 -->
                <div class="h-full w-full max-w-[1000px] mx-auto flex flex-col items-center justify-center py-10 px-4 relative" style="width: 100%; max-width: 1000px;">

                    <!-- 返回主页按钮 - 右上角 -->
                    <button
                        v-if="!isDetailMode"
                        @click="switchToCreateView()"
                        class="absolute top-4 right-4 px-4 py-2 bg-white dark:bg-[#111] border border-black/12 dark:border-white/20 rounded-full text-sm font-medium cursor-pointer transition-all duration-300 flex items-center gap-2 hover:opacity-90 hover:scale-105 [-webkit-appearance:none] [appearance:none] leading-none hover:bg-black/4 dark:hover:bg-white/8 backdrop-blur-sm z-10"
                        :title="t('goToHome')">
                        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" style="fill: currentColor;">
                            <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>
                        </svg>
                        <span>{{ t('goToHome') }}</span>
                    </button>

                    <!-- 列表模式：显示输入框和示例输入 -->
                    <template v-if="!isDetailMode">
                        <div class="text-center mb-10 w-full relative">
                        <h1 class="text-[32px] font-light mb-2 sm:text-[20px] md:text-[32px] lg:text-[32px]">{{ t('podcast.title') }}</h1>
                        <p class="text-[20px] text-[#888] font-light">{{ t('podcast.subtitle') }}</p>
                    </div>
                    <div
                        class="text-lg text-[#86868b] dark:text-[#888] mb-5 flex items-center justify-center gap-3 flex-nowrap"
                        ref="statusText">
                        <button
                            class="bg-black/6 dark:bg-white/20 rounded-full w-[40px] h-[40px] flex items-center justify-center cursor-pointer transition-all duration-300 text-base [-webkit-appearance:none] [appearance:none] leading-none border-none hover:scale-110 text-[#ff4444] dark:text-[#ff4444]"
                            v-show="showStopBtn"
                            @click="stopGeneration"
                            ref="stopBtn"
                            :title="t('podcast.stopGeneration')">
                            <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" style="fill: currentColor;">
                                <path d="M6 6h12v12H6z"></path>
                            </svg>
                        </button>
                        <button
                            class="bg-black/6 dark:bg-white/20 rounded-full w-[40px] h-[40px] flex items-center justify-center cursor-pointer transition-all duration-300 text-base [-webkit-appearance:none] [appearance:none] leading-none border-none hover:scale-110 text-[#1d1d1f] dark:text-white"
                            v-show="showDownloadBtn"
                            @click="downloadAudio"
                            ref="downloadBtn"
                            :title="t('podcast.downloadAudio')">
                            <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" style="fill: currentColor;">
                                <path d="M5 20h14v-2H5v2zm7-18v12l5-5 1.41 1.41L12 17.83 4.59 10.41 6 9l5 5V2h1z"/>
                            </svg>
                        </button>
                        <span ref="statusMessage" class="flex-1">{{ statusMsg }}</span>
                    </div>
                </template>

                                        <!-- 详情模式：显示返回按钮和操作按钮 -->
                                        <template v-if="isDetailMode">
                            <div class="flex justify-between w-full">
                        <button
                                class="mb-3 px-4 py-2 text-black bg-white rounded-full text-sm font-medium cursor-pointer transition-all duration-300 flex items-center gap-2 hover:opacity-90 hover:scale-105 [-webkit-appearance:none] [appearance:none] leading-none border mr-auto backdrop-blur-sm"
                                @click="router.push('/podcast_generate')"
                                ref="backToGenerateBtn"
                                :title="t('podcast.generateMore')">
                                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" style="fill: currentColor;">
                                    <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L11.83 12z"/>
                                </svg>
                                <span>{{ t('podcast.generateMore') }}</span>
                            </button>
                                                     <div
                                class="text-lg text-[#86868b] dark:text-[#888] mb-5 flex items-center justify-center gap-3 flex-nowrap"
                                ref="statusText">
                                <button
                                    class="bg-black/6 dark:bg-white/20 rounded-full w-[40px] h-[40px] flex items-center justify-center cursor-pointer transition-all duration-300 text-base [-webkit-appearance:none] [appearance:none] leading-none border-none hover:scale-110 text-[#1d1d1f] dark:text-white"
                                    @click="downloadAudio"
                                    ref="downloadBtn"
                                    :title="t('podcast.downloadAudio')">
                                    <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" style="fill: currentColor;">
                                        <path d="M5 20h14v-2H5v2zm7-18v12l5-5 1.41 1.41L12 17.83 4.59 10.41 6 9l5 5V2h1z"/>
                                    </svg>
                                </button>
                            </div>
                        <button
                                class="mb-5 px-4 py-2 text-black bg-white rounded-full text-sm font-medium cursor-pointer transition-all duration-300 flex items-center gap-2 hover:opacity-90 hover:scale-105 [-webkit-appearance:none] [appearance:none] leading-none border ml-auto backdrop-blur-sm"
                                @click="applyToDigitalHuman"
                                ref="applyBtn"
                                :title="t('podcast.applyToDigitalHuman')">
                                <span>{{ t('podcast.applyToDigitalHuman') }}</span>
                                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" style="fill: currentColor;">
                                    <path d="M8.59 16.59L13.17 12 8.59 7.41 10 6l6 6-6 6z"/>
                                </svg>
                            </button>
                            </div>
                        </template>
                    <!-- 播放器和字幕（详情模式或生成中时显示） -->
                    <template v-if="isDetailMode || showPlayer">

                    <div class="mb-5 w-full" v-show="showPlayer" ref="playerSection">
                        <div class="bg-white dark:bg-[#111] rounded-xl p-6 mb-6 w-full border border-black/6 dark:border-white/8 shadow-sm dark:shadow-none">
                            <div class="mb-1.5 text-s text-[#86868b] dark:text-[#888] opacity-90 flex items-center justify-center text-center whitespace-nowrap overflow-hidden text-ellipsis" ref="audioUserInputEl" :title="audioUserInput">{{ audioUserInput }}</div>
                            <div class="flex items-center gap-4 mb-4">
                                <button
                                    class="w-12 h-12 rounded-full bg-[#1d1d1f] dark:bg-white text-white dark:text-black border-none cursor-pointer flex items-center justify-center text-xl transition-all duration-300 hover:scale-105 [-webkit-appearance:none] [appearance:none] leading-none"
                                    :class="isPlaying ? 'playing' : 'paused'"
                                    @click="togglePlayback"
                                    ref="playBtn">
                                    <svg
                                        class="w-[22px] h-[22px] fill-current"
                                        :class="isPlaying ? 'hidden' : 'block'"
                                        viewBox="0 0 24 24"
                                        aria-hidden="true">
                                        <path d="M8 5v14l11-7z"></path>
                                    </svg>
                                    <svg
                                        class="w-[22px] h-[22px] fill-current"
                                        :class="isPlaying ? 'block' : 'hidden'"
                                        viewBox="0 0 24 24"
                                        aria-hidden="true">
                                        <path d="M6 5h4v14H6zM14 5h4v14h-4z"></path>
                                    </svg>
                                </button>
                                <div
                                    class="flex-1 relative h-2 bg-[#d1d1d6] dark:bg-[#333] rounded cursor-pointer group"
                                    ref="progressContainer"
                                    @mousedown="onProgressMouseDown"
                                    @touchstart="onProgressTouchStart">
                                    <div
                                        class="h-full bg-[#1d1d1f] dark:bg-white rounded transition-[width] duration-100 relative group-hover:bg-[#000] dark:group-hover:bg-[#ccc]"
                                        ref="progressBar"
                                        :style="{ width: progress + '%' }">
                                    </div>
                                    <div class="absolute top-1/2 left-0 -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-[#1d1d1f] dark:bg-white rounded-full opacity-0 transition-opacity duration-200 group-hover:opacity-100"></div>
                                </div>
                            </div>
                            <div
                                class="h-[80px] bg-[#f5f5f7] dark:bg-[#0a0a0a] rounded-lg mb-3 flex items-center justify-center gap-0.5 p-2.5 relative z-[1] overflow-hidden opacity-100"
                                ref="waveform">
                                <div
                                    v-for="bar in waveformBars"
                                    :key="bar.id"
                                    class="w-1 min-h-1 rounded-sm transition-[height] duration-3000 ease-out flex-shrink-0 relative z-[2] block visible opacity-100"
                                    :style="{
                                        height: bar.height + 'px',
                                        background: bar.isDark
                                            ? `linear-gradient(to top, #fff ${bar.intensity}%, #666 0%)`
                                            : `linear-gradient(to top, #1d1d1f ${bar.intensity}%, #d1d1d6 0%)`
                                    }">
                                </div>
                            </div>
                            <div class="flex justify-between text-xs text-[#86868b] dark:text-[#888]">
                                <span ref="currentTimeEl">{{ formatTime(currentTime) }}</span>
                                <span ref="durationEl">{{ formatTime(duration) }}</span>
                                </div>
                                <!-- 音频元素（隐藏，用于播放控制） -->
                                <audio
                                    ref="audioElement"
                                    :src="audioUrl"
                                    @loadedmetadata="onAudioLoadedMetadata"
                                    @canplay="onAudioCanPlay"
                                    @timeupdate="onAudioTimeUpdate"
                                    @ended="onAudioEnded"
                                    @play="onAudioPlay"
                                    @pause="onAudioPause"
                                    @error="onAudioError"
                                    preload="auto"
                                    class="hidden">
                                </audio>
                            </div>
                        </div>

                        <div class="flex justify-center">
                            <button
                                class="mb-5 px-4 py-2 bg-transparent text-[#1d1d1f] dark:text-white border border-black/12 dark:border-[#666] rounded-lg text-sm font-medium cursor-pointer transition-all duration-300 hover:bg-[#f5f5f7] dark:hover:bg-[#222] hover:border-black/20 dark:hover:border-[#444] hover:scale-105"
                                @click="toggleSubtitles"
                                ref="toggleSubtitlesBtn">
                                {{ showSubtitles ? t('podcast.hideSubtitles') : t('podcast.showSubtitles') }}
                            </button>
                        </div>

                        <div
                            class="w-full mb-5 bg-white dark:bg-[#111] rounded-xl p-6 max-h-[400px] sm:max-h-[400px] md:max-h-[500px] lg:max-h-[600px] border border-black/6 dark:border-white/8 shadow-sm dark:shadow-none overflow-y-auto [-webkit-overflow-scrolling:touch] [scrollbar-width:thin] [scrollbar-color:#d1d1d6_#f5f5f7] dark:[scrollbar-color:#333_#0a0a0a] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-[#f5f5f7] dark:[&::-webkit-scrollbar-track]:bg-[#0a0a0a] [&::-webkit-scrollbar-track]:rounded-sm [&::-webkit-scrollbar-thumb]:bg-[#d1d1d6] dark:[&::-webkit-scrollbar-thumb]:bg-[#333] [&::-webkit-scrollbar-thumb]:rounded-sm"
                            v-show="showSubtitles"
                            ref="subtitleSection"
                            id="subtitleSection"
                            @wheel="handleUserScroll"
                            @touchstart="handleUserScroll"
                            @scroll="handleUserScroll"
                            >
                            <!-- 字幕内容由 template 渲染 -->
                            <div
                                v-for="(subtitle, index) in subtitles"
                                :key="index"
                                :id="`subtitle-${index}`"
                                :class="[
                                    'subtitle mb-3 cursor-pointer flex items-center',
                                    subtitle.speaker === 'zh_male_dayixiansheng_v2_saturn_bigtts' ? 'text-left' : 'text-right'
                                ]"
                                @click="onSubtitleClick(index)">
                                <span class="inline-block text-[11px] text-[#86868b] dark:text-[#888] opacity-90 align-middle flex-[0_0_48px] text-center mr-2">
                                    {{ subtitleTimestamps[index] ? formatTime(subtitleTimestamps[index].start) : '--:--' }}
                                </span>
                                <div :class="[
                                    'inline-block max-w-[70%] px-4 py-2.5 rounded-xl text-sm leading-normal',
                                    subtitle.speaker === 'zh_male_dayixiansheng_v2_saturn_bigtts'
                                        ? 'rounded-bl-sm text-left'
                                        : 'rounded-br-sm text-left ml-auto',
                                    activeSubtitleIndex === index
                                        ? 'bg-[#d1d1d6] dark:bg-white text-[#1d1d1f] dark:text-black shadow-[0_0_20px_rgba(0,0,0,0.1)] dark:shadow-[0_0_20px_rgba(255,255,255,0.4)] scale-105 font-medium transition-[background-color,color,box-shadow,transform] duration-75 ease-out'
                                        : 'bg-[#f5f5f7] dark:bg-[#2a2a2a] text-[#1d1d1f] dark:text-[#ccc] transition-[background-color,color,box-shadow,transform] duration-75 ease-out'
                                ]"
                                :style="activeSubtitleIndex === index ? { willChange: 'background-color, color, box-shadow, transform' } : {}">
                                    {{ subtitle.text }}
                                </div>
                        </div>
                    </div>

                    </template>


                    <template v-if="!isDetailMode && !showPlayer">
                        <!-- 输入区域 -->
                        <div class="mb-10 w-full">
                        <div class="flex gap-3 mb-4 flex-row w-full">
                            <input
                                type="text"
                                class="flex-1 px-5 py-4 bg-white dark:bg-[#111] border border-black/12 dark:border-[#333] rounded-lg text-[#1d1d1f] dark:text-white text-md transition-all duration-300 focus:outline-none focus:border-black/20 dark:focus:border-[#666] focus:bg-[#fafafa] dark:focus:bg-[#1a1a1a] placeholder:text-[#86868b] dark:placeholder:text-[#666]"
                                v-model="input"
                                @keypress="onInputKeyPress"
                                :placeholder="t('podcast.inputPlaceholder')"
                                ref="inputField">
                            <button
                                class="px-8 py-4 bg-white text-black border-none rounded-lg text-sm font-medium cursor-pointer transition-all duration-300 whitespace-nowrap hover:bg-[#e0e0e0] hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:bg-white flex-shrink-0"
                                @click="generatePodcast"
                                ref="generateBtn">{{ t('podcast.generatePodcast') }}</button>
                        </div>

                        <!-- 示例输入气泡 -->
                        <div class="flex flex-wrap items-center justify-center gap-3 max-w-2xl mx-auto">
                            <button
                                v-for="(example, index) in exampleInputs"
                                :key="index"
                                @click="onExampleClick(example)"
                                class="relative px-4 py-2.5 bg-white/90 dark:bg-[#2c2c2e]/90 backdrop-blur-[10px] border border-black/8 dark:border-white/8 rounded-2xl text-sm text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] transition-all duration-200 cursor-pointer tracking-tight"
                                :class="{
                                    'rounded-br-sm': index % 2 === 0,
                                    'rounded-bl-sm': index % 2 === 1
                                }"
                            >
                                {{ example }}
                            </button>
                        </div>
                    </div>
                    </template>
                </div>
                <SiteFooter />
            </div>

        </div>

    </div>
    <Alert />
    <Confirm />
    <!-- 全局加载覆盖层 - Apple 风格 -->
    <div v-show="isLoading" class="fixed inset-0 bg-[#f5f5f7] dark:bg-[#000000] flex items-center justify-center z-[9999] transition-opacity duration-300">
      <Loading />
    </div>
</template>

<style scoped>
/* 所有样式已通过 Tailwind CSS 在 template 中定义 */
/* 波形图的动态样式（高度和渐变背景）通过 JavaScript 动态设置，保留 style 属性 */
</style>
