<template>
  <!-- 模态框遮罩和容器 - Apple 极简风格 -->
  <div class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-2">
    <div class="relative w-full h-full max-w-6xl max-h-[100vh] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden flex flex-col">
      <!-- 模态框头部 - Apple 风格 -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px] flex-shrink-0">
        <div class="flex items-center gap-3">
          <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
            <i class="fas fa-volume-up text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
            <span>{{ t('voiceSynthesis') }}</span>
          </h3>
          <!-- 模式切换开关 -->
          <div class="flex items-center gap-2">
            <button
              @click="toggleMode"
              class="relative w-14 h-7 rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-[color:var(--brand-primary)]/20 dark:focus:ring-[color:var(--brand-primary-light)]/20"
              :class="isMultiSegmentMode ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)]' : 'bg-[#86868b]/30 dark:bg-[#98989d]/30'"
              :title="isMultiSegmentMode ? t('tts.switchToSingleSegmentMode') : t('tts.switchToMultiSegmentMode')"
            >
              <!-- 滑动圆点 -->
              <span
                class="absolute top-0.5 left-0.5 w-6 h-6 bg-white rounded-full shadow-md transition-transform duration-300 flex items-center justify-center"
                :class="{ 'translate-x-7': isMultiSegmentMode, 'translate-x-0': !isMultiSegmentMode }"
              >
                <i :class="isMultiSegmentMode ? 'fas fa-layer-group text-[8px] text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]' : 'fas fa-exchange-alt text-[8px] text-[#86868b] dark:text-[#98989d]'"></i>
              </span>
            </button>
            <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight" :class="{ 'text-[#86868b] dark:text-[#98989d]': !isMultiSegmentMode }">
              {{ isMultiSegmentMode? t('tts.multiSegmentMode') : t('tts.singleSegmentMode') }}</span>
          </div>
          <button
            v-show="!isMultiSegmentMode"
            @click="openHistoryPanel"
            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100"
            :title="t('ttsHistoryTitle')"
          >
            <i class="fas fa-history text-sm"></i>
          </button>
        </div>
        <div class="flex items-center gap-2">
          <!-- 应用按钮 - Apple 风格 -->
          <button
            @click="isMultiSegmentMode ? applyMergedAudio() : applySelectedVoice()"
            :disabled="isMultiSegmentMode ? (audioSegments.filter(s => s.audioBlob).length === 0) : (!selectedVoice || !inputText.trim() || isGenerating)"
            class="w-9 h-9 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 active:scale-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            :title="isMultiSegmentMode ? t('applyMergedAudio') : t('applySelectedVoice')">
            <i class="fas fa-check text-sm"></i>
          </button>
          <!-- 关闭按钮 - Apple 风格 -->
          <button @click="closeModal"
            class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
            <i class="fas fa-times text-sm"></i>
          </button>
        </div>
      </div>

      <!-- 多段模式预览区 -->
      <div v-if="isMultiSegmentMode && audioSegments.length > 0" class="flex-shrink-0 bg-[#f5f5f7]/30 dark:bg-[#1c1c1e]/30">
        <div class="max-w-5xl mx-auto px-6 py-5">
          <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-4">
            <!-- 合并音频播放器 -->
            <div v-if="mergedAudioDuration > 0" class="mb-4">
              <div class="flex items-center gap-3 mb-3">
                <button
                  @click="toggleMergedAudioPlayback"
                  class="w-10 h-10 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-all duration-200"
                >
                  <i :class="isPlayingMerged ? 'fas fa-pause' : 'fas fa-play'" class="text-sm ml-0.5"></i>
                </button>
                <div class="flex-1">
                  <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('tts.mergedAudio') }}</div>
                  <div class="text-xs text-[#86868b] dark:text-[#98989d]">{{ formatAudioTime(mergedCurrentTime) }} / {{ formatAudioTime(mergedAudioDuration) }}</div>
                </div>
              </div>
              <input
                v-if="mergedAudioDuration > 0"
                type="range"
                :min="0"
                :max="mergedAudioDuration"
                :value="mergedCurrentTime"
                @input="onMergedProgressChange"
                class="w-full h-1 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full"
              />
            </div>
            <!-- 分段进度条 - 水平排列 -->
            <div class="flex items-start gap-4 flex-wrap max-h-[200px] overflow-y-auto main-scrollbar">
              <div
                v-for="(segment, index) in audioSegments"
                :key="segment.id"
                class="flex flex-col gap-2 flex-1 min-w-[140px] max-w-[200px]"
                :class="{ 'opacity-50': segment.isGenerating }"
              >
                <!-- 段落编号和播放按钮 -->
                <div class="flex items-center gap-2">
                  <span class="text-xs font-semibold text-[#86868b] dark:text-[#98989d] w-5 text-center">{{ index + 1 }}</span>
                  <button
                    @click.stop="playSegment(index)"
                    :disabled="!segment.audioUrl || segment.isGenerating"
                    class="w-7 h-7 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-full text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                    :class="{ 'bg-[color:var(--brand-primary)]/20 dark:bg-[color:var(--brand-primary-light)]/20 border-[color:var(--brand-primary)]/30 dark:border-[color:var(--brand-primary-light)]/30 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]': playingSegmentIndex === index && segmentAudioElements[index] && !segmentAudioElements[index].paused }"
                  >
                    <i v-if="segment.isGenerating" class="fas fa-spinner fa-spin text-[10px]"></i>
                    <i v-else-if="playingSegmentIndex === index && segmentAudioElements[index] && !segmentAudioElements[index].paused" class="fas fa-pause text-[10px]"></i>
                    <i v-else class="fas fa-play text-[10px] ml-0.5"></i>
                  </button>
                  <div class="text-xs text-[#86868b] dark:text-[#98989d] flex-shrink-0 font-mono">
                    {{ formatAudioTime(segment.currentTime || 0) }} / {{ formatAudioTime(segment.duration || 0) }}
                  </div>
                </div>
                <!-- 可点击的进度条 -->
                <div
                  @click.stop="handleSegmentProgressClick(index, $event)"
                  class="w-full h-3 bg-black/6 dark:bg-white/15 rounded-full relative overflow-hidden cursor-pointer hover:h-3.5 transition-all duration-200 group"
                  :class="{ 'ring-2 ring-[color:var(--brand-primary)]/50 dark:ring-[color:var(--brand-primary-light)]/50 ring-offset-1': playingSegmentIndex === index && segmentAudioElements[index] && !segmentAudioElements[index].paused }"
                >
                  <div
                    class="h-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] rounded-full transition-all duration-100"
                    :style="{ width: segment.duration > 0 ? `${((segment.currentTime || 0) / segment.duration) * 100}%` : '0%' }"
                  ></div>
                  <!-- 悬停时显示时间提示 -->
                  <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <span class="text-[10px] text-[#1d1d1f] dark:text-[#f5f5f7] font-medium bg-white/95 dark:bg-[#2c2c2e]/95 px-2 py-1 rounded shadow-sm">
                      {{ segment.duration > 0 ? formatAudioTime(segment.duration) : '--:--' }}
                    </span>
                  </div>
                </div>
                <audio
                  v-if="segment.audioUrl"
                  :ref="el => { if (el) segmentAudioElements[index] = el }"
                  :src="segment.audioUrl"
                  @loadedmetadata="() => onSegmentAudioLoaded(index)"
                  @timeupdate="() => onSegmentTimeUpdate(index)"
                  @ended="() => onSegmentAudioEnded(index)"
                  class="hidden"
                ></audio>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 固定区域：音频播放器和设置面板 - Apple 极简风格 -->
      <div v-if="!isMultiSegmentMode && (audioUrl || selectedVoice)" class="flex-shrink-0 bg-[#f5f5f7]/30 dark:bg-[#1c1c1e]/30">
        <div class="max-w-5xl mx-auto px-6 py-5">
          <div class="flex flex-col lg:flex-row gap-6 lg:gap-8">
            <!-- 音频播放器卡片 - Apple 风格 -->
            <div v-if="audioUrl || isGenerating" class="flex-1 lg:w-1/2">
              <div class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)] p-4">
                <div class="relative flex items-center mb-3">
                  <!-- 头像容器 -->
                  <div class="relative mr-3 flex-shrink-0">
                    <!-- 透明白色头像 -->
                    <div class="w-12 h-12 rounded-full bg-white/40 dark:bg-white/20 border border-white/30 dark:border-white/20 transition-all duration-200"></div>
                    <!-- Loading 指示器 - Apple 风格 -->
                    <div v-if="isGenerating" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white z-20">
                      <i class="fas fa-spinner fa-spin text-xs"></i>
                    </div>
                    <!-- 播放/暂停按钮 -->
                    <button
                      v-else
                      @click="toggleAudioPlayback"
                      class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white cursor-pointer hover:scale-110 transition-all duration-200 z-20 shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.3)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.4)]"
                    >
                      <i :class="isPlaying ? 'fas fa-pause' : 'fas fa-play'" class="text-xs ml-0.5"></i>
                    </button>
                  </div>

                  <!-- 音频信息 -->
                  <div class="flex-1 min-w-0">
                    <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight truncate">
                      {{ t('synthesizedAudio') }}<span v-if="selectedVoiceData"> - {{ selectedVoiceData.name }}</span>
                    </div>
                  </div>

                  <!-- 音频时长 -->
                  <div class="text-xs font-medium text-[#86868b] dark:text-[#98989d] tracking-tight flex-shrink-0">
                    {{ formatAudioTime(currentTime) }} / {{ formatAudioTime(audioDuration) }}
                  </div>
                </div>

                <!-- 进度条 -->
                <div class="flex items-center gap-2" v-if="audioDuration > 0">
                  <input
                    type="range"
                    :min="0"
                    :max="audioDuration"
                    :value="currentTime"
                    @input="onProgressChange"
                    @change="onProgressChange"
                    @mousedown="isDragging = true"
                    @mouseup="onProgressEnd"
                    @touchstart="isDragging = true"
                    @touchend="onProgressEnd"
                    class="flex-1 h-1 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                </div>
              </div>
              <!-- 隐藏的音频元素 -->
              <audio
                v-if="audioUrl"
                ref="audioElement"
                :src="audioUrl"
                @loadedmetadata="onAudioLoaded"
                @timeupdate="onTimeUpdate"
                @ended="onAudioEnded"
                @play="isPlaying = true"
                @pause="isPlaying = false"
                class="hidden"
              ></audio>
            </div>

            <!-- 设置面板 - Apple 极简风格（无卡片，直接显示） -->
            <div v-if="selectedVoice" class="flex-shrink-0 lg:w-1/2">
              <div class="space-y-3">
                <!-- 语速控制 -->
                <div class="flex items-center gap-3">
                  <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-14 tracking-tight">{{ t('speechRate') }}</label>
                  <input
                    type="range"
                    min="-50"
                    max="100"
                    v-model="speechRate"
                    class="flex-1 h-0.5 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getSpeechRateDisplayValue(speechRate) }}</span>
                </div>
                <!-- 音量控制 -->
                <div class="flex items-center gap-3">
                  <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-14 tracking-tight">{{ t('volume') }}</label>
                  <input
                    type="range"
                    min="-50"
                    max="100"
                    v-model="loudnessRate"
                    class="flex-1 h-0.5 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getLoudnessDisplayValue(loudnessRate) }}</span>
                </div>
                <!-- 音调控制 -->
                <div class="flex items-center gap-3">
                  <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-14 tracking-tight">{{ t('pitch') }}</label>
                  <input
                    type="range"
                    min="-12"
                    max="12"
                    v-model="pitch"
                    class="flex-1 h-0.5 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getPitchDisplayValue(pitch) }}</span>
                </div>
                <!-- 情感控制 - 仅当音色支持时显示 -->
                <div v-if="selectedVoiceData && selectedVoiceData.emotions && selectedVoiceData.emotions.length > 0" class="flex items-center gap-3">
                  <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-14 tracking-tight">{{ t('emotionIntensity') }}</label>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    v-model="emotionScale"
                    class="flex-1 h-0.5 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ emotionScale }}</span>
                </div>
                <div v-if="selectedVoiceData && selectedVoiceData.emotions && selectedVoiceData.emotions.length > 0" class="flex items-center gap-3">
                  <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-14 tracking-tight">{{ t('emotionType') }}</label>
                  <div class="flex-1">
                    <DropdownMenu
                      :items="emotionItems"
                      :selected-value="selectedEmotion"
                      :placeholder="t('neutral')"
                      @select-item="handleEmotionSelect"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 装饰性分割线 - Apple 风格（带V形图标） -->
        <div class="relative flex items-center justify-center py-3">
          <!-- 左侧线条 -->
          <div class="flex-1 h-px bg-gradient-to-r from-transparent via-black/20 dark:via-white/20 to-black/20 dark:to-white/20"></div>
          <!-- 中间V形图标 -->
          <div class="mx-4 flex items-center justify-center w-6 h-6 rounded-full bg-white/60 dark:bg-[#2c2c2e]/60 border border-black/10 dark:border-white/10">
            <i class="fas fa-chevron-down text-[8px] text-[#86868b] dark:text-[#98989d]"></i>
          </div>
          <!-- 右侧线条 -->
          <div class="flex-1 h-px bg-gradient-to-l from-transparent via-black/20 dark:via-white/20 to-black/20 dark:to-white/20"></div>
        </div>
      </div>

      <!-- 模态框内容 - Apple 风格（可滚动区域） -->
      <div class="flex-1 overflow-y-auto p-6 main-scrollbar">
        <div class="max-w-5xl mx-auto space-y-6">
          <!-- 多段模式输入区域 -->
          <template v-if="isMultiSegmentMode">
            <!-- 分割线 - 音频预览和段落输入区域之间 -->
            <div v-if="isMultiSegmentMode && audioSegments.length > 0" class="relative flex items-center justify-center py-3">
              <!-- 左侧线条 -->
              <div class="flex-1 h-px bg-gradient-to-r from-transparent via-black/20 dark:via-white/20 to-black/20 dark:to-white/20"></div>
              <!-- 中间V形图标 -->
              <div class="mx-4 flex items-center justify-center w-6 h-6 rounded-full bg-white/60 dark:bg-[#2c2c2e]/60 border border-black/10 dark:border-white/10">
                <i class="fas fa-chevron-down text-[8px] text-[#86868b] dark:text-[#98989d]"></i>
              </div>
              <!-- 右侧线条 -->
              <div class="flex-1 h-px bg-gradient-to-l from-transparent via-black/20 dark:via-white/20 to-black/20 dark:to-white/20"></div>
            </div>
            <!-- 添加段落按钮 -->
            <button
              @click="addSegment"
              class="w-full py-4 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 border-dashed rounded-xl text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 flex items-center justify-center gap-2 mb-6"
            >
              <i class="fas fa-plus text-sm"></i>
              <span class="text-sm font-medium">{{ t('tts.addSegment') }}</span>
            </button>
            <div
              v-for="(item, reversedIndex) in reversedSegments"
              :key="item.segment.id"
              class="space-y-4"
            >
              <div
                class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-4 transition-all duration-200"
                :class="{
                  'opacity-50 scale-95': draggingSegmentIndex === item.originalIndex,
                  'border-[color:var(--brand-primary)]/50 dark:border-[color:var(--brand-primary-light)]/50 ring-2 ring-[color:var(--brand-primary)]/20 dark:ring-[color:var(--brand-primary-light)]/20': dragOverSegmentIndex === item.originalIndex && draggingSegmentIndex !== item.originalIndex
                }"
                :draggable="audioSegments.length > 1"
                @dragstart="handleDragStart(item.originalIndex, $event)"
                @dragend="handleDragEnd"
                @dragover.prevent="handleDragOver(item.originalIndex, $event)"
                @dragleave="handleDragLeave(item.originalIndex)"
                @drop="handleDrop(item.originalIndex, $event)"
                style="position: relative; overflow: visible;"
              >
                <div class="flex items-center gap-3 mb-3">
                  <!-- 拖拽手柄 -->
                  <div
                    v-if="audioSegments.length > 1"
                    class="cursor-move text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] transition-colors"
                    :title="t('dragToReorder')"
                  >
                    <i class="fas fa-grip-vertical text-sm"></i>
                  </div>
                  <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('tts.segmentNumber', { index: item.originalIndex + 1 }) }}</span>
                  <button
                    @click="copySegment(item.originalIndex, $event)"
                    class="w-8 h-8 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200"
                    :title="t('tts.copySegment')"
                  >
                    <i class="fas fa-copy text-xs"></i>
                  </button>
                  <button
                    v-if="audioSegments.length > 1"
                    @click="removeSegment(item.originalIndex)"
                    class="w-8 h-8 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-red-500 dark:text-red-400 rounded-full hover:bg-red-50 dark:hover:bg-red-500/10 transition-all duration-200"
                    :title="t('tts.deleteSegment')"
                  >
                    <i class="fas fa-trash text-xs"></i>
                  </button>
                  <div class="flex-1"></div>
                  <div
                    class="relative voice-selector-container"
                    :data-segment-index="item.originalIndex"
                    :ref="el => setSegmentVoiceSelectorRef(item.originalIndex, el)"
                  >
                    <button
                      @click.stop="selectVoiceForSegment(item.originalIndex)"
                      class="flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-lg text-sm text-[#1d1d1f] dark:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200 whitespace-nowrap"
                    >
                      <!-- 头像显示 -->
                      <div v-if="item.segment.voiceData" class="relative flex-shrink-0">
                        <img
                          v-if="isFemaleVoice(item.segment.voiceData.voice_type)"
                          src="../../public/female.svg"
                          alt="Female Avatar"
                          class="w-6 h-6 rounded-full object-cover bg-white"
                        />
                        <img
                          v-else
                          src="../../public/male.svg"
                          alt="Male Avatar"
                          class="w-6 h-6 rounded-full object-cover bg-white"
                        />
                      </div>
                      <span>{{ item.segment.voiceData?.name || t('tts.selectVoice') }}</span>
                    </button>
                  </div>
                  <button
                    @click="generateSegmentTTS(item.originalIndex)"
                    :disabled="!item.segment.text.trim() || !item.segment.voice || item.segment.isGenerating"
                    class="w-10 h-10 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full hover:scale-110 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                    :title="t('tts.generate')"
                  >
                    <i v-if="item.segment.isGenerating" class="fas fa-spinner fa-spin text-sm"></i>
                    <i v-else class="fas fa-check text-sm"></i>
                  </button>
                </div>
                <!-- 文本输入区域 -->
                <div class="mb-3">
                  <label class="block text-xs text-[#86868b] dark:text-[#98989d] mb-1.5">{{ t('tts.text') }}</label>
                  <textarea
                    v-model="item.segment.text"
                    :placeholder="t('tts.placeholder')"
                    class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-lg px-4 py-3 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 transition-all duration-200 resize-none"
                    rows="2"
                  ></textarea>
                </div>
                <!-- 语音指令输入（仅2.0音色显示） -->
                <div v-if="item.segment.voiceData?.version === '2.0'" class="mt-3">
                  <label class="block text-xs text-[#86868b] dark:text-[#98989d] mb-1.5">{{ t('tts.voiceInstructionOptional') }}</label>
                  <textarea
                    v-model="item.segment.contextText"
                    :placeholder="t('tts.voiceInstructionPlaceholder')"
                    class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-lg px-4 py-2 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 transition-all duration-200 resize-none"
                    rows="2"
                  ></textarea>
                </div>
              </div>
            </div>
          </template>

          <!-- 单段模式输入区域 - Apple 风格 -->
          <template v-else>
          <!-- 文本输入区域 - Apple 风格 -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <i class="fas fa-keyboard text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('enterTextToConvert') }}</span>

              <button
                @click="openTextHistoryPanel"
                class="w-8 h-8 flex items-center justify-center rounded-full bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200"
                :title="t('ttsHistoryTabText')"
              >
                <i class="fas fa-history text-xs"></i>
              </button>
              </div>
            </div>
            <textarea
              v-model="inputText"
              :placeholder="t('tts.placeholder')"
              class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200 resize-none min-h-[100px]"
              rows="4"
            ></textarea>
          </div>

          <!-- 语音指令区域 - Apple 风格 -->
          <div>
            <div class="flex items-center justify-between mb-3">
              <div class="flex items-center gap-2">
                <i class="fas fa-magic text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
                <span class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('voiceInstruction') }}</span>
                <span class="text-xs text-[#86868b] dark:text-[#98989d]">{{ t('voiceInstructionHint') }}</span>

              <button
                @click="openInstructionHistoryPanel"
                class="w-8 h-8 flex items-center justify-center rounded-full bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] transition-all duration-200"
                :title="t('ttsHistoryTabInstruction')"
              >
                <i class="fas fa-history text-xs"></i>
              </button>
              </div>
            </div>
            <textarea
              v-model="contextText"
              :placeholder="t('voiceInstructionPlaceholder')"
              class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl px-5 py-3 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200 resize-none"
              rows="3"
            ></textarea>
          </div>

           <!-- 音色选择区域 - Apple 风格 -->
           <VoiceSelector
             :voices="voices"
             :filtered-voices="filteredVoices"
             :selected-voice="selectedVoice"
             :search-query="searchQuery"
             :is-generating="isGenerating"
             mode="full"
             :show-search="true"
             :show-filter="true"
             :show-history-button="true"
             @select-voice="onVoiceSelect"
             @update-search="searchQuery = $event"
             @toggle-filter="toggleFilterPanel"
             @open-history="openVoiceHistoryPanel"
             ref="voiceSelectorRef"
           />
          </template>
        </div>
      </div>
    </div>
  </div>

  <!-- 合并音频元素 - 放在模态框最外层，确保 ref 始终绑定 -->
  <audio
    ref="mergedAudioElement"
    :src="mergedAudioUrl"
    @loadedmetadata="onMergedAudioLoaded"
    @timeupdate="onMergedTimeUpdate"
    @ended="onMergedAudioEnded"
    @play="isPlayingMerged = true"
    @pause="isPlayingMerged = false"
    class="hidden"
  ></audio>

  <VoiceTtsHistoryPanel
    :visible="showHistoryPanel"
    :history="ttsHistory"
    mode="combined"
    :get-voice-name="getHistoryVoiceName"
    @close="closeHistoryPanel"
    @apply="applyCombinedHistoryEntry"
    @delete="handleDeleteHistoryEntry"
  />

  <VoiceTtsHistoryPanel
    :visible="showTextHistoryPanel"
    :history="ttsHistory"
    mode="text"
    @close="closeTextHistoryPanel"
    @apply="applyTextHistoryEntry"
  />

  <VoiceTtsHistoryPanel
    :visible="showInstructionHistoryPanel"
    :history="ttsHistory"
    mode="instruction"
    @close="closeInstructionHistoryPanel"
    @apply="applyInstructionHistoryEntry"
  />

  <VoiceTtsHistoryPanel
    :visible="showVoiceHistoryPanel"
    :history="ttsHistory"
    mode="voice"
    :get-voice-name="getHistoryVoiceName"
    @close="closeVoiceHistoryPanel"
    @apply="applyVoiceHistoryEntry"
  />

  <!-- 音色选择下拉菜单 - 直接在组件内渲染，使用固定定位 -->
  <div
    v-if="showVoiceSelector && selectedSegmentIndex >= 0 && audioSegments && audioSegments[selectedSegmentIndex]"
    ref="dropdownContainerRef"
    @click.stop
    @mousedown.stop
    class="fixed bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] overflow-hidden voice-selector-dropdown"
    :style="{
      position: 'fixed',
      top: dropdownStyle?.top || '0px',
      left: dropdownStyle?.left || '0px',
      width: dropdownStyle?.width || '256px',
      zIndex: '99999',
      minHeight: '200px',
      maxHeight: dropdownStyle?.maxHeight || '384px',
      overflowY: dropdownStyle?.overflowY || 'auto',
      pointerEvents: 'auto'
    }"
  >
    <VoiceSelector
      :voices="voices"
      :filtered-voices="filteredVoices"
      :selected-voice="(audioSegments && audioSegments[selectedSegmentIndex]) ? (audioSegments[selectedSegmentIndex].voice || '') : ''"
      :is-generating="(audioSegments && audioSegments[selectedSegmentIndex]) ? (audioSegments[selectedSegmentIndex].isGenerating || false) : false"
      mode="dropdown"
      :show-search="false"
      :show-filter="false"
      :show-history-button="false"
      @select-voice="onVoiceSelectForSegment"
      class="voice-selector-component"
    />
  </div>

  <!-- 筛选面板遮罩 - Apple 风格 -->
  <div v-if="showFilterPanel" class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[100] flex items-center justify-center p-4" @click="closeFilterPanel">
    <div class="bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl w-full max-w-2xl max-h-[85vh] overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.2)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.6)] flex flex-col" @click.stop>
      <!-- 筛选面板头部 - Apple 风格 -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
        <h3 class="text-lg font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-2 tracking-tight">
          <i class="fas fa-filter text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
          <span>{{ t('filterVoices') }}</span>
        </h3>
        <button @click="closeFilterPanel"
          class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
          <i class="fas fa-times text-sm"></i>
        </button>
      </div>

      <!-- 筛选内容 - Apple 风格 -->
      <div class="flex-1 overflow-y-auto p-6 main-scrollbar">
        <div class="space-y-6">
          <!-- 场景筛选 -->
          <div>
            <h4 class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">{{ t('scene') }}</h4>
            <div class="flex flex-wrap gap-2">
              <button
                v-for="category in categories"
                :key="category"
                @click="selectCategory(category)"
                class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                :class="selectedCategory === category
                  ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'"
              >
                {{ translateCategory(category) }}
              </button>
            </div>
          </div>

          <!-- 版本筛选 -->
          <div>
            <h4 class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">{{ t('version') }}</h4>
            <div class="flex flex-wrap gap-2">
              <button
                v-for="v in version"
                :key="v"
                @click="selectVersion(v)"
                class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                :class="selectedVersion === v
                  ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'"
              >
                {{ translateVersion(v) }}
              </button>
            </div>
          </div>

          <!-- 语言筛选 -->
          <div>
            <h4 class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">{{ t('language') }}</h4>
            <div class="flex flex-wrap gap-2">
              <button
                v-for="lang in languages"
                :key="lang"
                @click="selectLanguage(lang)"
                class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                :class="selectedLanguage === lang
                  ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'"
              >
                {{ translateLanguage(lang) }}
              </button>
            </div>
          </div>

          <!-- 性别筛选 -->
          <div>
            <h4 class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">{{ t('gender') }}</h4>
            <div class="flex flex-wrap gap-2">
              <button
                v-for="gender in genders"
                :key="gender"
                @click="selectGender(gender)"
                class="px-4 py-2 text-sm font-medium rounded-full transition-all duration-200 tracking-tight"
                :class="selectedGender === gender
                  ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white shadow-[0_2px_8px_rgba(var(--brand-primary-rgb),0.25)] dark:shadow-[0_2px_8px_rgba(var(--brand-primary-light-rgb),0.3)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'"
              >
                {{ translateGender(gender) }}
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- 筛选操作按钮 - Apple 风格 -->
      <div class="flex gap-3 px-6 py-4 border-t border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px]">
        <button @click="resetFilters"
          class="flex-1 px-5 py-3 bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full transition-all duration-200 font-medium text-[15px] tracking-tight hover:bg-white/80 dark:hover:bg-[#3a3a3c]/80 hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-[0.98]">
          {{ t('reset') }}
        </button>
        <button @click="applyFilters"
          class="flex-1 px-5 py-3 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 font-semibold text-[15px] tracking-tight hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100">
          {{ t('done') }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import DropdownMenu from './DropdownMenu.vue'
import VoiceTtsHistoryPanel from './VoiceTtsHistoryPanel.vue'
import VoiceSelector from './VoiceSelector.vue'
import { ttsHistory, loadTtsHistory, addTtsHistoryEntry, removeTtsHistoryEntry } from '../utils/other'

export default {
  name: 'VoiceTTS',
  components: {
    DropdownMenu,
    VoiceTtsHistoryPanel,
    VoiceSelector
  },
  emits: ['tts-complete', 'close-modal'],
  setup(props, { emit }) {
    const { t } = useI18n()
    const inputText = ref('')
    const contextText = ref('')
    const selectedVoice = ref('')
    const selectedVoiceResourceId = ref('')
    const searchQuery = ref('')
    const speechRate = ref(0)
    const loudnessRate = ref(0)
    const pitch = ref(0)
    const emotionScale = ref(3)
    const selectedEmotion = ref('neutral')
    const isGenerating = ref(false)
    const audioUrl = ref('')
    const currentAudio = ref(null) // 当前播放的音频对象
    const audioElement = ref(null) // 音频元素引用
    const isPlaying = ref(false) // 播放状态
    const audioDuration = ref(0) // 音频总时长
    const currentTime = ref(0) // 当前播放时间
    const shouldAutoPlay = ref(false) // 是否需要自动播放
    const isDragging = ref(false) // 是否正在拖拽进度条
    const voices = ref([])
    const emotions = ref([])
    const voiceListContainer = ref(null)
    const showControls = ref(false)
    const showFilterPanel = ref(false)
    const showHistoryPanel = ref(false)
    const showTextHistoryPanel = ref(false)
    const showInstructionHistoryPanel = ref(false)
    const showVoiceHistoryPanel = ref(false)

    // 多段语音合成模式
    const isMultiSegmentMode = ref(false)
    const audioSegments = ref([]) // 音频段列表 [{ id, text, voice, voiceData, audioUrl, audioBlob, duration, isGenerating }]
    const mergedAudioUrl = ref('') // 合并后的音频URL（已废弃，保留用于兼容）
    const mergedAudioElement = ref(null) // 合并音频元素（已废弃，保留用于兼容）
    const isMerging = ref(false) // 是否正在合并（已废弃，保留用于兼容）
    const isPlayingMerged = ref(false) // 是否正在播放合并音频
    const mergedAudioDuration = ref(0) // 合并音频总时长（所有段的总时长）
    const mergedCurrentTime = ref(0) // 合并音频当前时间（累计时间）
    const segmentStartTimes = ref([]) // 每段的开始时间（累计），用于计算总进度
    const playingSegmentIndex = ref(-1) // 当前播放的段索引
    const segmentAudioElements = ref({}) // 分段音频元素引用
    const showInstructionInput = ref(-1) // 显示语音指令输入的段索引
    const selectedSegmentIndex = ref(-1) // 当前选择音色的段索引
    const showVoiceSelector = ref(false) // 是否显示音色选择器
    const segmentVoiceSelectors = ref({}) // 分段音色选择器容器引用
    const dropdownStyle = ref({}) // 下拉菜单样式（不包含 display，由 v-show 控制）
    const dropdownContainerRef = ref(null) // 下拉菜单容器 ref
    const draggingSegmentIndex = ref(-1) // 正在拖拽的段落索引
    const dragOverSegmentIndex = ref(-1) // 拖拽悬停的段落索引

    // Category filtering - 存储原始中文值
    const selectedCategory = ref('全部场景')
    const categories = ref(['全部场景', '通用场景', '客服场景', '教育场景', '趣味口音', '角色扮演', '有声阅读', '多语种', '多情感', '视频配音'])
    const selectedVersion = ref('全部版本')
    const version = ref(['全部版本', '1.0', '2.0'])
    const selectedLanguage = ref('全部语言')
    const languages = ref(['全部语言'])
    const selectedGender = ref('全部性别')
    const genders = ref(['全部性别'])

    // 翻译映射函数
    const translateCategory = (category) => {
      const map = {
        '全部场景': t('allScenes'),
        '通用场景': t('generalScene'),
        '客服场景': t('customerServiceScene'),
        '教育场景': t('educationScene'),
        '趣味口音': t('funAccent'),
        '角色扮演': t('rolePlaying'),
        '有声阅读': t('audiobook'),
        '多语种': t('multilingual'),
        '多情感': t('multiEmotion'),
        '视频配音': t('videoDubbing')
      }
      return map[category] || category
    }

    const translateVersion = (ver) => {
      return ver === '全部版本' ? t('allVersions') : ver
    }

    const translateLanguage = (lang) => {
      if (lang === '全部语言') return t('allLanguages')

      // 语言名称映射 - 中文到翻译键（如果有的话直接显示）
      // 对于后端返回的中文语言名，直接显示即可，因为它们是通用的
      return lang
    }

    const translateGender = (gender) => {
      const map = {
        '全部性别': t('allGenders'),
        '女性': t('female'),
        '男性': t('male')
      }
      return map[gender] || gender
    }

    const openHistoryPanel = () => {
      loadTtsHistory()
      showHistoryPanel.value = true
    }

    const closeHistoryPanel = () => {
      showHistoryPanel.value = false
    }

    const openTextHistoryPanel = () => {
      loadTtsHistory()
      showTextHistoryPanel.value = true
    }

    const openInstructionHistoryPanel = () => {
      loadTtsHistory()
      showInstructionHistoryPanel.value = true
    }

    const openVoiceHistoryPanel = () => {
      loadTtsHistory()
      showVoiceHistoryPanel.value = true
    }

    const closeTextHistoryPanel = () => {
      showTextHistoryPanel.value = false
    }

    const closeInstructionHistoryPanel = () => {
      showInstructionHistoryPanel.value = false
    }

    const closeVoiceHistoryPanel = () => {
      showVoiceHistoryPanel.value = false
    }

    const handleDeleteHistoryEntry = (entry) => {
      if (!entry?.id) return
      removeTtsHistoryEntry(entry.id)
      loadTtsHistory()
    }


    // Load voices data
    onMounted(async () => {
      document.addEventListener('click', handleClickOutside)
      loadTtsHistory()
      try {
        const response = await fetch('/api/v1/voices/list')
        const data = await response.json()
        console.log('音色数据', data)
        voices.value = data.voices || []
        emotions.value = data.emotions || []

        // Map languages data to language options
        if (data.languages && Array.isArray(data.languages)) {
          const languageOptions = ['全部语言']
          data.languages.forEach(lang => {
            languageOptions.push(lang.zh) // Use Chinese name
          })
          languages.value = languageOptions
        }

        // Extract gender options from voices data
        if (voices.value && voices.value.length > 0) {
          const genderSet = new Set()
          voices.value.forEach(voice => {
            if (voice.gender) {
              genderSet.add(voice.gender)
            }
          })

          const genderOptions = ['全部性别']
          // Convert English gender to localized display - 保留中文作为内部值
          genderSet.forEach(gender => {
            if (gender === 'female') {
              genderOptions.push('女性')
            } else if (gender === 'male') {
              genderOptions.push('男性')
            } else {
              // For any other gender values, use as is
              genderOptions.push(gender)
            }
          })
          genders.value = genderOptions
        }
      } catch (error) {
        console.error('Failed to load voices:', error)
      }
    })

    // 组件卸载时清理音频资源
    onUnmounted(() => {
      if (currentAudio.value) {
        currentAudio.value.pause()
        currentAudio.value = null
      }
      // 清理音频URL
      if (audioUrl.value) {
        URL.revokeObjectURL(audioUrl.value)
      }
    })

    // 监听参数变化，自动重新生成音频
    watch([speechRate, loudnessRate, pitch, emotionScale, selectedEmotion], () => {
      if (selectedVoice.value && inputText.value.trim() && !isGenerating.value) {
        generateTTS()
      }
    })

    // 监听文本输入变化，使用防抖避免频繁生成
    let textTimeout = null
    watch([inputText, contextText], () => {
      if (textTimeout) {
        clearTimeout(textTimeout)
      }
      textTimeout = setTimeout(() => {
        if (selectedVoice.value && inputText.value.trim() && !isGenerating.value) {
          generateTTS()
        }
      }, 800) // 延迟800ms执行，给用户足够时间输入
    })

    // 监听搜索查询变化，重置滚动位置（延迟执行以避免频繁重置）
    let searchTimeout = null
    watch(searchQuery, () => {
      if (searchTimeout) {
        clearTimeout(searchTimeout)
      }
      searchTimeout = setTimeout(() => {
        resetScrollPosition()
      }, 300) // 延迟300ms执行
    })

    // 重置滚动位置（使用 VoiceSelector 组件的引用）
    const resetScrollPosition = () => {
      if (voiceSelectorRef.value && voiceSelectorRef.value.voiceListContainer) {
        voiceSelectorRef.value.voiceListContainer.scrollTop = 0
      }
      // 兼容旧的 voiceListContainer 引用
      if (voiceListContainer.value) {
        voiceListContainer.value.scrollTop = 0
      }
    }

    // Filter voices based on search query, category, version, language, and gender
    const filteredVoices = computed(() => {
      let filtered = [...voices.value] // 创建副本，避免修改原始数据

      console.log('原始音色数据:', voices.value.length)
      console.log('筛选条件:', {
        category: selectedCategory.value,
        version: selectedVersion.value,
        language: selectedLanguage.value,
        gender: selectedGender.value,
        search: searchQuery.value
      })

      // Filter by category
      if (selectedCategory.value !== '全部场景') {
        filtered = filtered.filter(voice => voice.scene === selectedCategory.value)
        console.log('分类筛选后:', filtered.length)
      }

      // Filter by version
      if (selectedVersion.value !== '全部版本') {
        filtered = filtered.filter(voice => voice.version === selectedVersion.value)
        console.log('版本筛选后:', filtered.length)
      }

      // Filter by language
      if (selectedLanguage.value !== '全部语言') {
        // Convert Chinese language display back to language code for filtering
        let languageFilter = selectedLanguage.value
        // Create a mapping from Chinese names to language codes
        const languageMap = {
          '中文': 'chinese',
          '美式英语': 'en_us',
          '英式英语': 'en_gb',
          '澳洲英语': 'en_au',
          '西语': 'es',
          '日语': 'ja'
        }

        if (languageMap[selectedLanguage.value]) {
          languageFilter = languageMap[selectedLanguage.value]
        }

        filtered = filtered.filter(voice => {
          // Check if voice.language array contains the language code
          return voice.language && Array.isArray(voice.language) && voice.language.includes(languageFilter)
        })
        console.log('语言筛选后:', filtered.length)
      }

      // Filter by gender
      if (selectedGender.value !== '全部性别') {
        // Convert Chinese gender display back to English for filtering
        let genderFilter = selectedGender.value
        if (selectedGender.value === '女性') {
          genderFilter = 'female'
        } else if (selectedGender.value === '男性') {
          genderFilter = 'male'
        }

        filtered = filtered.filter(voice => voice.gender === genderFilter)
        console.log('性别筛选后:', filtered.length)
      }

      // Filter by search query
      if (searchQuery.value) {
        filtered = filtered.filter(voice =>
          voice.name.toLowerCase().includes(searchQuery.value.toLowerCase())
        )
        console.log('搜索筛选后:', filtered.length)
      }

      console.log('最终筛选结果:', filtered.length)
      return filtered
    })

    // 反转段落数组，用于从下往上显示
    const reversedSegments = computed(() => {
      return audioSegments.value.map((segment, index) => ({
        segment,
        originalIndex: index
      })).reverse()
    })

    // Check if voice is female based on name
    const isFemaleVoice = (name) => {
      return name.toLowerCase().includes('female')
    }

    // Get selected voice data
    const selectedVoiceData = computed(() => {
      return voices.value.find(v => v.voice_type === selectedVoice.value)
    })

    // Emotion items for dropdown
    const emotionItems = computed(() => {
      const items = []
      if (selectedVoiceData.value && selectedVoiceData.value.emotions && emotions.value.length > 0) {
        selectedVoiceData.value.emotions.forEach(emotionName => {
          // Find the emotion data from emotions array
          const emotionData = emotions.value.find(emotion => emotion.name === emotionName)
          if (emotionData) {
            items.push({ value: emotionName, label: emotionData.zh })
          } else {
            // Fallback if emotion not found in emotions data
            items.push({ value: emotionName, label: emotionName })
          }
        })
      }

      // If no emotions found or no neutral emotion in the list, add neutral as default
      if (items.length === 0 || !items.find(item => item.value === 'neutral')) {
        items.unshift({ value: 'neutral', label: t('neutral') })
      }

      return items
    })

    // Get available emotions for selected voice
    const availableEmotions = computed(() => {
      return selectedVoiceData.value?.emotions || []
    })

    // Handle emotion selection
    const handleEmotionSelect = (item) => {
      selectedEmotion.value = item.value
    }

    // Handle voice selection and auto-generate TTS
    const onVoiceSelect = async (voice) => {
      selectedVoice.value = voice.voice_type
      selectedVoiceResourceId.value = voice.resource_id
      // Reset emotion if not available for this voice
      if (voice.emotions && !voice.emotions.includes(selectedEmotion.value)) {
        selectedEmotion.value = ''
      }

      // Auto-generate TTS when voice is selected and text is available
      await generateTTS()
    }

    // Generate TTS and auto-play
    const generateTTS = async () => {
      if (!inputText.value.trim()) {
        inputText.value = t('ttsPlaceholder')
      }

      if (!selectedVoice.value) return

      // 停止当前播放的音频
      if (audioElement.value) {
        audioElement.value.pause()
        audioElement.value.currentTime = 0
      }
      if (currentAudio.value) {
        currentAudio.value.pause()
        currentAudio.value.currentTime = 0
        currentAudio.value = null
      }

      console.log('contextText', contextText.value)
      isGenerating.value = true
      try {
        const response = await fetch('/api/v1/tts/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: inputText.value,
            voice_type: selectedVoice.value,
            context_texts: contextText.value,
            emotion: selectedEmotion.value,
            emotion_scale: emotionScale.value,
            speech_rate: speechRate.value,
            loudness_rate: loudnessRate.value,
            pitch: pitch.value,
            resource_id: selectedVoiceResourceId.value
          })
        })

        if (response.ok) {
          const blob = await response.blob()
          audioUrl.value = URL.createObjectURL(blob)
          // 标记需要自动播放
          shouldAutoPlay.value = true
          addTtsHistoryEntry(
            inputText.value,
            contextText.value,
            {
              voiceType: selectedVoice.value,
              voiceName: selectedVoiceData.value?.name || ''
            }
          )
        } else {
          throw new Error('TTS generation failed')
        }
      } catch (error) {
        console.error('TTS generation error:', error)
        alert(t('ttsGenerationFailed'))
      } finally {
        isGenerating.value = false
      }
    }

    const applyCombinedHistoryEntry = async (entry) => {
      if (!entry) return
      inputText.value = entry.text || ''
      contextText.value = entry.instruction || ''

      if (entry.voiceType) {
        const voice = voices.value.find(v => v.voice_type === entry.voiceType)
        if (voice) {
          await onVoiceSelect(voice)
          return
        }

        selectedVoice.value = entry.voiceType
        selectedVoiceResourceId.value = ''
      }

      nextTick(() => {
        generateTTS()
      })
      showHistoryPanel.value = false
    }

    const applyTextHistoryEntry = (value) => {
      if (!value) return
      inputText.value = value
      showTextHistoryPanel.value = false
    }

    const applyInstructionHistoryEntry = (value) => {
      if (!value) return
      contextText.value = value
      showInstructionHistoryPanel.value = false
    }

    const applyVoiceHistoryEntry = async (voiceType) => {
      if (!voiceType) return
      const voice = voices.value.find(v => v.voice_type === voiceType)
      if (voice) {
        await onVoiceSelect(voice)
      } else {
        selectedVoice.value = voiceType
        selectedVoiceResourceId.value = ''
        nextTick(() => {
          generateTTS()
        })
      }
      showVoiceHistoryPanel.value = false
    }

    const getHistoryVoiceName = (entry) => {
      if (!entry) return ''
      if (entry.voiceName) return entry.voiceName
      if (entry.voiceType) {
        const voice = voices.value.find(v => v.voice_type === entry.voiceType)
        return voice?.name || ''
      }
      return ''
    }

    // 格式化音频时间
    const formatAudioTime = (seconds) => {
      if (!seconds || isNaN(seconds)) return '0:00'
      const mins = Math.floor(seconds / 60)
      const secs = Math.floor(seconds % 60)
      return `${mins}:${secs.toString().padStart(2, '0')}`
    }

    // 切换播放/暂停
    const toggleAudioPlayback = () => {
      if (!audioElement.value) return

      if (audioElement.value.paused) {
        audioElement.value.play().catch(error => {
          console.log('播放失败:', error)
        })
      } else {
        audioElement.value.pause()
      }
    }

    // 音频加载完成
    const onAudioLoaded = () => {
      if (audioElement.value) {
        audioDuration.value = audioElement.value.duration || 0
        // 如果需要自动播放，则播放
        if (shouldAutoPlay.value) {
          setTimeout(() => {
            if (audioElement.value && !audioElement.value.paused) {
              return // 如果已经在播放，不重复播放
            }
            audioElement.value.play().catch(error => {
              console.log('自动播放被阻止:', error)
            })
            shouldAutoPlay.value = false // 重置自动播放标志
          }, 100)
        }
      }
    }

    // 时间更新
    const onTimeUpdate = () => {
      if (audioElement.value && !isDragging.value) {
        currentTime.value = audioElement.value.currentTime || 0
      }
    }

    // 进度条变化处理（点击或拖拽）
    const onProgressChange = (event) => {
      if (audioDuration.value > 0 && audioElement.value && event.target) {
        const newTime = parseFloat(event.target.value)
        currentTime.value = newTime
        // 立即更新音频位置
        audioElement.value.currentTime = newTime
      }
    }

    // 进度条拖拽结束处理
    const onProgressEnd = (event) => {
      if (audioElement.value && audioDuration.value > 0 && event.target) {
        const newTime = parseFloat(event.target.value)
        audioElement.value.currentTime = newTime
        currentTime.value = newTime
      }
      isDragging.value = false
    }

    // 播放结束
    const onAudioEnded = () => {
      isPlaying.value = false
      currentTime.value = 0
    }

    // 监听音频 URL 变化，重置状态
    watch(audioUrl, (newUrl) => {
      if (newUrl) {
        isPlaying.value = false
        currentTime.value = 0
        audioDuration.value = 0
        // 等待 DOM 更新后加载音频
        nextTick(() => {
          if (audioElement.value) {
            audioElement.value.load()
          }
        })
      } else {
        // URL 清空时重置自动播放标志
        shouldAutoPlay.value = false
      }
    })


    // Apply selected voice (emit the generated audio)
    const applySelectedVoice = () => {
      if (audioUrl.value) {
        // Convert the audio URL back to blob and emit
        fetch(audioUrl.value)
          .then(response => response.blob())
          .then(blob => {
            emit('tts-complete', blob)
          })
          .catch(error => {
            console.error('Error converting audio to blob:', error)
            alert(t('applyAudioFailed'))
          })
      }
    }

    // Close modal function
    const closeModal = () => {
      emit('close-modal')
    }

    // Toggle controls panel
    const toggleControls = () => {
      showControls.value = !showControls.value
    }

    // Filter panel functions
    const toggleFilterPanel = () => {
      showFilterPanel.value = !showFilterPanel.value
    }

    const closeFilterPanel = () => {
      showFilterPanel.value = false
    }

    const selectCategory = (category) => {
      selectedCategory.value = category
    }

    const selectVersion = (version) => {
      selectedVersion.value = version
    }

    const selectLanguage = (language) => {
      selectedLanguage.value = language
    }

    const selectGender = (gender) => {
      selectedGender.value = gender
    }

    const resetFilters = () => {
      selectedCategory.value = '全部场景'
      selectedVersion.value = '全部版本'
      selectedLanguage.value = '全部语言'
      selectedGender.value = '全部性别'
    }

    const applyFilters = () => {
      showFilterPanel.value = false
      resetScrollPosition()
    }

    // Convert speech rate to display value (0.5x to 2.0x)
    const getSpeechRateDisplayValue = (value) => {
      // Map -50 to 100 range to 0.5x to 2.0x
      const ratio = (parseInt(value) + 50) / 150 // Convert to 0-1 range
      const speechRate = 0.5 + (ratio * 1.5) // Convert to 0.5-2.0 range
      return `${speechRate.toFixed(1)}x`
    }

    // Convert loudness rate to display value (-100 to 100)
    const getLoudnessDisplayValue = (value) => {
      // Map -50 to 100 range to 50 to 200
      const apiValue = Math.round(parseInt(value)+100)
      return `${apiValue}%`
    }

    // Convert pitch to display value (-100 to 100)
    const getPitchDisplayValue = (value) => {
      // Map -12 to 12 range to -100 to 100 for API
      const apiValue = Math.round(parseInt(value) * 100 / 12)
      return `${apiValue}`
    }

    // Convert language code to Chinese display name
    const getLanguageDisplayName = (langCode) => {
      const languageMap = {
        'chinese': '中文',
        'en_us': '美式英语',
        'en_gb': '英式英语',
        'en_au': '澳洲英语',
        'es': '西语',
        'ja': '日语'
      }
      return languageMap[langCode] || langCode
    }

    // 多段模式相关函数
    const toggleMode = async () => {
      isMultiSegmentMode.value = !isMultiSegmentMode.value
      if (isMultiSegmentMode.value && audioSegments.value.length === 0) {
        // 添加默认示例
        await initDefaultSegments()
      }
    }

    // 初始化默认示例段落
    const initDefaultSegments = async () => {
      // 等待 voices 数据加载完成（最多等待 3 秒）
      let retryCount = 0
      while (voices.value.length === 0 && retryCount < 6) {
        await new Promise(resolve => setTimeout(resolve, 500))
        retryCount++
      }

      // 查找 Vivi 2.0 音色（尝试多种匹配方式）
      let viviVoice = voices.value.find(v =>
        v.name && v.name.toLowerCase().includes('vivi') && v.version === '2.0'
      )
      // 如果没找到，尝试只匹配 vivi（不限制版本）
      if (!viviVoice) {
        viviVoice = voices.value.find(v =>
          v.name && v.name.toLowerCase().includes('vivi')
        )
      }
      if (viviVoice) {
        console.log('找到 Vivi 音色:', viviVoice.name, viviVoice.voice_type)
      } else {
        console.warn('未找到 Vivi 音色，将创建段落但不会自动合成')
      }

      // 查找儒雅逸辰音色（尝试多种匹配方式）
      let ruyayiVoice = voices.value.find(v =>
        v.name && v.name.includes('儒雅逸辰')
      )
      // 如果没找到，尝试匹配逸辰
      if (!ruyayiVoice) {
        ruyayiVoice = voices.value.find(v =>
          v.name && v.name.includes('逸辰')
        )
      }
      // 如果还没找到，尝试匹配包含"儒雅"的
      if (!ruyayiVoice) {
        ruyayiVoice = voices.value.find(v =>
          v.name && v.name.includes('儒雅')
        )
      }
      if (ruyayiVoice) {
        console.log('找到儒雅逸辰音色:', ruyayiVoice.name, ruyayiVoice.voice_type)
      } else {
        console.warn('未找到儒雅逸辰音色，将创建段落但不会自动合成')
      }

      // 创建第一段：Vivi 2.0
      const segment1 = {
        id: Date.now() + Math.random(),
        text: '今天天气好好呀，要一起出去走走吗～',
        voice: viviVoice ? viviVoice.voice_type : '',
        voiceData: viviVoice || null,
        audioUrl: '',
        audioBlob: null,
        duration: 0,
        currentTime: 0,
        isGenerating: false,
        contextText: '用少女俏皮可爱的音色说'
      }
      audioSegments.value.push(segment1)

      // 创建第二段：儒雅逸辰
      const segment2 = {
        id: Date.now() + Math.random() + 1,
        text: '好啊，小傻瓜，晚上想不想吃火锅啊？',
        voice: ruyayiVoice ? ruyayiVoice.voice_type : '',
        voiceData: ruyayiVoice || null,
        audioUrl: '',
        audioBlob: null,
        duration: 0,
        currentTime: 0,
        isGenerating: false,
        contextText: '磁性的低音炮，宠溺的语气'
      }
      audioSegments.value.push(segment2)

      // 等待 DOM 更新后自动合成
      await nextTick()

      // 自动合成第一段（如果找到了音色）
      if (segment1.voice && segment1.text.trim()) {
        try {
          await generateSegmentTTS(0)
        } catch (error) {
          console.error('第一段合成失败:', error)
        }
      }

      // 等待第一段合成完成后再合成第二段（如果找到了音色）
      if (segment2.voice && segment2.text.trim()) {
        try {
          await generateSegmentTTS(1)
        } catch (error) {
          console.error('第二段合成失败:', error)
        }
      }
    }

    const addSegment = () => {
      audioSegments.value.push({
        id: Date.now() + Math.random(),
        text: '',
        voice: '',
        voiceData: null,
        audioUrl: '',
        audioBlob: null,
        duration: 0,
        currentTime: 0,
        isGenerating: false,
        contextText: ''
      })
    }

    const copySegment = (index, event) => {
      const segment = audioSegments.value[index]
      if (!segment) return

      // 创建段落的深拷贝，包括所有属性
      const copiedSegment = {
        id: Date.now() + Math.random(), // 新的唯一 ID
        text: segment.text || '',
        voice: segment.voice || '',
        voiceData: segment.voiceData ? { ...segment.voiceData } : null, // 浅拷贝 voiceData 对象
        audioUrl: '', // 新段落没有音频，需要重新生成
        audioBlob: null, // 不复制音频 blob，需要重新生成
        duration: 0, // 重置时长
        currentTime: 0, // 重置当前时间
        isGenerating: false, // 重置生成状态
        contextText: segment.contextText || '' // 复制语音指令
      }

      // 将复制的段落添加到列表末尾
      audioSegments.value.push(copiedSegment)

      // 添加视觉反馈：图标临时变为对勾
      if (event && event.target) {
        const button = event.target.closest('button')
        if (button) {
          const originalIcon = button.querySelector('i')
          if (originalIcon) {
            originalIcon.className = 'fas fa-check text-xs'
            setTimeout(() => {
              originalIcon.className = 'fas fa-copy text-xs'
            }, 1000)
          }
        }
      }

      console.log(t('segmentCopied'))
    }

    const removeSegment = (index) => {
      if (audioSegments.value[index].audioUrl) {
        URL.revokeObjectURL(audioSegments.value[index].audioUrl)
      }
      audioSegments.value.splice(index, 1)
      // 重新合并音频
      if (audioSegments.value.length > 0) {
        mergeAllSegments()
      } else {
        mergedAudioUrl.value = ''
      }
    }

    // 设置段落音色选择器的 ref
    const setSegmentVoiceSelectorRef = (index, el) => {
      if (el) {
        if (!segmentVoiceSelectors.value) {
          segmentVoiceSelectors.value = {}
        }
        segmentVoiceSelectors.value[index] = el
      } else {
        // 元素被卸载时，清理 ref
        if (segmentVoiceSelectors.value && segmentVoiceSelectors.value[index]) {
          delete segmentVoiceSelectors.value[index]
        }
      }
    }

    const selectVoiceForSegment = (index) => {
      if (selectedSegmentIndex.value === index && showVoiceSelector.value) {
        // 如果点击的是已选中的，则关闭
        selectedSegmentIndex.value = -1
        showVoiceSelector.value = false
      } else {
        // 先关闭其他可能打开的选择器
        if (showVoiceSelector.value) {
          showVoiceSelector.value = false
        }
        // 使用 setTimeout 延迟打开，避免点击事件立即触发 handleClickOutside
        setTimeout(() => {
          selectedSegmentIndex.value = index
          showVoiceSelector.value = true
          // 等待 DOM 更新后再更新位置
          nextTick(() => {
            updateDropdownPosition(index)
          })
        }, 10)
      }
    }

    const onVoiceSelectForSegment = (voice) => {
      if (selectedSegmentIndex.value >= 0) {
        const segment = audioSegments.value[selectedSegmentIndex.value]
        segment.voice = voice.voice_type
        segment.voiceData = voice
        segment.contextText = '' // 重置语音指令
        showInstructionInput.value = -1
      }
      selectedSegmentIndex.value = -1
      showVoiceSelector.value = false
    }

    // 更新下拉菜单位置
    const updateDropdownPosition = (index) => {
      // 使用双重 nextTick 确保 DOM 完全更新
      nextTick(() => {
        nextTick(() => {
          if (!segmentVoiceSelectors || !segmentVoiceSelectors.value) {
            return
          }

          const container = segmentVoiceSelectors.value[index]
          if (!container) {
            // 延迟重试，因为 ref 可能还没绑定
            setTimeout(() => {
              const retryContainer = segmentVoiceSelectors.value?.[index]
              if (retryContainer) {
                const rect = retryContainer.getBoundingClientRect()
                dropdownStyle.value = {
                  position: 'fixed',
                  top: `${rect.bottom + 8}px`,
                  left: `${rect.left}px`,
                  width: '256px',
                  zIndex: 10000,
                  maxHeight: '384px',
                  overflowY: 'auto',
                  pointerEvents: 'auto'
                }
              }
            }, 100)
            return
          }

          const rect = container.getBoundingClientRect()
          dropdownStyle.value = {
            position: 'fixed',
            top: `${rect.bottom + 8}px`,
            left: `${rect.left}px`,
            width: '256px',
            zIndex: '10000',
            maxHeight: '384px',
            overflowY: 'auto',
            pointerEvents: 'auto'
          }
        })
      })
    }

    // 计算是否应该显示下拉菜单（已移除，直接在模板中使用条件判断）

    // 监听窗口滚动和调整大小，更新下拉菜单位置
    const handleScrollOrResize = () => {
      if (showVoiceSelector.value && selectedSegmentIndex.value >= 0) {
        updateDropdownPosition(selectedSegmentIndex.value)
      }
    }

    // 在打开下拉菜单时添加监听器
    watch([showVoiceSelector, selectedSegmentIndex], () => {
      if (showVoiceSelector.value && selectedSegmentIndex.value >= 0) {
        window.addEventListener('scroll', handleScrollOrResize, true)
        window.addEventListener('resize', handleScrollOrResize)
        // 更新位置
        nextTick(() => {
          updateDropdownPosition(selectedSegmentIndex.value)
        })
      } else {
        window.removeEventListener('scroll', handleScrollOrResize, true)
        window.removeEventListener('resize', handleScrollOrResize)
      }
    })

    // 监听合并音频 URL 变化，自动加载音频
    watch(mergedAudioUrl, async (newUrl, oldUrl) => {
      if (newUrl) {
        // 等待 DOM 更新
        await nextTick()

        if (mergedAudioElement.value) {
          console.log('检测到合并音频 URL 变化，准备加载音频:', newUrl)

          // 如果 URL 改变，先清理旧的
          if (oldUrl && oldUrl !== newUrl) {
            mergedAudioElement.value.src = ''
            mergedAudioElement.value.load()
            await new Promise(resolve => setTimeout(resolve, 50))
          }

          // 设置新的 src
          mergedAudioElement.value.src = newUrl
          console.log('音频 src 已设置:', mergedAudioElement.value.src)

          // 监听加载错误
          const handleError = (e) => {
            console.error('合并音频加载错误:', {
              error: e,
              errorCode: mergedAudioElement.value.error?.code,
              errorMessage: mergedAudioElement.value.error?.message,
              src: mergedAudioElement.value.src,
              readyState: mergedAudioElement.value.readyState,
              networkState: mergedAudioElement.value.networkState
            })
            alert(t('mergedAudioLoadFailed', { error: mergedAudioElement.value.error?.message || t('unknownError') }))
          }
          mergedAudioElement.value.addEventListener('error', handleError, { once: true })

          // 监听加载成功
          const handleLoadedMetadata = () => {
            console.log('合并音频元数据加载成功:', {
              duration: mergedAudioElement.value.duration,
              readyState: mergedAudioElement.value.readyState
            })
            if (mergedAudioElement.value) {
              mergedAudioDuration.value = mergedAudioElement.value.duration || 0
            }
          }
          mergedAudioElement.value.addEventListener('loadedmetadata', handleLoadedMetadata, { once: true })

          const handleLoadedData = () => {
            console.log('合并音频数据加载成功，可以播放')
          }
          mergedAudioElement.value.addEventListener('loadeddata', handleLoadedData, { once: true })

          // 加载音频
          mergedAudioElement.value.load()
          console.log('已调用 load()，readyState:', mergedAudioElement.value.readyState)
        } else {
          console.warn('音频元素不存在，URL:', newUrl)
          // 如果元素不存在，稍后重试
          setTimeout(() => {
            if (mergedAudioElement.value && mergedAudioUrl.value === newUrl) {
              mergedAudioElement.value.src = newUrl
              mergedAudioElement.value.load()
            }
          }, 200)
        }
      } else {
        // URL 清空时，清理音频元素
        if (mergedAudioElement.value) {
          mergedAudioElement.value.src = ''
          mergedAudioElement.value.load()
        }
      }
    })

    // 点击外部关闭音色选择器
    const handleClickOutside = (event) => {
      if (!showVoiceSelector.value) return

      // 检查是否点击在音色选择器容器内（包括下拉菜单）
      const clickedInContainer = event.target.closest('.voice-selector-container') ||
                                  event.target.closest('.voice-selector-component') ||
                                  event.target.closest('.voice-selector-dropdown')

      if (!clickedInContainer) {
        showVoiceSelector.value = false
        selectedSegmentIndex.value = -1
      }
    }

    onUnmounted(() => {
      document.removeEventListener('click', handleClickOutside)
      if (currentAudio.value) {
        currentAudio.value.pause()
        currentAudio.value = null
      }
      // 清理音频URL
      if (audioUrl.value) {
        URL.revokeObjectURL(audioUrl.value)
      }
      // 清理多段模式的音频URL
      audioSegments.value.forEach(segment => {
        if (segment.audioUrl) {
          URL.revokeObjectURL(segment.audioUrl)
        }
      })
      if (mergedAudioUrl.value) {
        URL.revokeObjectURL(mergedAudioUrl.value)
      }
    })

    const generateSegmentTTS = async (index) => {
      const segment = audioSegments.value[index]
      if (!segment.text.trim() || !segment.voice) return

      segment.isGenerating = true
      try {
        const response = await fetch('/api/v1/tts/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            text: segment.text,
            voice_type: segment.voice,
            context_texts: segment.contextText || '',
            emotion: 'neutral',
            emotion_scale: 3,
            speech_rate: 0,
            loudness_rate: 0,
            pitch: 0,
            resource_id: segment.voiceData?.resource_id || ''
          })
        })

        if (response.ok) {
          const blob = await response.blob()
          segment.audioBlob = blob
          segment.audioUrl = URL.createObjectURL(blob)
          // 等待音频加载后获取时长
          await nextTick()
          if (segmentAudioElements.value[index]) {
            segmentAudioElements.value[index].load()
          }
        } else {
          throw new Error('TTS generation failed')
        }
      } catch (error) {
        console.error('TTS generation error:', error)
        alert(t('ttsGenerationFailed'))
      } finally {
        segment.isGenerating = false
        // 自动合并所有段
        await mergeAllSegments()
      }
    }

    const onSegmentAudioLoaded = (index) => {
      const audioEl = segmentAudioElements.value[index]
      if (audioEl) {
        audioSegments.value[index].duration = audioEl.duration || 0
        if (!audioSegments.value[index].currentTime) {
          audioSegments.value[index].currentTime = 0
        }
        // 重新计算合并音频的总时长
        mergeAllSegments()
      }
    }

    const onSegmentTimeUpdate = (index) => {
      const audioEl = segmentAudioElements.value[index]
      if (audioEl && audioSegments.value[index]) {
        audioSegments.value[index].currentTime = audioEl.currentTime || 0

        // 如果是连续播放模式，更新累计时间
        if (isPlayingMerged.value && playingSegmentIndex.value === index) {
          let segmentIndexInSequence = 0
          for (let i = 0; i < index; i++) {
            if (audioSegments.value[i].audioUrl && audioSegments.value[i].duration > 0) {
              segmentIndexInSequence++
            }
          }

          if (segmentIndexInSequence < segmentStartTimes.value.length) {
            const segmentStartTime = segmentStartTimes.value[segmentIndexInSequence]
            mergedCurrentTime.value = segmentStartTime + (audioEl.currentTime || 0)
          }
        }
      }
    }

    const onSegmentAudioEnded = (index) => {
      // 如果是连续播放模式，自动播放下一段
      if (isPlayingMerged.value && playingSegmentIndex.value === index) {
        let segmentIndexInSequence = 0
        for (let i = 0; i < index; i++) {
          if (audioSegments.value[i].audioUrl && audioSegments.value[i].duration > 0) {
            segmentIndexInSequence++
          }
        }
        playNextSegment(segmentIndexInSequence)
      } else {
        // 单独播放模式
        playingSegmentIndex.value = -1
        if (audioSegments.value[index]) {
          audioSegments.value[index].currentTime = 0
        }
      }
    }

    // 处理进度条点击
    const handleSegmentProgressClick = (index, event) => {
      const segment = audioSegments.value[index]
      if (!segment.audioUrl || segment.isGenerating || !segment.duration) return

      const audioEl = segmentAudioElements.value[index]
      if (!audioEl) return

      const progressBar = event.currentTarget
      const rect = progressBar.getBoundingClientRect()
      const clickX = event.clientX - rect.left
      const percentage = clickX / rect.width
      const newTime = Math.max(0, Math.min(segment.duration, percentage * segment.duration))

      audioEl.currentTime = newTime
      segment.currentTime = newTime

      // 如果音频未播放，则开始播放
      if (audioEl.paused) {
        // 停止其他正在播放的段
        Object.values(segmentAudioElements.value).forEach((el, i) => {
          if (el && i !== index && !el.paused) {
            el.pause()
          }
        })
        audioEl.play().catch(error => {
          console.log('播放失败:', error)
        })
        playingSegmentIndex.value = index
      }
    }

    const playSegment = (index) => {
      const audioEl = segmentAudioElements.value[index]
      if (!audioEl) return

      // 停止其他正在播放的段
      Object.values(segmentAudioElements.value).forEach((el, i) => {
        if (el && i !== index && !el.paused) {
          el.pause()
          if (audioSegments.value[i]) {
            audioSegments.value[i].currentTime = el.currentTime
          }
        }
      })

      if (audioEl.paused) {
        audioEl.play().catch(error => {
          console.log('播放失败:', error)
        })
        playingSegmentIndex.value = index
      } else {
        audioEl.pause()
        playingSegmentIndex.value = -1
        if (audioSegments.value[index]) {
          audioSegments.value[index].currentTime = audioEl.currentTime
        }
      }
    }

    const mergeAllSegments = async () => {
      const segmentsWithAudio = audioSegments.value.filter(s => s.audioUrl && s.duration > 0)
      if (segmentsWithAudio.length === 0) {
        mergedAudioDuration.value = 0
        mergedCurrentTime.value = 0
        segmentStartTimes.value = []
        return
      }

      // 只计算总时长和每段的开始时间，不实际合并文件
      let totalDuration = 0
      const startTimes = [0] // 第一段从 0 开始

      for (let i = 0; i < segmentsWithAudio.length; i++) {
        const segment = segmentsWithAudio[i]
        if (i > 0) {
          startTimes.push(totalDuration)
        }
        totalDuration += segment.duration || 0
      }

      mergedAudioDuration.value = totalDuration
      segmentStartTimes.value = startTimes

      console.log('计算总时长:', totalDuration, '秒，共', segmentsWithAudio.length, '段')
    }

    // 将 AudioBuffer 转换为 WAV
    const audioBufferToWav = (buffer) => {
      const length = buffer.length
      const numberOfChannels = buffer.numberOfChannels
      const sampleRate = buffer.sampleRate
      const bytesPerSample = 2 // 16-bit
      const blockAlign = numberOfChannels * bytesPerSample
      const byteRate = sampleRate * blockAlign
      const dataSize = length * blockAlign
      // RIFF chunk size = 文件总大小 - 8 (不包括 RIFF 和 size 字段本身)
      // 文件总大小 = 44 (文件头) + dataSize
      const riffChunkSize = 36 + dataSize // 36 = 44 - 8

      const arrayBuffer = new ArrayBuffer(44 + dataSize)
      const view = new DataView(arrayBuffer)

      // WAV 文件头写入函数
      const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i))
        }
      }

      // RIFF header (12 bytes: 4 + 4 + 4)
      writeString(0, 'RIFF')
      view.setUint32(4, riffChunkSize, true) // RIFF chunk size (little-endian)
      writeString(8, 'WAVE')

      // fmt chunk (24 bytes: 4 + 4 + 16)
      writeString(12, 'fmt ')
      view.setUint32(16, 16, true) // fmt chunk size (little-endian)
      view.setUint16(20, 1, true) // Audio format: 1 = PCM (little-endian)
      view.setUint16(22, numberOfChannels, true) // Number of channels (little-endian)
      view.setUint32(24, sampleRate, true) // Sample rate (little-endian)
      view.setUint32(28, byteRate, true) // Byte rate (little-endian)
      view.setUint16(32, blockAlign, true) // Block align (little-endian)
      view.setUint16(34, 16, true) // Bits per sample (little-endian)

      // data chunk header (8 bytes: 4 + 4)
      writeString(36, 'data')
      view.setUint32(40, dataSize, true) // Data size (little-endian)

      // 写入音频数据 (PCM 16-bit little-endian)
      let offset = 44
      for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
          const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]))
          // 转换为 16-bit PCM (little-endian)
          // 范围: -32768 到 32767
          const int16Sample = sample < 0
            ? Math.max(-0x8000, Math.floor(sample * 0x8000))
            : Math.min(0x7FFF, Math.floor(sample * 0x7FFF))
          view.setInt16(offset, int16Sample, true) // little-endian
          offset += 2
        }
      }

      return arrayBuffer
    }

    // 播放下一段音频
    const playNextSegment = (currentIndex) => {
      const segmentsWithAudio = audioSegments.value.filter(s => s.audioUrl && s.duration > 0)
      const nextIndex = currentIndex + 1

      if (nextIndex >= segmentsWithAudio.length) {
        // 所有段播放完成
        isPlayingMerged.value = false
        playingSegmentIndex.value = -1
        mergedCurrentTime.value = 0
        return
      }

      // 找到下一段在 audioSegments 中的实际索引
      let actualIndex = -1
      let foundCount = 0
      for (let i = 0; i < audioSegments.value.length; i++) {
        if (audioSegments.value[i].audioUrl && audioSegments.value[i].duration > 0) {
          if (foundCount === nextIndex) {
            actualIndex = i
            break
          }
          foundCount++
        }
      }

      if (actualIndex >= 0) {
        playingSegmentIndex.value = actualIndex
        const audioEl = segmentAudioElements.value[actualIndex]
        if (audioEl) {
          audioEl.currentTime = 0
          audioEl.play().catch(error => {
            console.error('播放下一段失败:', error)
            isPlayingMerged.value = false
            playingSegmentIndex.value = -1
          })
        }
      }
    }

    const toggleMergedAudioPlayback = () => {
      const segmentsWithAudio = audioSegments.value.filter(s => s.audioUrl && s.duration > 0)
      if (segmentsWithAudio.length === 0) {
        console.warn('没有可播放的音频段')
        return
      }

      if (isPlayingMerged.value) {
        // 暂停播放：停止当前播放的段
        if (playingSegmentIndex.value >= 0) {
          const audioEl = segmentAudioElements.value[playingSegmentIndex.value]
          if (audioEl) {
            audioEl.pause()
          }
        }
        isPlayingMerged.value = false
        playingSegmentIndex.value = -1
      } else {
        // 开始播放：从第一段开始，或从上次暂停的位置继续
        let startIndex = 0
        let foundCount = 0

        // 如果之前有播放位置，找到对应的段
        if (mergedCurrentTime.value > 0 && segmentStartTimes.value.length > 0) {
          for (let i = segmentStartTimes.value.length - 1; i >= 0; i--) {
            if (mergedCurrentTime.value >= segmentStartTimes.value[i]) {
              startIndex = i
              break
            }
          }
        }

        // 找到实际索引
        let actualIndex = -1
        for (let i = 0; i < audioSegments.value.length; i++) {
          if (audioSegments.value[i].audioUrl && audioSegments.value[i].duration > 0) {
            if (foundCount === startIndex) {
              actualIndex = i
              break
            }
            foundCount++
          }
        }

        if (actualIndex >= 0) {
          playingSegmentIndex.value = actualIndex
          const audioEl = segmentAudioElements.value[actualIndex]
          if (audioEl) {
            // 如果从中间开始，计算当前段内的位置
            if (startIndex > 0 && segmentStartTimes.value[startIndex] > 0) {
              const segmentStartTime = segmentStartTimes.value[startIndex]
              const segmentOffset = mergedCurrentTime.value - segmentStartTime
              audioEl.currentTime = Math.max(0, Math.min(segmentOffset, audioSegments.value[actualIndex].duration))
            } else {
              audioEl.currentTime = 0
            }

            audioEl.play().catch(error => {
              console.error('播放失败:', error)
              alert(t('playbackFailed', { error: error.message || t('unknownError') }))
            })
            isPlayingMerged.value = true
          }
        }
      }
    }

    const onMergedAudioLoaded = () => {
      // 已废弃，保留用于兼容
    }

    const onMergedTimeUpdate = () => {
      // 已废弃，时间更新由 onSegmentTimeUpdate 处理
    }

    const onMergedProgressChange = (event) => {
      if (mergedAudioDuration.value > 0 && event.target) {
        const newTime = parseFloat(event.target.value)
        mergedCurrentTime.value = newTime

        // 找到对应的段和段内位置
        if (segmentStartTimes.value.length > 0) {
          let targetSegmentIndex = -1
          let segmentIndexInSequence = 0

          // 找到目标段
          for (let i = segmentStartTimes.value.length - 1; i >= 0; i--) {
            if (newTime >= segmentStartTimes.value[i]) {
              segmentIndexInSequence = i
              break
            }
          }

          // 找到实际索引
          let foundCount = 0
          for (let i = 0; i < audioSegments.value.length; i++) {
            if (audioSegments.value[i].audioUrl && audioSegments.value[i].duration > 0) {
              if (foundCount === segmentIndexInSequence) {
                targetSegmentIndex = i
                break
              }
              foundCount++
            }
          }

          if (targetSegmentIndex >= 0) {
            const segmentStartTime = segmentStartTimes.value[segmentIndexInSequence]
            const segmentOffset = newTime - segmentStartTime
            const audioEl = segmentAudioElements.value[targetSegmentIndex]

            if (audioEl) {
              audioEl.currentTime = Math.max(0, Math.min(segmentOffset, audioSegments.value[targetSegmentIndex].duration))

              // 如果正在播放，切换到目标段
              if (isPlayingMerged.value) {
                // 停止当前播放的段
                if (playingSegmentIndex.value >= 0 && playingSegmentIndex.value !== targetSegmentIndex) {
                  const currentAudioEl = segmentAudioElements.value[playingSegmentIndex.value]
                  if (currentAudioEl) {
                    currentAudioEl.pause()
                  }
                }

                playingSegmentIndex.value = targetSegmentIndex
                audioEl.play().catch(error => {
                  console.error('跳转播放失败:', error)
                })
              }
            }
          }
        }
      }
    }

    const onMergedAudioEnded = () => {
      // 已废弃，结束处理由 onSegmentAudioEnded 处理
      isPlayingMerged.value = false
      mergedCurrentTime.value = 0
    }

    const applyMergedAudio = async () => {
      const segmentsWithAudio = audioSegments.value.filter(s => s.audioBlob)
      if (segmentsWithAudio.length === 0) {
        alert(t('noSegmentsToApply'))
        return
      }

      try {
        // 临时合并所有段用于应用
        const audioContext = new (window.AudioContext || window.webkitAudioContext)()
        const audioBuffers = []

        // 加载并解码所有音频段
        for (const segment of segmentsWithAudio) {
          try {
            const arrayBuffer = await segment.audioBlob.arrayBuffer()
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)
            audioBuffers.push(audioBuffer)
          } catch (error) {
            console.error('音频段解码失败:', error)
            throw new Error(t('audioDecodeFailed', { error: error.message || t('unknownError') }))
          }
        }

        if (audioBuffers.length === 0) {
          throw new Error('没有可用的音频段')
        }

        // 使用第一个音频的参数作为目标格式
        const targetSampleRate = audioBuffers[0].sampleRate
        const targetChannels = audioBuffers[0].numberOfChannels

        // 计算总长度
        let totalLength = 0
        for (const buffer of audioBuffers) {
          totalLength += buffer.length
        }

        // 创建合并后的音频缓冲区
        const mergedBuffer = audioContext.createBuffer(
          targetChannels,
          totalLength,
          targetSampleRate
        )

        // 合并所有音频数据
        let offset = 0
        for (const buffer of audioBuffers) {
          const bufferLength = buffer.length

          for (let channel = 0; channel < targetChannels; channel++) {
            const mergedChannelData = mergedBuffer.getChannelData(channel)

            if (channel < buffer.numberOfChannels) {
              const sourceChannelData = buffer.getChannelData(channel)
              mergedChannelData.set(sourceChannelData, offset)
            } else {
              const sourceChannelData = buffer.getChannelData(0)
              mergedChannelData.set(sourceChannelData, offset)
            }
          }

          offset += bufferLength
        }

        // 转换为 WAV
        const wav = audioBufferToWav(mergedBuffer)
        const blob = new Blob([wav], { type: 'audio/wav' })

        emit('tts-complete', blob)
      } catch (error) {
        console.error('合并音频失败:', error)
        alert(t('mergedAudioFailed', { error: error.message || t('unknownError') }))
      }
    }

    // 拖拽排序函数
    const handleDragStart = (index, event) => {
      draggingSegmentIndex.value = index
      event.dataTransfer.effectAllowed = 'move'
      event.dataTransfer.setData('text/plain', index.toString())
    }

    const handleDragEnd = () => {
      draggingSegmentIndex.value = -1
      dragOverSegmentIndex.value = -1
    }

    const handleDragOver = (index, event) => {
      event.preventDefault()
      event.dataTransfer.dropEffect = 'move'
      if (draggingSegmentIndex.value !== index && draggingSegmentIndex.value >= 0) {
        dragOverSegmentIndex.value = index
      }
    }

    const handleDragLeave = (index) => {
      // 只有当离开整个拖拽区域时才清除 dragOverSegmentIndex
      // 这里简化处理，在 drop 时再清除
    }

    const handleDrop = (targetIndex, event) => {
      event.preventDefault()
      event.stopPropagation()

      const draggedIndex = draggingSegmentIndex.value
      if (draggedIndex === -1 || draggedIndex === targetIndex) {
        draggingSegmentIndex.value = -1
        dragOverSegmentIndex.value = -1
        return
      }

      // 重新排序段落
      const segments = [...audioSegments.value]
      const draggedSegment = segments[draggedIndex]
      segments.splice(draggedIndex, 1)
      segments.splice(targetIndex, 0, draggedSegment)

      // 更新 audioSegments
      audioSegments.value = segments

      // 重新计算合并音频的总时长
      mergeAllSegments()

      draggingSegmentIndex.value = -1
      dragOverSegmentIndex.value = -1
    }

    return {
      t,
      inputText,
      contextText,
      selectedVoice,
      searchQuery,
      speechRate,
      loudnessRate,
      pitch,
      emotionScale,
      selectedEmotion,
      isGenerating,
      audioUrl,
      audioElement,
      isPlaying,
      audioDuration,
      currentTime,
      isDragging,
      onProgressChange,
      onProgressEnd,
      voices,
      voiceListContainer,
      showControls,
      showFilterPanel,
      filteredVoices,
      isFemaleVoice,
      selectedVoiceData,
      formatAudioTime,
      toggleAudioPlayback,
      onAudioLoaded,
      onTimeUpdate,
      onAudioEnded,
      availableEmotions,
      onVoiceSelect,
      generateTTS,
      applySelectedVoice,
      closeModal,
      toggleControls,
      toggleFilterPanel,
      closeFilterPanel,
      selectCategory,
      selectVersion,
      selectLanguage,
      selectGender,
      resetFilters,
      applyFilters,
      getSpeechRateDisplayValue,
      getLoudnessDisplayValue,
      getPitchDisplayValue,
      getLanguageDisplayName,
      emotionItems,
      handleEmotionSelect,
      selectedCategory,
      categories,
      selectedVoiceResourceId,
      version,
      selectedVersion,
      selectedLanguage,
      languages,
      selectedGender,
      genders,
      resetScrollPosition,
      translateCategory,
      translateVersion,
      translateLanguage,
      translateGender,
      ttsHistory,
      showHistoryPanel,
      openHistoryPanel,
      closeHistoryPanel,
      applyCombinedHistoryEntry,
      applyTextHistoryEntry,
      applyInstructionHistoryEntry,
      applyVoiceHistoryEntry,
      getHistoryVoiceName,
      handleDeleteHistoryEntry,
      showTextHistoryPanel,
      showInstructionHistoryPanel,
      showVoiceHistoryPanel,
      openTextHistoryPanel,
      openInstructionHistoryPanel,
      openVoiceHistoryPanel,
      closeTextHistoryPanel,
      closeInstructionHistoryPanel,
      closeVoiceHistoryPanel,
      // 多段模式相关
      isMultiSegmentMode,
      audioSegments,
      reversedSegments,
      mergedAudioUrl,
      isMerging,
      isPlayingMerged,
      mergedAudioDuration,
      mergedCurrentTime,
      playingSegmentIndex,
      segmentAudioElements,
      showInstructionInput,
      toggleMode,
      addSegment,
      copySegment,
      removeSegment,
      setSegmentVoiceSelectorRef,
      selectVoiceForSegment,
      onVoiceSelectForSegment,
      dropdownStyle,
      generateSegmentTTS,
      onSegmentAudioLoaded,
      onSegmentTimeUpdate,
      onSegmentAudioEnded,
      handleSegmentProgressClick,
      playSegment,
      toggleMergedAudioPlayback,
      onMergedAudioLoaded,
      onMergedTimeUpdate,
      onMergedProgressChange,
      onMergedAudioEnded,
      applyMergedAudio,
      dropdownContainerRef,
      showVoiceSelector,
      selectedSegmentIndex,
      // 拖拽相关
      draggingSegmentIndex,
      dragOverSegmentIndex,
      handleDragStart,
      handleDragEnd,
      handleDragOver,
      handleDragLeave,
      handleDrop
    }
  }
}
</script>

<style scoped>
/* Apple 风格极简设计 - 大部分样式已通过 Tailwind CSS 在 template 中定义 */

/* 隐藏 radio input */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* 深色模式下增强滑动条可见性 */
.dark input[type="range"]::-webkit-slider-thumb {
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.15);
}

.dark input[type="range"]::-moz-range-thumb {
  box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.15);
}
</style>
