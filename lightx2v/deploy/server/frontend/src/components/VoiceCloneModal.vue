<template>
  <div class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[150] flex items-center justify-center p-2">
    <div class="relative w-full h-full max-w-4xl max-h-[100vh] bg-white/95 dark:bg-[#1e1e1e]/95 backdrop-blur-[40px] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.2)] overflow-hidden flex flex-col">

      <!-- Method Selection Screen -->
      <div v-if="currentScreen === 'METHOD_SELECT'" class="flex-1 flex flex-col">
        <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8">
          <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('selectCloneMethod') }}</h3>
          <button @click="closeModal" class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] rounded-full transition-all">
            <i class="fas fa-times text-sm"></i>
          </button>
        </div>
        <div class="flex-1 overflow-y-auto p-6">
          <p class="text-sm text-[#86868b] dark:text-[#98989d] mb-6">{{ t('cloneMethodHint') }}</p>
          <div class="space-y-4">
            <button @click="selectMethod('RECORD')" class="w-full p-6 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-xl hover:bg-white dark:hover:bg-[#3a3a3c] transition-all text-left flex items-center gap-4">
              <div class="w-14 h-14 rounded-xl bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                <i class="fas fa-microphone text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
              </div>
              <div class="flex-1">
                <h4 class="font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-1">{{ t('recordAudio') }}</h4>
                <p class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('recordAudioHint') }}</p>
              </div>
              <i class="fas fa-chevron-right text-[#86868b] dark:text-[#98989d]"></i>
            </button>

            <button @click="videoInputRef?.click()" class="w-full p-6 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-xl hover:bg-white dark:hover:bg-[#3a3a3c] transition-all text-left flex items-center gap-4">
              <div class="w-14 h-14 rounded-xl bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                <i class="fas fa-video text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
              </div>
              <div class="flex-1">
                <h4 class="font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-1">{{ t('extractFromVideo') }}</h4>
                <p class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('extractFromVideoHint') }}</p>
              </div>
              <input type="file" ref="videoInputRef" accept="video/*" class="hidden" @change="handleVideoFile" />
              <i class="fas fa-chevron-right text-[#86868b] dark:text-[#98989d]"></i>
            </button>

            <button @click="audioInputRef?.click()" class="w-full p-6 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-xl hover:bg-white dark:hover:bg-[#3a3a3c] transition-all text-left flex items-center gap-4">
              <div class="w-14 h-14 rounded-xl bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
                <i class="fas fa-upload text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-xl"></i>
              </div>
              <div class="flex-1">
                <h4 class="font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-1">{{ t('importAudioFile') }}</h4>
                <p class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('importAudioFileHint') }}</p>
              </div>
              <input type="file" ref="audioInputRef" accept="audio/*" class="hidden" @change="handleAudioFile" />
              <i class="fas fa-chevron-right text-[#86868b] dark:text-[#98989d]"></i>
            </button>
          </div>
        </div>
      </div>

      <!-- Recording Screen -->
      <div v-else-if="currentScreen === 'RECORDING'" class="flex-1 flex flex-col">
        <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8">
          <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('recordAudio') }}</h3>
          <button @click="currentScreen = 'METHOD_SELECT'" class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] rounded-full transition-all">
            <i class="fas fa-arrow-left text-sm"></i>
          </button>
        </div>
        <div class="flex-1 flex flex-col items-center justify-center p-8 space-y-8">
          <div class="w-full max-w-lg space-y-4">
            <!-- 切换按钮：示例句子 / 自定义文本 -->
            <div class="flex items-center justify-center gap-2 mb-2">
              <button
                @click="useSampleSentence = true"
                :class="`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  useSampleSentence
                    ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
                    : 'bg-white/80 dark:bg-[#2c2c2e]/80 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c]'
                }`"
              >
                {{ t('sampleSentence') }}
              </button>
              <button
                @click="useSampleSentence = false"
                :class="`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  !useSampleSentence
                    ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'
                    : 'bg-white/80 dark:bg-[#2c2c2e]/80 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c]'
                }`"
              >
                {{ t('customText') }}
              </button>
            </div>

            <!-- 示例句子模式 -->
            <div v-if="useSampleSentence" class="space-y-4">
              <div class="flex items-center justify-between text-xs text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] font-semibold">
                <span>{{ t('pleaseReadSentence') }}</span>
                <button @click="changeSentence" class="flex items-center gap-1 hover:opacity-80 transition-opacity">
                  <i class="fas fa-random text-xs"></i>
                  <span>{{ t('changeSentence') }}</span>
                </button>
              </div>
              <div class="min-h-[160px] flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-2xl p-6">
                <p class="text-2xl font-medium text-[#1d1d1f] dark:text-[#f5f5f7] text-center leading-relaxed">"{{ currentSentence }}"</p>
              </div>
            </div>

            <!-- 自定义文本模式 -->
            <div v-else class="space-y-4">
              <div class="text-xs text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] font-semibold">
                <span>{{ t('pleaseReadSentence') }}</span>
              </div>
              <textarea
                v-model="customText"
                :placeholder="t('customTextPlaceholder')"
                class="w-full min-h-[160px] bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-2xl p-6 text-lg font-medium text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all resize-none leading-relaxed"
              ></textarea>
            </div>
          </div>

          <div class="relative">
            <div v-if="isRecording" class="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-30"></div>
            <button
              @click="isRecording ? stopRecording() : startRecording()"
              :class="`relative z-10 w-24 h-24 rounded-full flex items-center justify-center transition-all ${
                isRecording
                  ? 'bg-red-500 text-white scale-110'
                  : 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white hover:scale-105'
              }`"
            >
              <i v-if="isRecording" class="fas fa-stop text-2xl"></i>
              <i v-else class="fas fa-microphone text-2xl"></i>
            </button>
          </div>
          <p class="text-sm text-[#86868b] dark:text-[#98989d]">
            {{ isRecording ? t('clickToStopRecording') : t('clickToStartRecording') }}
          </p>
        </div>
      </div>

      <!-- Audio Editor Screen -->
      <div v-else-if="currentScreen === 'AUDIO_EDITOR' && audioUrl" class="flex-1 flex flex-col">
        <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8">
          <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('editAudio') }}</h3>
          <button @click="currentScreen = 'METHOD_SELECT'" class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] rounded-full transition-all">
            <i class="fas fa-arrow-left text-sm"></i>
          </button>
        </div>
        <div class="flex-1 overflow-y-auto p-6">
          <audio ref="audioRef" :src="audioUrl" class="hidden" @timeupdate="updateTime" @loadedmetadata="updateDuration" @ended="onEnded" />

          <div class="max-w-2xl mx-auto space-y-6">
            <!-- Duration Warning -->
            <div v-if="showDurationWarning" class="flex justify-center animate-in slide-in-from-top-2 fade-in">
              <div class="bg-amber-500/10 border border-amber-500/50 text-amber-500 px-4 py-2 rounded-full flex items-center gap-2 text-sm font-medium shadow-lg backdrop-blur-md">
                <i class="fas fa-exclamation-circle text-xs"></i>
                最多选择15s的音频片段
              </div>
            </div>

            <!-- Audio Visualizer with Interactive Trim Controls -->
            <div class="w-full flex flex-col gap-4">
              <div
                ref="visualizerRef"
                class="w-full h-40 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-2xl flex items-center justify-center relative select-none touch-none overflow-hidden"
              >
                <!-- Waveform Bars -->
                <div
                  ref="waveformContainerRef"
                  class="flex items-center gap-1.5 h-20 w-full justify-center relative z-0"
                >
                  <div
                    v-for="i in waveformBarCount"
                    :key="i"
                    class="w-1.5 rounded-full transition-all duration-300 flex-shrink-0"
                    :style="getWaveformBarStyle(i)"
                  ></div>
                </div>

                <!-- Left Trim Mask (Visible in Trim Mode, not when trimmed) -->
                <div
                  v-if="isTrimMode && !isTrimmed"
                  class="absolute left-0 top-0 bottom-0 bg-white/80 dark:bg-[#2c2c2e]/80 border-r-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] transition-all duration-75 backdrop-blur-sm z-10"
                  :style="{ width: `${trimStartPercent}%` }"
                >
                  <div
                    class="absolute right-[-12px] top-1/2 -translate-y-1/2 w-8 h-12 cursor-ew-resize z-20 flex items-center justify-center group outline-none"
                    @pointerdown="handlePointerDown($event, 'start')"
                    @pointermove="handlePointerMove"
                    @pointerup="handlePointerUp"
                  >
                    <div
                      :class="`w-1.5 h-8 bg-white dark:bg-white rounded-full shadow-lg transition-colors ${
                        dragTarget === 'start' ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)]' : 'group-hover:bg-[color:var(--brand-primary)] dark:group-hover:bg-[color:var(--brand-primary-light)]'
                      }`"
                    ></div>
                  </div>
                </div>

                <!-- Right Trim Mask (Visible in Trim Mode, not when trimmed) -->
                <div
                  v-if="isTrimMode && !isTrimmed"
                  class="absolute right-0 top-0 bottom-0 bg-white/80 dark:bg-[#2c2c2e]/80 border-l-2 border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] transition-all duration-75 z-10"
                  :style="{ width: `${100 - trimEndPercent}%` }"
                >
                  <div
                    class="absolute left-[-12px] top-1/2 -translate-y-1/2 w-8 h-12 cursor-ew-resize z-20 flex items-center justify-center group outline-none"
                    @pointerdown="handlePointerDown($event, 'end')"
                    @pointermove="handlePointerMove"
                    @pointerup="handlePointerUp"
                  >
                    <div
                      :class="`w-1.5 h-8 bg-white dark:bg-white rounded-full shadow-lg transition-colors ${
                        dragTarget === 'end' ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)]' : 'group-hover:bg-[color:var(--brand-primary)] dark:group-hover:bg-[color:var(--brand-primary-light)]'
                      }`"
                    ></div>
                  </div>
                </div>

                <!-- Selection Drag Area (Only in Trim Mode, not when trimmed) -->
                <div
                  v-if="isTrimMode && !isTrimmed"
                  class="absolute top-0 bottom-0 z-0 cursor-grab active:cursor-grabbing hover:bg-white/5 dark:hover:bg-white/5 transition-colors"
                  :style="{ left: `${trimStartPercent}%`, width: `${trimEndPercent - trimStartPercent}%` }"
                  @pointerdown="handlePointerDown($event, 'selection')"
                  @pointermove="handlePointerMove"
                  @pointerup="handlePointerUp"
                  :title="t('dragToMoveSelection')"
                ></div>
              </div>

              <!-- Time Info Row -->
              <div class="flex items-center justify-between text-xs font-mono px-2">
                <div class="text-[#86868b] dark:text-[#98989d]">
                  <span class="block text-[10px] opacity-50">{{ t('start') }}</span>
                  {{ formatTime(trimStart) }}
                </div>

                <div class="flex flex-col items-center justify-center">
                  <span class="text-[10px] text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] mb-0.5">{{ t('selectedDuration') }}</span>
                  <span
                    :class="`text-sm font-bold text-white bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/10 px-2 py-0.5 rounded border ${
                      selectionDuration > 15 ? 'border-red-500/50 text-red-400' : 'border-[color:var(--brand-primary)]/20 dark:border-[color:var(--brand-primary-light)]/20'
                    }`"
                  >
                    {{ selectionDuration.toFixed(1) }}s
                  </span>
                </div>

                <div class="text-[#86868b] dark:text-[#98989d] text-right">
                  <span class="block text-[10px] opacity-50">{{ t('end') }}</span>
                  {{ formatTime(trimEnd) }}
                </div>
              </div>
            </div>

            <!-- Global Progress Bar -->
            <div class="w-full space-y-2 max-w-sm mx-auto">
              <input
                type="range"
                :min="0"
                :max="sliderMax || 100"
                step="0.01"
                :value="sliderValue"
                @input="handleSliderChange"
                @mousedown="isDraggingProgress = true"
                @mouseup="isDraggingProgress = false"
                class="w-full h-1.5 bg-black/6 dark:bg-white/15 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <div class="flex justify-between text-xs text-[#86868b] dark:text-[#98989d] font-mono">
                <span>{{ formatTime(sliderValue) }}</span>
                <span>{{ formatTime(sliderMax) }}</span>
              </div>
            </div>

            <!-- Control Buttons -->
            <div class="flex items-center justify-center gap-6">
              <button
                @click="toggleLoop"
                :class="`p-3 rounded-full transition-all border ${
                  isLooping
                    ? 'text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] border-[color:var(--brand-primary)]/30 dark:border-[color:var(--brand-primary-light)]/30 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/10 shadow-[0_0_10px_rgba(var(--brand-primary-rgb),0.1)]'
                    : 'text-[#86868b] dark:text-[#98989d] border-transparent hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7]'
                }`"
                :title="t('loopPlayback')"
              >
                <i class="fas fa-redo text-lg"></i>
              </button>

              <button
                @click="handleToggleTrimMode"
                :class="`p-4 rounded-full border transition-all ${
                  isTrimMode
                    ? 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-[color:var(--brand-primary)] text-white shadow-[0_0_20px_rgba(var(--brand-primary-rgb),0.4)]'
                    : 'bg-white/80 dark:bg-[#2c2c2e]/80 border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c]'
                }`"
                :title="isTrimMode ? t('confirmTrim') : t('trimMode')"
              >
                <i v-if="isTrimMode" class="fas fa-check text-lg"></i>
                <i v-else class="fas fa-cut text-lg"></i>
              </button>

              <button
                @click="togglePlay"
                class="w-20 h-20 rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white flex items-center justify-center shadow-[0_0_25px_rgba(var(--brand-primary-rgb),0.4)] hover:scale-105 transition-transform hover:shadow-[0_0_35px_rgba(var(--brand-primary-rgb),0.6)]"
              >
                <i v-if="isPlaying" class="fas fa-pause text-2xl"></i>
                <i v-else class="fas fa-play text-2xl ml-1"></i>
              </button>

              <button
                @click="handleReset"
                class="p-4 rounded-full bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:bg-white dark:hover:bg-[#3a3a3c] transition-colors"
                :title="t('reset')"
              >
                <i class="fas fa-redo text-lg"></i>
              </button>

              <button
                @click="cycleSpeed"
                :class="`p-3 w-12 flex flex-col items-center justify-center rounded-full font-mono text-xs font-bold border transition-all ${
                  playbackRate !== 1
                    ? 'text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] border-[color:var(--brand-primary)]/30'
                    : 'text-[#86868b] dark:text-[#98989d] border-transparent hover:border-black/8 dark:hover:border-white/8'
                }`"
                :title="t('playbackSpeed')"
              >
                <span>{{ playbackRate }}x</span>
              </button>
            </div>

            <!-- Hint Text -->
            <p class="text-sm text-[#86868b] dark:text-[#98989d] text-center h-4">
              {{ isTrimMode ? t('dragToAdjustSelection') : isTrimmed ? t('trimConfirmed') : ' ' }}
            </p>

            <!-- User Text Input -->
            <div class="w-full space-y-2 max-w-2xl mx-auto">
              <label class="text-sm font-semibold text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] flex items-center gap-2">
                <i class="fas fa-keyboard text-xs"></i>
                <span>{{ t('audioText') }}</span>
                <span class="text-xs text-[#86868b] dark:text-[#98989d] font-normal">{{ t('optional') }}</span>
              </label>
              <textarea
                v-model="userInputText"
                :placeholder="t('audioTextPlaceholder')"
                class="w-full min-h-[80px] bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 rounded-xl p-4 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all resize-none leading-relaxed"
              ></textarea>
              <p class="text-xs text-[#86868b] dark:text-[#98989d] text-center">
                {{ t('audioTextHint') }}
              </p>
            </div>
          </div>
        </div>
        <div class="px-6 py-4 border-t border-black/8 dark:border-white/8">
          <button
            @click="startCloning"
            :disabled="selectionDuration > 15"
            class="w-full px-5 py-3 rounded-full font-semibold transition-all hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            :class="selectionDuration > 15
              ? 'bg-gray-400 dark:bg-gray-600 text-white'
              : 'bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white'"
          >
            {{ selectionDuration > 15 ? t('pleaseSelect15sOrLess') : t('startCloning') }}
          </button>
        </div>
      </div>

      <!-- Processing Screen -->
      <div v-else-if="currentScreen === 'PROCESSING'" class="flex-1 flex flex-col items-center justify-center p-8">
        <div class="w-24 h-24 rounded-full bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center mb-6">
          <i class="fas fa-spinner fa-spin text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-3xl"></i>
        </div>
        <h3 class="text-2xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] mb-2">{{ t('generatingVoice') }}</h3>
        <p class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('generatingVoiceHint') }}</p>
      </div>

      <!-- Success Screen -->
      <div v-else-if="currentScreen === 'SUCCESS'" class="flex-1 flex flex-col">
        <div class="flex-1 overflow-y-auto p-6">
          <div class="max-w-md mx-auto flex flex-col items-center text-center space-y-6">
            <div class="w-24 h-24 rounded-full bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 flex items-center justify-center">
              <i class="fas fa-check text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] text-3xl"></i>
            </div>
            <h3 class="text-3xl font-bold text-[#1d1d1f] dark:text-[#f5f5f7]">{{ t('voiceCloneGenerated') }}</h3>
            <p class="text-sm text-[#86868b] dark:text-[#98989d]">{{ t('voiceCloneGeneratedHint') }}</p>

            <!-- Preview Buttons -->
            <div class="w-full space-y-3">
              <button @click="playPreview('cn')" :class="`w-full p-5 rounded-2xl border flex items-center justify-between transition-all ${
                isPlayingPreview === 'cn'
                  ? 'bg-[color:var(--brand-primary)]/20 dark:bg-[color:var(--brand-primary-light)]/20 border-[color:var(--brand-primary)] text-[color:var(--brand-primary)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border-black/8 dark:border-white/8'
              }`">
                <div class="flex items-center gap-3">
                  <span :class="`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                    isPlayingPreview === 'cn' ? 'bg-[color:var(--brand-primary)] text-white' : 'bg-black/5 dark:bg-white/5 text-[#86868b]'
                  }`">CN</span>
                  <span class="font-medium">{{ t('chinesePreview') }}</span>
                </div>
                <i v-if="isPlayingPreview === 'cn'" class="fas fa-volume-up animate-pulse"></i>
                <i v-else class="fas fa-play text-[#86868b]"></i>
              </button>

              <button @click="playPreview('en')" :class="`w-full p-5 rounded-2xl border flex items-center justify-between transition-all ${
                isPlayingPreview === 'en'
                  ? 'bg-[color:var(--brand-primary)]/20 dark:bg-[color:var(--brand-primary-light)]/20 border-[color:var(--brand-primary)] text-[color:var(--brand-primary)]'
                  : 'bg-white/80 dark:bg-[#2c2c2e]/80 border-black/8 dark:border-white/8'
              }`">
                <div class="flex items-center gap-3">
                  <span :class="`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${
                    isPlayingPreview === 'en' ? 'bg-[color:var(--brand-primary)] text-white' : 'bg-black/5 dark:bg-white/5 text-[#86868b]'
                  }`">EN</span>
                  <span class="font-medium">{{ t('englishPreview') }}</span>
                </div>
                <i v-if="isPlayingPreview === 'en'" class="fas fa-volume-up animate-pulse"></i>
                <i v-else class="fas fa-play text-[#86868b]"></i>
              </button>
            </div>

            <!-- Name Input -->
            <div class="w-full space-y-2">
              <label class="text-sm font-semibold text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]">{{ t('voiceName') }}</label>
              <input
                type="text"
                v-model="voiceName"
                :placeholder="t('voiceNamePlaceholder')"
                class="w-full p-4 rounded-xl bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] focus:outline-none focus:border-[color:var(--brand-primary)] transition-all"
              />
            </div>
          </div>
        </div>
        <div class="px-6 py-4 border-t border-black/8 dark:border-white/8 space-y-3">
          <button @click="saveVoice" :disabled="!voiceName.trim()" class="w-full px-5 py-3 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full font-semibold transition-all hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed">
            {{ t('saveVoice') }}
          </button>
          <button @click="reclone" class="w-full px-5 py-3 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#1d1d1f] dark:text-[#f5f5f7] rounded-full font-medium transition-all">
            {{ t('reclone') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { showAlert } from '../utils/other'

const SAMPLE_SENTENCES_ZH = [
  "晚上躺在床上但听故事，妈妈的声音轻轻的。窗外有虫儿在叫，像在跟着故事哼调子，我抱着小熊，慢慢就眯起了眼睛。",
  "傍晚里小区真热闹，张奶奶带着小孙子学走路，宝宝摇摇晃晃扑进她怀里。王爷爷和李爷爷坐在石桌上下棋，棋子敲得啪啪响。",
  "下雨啦，我趴在窗边看。雨点打在树叶上沙沙响，路上的行人举着伞快步走，像一朵朵会移动的小花，真好看。",
  "周末去公园玩，我看到小朋友们在放风筝，风筝飞得高高的，像一只只小鸟在天上飞。",
  "怎么就这么难呢。我已经很努力了啊，可还是什么都留不住，心里面空落落的，真难受……",
  "说不在乎都是假的，我装作不在乎，其实心里难受得要命。夜里翻来覆去睡不着，心里头跟堵了块石头似的，压得喘不过气来。",
  "哎，这照片都泛黄了。那时候多好啊，谁能想到呢，说散就散了。现在就我一个人，守着这些念想，唉……",
  "咱们球队夺冠啦！最后那个绝杀球看得我跳起来撞到天花板！现在嗓子喊哑了都不觉得疼，就想抱着队友们哭，这感觉太爽了，我们是最棒的！",
  "妹妹的画被选去参加画展了！她站在台上看着自己的画，眼睛亮晶晶的，像星星一样。她跟我说，以后要当画家，画出最美的画给大家看。",
  "和闺蜜逛街，看到好看的衣服就想试，试完又觉得不合适，最后啥也没买。闺蜜说，我们这是在培养审美，以后买东西才不会冲动。",
  "去参加婚礼，新娘和新郎互相表白，台下好多人都在哭，我也忍不住红了眼眶。他们说，以后要一直幸福下去，要一直牵着手走下去。",
  "和朋友们一起玩游戏，玩到凌晨两点，大家都累得不行，但还是特别开心。我们说，以后要经常一起玩游戏，要一直一起玩下去。",
  "凭什么这么做？明明占着理，偏要胡搅蛮缠！气得太阳穴突突跳，嗓门都拔高了，今天非要讨个说法不可，谁也别想蒙混过关！",
  "这个项目真棘手，眼看就要黄了，偏偏还碰上这么个不省心的货，天天给我找麻烦。这项目要是黄了，我非得让他好看！",
  "这都什么事儿啊！昨晚喝多了，早上起来头疼得要命，胃里翻江倒海的，这酒量真是不行。下次再也不敢了，得戒酒了。",
  "哪有这样蛮不讲理的？明明错在你，倒反过来怪别人！气得我浑身发烫，话都说不利索，这理必须掰扯清楚！"
]

const SAMPLE_SENTENCES_EN = [
  "Lying in bed at night, listening to stories, mom's voice is gentle. Outside the window, insects are chirping, as if humming along with the story. I hold my little bear and slowly close my eyes.",
  "The neighborhood is really lively in the evening. Grandma Zhang is teaching her little grandson to walk, and the baby wobbles into her arms. Grandpa Wang and Grandpa Li are playing chess at the stone table, the pieces clicking loudly.",
  "It's raining! I'm lying by the window watching. Raindrops patter on the leaves, and people on the street walk quickly with umbrellas, like moving little flowers. So beautiful.",
  "Going to the park on the weekend, I see children flying kites. The kites fly so high, like little birds in the sky.",
  "Why is it so difficult? I've tried so hard, but I still can't keep anything. My heart feels empty, so sad...",
  "Saying I don't care is all fake. I pretend not to care, but deep down I'm hurting so much. Tossing and turning at night, unable to sleep, my heart feels blocked like a stone, suffocating me.",
  "Ah, this photo has turned yellow. Those were such good times. Who would have thought we'd drift apart? Now I'm alone, holding onto these memories, sigh...",
  "Our team won the championship! That last winning shot made me jump up and hit the ceiling! My voice is hoarse from shouting, but I don't feel any pain. I just want to hug my teammates and cry. This feeling is amazing, we're the best!",
  "My sister's painting was selected for the art exhibition! She stood on stage looking at her painting, her eyes sparkling like stars. She told me she wants to be a painter and create the most beautiful paintings for everyone to see.",
  "Shopping with my best friend, we see beautiful clothes and want to try them on, but after trying them, we feel they're not quite right. We end up buying nothing. My friend says we're cultivating our taste, so we won't be impulsive when shopping later.",
  "Attending a wedding, the bride and groom express their love to each other. Many people in the audience are crying, and I can't help but tear up too. They say they'll be happy together forever, holding hands and walking forward together.",
  "Playing games with friends until 2 AM, everyone is exhausted but still so happy. We say we'll play games together often, always together.",
  "How can you do this? I'm clearly in the right, but you're being unreasonable! My temples are throbbing, my voice is raised. Today I must get an explanation, no one can get away with this!",
  "This project is really tricky. It's about to fail, and I've run into this troublesome person who keeps causing me problems every day. If this project fails, I'll make sure they pay!",
  "What's going on? I drank too much last night, woke up this morning with a terrible headache, and my stomach is churning. My alcohol tolerance is really poor. I'll never do this again, I need to quit drinking.",
  "How can you be so unreasonable? You're clearly in the wrong, but you're blaming others! I'm so angry I'm burning up, can't even speak clearly. We must sort this out!"
]

export default {
  name: 'VoiceCloneModal',
  emits: ['close', 'saved'],
  setup(props, { emit }) {
    const { t, locale } = useI18n()
    const currentScreen = ref('METHOD_SELECT')
    const videoInputRef = ref(null)
    const audioInputRef = ref(null)
    const audioRef = ref(null)
    const currentSentence = ref('')
    const customText = ref('')
    const useSampleSentence = ref(true)  // true: 使用示例句子, false: 使用自定义文本
    const isRecording = ref(false)
    const mediaRecorder = ref(null)
    const audioUrl = ref(null)
    const audioBlob = ref(null)
    const isPlaying = ref(false)
    const currentTime = ref(0)
    const duration = ref(0)
    const isTrimMode = ref(false)
    const trimStart = ref(0)  // 秒数
    const trimEnd = ref(0)    // 秒数
    const isDragging = ref(false)
    const dragType = ref(null)  // 'start' or 'end'
    const isDraggingProgress = ref(false)  // 是否正在拖动播放进度条
    const isTrimming = ref(false)  // 是否正在裁剪音频
    const originalAudioBlob = ref(null)  // 保存原始音频，用于取消裁剪
    const visualizerRef = ref(null)  // 可视化波形图引用
    const waveformContainerRef = ref(null)  // 波形条容器引用
    const dragTarget = ref(null)  // 'start' | 'end' | 'selection' | null
    const dragStartRef = ref(null)  // { x: number, start: number, end: number } | null
    const isTrimmed = ref(false)  // 是否已确认裁剪
    const showDurationWarning = ref(false)  // 是否显示时长警告
    const isLooping = ref(false)  // 是否循环播放
    const playbackRate = ref(1)  // 播放倍速
    const waveformBarCount = ref(50)  // 波形条数量，根据容器动态计算
    const voiceName = ref('')
    const speakerId = ref('')
    const isSaved = ref(false)  // 标记音色是否已保存
    const isPlayingPreview = ref(null)
    const previewAudioRef = ref(null)
    const cnPreviewUrl = ref('')
    const enPreviewUrl = ref('')
    const userInputText = ref('')  // 用户输入的文本

    const changeSentence = () => {
      // 根据当前语言环境选择对应的句子数组
      const sentences = locale.value === 'zh' || locale.value === 'zh-CN' ? SAMPLE_SENTENCES_ZH : SAMPLE_SENTENCES_EN
      const randomIndex = Math.floor(Math.random() * sentences.length)
      currentSentence.value = sentences[randomIndex]
    }

    const selectMethod = (method) => {
      if (method === 'RECORD') {
        currentScreen.value = 'RECORDING'
        // 如果使用示例句子模式，初始化一个随机句子
        if (useSampleSentence.value) {
          changeSentence()
        }
      }
    }

    const startRecording = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        const recorder = new MediaRecorder(stream)
        const chunks = []

        recorder.ondataavailable = (e) => {
          if (e.data.size > 0) chunks.push(e.data)
        }

        recorder.onstop = () => {
          const blob = new Blob(chunks, { type: 'audio/webm' })
          stream.getTracks().forEach(track => track.stop())
          handleAudioReady(blob)
        }

        recorder.start()
        mediaRecorder.value = recorder
        isRecording.value = true
      } catch (err) {
        showAlert(t('microphonePermissionError'), 'warning')
      }
    }

    const stopRecording = () => {
      if (mediaRecorder.value && isRecording.value) {
        mediaRecorder.value.stop()
        isRecording.value = false
        mediaRecorder.value = null
      }
    }

    const handleVideoFile = (e) => {
      const file = e.target.files?.[0]
      if (file) {
        handleAudioReady(file)
      }
    }

    const handleAudioFile = (e) => {
      const file = e.target.files?.[0]
      if (file) {
        handleAudioReady(file)
      }
    }

    const handleAudioReady = (blob) => {
      audioBlob.value = blob
      originalAudioBlob.value = blob  // 保存原始音频
      audioUrl.value = URL.createObjectURL(blob)
      currentScreen.value = 'AUDIO_EDITOR'
    }

    const confirmTrim = async () => {
      if (!audioBlob.value || isTrimming.value) return

      isTrimming.value = true
      try {
        // 保存当前裁剪范围
        const startTime = trimStart.value
        const endTime = trimEnd.value

        // 裁剪音频
        const trimmedBlob = await trimAudio(audioBlob.value, startTime, endTime)

        // 更新音频
        if (audioUrl.value) {
          URL.revokeObjectURL(audioUrl.value)
        }
        audioBlob.value = trimmedBlob
        originalAudioBlob.value = trimmedBlob  // 更新原始音频为裁剪后的音频
        audioUrl.value = URL.createObjectURL(trimmedBlob)

        // 等待音频加载完成并更新时长
        if (audioRef.value) {
          await new Promise((resolve) => {
            const onLoadedMetadata = () => {
              // 更新时长
              duration.value = audioRef.value.duration || 0
              // 重置裁剪范围（因为音频已经被裁剪了，新的范围应该是整个音频）
              trimStart.value = 0
              trimEnd.value = duration.value
              audioRef.value.removeEventListener('loadedmetadata', onLoadedMetadata)
              resolve()
            }
            audioRef.value.addEventListener('loadedmetadata', onLoadedMetadata)
            audioRef.value.load()
          })
        }

        // 重置播放状态
        if (audioRef.value) {
          audioRef.value.currentTime = 0
        }
        currentTime.value = 0
        isPlaying.value = false

        // 退出裁剪模式
        isTrimMode.value = false

        console.log(`Audio trimmed successfully from ${formatTime(startTime)} to ${formatTime(endTime)}`)
      } catch (error) {
        console.error('Failed to trim audio:', error)
        const errorMessage = error.message || error.toString() || 'Unknown error'
        const processedMessage = processErrorMessage(errorMessage)
        showAlert(t('cloneFailed') + ': ' + processedMessage, 'danger')
      } finally {
        isTrimming.value = false
      }
    }

    const cancelTrim = () => {
      // 恢复原始裁剪范围
      trimStart.value = 0
      trimEnd.value = Math.min(duration.value, 15.0)
      // 退出裁剪模式
      isTrimMode.value = false
    }

    const togglePlay = () => {
      if (audioRef.value) {
        const startSec = trimStart.value
        const endSec = trimEnd.value

        if (isPlaying.value) {
          audioRef.value.pause()
          isPlaying.value = false
        } else {
          // 如果处于裁剪模式或已裁剪，确保从裁剪开始位置播放
          if (isTrimMode.value || isTrimmed.value) {
            const curr = audioRef.value.currentTime
            if (curr < startSec || curr >= endSec) {
              audioRef.value.currentTime = startSec
            }
          }
          audioRef.value.play()
          isPlaying.value = true
        }
      }
    }

    const toggleLoop = () => {
      isLooping.value = !isLooping.value
    }

    const cycleSpeed = () => {
      const speeds = [0.5, 1, 1.5, 2]
      const currentIndex = speeds.indexOf(playbackRate.value)
      const nextIndex = (currentIndex + 1) % speeds.length
      playbackRate.value = speeds[nextIndex]

      if (audioRef.value) {
        audioRef.value.playbackRate = playbackRate.value
      }
    }

    const handleToggleTrimMode = () => {
      if (isTrimMode.value) {
        // 确认裁剪
        isTrimMode.value = false
        isTrimmed.value = true
        // 跳转到选择开始位置并暂停
        if (audioRef.value) {
          audioRef.value.pause()
          audioRef.value.currentTime = trimStart.value
          currentTime.value = 0
          isPlaying.value = false
        }
      } else {
        // 启用裁剪模式（重新编辑）
        isTrimMode.value = true
        isTrimmed.value = false
        if (audioRef.value) {
          currentTime.value = audioRef.value.currentTime
        }
      }
    }

    const handleSliderChange = (e) => {
      const val = Number(e.target.value)

      if (audioRef.value) {
        if (isTrimmed.value) {
          const startSec = trimStart.value
          audioRef.value.currentTime = startSec + val
        } else {
          audioRef.value.currentTime = val
        }
      }
    }

    const updateTime = () => {
      if (audioRef.value && !isDraggingProgress.value) {
        const curr = audioRef.value.currentTime
        const startSec = trimStart.value
        const endSec = trimEnd.value

        // 如果已裁剪，强制在裁剪范围内
        if (isTrimmed.value) {
          if (curr < startSec - 0.1) {
            audioRef.value.currentTime = startSec
          } else if (curr >= endSec) {
            if (isLooping.value) {
              audioRef.value.currentTime = startSec
            } else {
              audioRef.value.pause()
              isPlaying.value = false
              audioRef.value.currentTime = startSec
            }
          }
          // 更新相对时间
          currentTime.value = Math.max(0, curr - startSec)
        } else if (isTrimMode.value) {
          // 裁剪模式下也限制在范围内
          if (curr >= endSec) {
            if (isLooping.value) {
              audioRef.value.currentTime = startSec
            } else {
              audioRef.value.pause()
              isPlaying.value = false
              audioRef.value.currentTime = startSec
            }
          }
          currentTime.value = curr
        } else {
          currentTime.value = curr
        }
      }
    }

    const updateDuration = () => {
      if (audioRef.value) {
        duration.value = audioRef.value.duration || 0
        // 初始化裁剪范围
        if (trimEnd.value === 0 || trimEnd.value > duration.value) {
          trimEnd.value = Math.min(duration.value, 15.0)  // 默认最多15秒
        }
        if (trimStart.value < 0) {
          trimStart.value = 0
        }

        // 如果音频超过15秒，自动进入裁剪模式
        if (duration.value > 15) {
          isTrimMode.value = true
          isTrimmed.value = false
          trimStart.value = 0
          trimEnd.value = 15.0
          showDurationWarning.value = true
        } else {
          showDurationWarning.value = false
        }
      }
    }

    // 计算属性
    const trimStartPercent = computed(() => {
      return duration.value > 0 ? (trimStart.value / duration.value) * 100 : 0
    })

    const trimEndPercent = computed(() => {
      return duration.value > 0 ? (trimEnd.value / duration.value) * 100 : 100
    })

    // 计算波形条数量，使其刚好填满容器
    const calculateWaveformBarCount = () => {
      if (!waveformContainerRef.value) return 50

      const container = waveformContainerRef.value
      const containerWidth = container.offsetWidth

      // 如果容器宽度为 0，说明还没有渲染完成
      if (containerWidth === 0) return 50

      // 已去除 padding，直接使用容器宽度
      const availableWidth = containerWidth

      // 每个波形条宽度 6px (w-1.5 = 0.375rem = 6px)，gap 6px (gap-1.5 = 0.375rem = 6px)
      const barWidth = 6
      const gapWidth = 6

      // 计算能放多少个波形条：n * barWidth + (n-1) * gapWidth <= availableWidth
      // n * barWidth + n * gapWidth - gapWidth <= availableWidth
      // n * (barWidth + gapWidth) <= availableWidth + gapWidth
      // n <= (availableWidth + gapWidth) / (barWidth + gapWidth)
      const maxBars = Math.floor((availableWidth + gapWidth) / (barWidth + gapWidth))

      // 确保至少有一些波形条，最多不超过合理范围
      return Math.max(20, Math.min(maxBars, 100))
    }

    // 更新波形条数量
    const updateWaveformBarCount = () => {
      if (waveformContainerRef.value) {
        const newCount = calculateWaveformBarCount()
        if (newCount !== waveformBarCount.value) {
          waveformBarCount.value = newCount
        }
      }
    }

    // 获取波形条样式
    const getWaveformBarStyle = (index) => {
      // 计算波形条的位置百分比
      // 手柄位置是基于 visualizerRef（整个容器）宽度的百分比
      // 波形条在 visualizerRef 内部，有 padding，但位置应该映射到整个容器宽度
      // 使用 index / waveformBarCount 来计算波形条位置（从 0 到 1）
      // 注意：index 从 1 开始，所以第一个波形条位置是 1/waveformBarCount，最后一个接近 1
      const barPosition = index / waveformBarCount.value  // 波形条位置，范围约 1/n 到 1.0
      const startThreshold = trimStartPercent.value / 100  // 转换为 0-1 范围
      const endThreshold = trimEndPercent.value / 100      // 转换为 0-1 范围

      // 计算透明度
      // 判断波形条是否在裁剪范围内
      let opacity = 1
      if (!isTrimmed.value && isTrimMode.value) {
        opacity = (barPosition >= startThreshold && barPosition <= endThreshold) ? 1 : 0.3
      }

      return {
        height: `${20 + Math.random() * 70}%`,
        backgroundColor: 'var(--brand-primary)',
        opacity: opacity
      }
    }

    const selectionDuration = computed(() => {
      return trimEnd.value - trimStart.value
    })

    const sliderMax = computed(() => {
      return isTrimmed ? selectionDuration.value : duration.value
    })

    const sliderValue = computed(() => {
      if (isTrimmed) {
        // 如果已裁剪，显示相对时间
        const startSec = trimStart.value
        return Math.max(0, currentTime.value - startSec)
      } else {
        return currentTime.value
      }
    })

    // Pointer事件处理（用于拖动）
    const handlePointerDown = (e, target) => {
      e.preventDefault()
      e.stopPropagation()
      dragTarget.value = target
      isTrimMode.value = true
      isTrimmed.value = false

      if (target === 'selection') {
        dragStartRef.value = {
          x: e.clientX,
          start: trimStart.value,
          end: trimEnd.value
        }
      }

      if (e.currentTarget) {
        e.currentTarget.setPointerCapture(e.pointerId)
      }
    }

    const handlePointerMove = (e) => {
      if (!dragTarget.value || !visualizerRef.value) return

      const rect = visualizerRef.value.getBoundingClientRect()

      // 处理选择区域拖动
      if (dragTarget.value === 'selection' && dragStartRef.value) {
        const deltaX = e.clientX - dragStartRef.value.x
        const deltaPct = (deltaX / rect.width) * 100
        const deltaTime = (deltaPct / 100) * duration.value

        let newStart = dragStartRef.value.start + deltaTime
        let newEnd = dragStartRef.value.end + deltaTime

        if (newStart < 0) {
          newEnd = newEnd - newStart
          newStart = 0
        }
        if (newEnd > duration.value) {
          newStart = newStart - (newEnd - duration.value)
          newEnd = duration.value
        }

        trimStart.value = Math.max(0, newStart)
        trimEnd.value = Math.min(duration.value, newEnd)
        return
      }

      // 处理单独调整
      const x = e.clientX - rect.left
      let percentage = (x / rect.width) * 100
      percentage = Math.max(0, Math.min(100, percentage))
      const time = (percentage / 100) * duration.value

      if (dragTarget.value === 'start') {
        trimStart.value = Math.max(0, Math.min(time, trimEnd.value - 0.5))
      } else if (dragTarget.value === 'end') {
        trimEnd.value = Math.max(trimStart.value + 0.5, Math.min(time, duration.value))
      }

      // 检查时长警告
      if (selectionDuration.value > 15.1) {
        showDurationWarning.value = true
      } else {
        showDurationWarning.value = false
      }
    }

    const handlePointerUp = (e) => {
      dragTarget.value = null
      dragStartRef.value = null
      if (e.currentTarget) {
        e.currentTarget.releasePointerCapture(e.pointerId)
      }

      if (selectionDuration.value > 15.1) {
        showDurationWarning.value = true
      } else {
        showDurationWarning.value = false
      }
    }

    const onEnded = () => {
      if (isLooping.value) {
        const startSec = trimStart.value
        if (audioRef.value) {
          audioRef.value.currentTime = startSec
          audioRef.value.play()
        }
      } else {
        isPlaying.value = false
        if (isTrimmed.value) {
          const startSec = trimStart.value
          if (audioRef.value) {
            audioRef.value.currentTime = startSec
          }
          currentTime.value = 0
        } else {
          currentTime.value = 0
        }
      }
    }

    const onRangeChange = (e) => {
      const val = Number(e.target.value)
      currentTime.value = val
      if (audioRef.value) audioRef.value.currentTime = val
    }

    const handleReset = () => {
      // 如果时长超过15秒，不能完全重置，但可以重置选择到开始
      if (duration.value > 15) {
        trimStart.value = 0
        trimEnd.value = 15.0
        isTrimMode.value = true
        isTrimmed.value = false
        showDurationWarning.value = true
      } else {
        trimStart.value = 0
        trimEnd.value = duration.value
        isTrimMode.value = false
        isTrimmed.value = false
        showDurationWarning.value = false
      }

      playbackRate.value = 1
      isLooping.value = false
      if (audioRef.value) {
        audioRef.value.pause()
        audioRef.value.currentTime = 0
        audioRef.value.playbackRate = 1
      }
      isPlaying.value = false
      currentTime.value = 0
    }

    // 裁剪音频
    const trimAudio = async (audioBlob, startTime, endTime) => {
      return new Promise((resolve, reject) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)()
        const fileReader = new FileReader()

        fileReader.onload = async (e) => {
          try {
            const audioBuffer = await audioContext.decodeAudioData(e.target.result)
            const sampleRate = audioBuffer.sampleRate
            const startSample = Math.floor(startTime * sampleRate)
            const endSample = Math.floor(endTime * sampleRate)
            const length = endSample - startSample

            // 创建新的音频缓冲区
            const newBuffer = audioContext.createBuffer(
              audioBuffer.numberOfChannels,
              length,
              sampleRate
            )

            // 复制音频数据
            for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
              const oldData = audioBuffer.getChannelData(channel)
              const newData = newBuffer.getChannelData(channel)
              for (let i = 0; i < length; i++) {
                newData[i] = oldData[startSample + i]
              }
            }

            // 转换为 WAV
            const wav = audioBufferToWav(newBuffer)
            const blob = new Blob([wav], { type: 'audio/wav' })
            resolve(blob)
          } catch (error) {
            reject(error)
          }
        }

        fileReader.onerror = reject
        fileReader.readAsArrayBuffer(audioBlob)
      })
    }

    // 将 AudioBuffer 转换为 WAV
    const audioBufferToWav = (buffer) => {
      const length = buffer.length
      const numberOfChannels = buffer.numberOfChannels
      const sampleRate = buffer.sampleRate
      const bytesPerSample = 2
      const blockAlign = numberOfChannels * bytesPerSample
      const byteRate = sampleRate * blockAlign
      const dataSize = length * blockAlign
      const bufferSize = 44 + dataSize
      const arrayBuffer = new ArrayBuffer(bufferSize)
      const view = new DataView(arrayBuffer)

      // WAV header
      const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i))
        }
      }

      writeString(0, 'RIFF')
      view.setUint32(4, bufferSize - 8, true)
      writeString(8, 'WAVE')
      writeString(12, 'fmt ')
      view.setUint32(16, 16, true) // fmt chunk size
      view.setUint16(20, 1, true) // audio format (PCM)
      view.setUint16(22, numberOfChannels, true)
      view.setUint32(24, sampleRate, true)
      view.setUint32(28, byteRate, true)
      view.setUint16(32, blockAlign, true)
      view.setUint16(34, 16, true) // bits per sample
      writeString(36, 'data')
      view.setUint32(40, dataSize, true)

      // Write audio data
      let offset = 44
      for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
          const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]))
          view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true)
          offset += 2
        }
      }

      return arrayBuffer
    }

    const formatTime = (time) => {
      if (!time && time !== 0) return "0:00"
      const min = Math.floor(time / 60)
      const sec = Math.floor(time % 60)
      return `${min}:${sec.toString().padStart(2, '0')}`
    }

    // 处理错误信息，将英文错误转换为中文
    const processErrorMessage = (errorMessage) => {
      if (!errorMessage) return ''

      // 提取实际错误信息（去除 "Voice clone failed: " 等前缀）
      let message = errorMessage
      if (message.includes('Voice clone failed:')) {
        message = message.split('Voice clone failed:')[1]?.trim() || message
      }
      if (message.includes('Clone error: Error:')) {
        message = message.split('Clone error: Error:')[1]?.trim() || message
      }
      if (message.includes('Error:')) {
        message = message.split('Error:')[1]?.trim() || message
      }

      // 检查是否包含中文（简单判断：包含中文字符）
      const hasChinese = /[\u4e00-\u9fa5]/.test(message)

      // 如果已经是中文，直接返回
      if (hasChinese) {
        return message
      }

      // 错误信息映射（英文 -> 中文），按优先级匹配
      const lowerMessage = message.toLowerCase()

      // 1. 字符错误率过高
      if (lowerMessage.includes('char error rate high')) {
          return '自动识别音频文本错误率过高，请上传清晰音频或尝试手动输入音频文本。'
        }

      // 2. 文本长度过长
      if (lowerMessage.includes('text length') && lowerMessage.includes('too long')) {
        if (lowerMessage.includes('greater than 50') || lowerMessage.includes('> 50')) {
          return '文本长度过长（超过50个字符），请选择更短的音频片段（建议15秒以内）。'
        }
        return '文本长度过长，请选择更短的音频片段（建议15秒以内）。'
      }

      // 3. 其他常见错误
      const errorMap = {
        'failed to extract text from audio': '无法从音频中提取文本',
        'asr failed': '语音识别失败',
        'no file uploaded': '未上传文件',
        'unsupported file format': '不支持的文件格式',
        'voice clone client not initialized': '语音克隆客户端未初始化',
        'asr client not initialized': '语音识别客户端未初始化',
        'voice clone failed': '语音克隆失败',
        'clone failed': '克隆失败'
      }

      // 尝试匹配错误信息
      for (const [en, zh] of Object.entries(errorMap)) {
        if (lowerMessage.includes(en)) {
          return zh
        }
      }

      // 如果没有匹配到，返回原始错误信息
      return message
    }

    const startCloning = async () => {
      if (!audioBlob.value) return

      currentScreen.value = 'PROCESSING'

      try {
        const formData = new FormData()

        // 裁剪音频（如果已确认裁剪或在裁剪模式下有选择）
        let fileToUpload = audioBlob.value
        if (isTrimmed.value || (isTrimMode.value && (trimStart.value > 0 || trimEnd.value < duration.value))) {
          try {
            const trimmedBlob = await trimAudio(audioBlob.value, trimStart.value, trimEnd.value)
            fileToUpload = new File([trimmedBlob], 'trimmed_audio.wav', { type: 'audio/wav' })
            console.log(`Audio trimmed from ${formatTime(trimStart.value)} to ${formatTime(trimEnd.value)}`)
          } catch (error) {
            console.error('Failed to trim audio, using original:', error)
            // 如果裁剪失败，使用原始音频
          }
        }

        // 如果 audioBlob 是 Blob（没有文件名），创建一个 File 对象
        if (fileToUpload instanceof Blob && !(fileToUpload instanceof File)) {
          // 根据 MIME 类型确定文件扩展名
          let extension = '.webm' // 默认扩展名
          if (fileToUpload.type === 'audio/webm') {
            extension = '.webm'
          } else if (fileToUpload.type === 'audio/wav' || fileToUpload.type === 'audio/wave') {
            extension = '.wav'
          } else if (fileToUpload.type === 'audio/mp3' || fileToUpload.type === 'audio/mpeg') {
            extension = '.mp3'
          } else if (fileToUpload.type.startsWith('video/')) {
            extension = '.mp4' // 视频文件默认使用 .mp4
          }

          // 创建 File 对象，指定文件名
          fileToUpload = new File([fileToUpload], `audio${extension}`, { type: fileToUpload.type })
        }

        formData.append('file', fileToUpload)
        // 如果用户输入了文本，也一起发送
        if (userInputText.value && userInputText.value.trim()) {
          formData.append('text', userInputText.value.trim())
        }
        // 不再在克隆时传递名称，名称在保存时传递

        const token = localStorage.getItem('accessToken')
        const response = await fetch('/api/v1/voice/clone', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`
          },
          body: formData
        })

        if (!response.ok) {
          const error = await response.json()
          const errorMessage = error.error || 'Clone failed'
          // 如果错误信息中包含 char error rate high，标记为建议手动输入
          const errorObj = new Error(errorMessage)
          if (errorMessage.toLowerCase().includes('char error rate high')) {
            errorObj.suggestManualInput = true
          }
          throw errorObj
        }

        const result = await response.json()
        speakerId.value = result.speaker_id
        isSaved.value = false  // 重置保存标志，新克隆的音色还未保存

        // Generate previews
        await generatePreviews()

        currentScreen.value = 'SUCCESS'
      } catch (error) {
        console.error('Clone error:', error)
        const errorMessage = error.message || error.toString() || 'Unknown error'
        const processedMessage = processErrorMessage(errorMessage)

        // 如果后端建议手动输入文本，显示更友好的提示并聚焦到文本输入框
        if (error.suggestManualInput || errorMessage.includes('manually enter') || errorMessage.includes('text input box')) {
          showAlert(t('cloneFailed') + ': ' + processedMessage, 'warning')
          // 确保在音频编辑器界面，并聚焦到文本输入框
          currentScreen.value = 'AUDIO_EDITOR'
          // 使用 nextTick 确保 DOM 更新后再聚焦
          nextTick(() => {
            const textInput = document.querySelector('textarea[placeholder*="音频对应的文本"]') ||
                             document.querySelector('textarea[placeholder*="text content"]')
            if (textInput) {
              textInput.focus()
              textInput.scrollIntoView({ behavior: 'smooth', block: 'center' })
            }
          })
        } else {
          showAlert(t('cloneFailed') + ': ' + processedMessage, 'danger')
          currentScreen.value = 'AUDIO_EDITOR'
        }
      }
    }

    const generatePreviews = async () => {
      if (!speakerId.value) return

        const token = localStorage.getItem('accessToken')
        const cnText = "这是你的专属克隆音色，希望你能喜欢。"
        const enText = "This is your exclusive clone voice, I hope you like it."

      try {
        // Generate Chinese preview
        const cnResponse = await fetch('/api/v1/voice/clone/tts', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            text: cnText,
            speaker_id: speakerId.value,
            language: 'ZH_CN'
          })
        })
        if (cnResponse.ok) {
          const cnBlob = await cnResponse.blob()
          cnPreviewUrl.value = URL.createObjectURL(cnBlob)
        }

        // Generate English preview (Note: SenseTime may not support EN_US, use ZH_CN with English text)
        const enResponse = await fetch('/api/v1/voice/clone/tts', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            text: enText,
            speaker_id: speakerId.value,
            language: 'EN_US' // Use ZH_CN as fallback
          })
        })
        if (enResponse.ok) {
          const enBlob = await enResponse.blob()
          enPreviewUrl.value = URL.createObjectURL(enBlob)
        }
      } catch (error) {
        console.error('Preview generation error:', error)
      }
    }

    const playPreview = (lang) => {
      if (previewAudioRef.value) {
        previewAudioRef.value.pause()
      }

      const url = lang === 'cn' ? cnPreviewUrl.value : enPreviewUrl.value
      if (!url) return

      const audio = new Audio(url)
      previewAudioRef.value = audio
      isPlayingPreview.value = lang

      audio.play()
      audio.onended = () => {
        isPlayingPreview.value = null
      }
    }

    const saveVoice = async () => {
      if (!voiceName.value.trim() || !speakerId.value) return

      try {
        const token = localStorage.getItem('accessToken')
        const response = await fetch('/api/v1/voice/clone/save', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            speaker_id: speakerId.value,
            name: voiceName.value.trim()
          })
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.error || 'Save failed')
        }

        // 保存成功，标记为已保存
        isSaved.value = true

        // 通知父组件
        emit('saved', {
          speaker_id: speakerId.value,
          name: voiceName.value.trim()
        })
        closeModal()
      } catch (error) {
        console.error('Save voice error:', error)
        const errorMessage = error.message || error.toString() || 'Unknown error'
        const processedMessage = processErrorMessage(errorMessage)
        showAlert(t('cloneFailed') + ': ' + processedMessage, 'danger')
      }
    }

    const deleteVoiceClone = async (speakerIdToDelete) => {
      if (!speakerIdToDelete) return

      try {
        const token = localStorage.getItem('accessToken')
        const response = await fetch(`/api/v1/voice/clone/${speakerIdToDelete}`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (response.ok) {
          console.log('Voice clone deleted:', speakerIdToDelete)
        } else {
          console.warn('Failed to delete voice clone:', speakerIdToDelete)
        }
      } catch (error) {
        console.error('Delete voice clone error:', error)
        // 静默失败，不影响用户体验
      }
    }

    const reclone = async () => {
      // 如果已有克隆的音色，先删除
      if (speakerId.value) {
        await deleteVoiceClone(speakerId.value)
      }

      // Reset to method selection
      currentScreen.value = 'METHOD_SELECT'
      audioUrl.value = null
      audioBlob.value = null
      speakerId.value = ''
      voiceName.value = ''
      isSaved.value = false  // 重置保存标志
      customText.value = ''
      useSampleSentence.value = true
      currentSentence.value = ''
      userInputText.value = ''  // 重置用户输入的文本
      isPlayingPreview.value = null
      if (previewAudioRef.value) {
        previewAudioRef.value.pause()
        previewAudioRef.value = null
      }
      if (cnPreviewUrl.value) URL.revokeObjectURL(cnPreviewUrl.value)
      if (enPreviewUrl.value) URL.revokeObjectURL(enPreviewUrl.value)
      cnPreviewUrl.value = ''
      enPreviewUrl.value = ''
    }

    const closeModal = async () => {
      // 如果有关闭时未保存的音色，删除它
      if (speakerId.value && currentScreen.value === 'SUCCESS' && !isSaved.value) {
        // 只有在成功界面关闭且未保存时才删除（说明用户没有点击保存）
        await deleteVoiceClone(speakerId.value)
      }
      emit('close')
    }

    // ResizeObserver 实例
    let resizeObserver = null

    // 初始化波形条和监听器
    const initWaveformObserver = () => {
      if (!waveformContainerRef.value) return

      // 先更新一次
      updateWaveformBarCount()

      // 如果已经有观察器，先断开
      if (resizeObserver) {
        resizeObserver.disconnect()
      }

      // 创建新的观察器
      resizeObserver = new ResizeObserver(() => {
        updateWaveformBarCount()
      })
      resizeObserver.observe(waveformContainerRef.value)
    }

    // 监听屏幕切换，当进入音频编辑器时初始化
    watch(currentScreen, (newScreen) => {
      if (newScreen === 'AUDIO_EDITOR') {
        // 等待 DOM 更新后再初始化
        nextTick(() => {
          initWaveformObserver()
        })
      }
    }, { immediate: true })

    onMounted(() => {
      changeSentence()

      // 如果初始就在音频编辑器屏幕，初始化波形条
      if (currentScreen.value === 'AUDIO_EDITOR') {
        nextTick(() => {
          initWaveformObserver()
        })
      }
    })

    onUnmounted(() => {
      // 断开 ResizeObserver
      if (resizeObserver) {
        resizeObserver.disconnect()
        resizeObserver = null
      }

      if (audioUrl.value) URL.revokeObjectURL(audioUrl.value)
      if (cnPreviewUrl.value) URL.revokeObjectURL(cnPreviewUrl.value)
      if (enPreviewUrl.value) URL.revokeObjectURL(enPreviewUrl.value)
      if (previewAudioRef.value) {
        previewAudioRef.value.pause()
      }
    })

    return {
      t,
      currentScreen,
      videoInputRef,
      audioInputRef,
      audioRef,
      currentSentence,
      customText,
      useSampleSentence,
      isRecording,
      audioUrl,
      isPlaying,
      currentTime,
      duration,
      isTrimMode,
      trimStart,
      trimEnd,
      voiceName,
      isPlayingPreview,
      changeSentence,
      selectMethod,
      startRecording,
      stopRecording,
      handleVideoFile,
      handleAudioFile,
      togglePlay,
      updateTime,
      updateDuration,
      onEnded,
      onRangeChange,
      handleReset,
      formatTime,
      trimAudio,
      confirmTrim,
      cancelTrim,
      handleToggleTrimMode,
      toggleLoop,
      cycleSpeed,
      handleSliderChange,
      handlePointerDown,
      handlePointerMove,
      handlePointerUp,
      trimStartPercent,
      trimEndPercent,
      selectionDuration,
      sliderMax,
      sliderValue,
      visualizerRef,
      waveformContainerRef,
      waveformBarCount,
      playbackRate,
      isLooping,
      isTrimmed,
      showDurationWarning,
      getWaveformBarStyle,
      updateWaveformBarCount,
      startCloning,
      playPreview,
      saveVoice,
      reclone,
      closeModal,
      userInputText
    }
  }
}
</script>
