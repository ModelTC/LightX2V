<template>
  <!-- 模态框容器 - Apple 极简风格 -->
  <div class="flex flex-col h-full bg-white dark:bg-[#1e1e1e] overflow-hidden">
    <!-- 模态框头部 - Apple 风格 -->
    <div class="flex items-center justify-between px-6 py-4 border-b border-black/8 dark:border-white/8 bg-white/50 dark:bg-[#1e1e1e]/50 backdrop-blur-[20px] flex-shrink-0">
      <h3 class="text-xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] flex items-center gap-3 tracking-tight">
        <i class="fas fa-volume-up text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
        <span>{{ t('voiceSynthesis') }}</span>
      </h3>
      <div class="flex items-center gap-2">
        <!-- 应用按钮 - Apple 风格 -->
        <button
          @click="applySelectedVoice"
          :disabled="!selectedVoice || !inputText.trim() || isGenerating"
          class="w-9 h-9 flex items-center justify-center bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white rounded-full transition-all duration-200 hover:scale-110 active:scale-100 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          :title="t('applySelectedVoice')">
          <i class="fas fa-check text-sm"></i>
        </button>
        <!-- 关闭按钮 - Apple 风格 -->
        <button @click="closeModal"
          class="w-9 h-9 flex items-center justify-center bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-full transition-all duration-200 hover:scale-110 active:scale-100">
          <i class="fas fa-times text-sm"></i>
        </button>
      </div>
    </div>

    <!-- 模态框内容 - Apple 风格 -->
    <div class="flex-1 overflow-y-auto p-6 main-scrollbar">
      <div class="max-w-5xl mx-auto space-y-6">
        <!-- 音频播放器 - Apple 风格 -->
            <div v-if="audioUrl" class="bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-2xl p-5 transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]">
          <div class="flex items-center gap-3 mb-3">
            <i class="fas fa-music text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
            <h6 class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('generatedAudio') }}</h6>
          </div>
          <audio :src="audioUrl" controls class="w-full rounded-xl"></audio>
        </div>
        <!-- 文本输入区域 - Apple 风格 -->
        <div>
          <label class="block text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">
            <i class="fas fa-keyboard mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
            {{ t('enterTextToConvert') }}
          </label>
          <textarea
            v-model="inputText"
            :placeholder="t('ttsPlaceholder')"
            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200 resize-none min-h-[100px]"
            rows="4"
          ></textarea>
        </div>

        <!-- 语音指令区域 - Apple 风格 -->
        <div>
          <label class="block text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 tracking-tight">
            <i class="fas fa-magic mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
            {{ t('voiceInstruction') }}
            <span class="text-xs text-[#86868b] dark:text-[#98989d] ml-2">{{ t('voiceInstructionHint') }}</span>
          </label>
          <textarea
            v-model="contextText"
            :placeholder="t('voiceInstructionPlaceholder')"
            class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl px-5 py-3 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] transition-all duration-200 resize-none"
            rows="2"
          ></textarea>
        </div>

        <!-- 音色选择区域 - Apple 风格 -->
        <div>
          <div class="flex items-center justify-between mb-4">
            <label class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">
              <i class="fas fa-microphone-alt mr-2 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)]"></i>
              {{ t('selectVoice') }}
            </label>
            <div class="flex items-center gap-3">
              <!-- 搜索框 - Apple 风格 -->
              <div class="relative w-52">
                <i class="fas fa-search absolute left-3 top-1/2 -translate-y-1/2 text-[#86868b] dark:text-[#98989d] text-xs pointer-events-none z-10"></i>
                <input
                  v-model="searchQuery"
                  :placeholder="t('searchVoice')"
                  class="w-full bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-lg py-2 pl-9 pr-3 text-sm text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] tracking-tight hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 transition-all duration-200"
                  type="text"
                />
              </div>

              <!-- 筛选按钮 - Apple 风格 -->
              <button @click="toggleFilterPanel"
                class="flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-[#2c2c2e]/80 border border-black/8 dark:border-white/8 text-[#86868b] dark:text-[#98989d] hover:text-[#1d1d1f] dark:hover:text-[#f5f5f7] hover:bg-white dark:hover:bg-[#3a3a3c] rounded-lg transition-all duration-200 text-sm font-medium tracking-tight">
                <i class="fas fa-filter text-xs"></i>
                <span>{{ t('filter') }}</span>
              </button>
            </div>
          </div>
          <!-- 音色列表容器 - Apple 风格 -->
          <div class="bg-white/50 dark:bg-[#2c2c2e]/50 backdrop-blur-[10px] border border-black/6 dark:border-white/6 rounded-2xl p-5 max-h-[500px] overflow-y-auto main-scrollbar pr-3" ref="voiceListContainer">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              <label
                v-for="(voice, index) in filteredVoices"
                :key="index"
                class="relative m-0 p-0 cursor-pointer"
              >
                <input
                  type="radio"
                  :value="voice.voice_type"
                  v-model="selectedVoice"
                  @change="onVoiceSelect(voice)"
                  class="sr-only"
                />
                <div
                  class="relative flex items-center p-4 bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.2)]"
                  :class="{
                    'border-[color:var(--brand-primary)] dark:border-[color:var(--brand-primary-light)] bg-[color:var(--brand-primary)]/5 dark:bg-[color:var(--brand-primary-light)]/10 shadow-[0_4px_12px_rgba(var(--brand-primary-rgb),0.15)] dark:shadow-[0_4px_12px_rgba(var(--brand-primary-light-rgb),0.2)]': selectedVoice === voice.voice_type
                  }"
                >
                  <!-- V2 标签 - Apple 风格 -->
                  <div v-if="voice.version === '2.0'" class="absolute top-2 right-2 px-2 py-1 bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] text-white text-[10px] font-semibold rounded-md z-10">
                    v2.0
                  </div>

                  <!-- 头像容器 -->
                  <div class="relative mr-3 flex-shrink-0">
                    <!-- Female Avatar -->
                    <img
                      v-if="isFemaleVoice(voice.voice_type)"
                      src="../../public/female.svg"
                      alt="Female Avatar"
                      class="w-12 h-12 rounded-full object-cover bg-white transition-all duration-200"
                      :class="{ 'opacity-60': selectedVoice === voice.voice_type }"
                    />
                    <!-- Male Avatar -->
                    <img
                      v-else
                      src="../../public/male.svg"
                      alt="Male Avatar"
                      class="w-12 h-12 rounded-full object-cover bg-white transition-all duration-200"
                      :class="{ 'opacity-60': selectedVoice === voice.voice_type }"
                    />
                    <!-- Loading 指示器 - Apple 风格 -->
                    <div v-if="isGenerating && selectedVoice === voice.voice_type" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white z-20">
                      <i class="fas fa-spinner fa-spin text-xs"></i>
                    </div>
                    <!-- 设置按钮 - Apple 风格 -->
                    <div v-if="!isGenerating && selectedVoice === voice.voice_type" @click.stop="toggleControls" class="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-[color:var(--brand-primary)]/90 dark:bg-[color:var(--brand-primary-light)]/90 rounded-full flex items-center justify-center text-white cursor-pointer hover:scale-110 transition-all duration-200 z-20">
                      <i class="fas fa-cog text-xs"></i>
                    </div>
                  </div>

                  <!-- 音色信息 -->
                  <div class="flex-1 min-w-0">
                    <div class="text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-1 tracking-tight truncate">
                      {{ voice.name }}
                    </div>
                    <div class="flex flex-wrap gap-1.5">
                      <span v-if="voice.scene" class="inline-block px-2 py-0.5 bg-black/5 dark:bg-white/5 text-[#86868b] dark:text-[#98989d] rounded text-[11px] font-medium">
                        {{ voice.scene }}
                      </span>
                      <span
                        v-for="langCode in voice.language"
                        :key="langCode"
                        class="inline-block px-2 py-0.5 bg-[color:var(--brand-primary)]/10 dark:bg-[color:var(--brand-primary-light)]/15 text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] rounded text-[11px] font-medium"
                      >
                        {{ getLanguageDisplayName(langCode) }}
                      </span>
                    </div>
                  </div>

                  <!-- TTS 控制面板 - Apple 风格 -->
                  <div v-if="selectedVoice === voice.voice_type && showControls" class="absolute top-full left-0 right-0 mt-2 bg-white/95 dark:bg-[#2c2c2e]/95 backdrop-blur-[20px] border border-black/8 dark:border-white/8 rounded-xl p-4 shadow-[0_8px_24px_rgba(0,0,0,0.12)] dark:shadow-[0_8px_24px_rgba(0,0,0,0.4)] z-50 space-y-3">
                    <!-- 语速控制 -->
                    <div class="flex items-center gap-3">
                      <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-16 tracking-tight">{{ t('speechRate') }}:</label>
                      <input
                        type="range"
                        min="-50"
                        max="100"
                        v-model="speechRate"
                        class="flex-1 h-1 bg-black/8 dark:bg-white/8 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                      <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getSpeechRateDisplayValue(speechRate) }}</span>
                    </div>
                    <!-- 音量控制 -->
                    <div class="flex items-center gap-3">
                      <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-16 tracking-tight">{{ t('volume') }}:</label>
                      <input
                        type="range"
                        min="-50"
                        max="100"
                        v-model="loudnessRate"
                        class="flex-1 h-1 bg-black/8 dark:bg-white/8 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                      <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getLoudnessDisplayValue(loudnessRate) }}</span>
                    </div>
                    <!-- 音调控制 -->
                    <div class="flex items-center gap-3">
                      <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-16 tracking-tight">{{ t('pitch') }}:</label>
                      <input
                        type="range"
                        min="-12"
                        max="12"
                        v-model="pitch"
                        class="flex-1 h-1 bg-black/8 dark:bg-white/8 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                      <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ getPitchDisplayValue(pitch) }}</span>
                    </div>
                    <!-- 情感控制 - 仅当音色支持时显示 -->
                    <div v-if="voice.emotions && voice.emotions.length > 0" class="flex items-center gap-3">
                      <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-16 tracking-tight">{{ t('emotionIntensity') }}:</label>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        v-model="emotionScale"
                        class="flex-1 h-1 bg-black/8 dark:bg-white/8 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[color:var(--brand-primary)] dark:[&::-webkit-slider-thumb]:bg-[color:var(--brand-primary-light)] [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer"
                      />
                      <span class="text-xs font-medium text-[#1d1d1f] dark:text-[#f5f5f7] w-12 text-right tracking-tight">{{ emotionScale }}</span>
                    </div>
                    <div v-if="voice.emotions && voice.emotions.length > 0" class="flex items-center gap-3">
                      <label class="text-xs font-medium text-[#86868b] dark:text-[#98989d] w-16 tracking-tight">{{ t('emotionType') }}:</label>
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
              </label>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>

  <!-- 筛选面板遮罩 - Apple 风格 -->
  <div v-if="showFilterPanel" class="fixed inset-0 bg-black/50 dark:bg-black/60 backdrop-blur-sm z-[9999] flex items-center justify-center p-4" @click="closeFilterPanel">
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
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import DropdownMenu from './DropdownMenu.vue'

export default {
  name: 'VoiceTTS',
  components: {
    DropdownMenu
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
    const voices = ref([])
    const emotions = ref([])
    const voiceListContainer = ref(null)
    const showControls = ref(false)
    const showFilterPanel = ref(false)

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


    // Load voices data
    onMounted(async () => {
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

    // Filter voices based on search query, category, version, language, and gender
    // Emotion items for dropdown
    const emotionItems = computed(() => {
      const items = []
      const selectedVoiceData = voices.value.find(v => v.voice_type === selectedVoice.value)
      if (selectedVoiceData && selectedVoiceData.emotions && emotions.value.length > 0) {
        selectedVoiceData.emotions.forEach(emotionName => {
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

    // Check if voice is female based on name
    const isFemaleVoice = (name) => {
      return name.toLowerCase().includes('female')
    }

    // Get available emotions for selected voice
    const availableEmotions = computed(() => {
      const selectedVoiceData = voices.value.find(v => v.voice_type === selectedVoice.value)
      return selectedVoiceData?.emotions || []
    })

    // Reset scroll position
    // Handle emotion selection
    const handleEmotionSelect = (item) => {
      selectedEmotion.value = item.value
    }

    const resetScrollPosition = () => {
      // Reset voice list scroll to top
      if (voiceListContainer.value) {
        voiceListContainer.value.scrollTop = 0
      }
    }

    // Handle voice selection and auto-generate TTS
    const onVoiceSelect = async (voice) => {
      selectedVoice.value = voice.voice_type
      selectedVoiceResourceId.value = voice.resource_id
      // Reset emotion if not available for this voice
      if (voice.emotions && !voice.emotions.includes(selectedEmotion.value)) {
        selectedEmotion.value = ''
      }

      // Reset scroll position when voice is selected
      resetScrollPosition()

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

          // 自动播放生成的音频
          const audio = new Audio(audioUrl.value)
          currentAudio.value = audio // 保存当前播放的音频对象
          audio.play().catch(error => {
            console.log('自动播放被阻止:', error)
            // 如果自动播放失败，用户仍可以手动播放
          })
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
      voices,
      voiceListContainer,
      showControls,
      showFilterPanel,
      filteredVoices,
      isFemaleVoice,
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
      translateGender
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
</style>
