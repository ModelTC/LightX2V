<script setup>
import { useI18n } from 'vue-i18n'
const { t, locale } = useI18n()
import {    // 登录相关
            loginWithGitHub,
            loginWithGoogle,
            loginWithSms,
            phoneNumber,
            verifyCode,
            smsCountdown,
            showSmsForm,
            sendSmsCode,
            handleLoginCallback,
            handlePhoneEnter,
            handleVerifyCodeEnter,
            toggleSmsLogin,
            isLoggedIn,
            loginLoading,
            isLoading,
            initLoading,
            downloadLoading} from '../utils/other'
import { ref } from 'vue';
import { useRouter } from 'vue-router'
const router = useRouter();

</script>

<template>
            <!-- Apple 极简风格登录卡片 -->
            <div class="w-full max-w-xl mx-auto">
                <div class="bg-white/85 dark:bg-[#1e1e1e]/85 backdrop-blur-[40px] backdrop-saturate-[180%] border border-black/10 dark:border-white/10 rounded-3xl shadow-[0_20px_60px_rgba(0,0,0,0.15)] dark:shadow-[0_20px_60px_rgba(0,0,0,0.5)] px-14 py-16 sm:px-12 sm:py-14">
                    <!-- Logo和标题 - Apple 风格 -->
                    <div class="text-center mb-12">
                        <div class="flex items-center justify-center gap-3 mb-5">
                            <img src="../../public/logo.svg" alt="LightX2V" class="w-14 h-12" loading="lazy" />
                            <h1 class="text-4xl font-semibold text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">LightX2V</h1>
                        </div>
                        <p class="text-base text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('loginSubtitle') }}</p>
                    </div>

                    <!-- 表单区域 -->
                    <div class="space-y-6">
                        <!-- 手机号输入框 - Apple 风格 -->
      <div>
                            <label for="phoneNumber" class="block text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] mb-3 text-left tracking-tight">{{ t('phoneNumber') }}</label>
                            <input
                                v-model="phoneNumber"
                                type="tel"
                                name="phoneNumber"
                                required
                                maxlength="11"
                                @keyup.enter="handlePhoneEnter"
                                class="block w-full rounded-xl bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] tracking-tight"
                            />
      </div>

                        <!-- 验证码输入框 - Apple 风格 -->
      <div>
                            <div class="flex items-center justify-between mb-3">
                                <label for="verifyCode" class="block text-sm font-medium text-[#1d1d1f] dark:text-[#f5f5f7] tracking-tight">{{ t('verifyCode') }}</label>
            <button
                @click="sendSmsCode"
                                    class="text-sm font-medium text-[color:var(--brand-primary)] dark:text-[color:var(--brand-primary-light)] hover:text-[color:var(--brand-primary)]/80 dark:hover:text-[color:var(--brand-primary-light)]/80 transition-colors tracking-tight"
                :disabled="!phoneNumber || smsCountdown > 0 || loginLoading"
            >
                {{ smsCountdown > 0 ? `${smsCountdown}s` : t('sendSmsCode') }}
            </button>
        </div>
                            <input
                                v-model="verifyCode"
                                type="text"
                                name="verifyCode"
                                required
                                maxlength="6"
                                @keyup.enter="handleVerifyCodeEnter"
                                class="block w-full rounded-xl bg-white/80 dark:bg-[#2c2c2e]/80 backdrop-blur-[20px] border border-black/8 dark:border-white/8 px-5 py-4 text-[15px] text-[#1d1d1f] dark:text-[#f5f5f7] placeholder-[#86868b] dark:placeholder-[#98989d] transition-all duration-200 hover:bg-white dark:hover:bg-[#3a3a3c] hover:border-black/12 dark:hover:border-white/12 focus:outline-none focus:border-[color:var(--brand-primary)]/50 dark:focus:border-[color:var(--brand-primary-light)]/60 focus:shadow-[0_4px_16px_rgba(var(--brand-primary-rgb),0.12)] dark:focus:shadow-[0_4px_16px_rgba(var(--brand-primary-light-rgb),0.2)] tracking-tight"
                            />
      </div>

                        <!-- 登录按钮 - Apple 风格 -->
                        <div class="pt-4">
                            <button
                                @click="loginWithSms"
                                :disabled="!phoneNumber || !verifyCode || loginLoading"
                                class="w-full rounded-full bg-[color:var(--brand-primary)] dark:bg-[color:var(--brand-primary-light)] border-0 px-6 py-4 text-base font-semibold text-white hover:scale-[1.02] hover:shadow-[0_8px_24px_rgba(var(--brand-primary-rgb),0.35)] dark:hover:shadow-[0_8px_24px_rgba(var(--brand-primary-light-rgb),0.4)] active:scale-100 transition-all duration-200 ease-out disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none tracking-tight"
                            >
            {{ loginLoading ? t('loginLoading') : t('login') }}
        </button>
      </div>
    </div>

                    <!-- 分隔线 - Apple 风格 -->
                    <div class="relative my-10">
                        <div class="absolute inset-0 flex items-center">
                            <div class="w-full border-t border-black/8 dark:border-white/8"></div>
                        </div>
                        <div class="relative flex justify-center">
                            <span class="bg-white/95 dark:bg-[#1e1e1e]/95 px-4 text-sm text-[#86868b] dark:text-[#98989d] tracking-tight">{{ t('orLoginWith') }}</span>
                        </div>
                    </div>

                    <!-- 第三方登录按钮 - Apple 风格 -->
                    <div class="flex justify-center gap-5">
                        <button
                            @click="loginWithGitHub"
                            class="w-14 h-14 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 rounded-full text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100 transition-all duration-200"
                            :disabled="loginLoading"
                            :title="t('loginWithGitHub')"
                        >
                            <i class="fab fa-github text-2xl"></i>
                        </button>

                        <button
                            @click="loginWithGoogle"
                            class="w-14 h-14 flex items-center justify-center bg-white dark:bg-[#3a3a3c] border border-black/8 dark:border-white/8 rounded-full text-[#1d1d1f] dark:text-[#f5f5f7] hover:scale-110 hover:shadow-[0_4px_12px_rgba(0,0,0,0.1)] dark:hover:shadow-[0_4px_12px_rgba(0,0,0,0.3)] active:scale-100 transition-all duration-200"
                            :disabled="loginLoading"
                            :title="t('loginWithGoogle')"
                        >
                            <i class="fab fa-google text-2xl"></i>
                        </button>
                    </div>
                </div>
            </div>
</template>
