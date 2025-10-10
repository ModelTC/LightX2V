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
            <div class="login-card">
                <div class="card-body text-center p-8">
                    <!-- Logo和标题 -->
                    <div>
                        <div class="login-logo">
                            <i class="fas fa-film me-3"></i>
                            LightX2V
                        </div>
                        <p class="login-subtitle">{{ t('loginSubtitle') }}</p>
                    </div>
    <div class="space-y-6 w-[80%] mx-auto">
      <div>
        <label for="phoneNumber" class="block text-sm/6 font-medium text-gray-100 text-left">{{ t('phoneNumber') }}</label>
        <div class="mt-2">
          <input v-model="phoneNumber" type="tel" name="phoneNumber" required maxlength="11" @keyup.enter="handlePhoneEnter" class="block w-full rounded-md bg-white/5 px-3 py-1.5 text-base text-white outline-1 -outline-offset-1 outline-white/10 placeholder:text-gray-500 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-500 sm:text-sm/6" />
        </div>
      </div>

      <div>
        <div class="flex items-center justify-between">
          <label for="verifyCode" class="block text-sm/6 font-medium text-gray-100">{{ t('verifyCode') }}</label>
          <div class="text-sm">
            <button
                @click="sendSmsCode"
                class="font-semibold text-[#9a72ff] hover:text-indigo-300"
                :disabled="!phoneNumber || smsCountdown > 0 || loginLoading"
            >
                {{ smsCountdown > 0 ? `${smsCountdown}s` : t('sendSmsCode') }}
            </button>
        </div>
        </div>
        <div class="mt-2">
          <input v-model="verifyCode" type="text" name="verifyCode" required maxlength="6" @keyup.enter="handleVerifyCodeEnter" class="block w-full rounded-md bg-white/5 px-3 py-1.5 text-base text-white outline-1 -outline-offset-1 outline-white/10 placeholder:text-gray-500 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-500 sm:text-sm/6" />
        </div>
      </div>

      <div>
        <button @click="loginWithSms" :disabled="!phoneNumber || !verifyCode || loginLoading" class="btn-submit flex w-full justify-center rounded-md bg-laser-purple px-3 py-1.5 text-sm/6 font-semibold text-white hover:bg-indigo-400 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500">
            {{ loginLoading ? t('loginLoading') : t('login') }}
        </button>
      </div>
    </div>

                    <!-- 分隔线 -->
                    <div class="divider mb-4 pt-4">
                        <span class="divider-text">{{ t('orLoginWith') }}</span>
                    </div>

                    <!-- 第三方登录按钮 -->
                    <div class="social-login-buttons">
                        <button @click="loginWithGitHub" class="btn btn-icon"
                            :disabled="loginLoading" :title="t('loginWithGitHub')">
                            <i class="fab fa-github"></i>
                        </button>

                        <button @click="loginWithGoogle" class="btn btn-icon"
                            :disabled="loginLoading" :title="t('loginWithGoogle')">
                            <i class="fab fa-google"></i>
                        </button>
                    </div>
                </div>
            </div>
</template>
