import { createApp } from 'vue'
import router from './router'

import './style.css'
import App from './App.vue'
import { createPinia } from 'pinia'

import i18n, { initLanguage } from './utils/i18n'

const app = createApp(App)
const pinia = createPinia()

app.use(i18n)
app.use(pinia)
app.use(router)

// 初始化语言
initLanguage().then(() => {
    app.mount('#app')
  })
