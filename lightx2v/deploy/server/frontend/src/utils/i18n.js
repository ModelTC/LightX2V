import { createI18n } from 'vue-i18n'
import { ref } from 'vue'

const loadedLanguages = new Set()

// 创建 i18n 实例（初始只设置 locale，不加载全部语言）
const i18n = createI18n({
  legacy: false,
  globalInjection: true,
  locale: 'zh',
  fallbackLocale: 'en',
  messages: {}
})

// 异步加载语言文件
async function loadLanguageAsync(lang) {
  if (!loadedLanguages.has(lang)) {
    const messages = await import(`../locales/${lang}.json`)
    i18n.global.setLocaleMessage(lang, messages.default)
    loadedLanguages.add(lang)
  }
  if (i18n.global.locale.value === lang) return lang
  i18n.global.locale.value = lang
  localStorage.setItem('app-lang', lang) // ✅ 记住用户选择
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  return lang
}

// 初始化默认语言
async function initLanguage() {
  const savedLang = localStorage.getItem('app-lang') || 'zh'
  return loadLanguageAsync(savedLang)
}
async function switchLang() {
  const newLang = i18n.global.locale.value === 'zh' ? 'en' : 'zh'
  await loadLanguageAsync(newLang)
}

      //   // 语言切换功能
      //   const switchLanguage = (langCode) => {
      //     currentLanguage.value = langCode;
      //     localStorage.setItem('preferredLanguage', langCode);

      //     // 更新页面标题
      //     document.title = t('pageTitle');

      //     // 更新HTML lang属性
      //     document.documentElement.lang = langCode === 'zh' ? 'zh-CN' : 'en';
      // };

      // // 简单语言切换功能（中英文切换）
      // const toggleLanguage = () => {
      //     const newLang = currentLanguage.value === 'zh' ? 'en' : 'zh';
      //     switchLanguage(newLang);
      // };

  const languageOptions = ref([
    { code: 'zh', name: '中文', flag: 'ZH' },
    { code: 'en', name: 'English', flag: 'EN' }
]);

export { i18n as default, loadLanguageAsync, initLanguage, switchLang, languageOptions }
