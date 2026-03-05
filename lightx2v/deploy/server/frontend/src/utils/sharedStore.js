/**
 * 跨框架共享状态管理
 * 用于 Vue 和 React 应用之间的状态同步
 */
class SharedStore {
  constructor() {
    this.listeners = new Set()
    // 从 localStorage 读取初始状态
    this.state = {
      user: this.getLocalStorageItem('currentUser', null),
      token: this.getLocalStorageItem('accessToken', null),
      theme: this.getLocalStorageItem('theme', 'light'),
      language: this.getLocalStorageItem('app-lang', 'zh'),
      isLoggedIn: !!this.getLocalStorageItem('accessToken', null)
    }

    // 监听 localStorage 变化（跨标签页同步）
    window.addEventListener('storage', (e) => {
      if (e.key === 'currentUser' || e.key === 'accessToken') {
        this.syncFromLocalStorage()
        this.notify()
      }
    })
  }

  getLocalStorageItem(key, defaultValue = null) {
    try {
      const item = localStorage.getItem(key)
      if (item === null) return defaultValue
      // 尝试解析 JSON
      if (item.startsWith('{') || item.startsWith('[')) {
        return JSON.parse(item)
      }
      return item
    } catch (e) {
      return defaultValue
    }
  }

  // 从 localStorage 同步状态
  syncFromLocalStorage() {
    const token = localStorage.getItem('accessToken')
    const user = this.getLocalStorageItem('currentUser', null)
    const theme = localStorage.getItem('theme') || 'light'
    const language = localStorage.getItem('app-lang') || 'zh'

    this.state = {
      ...this.state,
      token,
      user,
      theme,
      language,
      isLoggedIn: !!token
    }
  }

  // 设置状态
  setState(key, value) {
    this.state[key] = value

    // 同步到 localStorage
    if (key === 'token') {
      if (value) {
        localStorage.setItem('accessToken', value)
      } else {
        localStorage.removeItem('accessToken')
      }
      this.state.isLoggedIn = !!value
    } else if (key === 'user') {
      if (value) {
        localStorage.setItem('currentUser', JSON.stringify(value))
      } else {
        localStorage.removeItem('currentUser')
      }
    } else if (key === 'theme') {
      localStorage.setItem('theme', value)
    } else if (key === 'language') {
      localStorage.setItem('app-lang', value)
    }

    this.notify()
  }

  // 批量设置状态
  setStates(states) {
    Object.keys(states).forEach(key => {
      this.setState(key, states[key])
    })
  }

  // 获取状态
  getState(key) {
    // 如果 key 为空，返回整个 state
    if (!key) {
      return { ...this.state }
    }
    return this.state[key]
  }

  // 订阅变化
  subscribe(callback) {
    this.listeners.add(callback)
    // 返回取消订阅函数
    return () => {
      this.listeners.delete(callback)
    }
  }

  // 通知所有监听者
  notify() {
    this.listeners.forEach(cb => {
      try {
        cb(this.state)
      } catch (e) {
        console.error('Error in SharedStore listener:', e)
      }
    })
  }

  // 清除所有状态
  clear() {
    this.setState('token', null)
    this.setState('user', null)
    this.setState('isLoggedIn', false)
  }
}

// 单例
export const sharedStore = new SharedStore()

// 如果 localStorage 中有数据，同步一次
sharedStore.syncFromLocalStorage()
