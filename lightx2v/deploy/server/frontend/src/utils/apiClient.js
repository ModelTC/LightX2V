/**
 * 统一 API 客户端
 * 用于 Vue 和 React 应用的统一后端调用
 */
import { sharedStore } from './sharedStore'

class ApiClient {
  constructor() {
    // 从环境变量或配置中获取后端地址
    this.baseURL = import.meta.env.VITE_API_BASE_URL || "" // empty string means relative path
  }

  /**
   * 通用请求方法
   */
  async request(url, options = {}) {
    const token = sharedStore.getState('token')
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers
    }

    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    try {
      const response = await fetch(`${this.baseURL}${url}`, {
        ...options,
        headers
      })

      // 处理认证失败
      if (response.status === 401) {
        // token 过期，清除状态
        sharedStore.clear()
        // 如果在非登录页面，跳转到登录页
        if (!window.location.pathname.includes('/login')) {
          window.location.href = '/login'
        }
        throw new Error('未授权，请重新登录')
      }

      if (!response.ok) {
        const errorText = await response.text()
        let errorMessage = `API Error: ${response.status}`
        try {
          const errorData = JSON.parse(errorText)
          errorMessage = errorData.message || errorData.error || errorMessage
        } catch (e) {
          errorMessage = errorText || errorMessage
        }
        throw new Error(errorMessage)
      }

      // 处理空响应
      const contentType = response.headers.get('content-type')
      if (contentType && contentType.includes('application/json')) {
        return await response.json()
      }

      return await response.text()
    } catch (error) {
      console.error('API Request Error:', error)
      throw error
    }
  }

  /**
   * GET 请求
   */
  async get(url, params = {}) {
    const queryString = new URLSearchParams(params).toString()
    const fullUrl = queryString ? `${url}?${queryString}` : url
    return this.request(fullUrl, { method: 'GET' })
  }

  /**
   * POST 请求
   */
  async post(url, data = {}) {
    return this.request(url, {
      method: 'POST',
      body: JSON.stringify(data)
    })
  }

  /**
   * PUT 请求
   */
  async put(url, data = {}) {
    return this.request(url, {
      method: 'PUT',
      body: JSON.stringify(data)
    })
  }

  /**
   * DELETE 请求
   */
  async delete(url) {
    return this.request(url, { method: 'DELETE' })
  }

  /**
   * PATCH 请求
   */
  async patch(url, data = {}) {
    return this.request(url, {
      method: 'PATCH',
      body: JSON.stringify(data)
    })
  }
}

// 单例
export const apiClient = new ApiClient()
