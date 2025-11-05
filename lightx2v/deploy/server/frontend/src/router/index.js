// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import Generate from '../components/Generate.vue'
import Projects from '../components/Projects.vue'
import Inspirations from '../components/Inspirations.vue'
import Share from '../views/Share.vue'
import { showAlert } from '../utils/other'

const routes = [
  {
    path: '/',
    redirect: (to) => {
      // 保留查询参数（用于 OAuth 回调）
      return { path: '/generate', query: to.query }
    }
  },
  {
    path: '/login', name: 'Login', component: Login, meta: { requiresAuth: false }
  },
  {
    path: '/share/:shareId', name: 'Share', component: Share, meta: { requiresAuth: false }
  },
  {
    path: '/home',
    component: Layout,
    meta: {
      requiresAuth: true
    },
    children: [
      {
        path: '/generate',
        name: 'Generate',
        component: Generate,
        meta: { requiresAuth: true },
        props: route => ({ query: route.query })
      },
      {
        path: '/projects',
        name: 'Projects',
        component: Projects,
        meta: { requiresAuth: true },
        props: route => ({ query: route.query })
      },
      {
        path: '/inspirations',
        name: 'Inspirations',
        component: Inspirations,
        meta: { requiresAuth: true },
        props: route => ({ query: route.query })
      },
      {
        path: '/task/:taskId',
        name: 'TaskDetail',
        component: Projects,
        meta: { requiresAuth: true },
        props: route => ({ taskId: route.params.taskId, query: route.query })
      },
      {
        path: '/template/:templateId',
        name: 'TemplateDetail',
        component: Inspirations,
        meta: { requiresAuth: true },
        props: route => ({ templateId: route.params.templateId, query: route.query })
      },
    ]
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('../views/404.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫 - 整合和优化后的逻辑
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('accessToken')
  console.log('token', token)
  // 检查 URL 中是否有 code 参数（OAuth 回调）
  // 可以从路由查询参数或实际 URL 中获取
  const hasOAuthCode = to.query?.code !== undefined ||
                      (typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('code') !== null)

  // 1. OAuth 回调处理：如果有 code 参数（GitHub/Google 登录回调），直接放行
  // App.vue 的 onMounted 会处理登录回调逻辑
  if (hasOAuthCode) {
    console.log('检测到 OAuth 回调，放行让 App.vue 处理')
    next()
    return
  }

  // 2. 不需要登录的页面（登录页、分享页等）
  if (to.meta.requiresAuth === false) {
    // 如果已登录用户访问登录页，重定向到生成页面
    if (token && to.path === '/login') {
      console.log('已登录用户访问登录页，重定向到生成页')
      next('/generate')
    } else {
      next()
    }
    return
  }

  // 3. 需要登录的页面处理
  if (!token) {
    // 未登录但访问需要登录的页面，跳转到登录页
    console.log('需要登录但未登录，跳转到登录页')
    next('/login')
    // 延迟显示提示，确保路由跳转完成
    setTimeout(() => {
      showAlert('请先登录', 'warning')
    }, 100)
    return
  }

  // 4. 已登录用户处理
  // 已登录用户访问首页，重定向到生成页（保留查询参数）
  if (to.path === '/') {
    console.log('已登录用户访问首页，重定向到生成页')
    const query = to.query && Object.keys(to.query).length > 0 ? to.query : {}
    next({ path: '/generate', query })
  } else {
    // 已登录且访问其他页面，正常放行
    next()
  }
})



export default router;
