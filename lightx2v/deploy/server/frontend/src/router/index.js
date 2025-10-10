// src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import Login from '../views/Login.vue'
import Layout from '../views/Layout.vue'
import Generate from '../components/Generate.vue'
import Projects from '../components/Projects.vue'
import Inspirations from '../components/Inspirations.vue'
import Share from '../views/Share.vue'

const routes = [
  { path: '/', redirect: '/login' },
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

// 路由守卫
// router/index.js
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('accessToken')
  console.log('路由守卫 - token:', token)
  console.log('路由守卫 - to.path:', to.path)

  // 如果是不需要登录的页面，直接放行
  if (to.meta.requiresAuth === false) {
    console.log('不需要登录的页面，直接放行')
    // 如果已登录用户访问登录页面，重定向到生成页面
    if (token && to.path === '/login') {
      console.log('已登录用户访问登录页，重定向到生成页')
      next('/generate');
    } else {
      next();
    }
    return;
  }

  // 需要登录的页面
  if (!token) { // 未登录
    console.log('需要登录但未登录，跳转到登录页')
    next('/login');
  } else {
    console.log('已登录')
    if (to.path === '/') { // 已登录且在首页
      console.log('已登录且在首页，跳转到生成页')
      next('/generate');
    } else { // 已登录且不在首页
      console.log('已登录，放行')
      next();
    }
  }
})



export default router;
