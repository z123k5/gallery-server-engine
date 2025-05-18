import { createRouter, createWebHistory } from 'vue-router'
import Login from '@/views/Login.vue'
import Admin from '@/views/Admin.vue'

const routes = [
    { path: '/', redirect: '/login' },
    { path: '/login', component: Login },
    { path: '/admin', component: Admin, meta: { requiresAuth: true } }
]

const router = createRouter({
    history: createWebHistory(),
    routes
})

// 全局前置守卫
router.beforeEach((to, from, next) => {
    const token = localStorage.getItem('token')

    if (to.meta.requiresAuth && !token) {
        // 需要登录但没 token，跳到登录页
        next('/login')
    } else {
        // 有 token 或者不需要登录，直接放行
        next()
    }
})

export default router
