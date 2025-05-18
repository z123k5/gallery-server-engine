<template>
    <n-layout has-sider class="h-screen">
        <!-- 左侧菜单 -->
        <n-layout-sider width="220" bordered>
            <n-menu v-model:value="activeMenu" :options="menuOptions" :collapsed-width="64" :collapsed-icon-size="22" />
        </n-layout-sider>

        <!-- 内容区域 -->
        <n-layout-content class="p-6 overflow-auto">
            <n-card :title="currentTitle">
                <component :is="currentComponent" />
            </n-card>
        </n-layout-content>
    </n-layout>
</template>

<script setup lang="ts">
import { ref, computed, defineAsyncComponent } from 'vue'
import { NLayout, NLayoutSider, NLayoutContent, NMenu, NCard } from 'naive-ui'

// 菜单
const activeMenu = ref('user')
const menuOptions = [
    { label: '用户管理', key: 'user' },
    { label: '模型管理', key: 'model' },
    { label: '用户图库管理', key: 'gallery' },
    { label: '日志下载', key: 'log' }
]

// 显示标题
const currentTitle = computed(() => {
    const item = menuOptions.find(i => i.key === activeMenu.value)
    return item ? item.label : ''
})

// 每个板块的组件
const componentsMap: Record<string, any> = {
    user: defineAsyncComponent(() => import('./components/UserManage.vue')),
    model: defineAsyncComponent(() => import('./components/ModelManage.vue')),
    gallery: defineAsyncComponent(() => import('./components/GalleryManage.vue')),
    log: defineAsyncComponent(() => import('./components/LogManage.vue'))
}

// 当前显示的组件
const currentComponent = computed(() => componentsMap[activeMenu.value])
</script>

<style scoped>
.h-screen {
    height: 100vh;
}

.p-6 {
    padding: 24px;
}

.overflow-auto {
    overflow: auto;
}
</style>
