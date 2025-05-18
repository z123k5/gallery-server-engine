<template>
    <div class="flex flex-col gap-4">
        <n-button type="info" block @click="downloadLog">下载异常日志</n-button>
    </div>
</template>

<script setup lang="ts">
import { NButton } from 'naive-ui'

function downloadLog() {
    // 简单模拟下载
    window.open('/api/download-log', '_blank')
}
</script>
