<template>
    <div class="flex flex-col gap-4">
        <n-upload :action="'/api/upload-model'" :max="1">
            <n-button type="primary">上传新模型</n-button>
        </n-upload>
        <n-select :options="modelOptions" placeholder="切换当前模型" />
        <n-statistic label="CPU占用率" :value="56" suffix="%" />
        <n-statistic label="内存使用" :value="3.2" suffix="GB" />
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { NButton, NUpload, NSelect, NStatistic } from 'naive-ui'

const modelOptions = ref([
    { label: 'VLM-ResNet50', value: 'resnet50' },
    { label: 'VLM-ViT-B/32', value: 'vit-b32' }
])
</script>