<template>
    <n-layout style="height: 100vh; display: flex; justify-content: center; align-items: center;">
        <n-card title="管理员登录" style="width: 320px;" :bordered="false" size="large">
            <n-form :model="form" :rules="rules" ref="formRef">
                <n-form-item label="账号" path="username">
                    <n-input v-model:value="form.username" placeholder="请输入账号" />
                </n-form-item>
                <n-form-item label="密码" path="password">
                    <n-input v-model:value="form.password" type="password" show-password-on="click"
                        placeholder="请输入密码" />
                </n-form-item>
            </n-form>

            <div style="margin-top: 20px;">
                <n-button type="primary" block @click="handleLogin">
                    登录
                </n-button>
            </div>
        </n-card>
    </n-layout>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

// 表单数据
const form = ref({
    username: '',
    password: ''
})

// 表单校验
const rules = {
    username: [
        { required: true, message: '请输入账号', trigger: 'blur' }
    ],
    password: [
        { required: true, message: '请输入密码', trigger: 'blur' }
    ]
}

// 登录逻辑（这里直接简单模拟一下）
const formRef = ref()

function handleLogin() {
    formRef.value?.validate((errors) => {
        if (!errors) {

            axios.post(
                '/users/login',
                new URLSearchParams({
                    username: form.value.username,
                    password: form.value.password
                }),
                {
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
                }
            )
                .then(({ data }) => {
                    localStorage.setItem('access_token', data.access_token)
                    router.push('/admin')
                })
                .catch(error => {
                    console.error(error)
                })
        }
    })
}

</script>
