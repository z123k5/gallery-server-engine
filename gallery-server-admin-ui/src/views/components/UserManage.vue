<template>
    <div class="grid grid-cols-1 gap-4">
        <n-p>你选中了 {{ checkedRowKeys.length }} 行。</n-p>
        <n-data-table v-model:checked-row-keys="checkedRowKeys" :columns="columns" :data="data" :pagination="pagination"
            :row-key="rowKey" />
        <n-card title="流量统计">
            <div>在此处展示统计数据...</div>
        </n-card>
    </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, h } from 'vue'
import { useMessage, NButton } from 'naive-ui'
import { DataTableColumns, useDialog, NInputNumber, NModal } from 'naive-ui'
import * as echarts from 'echarts'
import axios from 'axios'

const dialog = useDialog()

interface User {
    id: number
    username: string
    email: string
    is_audited: boolean
    is_admin: boolean
    upload_limit_a_day: number
    key: number
}

export default defineComponent({
    setup() {
        const message = useMessage()
        const checkedRowKeys = ref<Array<string | number>>([])
        const data = ref<User[]>([])
        const pagination = {
            pageSize: 6
        }
        const dialog = useDialog()

        const rowKey = (row: User) => row.key

        const columns: DataTableColumns<User> = [
            {
                type: 'selection',
                options: [
                    'all',
                    'none',
                    {
                        label: '选中前 2 行',
                        key: 'f2',
                        onSelect: (pageData) => {
                            checkedRowKeys.value = pageData.map(row => row.key).slice(0, 2)
                        }
                    }
                ]
            },
            {
                title: '用户名',
                key: 'username'
            },
            {
                title: '邮箱',
                key: 'email'
            },
            {
                title: '审核状态',
                key: 'is_audited',
                render(row) {
                    return h('span', row.is_audited ? '已审核' : '未审核')
                }
            },
            {
                title: '是否管理员',
                key: 'is_admin',
                render(row) {
                    return h('span', row.is_admin ? '是' : '否')
                }
            },
            {
                title: '日上传限制',
                key: 'upload_limit_a_day',
                render(row) {
                    return row.upload_limit_a_day === -1 ? h('span', '无限制')
                        : h('span', row.upload_limit_a_day + ' 次')
                }
            },
            {
                title: '操作',
                key: 'actions',
                render(row) {
                    return h('div', [
                        h(
                            NButton,
                            {
                                size: 'small',
                                type: 'success',
                                onClick: () => auditUser(row.id)
                            },
                            { default: () => '审核' }
                        ),
                        h(
                            NButton,
                            {
                                size: 'small',
                                type: 'error',
                                onClick: () => deleteUser(row.id)
                            },
                            { default: () => '删除' }
                        ),
                        h(
                            NButton,
                            {
                                size: 'small',
                                type: 'primary',
                                onClick: () => resetPwd(row.id, row.username)
                            },
                            { default: () => '重置密码' }
                        ),
                        h(
                            NButton,
                            {
                                size: 'small',
                                type: 'warning',
                                onClick: () => limitFreq(row.id)
                            },
                            { default: () => '限制频率' }
                        ),
                        h(
                            NButton,
                            {
                                size: 'small',
                                type: 'info',
                                onClick: () => reviewFlow(row.id)
                            },
                            { default: () => '审查流量' }
                        )
                    ])
                }
            }
        ]

        const authHeader = {
            headers: {
                Authorization: 'Bearer ' + localStorage.getItem('access_token')
            }
        }

        onMounted(async () => {
            try {
                const res = await axios.get('/admin/get_user_list', authHeader)
                data.value = res.data.data.map((user: User) => ({
                    ...user,
                    key: user.id
                }))
            } catch (error) {
                message.error('获取用户列表失败')
            }
        })

        async function auditUser(userId: number) {
            try {
                await axios.post(`/admin/audit_new_user?userId=${userId}`, null, authHeader)
                message.success('审核成功')
                data.value = data.value.map(user => {
                    if (user.id === userId) {
                        return {
                            ...user,
                            is_audited: true
                        } as User
                    }
                    return user
                })
            } catch {
                message.error('审核失败')
            }

        }

        async function deleteUser(userId: number) {
            const d = dialog.create({
                title: '确认删除',
                content: '确定要删除该用户吗？此操作不可恢复。',
                positiveText: '删除',
                negativeText: '取消',
                onPositiveClick: async () => {
                    try {
                        await axios.post(`/admin/delete_user?userId=${userId}`, null, authHeader)
                        message.success('删除成功')
                        data.value = data.value.filter(user => user.id !== userId)
                    } catch {
                        message.error('删除失败')
                    }
                    d.destroy()
                },
                onNegativeClick: () => {
                    d.destroy()
                }
            })
        }

        async function resetPwd(userId: number, userName: string) {
            try {
                await axios.post(`/admin/reset_user_password?userId=${userId}&userName=${userName}`
                    , authHeader)
                message.success('密码已重置')
            } catch {
                message.error('重置密码失败')
            }
        }

        async function limitFreq(userId: number) {
            const limitValue = ref<number | null>(null)

            const d = dialog.create({
                title: '设置上传限制',
                content: () => h(NInputNumber, {
                    value: limitValue.value,
                    'onUpdate:value': (value: number | null) => {
                        limitValue.value = value
                    },
                    placeholder: '请输入每日上传限制'
                }),
                action: () => h(NButton, {
                    onClick: async () => {
                        console.log(limitValue.value)
                        const numValue = Number(limitValue.value)

                        if (!isNaN(numValue) && numValue >= 0) {
                            try {
                                await axios.post(
                                    `/admin/limit_user_frequency?userId=${userId}&upload_limit_a_day=${numValue}`,
                                    null,
                                    authHeader
                                )
                                data.value = data.value.map(user => {
                                    if (user.id === userId) {
                                        return {
                                            ...user,
                                            upload_limit_a_day: numValue
                                        } as User
                                    }
                                    return user
                                })
                                message.success('限制已设置')
                            } catch {
                                message.error('限制失败')
                            }
                            d.destroy()
                        } else {
                            message.warning('请输入有效的数值')
                        }
                    }
                }, '确定'),
                onClose: () => d.destroy()
            })
        }

        function renderChart(container: HTMLElement, data: Array<{ date: string; count: number }>) {
            const chart = echarts.init(container)
            const dates = data.map(d => d.date)
            const counts = data.map(d => d.count)

            chart.setOption({
                tooltip: { trigger: 'axis' },
                xAxis: { type: 'category', data: dates },
                yAxis: { type: 'value' },
                series: [{ name: '上传次数', type: 'line', data: counts }]
            })
        }

        async function fetchUploadStats(userId: number, container: HTMLElement, dialogInstance: any) {
            try {
                const res = await axios.get(`/admin/audit_user_upload?userIds=[${userId}]`, authHeader)
                if (res.data.status === 'success') {
                    const data = res.data.data // 假设返回格式为 { date: '2024-07-01', count: 12 }[]
                    renderChart(container, data)
                } else {
                    message.error(res.data.message || '获取数据失败')
                    dialogInstance.destroy()
                }
            } catch (e) {
                message.error('请求失败')
                dialogInstance.destroy()
            }
        }

        function reviewFlow(userId: number) {
            const d = dialog.create({
                title: '用户上传流量统计',
                content: () => {
                    const chartContainer = document.createElement('div')
                    chartContainer.style.width = '500px'
                    chartContainer.style.height = '300px'

                    // 使用 h 包裹 chartContainer，返回合法的 VNode
                    return h('div', {
                        style: {
                            width: '500px',
                            height: '300px'
                        },
                        ref: (el) => {
                            if (el) {
                                fetchUploadStats(userId, el as HTMLElement, d)
                            }
                        }
                    })
                },
                action: () => h(NButton, { onClick: () => d.destroy() }, '关闭')
            })
        }

        return {
            data,
            checkedRowKeys,
            columns,
            pagination,
            rowKey
        }
    }
})
</script>
