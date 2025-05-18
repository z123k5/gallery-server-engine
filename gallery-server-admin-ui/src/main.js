import './assets/main.css'
import naive from 'naive-ui'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import axios from 'axios'

const app = createApp(App)
app.use(naive)
app.use(router)

// Set up axios
// 设置全局地址
axios.defaults.baseURL = 'http://127.0.0.1:8443';
// axios.defaults.baseURL = 'https://frp-dad:34952';


app.mount('#app')
