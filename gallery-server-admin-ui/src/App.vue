<template>
  <n-message-provider>
    <n-notification-provider>
      <n-dialog-provider>
        <router-view />
      </n-dialog-provider>
    </n-notification-provider>
  </n-message-provider>
</template>

<script setup lang="ts">
// 这里啥都不用写，保持干净
</script>

<style>
html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background-color: #f5f7fa;
}
</style>
