<template>
  <div class="memento-app">
    <!-- 顶部栏 -->
    <header class="topbar">
      <div class="topbar-left">
        <button class="menu-btn" @click="sidebarOpen = !sidebarOpen" title="菜单">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M3 12h18M3 6h18M3 18h18"/>
          </svg>
        </button>
        <span class="brand">Memento</span>
        <span class="badge">v2.1</span>
      </div>

      <div class="topbar-center">
        <span class="status-dot" :class="cloudOnline ? 'online' : 'offline'"></span>
        <span class="status-text">{{ cloudOnline ? '云端在线' : '云端离线' }}</span>
        <span class="sep">|</span>
        <span class="status-dot" :class="comfyReady ? 'online' : 'offline'"></span>
        <span class="status-text">{{ comfyReady ? 'ComfyUI 就绪' : 'ComfyUI 未连接' }}</span>
      </div>

      <div class="topbar-right">
        <button class="topbar-btn" @click="activePanel = 'cloud'" :class="{ active: activePanel === 'cloud' }">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>
          云端
        </button>
        <button class="topbar-btn" @click="activePanel = 'task'" :class="{ active: activePanel === 'task' }">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="1"/></svg>
          任务
        </button>
        <button class="topbar-btn" @click="activePanel = 'model'" :class="{ active: activePanel === 'model' }">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><circle cx="8" cy="6" r="1"/><circle cx="8" cy="18" r="1"/></svg>
          模型
        </button>
      </div>
    </header>

    <div class="main-area">
      <!-- 侧边栏 -->
      <aside class="sidebar" :class="{ open: sidebarOpen }">
        <div class="panel" v-if="sidebarOpen">
          <CloudPanel v-if="activePanel === 'cloud'" :api-url="apiUrl" />
          <TaskPanel v-if="activePanel === 'task'" :api-url="apiUrl" />
          <ModelPanel v-if="activePanel === 'model'" :local-url="localUrl" />
        </div>
      </aside>

      <!-- ComfyUI iframe -->
      <div class="comfy-container" :class="{ 'sidebar-open': sidebarOpen }">
        <div v-if="!comfyReady" class="comfy-loading">
          <div class="spinner"></div>
          <p>正在连接 ComfyUI...</p>
          <p class="hint">请确保 ComfyUI 已启动 (端口 8188)</p>
        </div>
        <iframe
          ref="comfyFrame"
          :src="comfyUrl"
          class="comfy-iframe"
          @load="onComfyLoad"
          allow="clipboard-read; clipboard-write"
        ></iframe>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import CloudPanel from './components/CloudPanel.vue'
import TaskPanel from './components/TaskPanel.vue'
import ModelPanel from './components/ModelPanel.vue'

const sidebarOpen = ref(true)
const activePanel = ref('cloud')
const comfyReady = ref(false)
const cloudOnline = ref(false)
const comfyUrl = ref('http://127.0.0.1:8188')
const apiUrl = ref(localStorage.getItem('memento_api_url') || 'http://118.31.189.101:8000/api/v1')
const localUrl = ref(window.location.port === '8189' ? '' : 'http://127.0.0.1:8189')
const comfyFrame = ref(null)

let pollingTimer = null

function onComfyLoad() {
  comfyReady.value = true
}

function checkComfy() {
  fetch(comfyUrl.value + '/system_stats', { signal: AbortSignal.timeout(3000) })
    .then(r => { if (r.ok) comfyReady.value = true })
    .catch(() => { comfyReady.value = false })
}

function checkCloud() {
  fetch(apiUrl.value + '/health', { signal: AbortSignal.timeout(5000) })
    .then(r => r.json())
    .then(d => { cloudOnline.value = d.status === 'ok' || d.healthy === true })
    .catch(() => { cloudOnline.value = false })
}

onMounted(() => {
  checkComfy()
  checkCloud()
  pollingTimer = setInterval(() => {
    checkComfy()
    checkCloud()
  }, 15000)
})

onUnmounted(() => {
  if (pollingTimer) clearInterval(pollingTimer)
})
</script>

<style>
.memento-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #0a0a0f;
  color: #e0e0e0;
}

/* ── Top Bar ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 44px;
  padding: 0 12px;
  background: #14141a;
  border-bottom: 1px solid #222;
  flex-shrink: 0;
  z-index: 10;
}
.topbar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}
.menu-btn {
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  display: flex;
}
.menu-btn:hover { color: #fff; background: #222; }
.brand {
  font-size: 15px;
  font-weight: 700;
  color: #00d4aa;
  letter-spacing: -0.5px;
}
.badge {
  font-size: 10px;
  background: #00d4aa22;
  color: #00d4aa;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
}
.topbar-center {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #666;
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}
.status-dot.online { background: #00d4aa; box-shadow: 0 0 6px #00d4aa66; }
.status-dot.offline { background: #555; }
.status-text { color: #888; }
.sep { color: #333; }
.topbar-right {
  display: flex;
  gap: 4px;
}
.topbar-btn {
  background: none;
  border: none;
  color: #777;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
  transition: all 0.15s;
}
.topbar-btn:hover { color: #ccc; background: #1a1a22; }
.topbar-btn.active { color: #00d4aa; background: #00d4aa15; }

/* ── Main Area ── */
.main-area {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 0;
  overflow: hidden;
  background: #111118;
  border-right: 1px solid #222;
  transition: width 0.2s ease;
  flex-shrink: 0;
}
.sidebar.open {
  width: 320px;
}
.panel {
  width: 320px;
  height: 100%;
  overflow-y: auto;
}

/* ── ComfyUI Container ── */
.comfy-container {
  flex: 1;
  position: relative;
  background: #0a0a0f;
}
.comfy-iframe {
  width: 100%;
  height: 100%;
  border: none;
}
.comfy-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #666;
  font-size: 14px;
}
.comfy-loading .hint {
  font-size: 12px;
  color: #444;
}
.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #222;
  border-top-color: #00d4aa;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
