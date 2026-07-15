<template>
  <div class="cloud-panel">
    <h3 class="panel-title">云端连接</h3>

    <div class="section">
      <label class="label">API 地址</label>
      <input v-model="apiUrl" class="input" placeholder="http://memento.asia/api/v1" @change="save" />
    </div>

    <div class="section">
      <label class="label">用户 Token</label>
      <div class="input-row">
        <input v-model="token" :type="showToken ? 'text' : 'password'" class="input" placeholder="粘贴你的 Memento Token" @change="save" />
        <button class="icon-btn" @click="showToken = !showToken" :title="showToken ? '隐藏' : '显示'">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        </button>
      </div>
      <p class="hint">从 Web 端获取 Token，或联系管理员</p>
    </div>

    <div class="section">
      <button class="btn primary" @click="connect" :disabled="connecting">
        {{ connecting ? '连接中...' : (connected ? '重新连接' : '连接云端') }}
      </button>
      <button v-if="connected" class="btn danger" @click="disconnect">断开</button>
    </div>

    <div v-if="statusMsg" class="status-msg" :class="statusType">
      {{ statusMsg }}
    </div>

    <div class="divider"></div>

    <div class="section">
      <label class="label">连接状态</label>
      <div class="info-grid">
        <div class="info-item">
          <span class="info-label">云端</span>
          <span class="info-value" :class="cloudOk ? 'ok' : 'err'">{{ cloudOk ? '在线' : '离线' }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">注册</span>
          <span class="info-value" :class="connected ? 'ok' : ''">{{ connected ? '已注册' : '未注册' }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">本机</span>
          <span class="info-value" :class="comfyOk ? 'ok' : 'err'">{{ comfyOk ? 'ComfyUI 在线' : 'ComfyUI 离线' }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">GPU</span>
          <span class="info-value" :class="gpuOk ? 'ok' : 'err'">{{ gpuModel || '检测中...' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const props = defineProps({ apiUrl: String })

const apiUrl = ref(props.apiUrl || 'http://memento.asia/api/v1')
const token = ref(localStorage.getItem('memento_token') || '')
const showToken = ref(false)
const connecting = ref(false)
const connected = ref(false)
const statusMsg = ref('')
const statusType = ref('info')
const cloudOk = ref(false)
const comfyOk = ref(false)
const gpuModel = ref('')
const gpuOk = ref(false)

function save() {
  localStorage.setItem('memento_api_url', apiUrl.value)
  localStorage.setItem('memento_token', token.value)
}

async function connect() {
  if (!token.value.trim()) {
    statusMsg.value = '请输入 Token'
    statusType.value = 'error'
    return
  }
  connecting.value = true
  statusMsg.value = ''
  try {
    const resp = await fetch(apiUrl.value + '/workflow/local/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: token.value.trim(),
        host: '127.0.0.1',
        port: 8188,
        version: '2.1.0'
      })
    })
    const data = await resp.json()
    if (data.status === 'ok') {
      connected.value = true
      statusMsg.value = '✅ 注册成功 — ' + (data.message || '')
      statusType.value = 'success'
      save()
    } else {
      statusMsg.value = '注册失败: ' + (data.message || data.detail || '未知错误')
      statusType.value = 'error'
    }
  } catch (e) {
    statusMsg.value = '连接失败: ' + e.message
    statusType.value = 'error'
  }
  connecting.value = false
}

async function disconnect() {
  if (!token.value.trim()) return
  try {
    await fetch(apiUrl.value + '/workflow/local/unregister', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: token.value.trim() })
    })
    connected.value = false
    statusMsg.value = '已断开连接'
    statusType.value = 'info'
  } catch (e) {
    statusMsg.value = '断开失败: ' + e.message
    statusType.value = 'error'
  }
}

async function checkHealth() {
  try {
    const r = await fetch(apiUrl.value + '/health', { signal: AbortSignal.timeout(5000) })
    const d = await r.json()
    cloudOk.value = d.status === 'ok' || d.healthy === true
  } catch { cloudOk.value = false }

  try {
    const r = await fetch('http://127.0.0.1:8188/system_stats', { signal: AbortSignal.timeout(3000) })
    if (r.ok) {
      comfyOk.value = true
      const d = await r.json()
      if (d.system && d.system.gpu) {
        const gpu = d.system.gpu
        gpuModel.value = gpu.name || gpu.gpu_name || 'GPU'
        gpuOk.value = true
      }
    }
  } catch { comfyOk.value = false; gpuOk.value = false }

  // 如果已连接，发心跳
  if (connected.value && token.value.trim()) {
    try {
      await fetch(apiUrl.value + '/workflow/local/heartbeat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: token.value.trim(),
          host: '127.0.0.1',
          port: 8188,
          version: '2.1.0',
          active_tasks: 0,
          gpu_available: gpuOk.value
        })
      })
    } catch {}
  }
}

let timer = null
onMounted(() => {
  checkHealth()
  timer = setInterval(checkHealth, 30000)
})
</script>

<style scoped>
.cloud-panel { padding: 16px; }
.panel-title {
  font-size: 14px;
  font-weight: 700;
  color: #ccc;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid #222;
}
.section { margin-bottom: 14px; }
.label {
  display: block;
  font-size: 11px;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.input {
  width: 100%;
  padding: 8px 10px;
  background: #1a1a24;
  border: 1px solid #333;
  border-radius: 6px;
  color: #ddd;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}
.input:focus { border-color: #00d4aa; }
.input-row { display: flex; gap: 4px; align-items: center; }
.input-row .input { flex: 1; }
.icon-btn {
  background: none;
  border: 1px solid #333;
  color: #666;
  padding: 6px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
}
.icon-btn:hover { color: #999; border-color: #555; }
.hint { font-size: 11px; color: #444; margin-top: 4px; }
.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.15s;
  margin-right: 8px;
}
.btn.primary { background: #00d4aa; color: #0a0a0f; }
.btn.primary:hover { background: #00e6b8; }
.btn.primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn.danger { background: #3a1a1a; color: #ef4444; }
.btn.danger:hover { background: #4a2020; }
.status-msg {
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  margin-top: 8px;
}
.status-msg.success { background: #00d4aa15; color: #00d4aa; }
.status-msg.error { background: #ef444415; color: #ef4444; }
.status-msg.info { background: #1a1a24; color: #888; }
.divider { height: 1px; background: #222; margin: 16px 0; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.info-item {
  background: #1a1a24;
  padding: 8px 10px;
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.info-label { font-size: 10px; color: #555; text-transform: uppercase; }
.info-value { font-size: 12px; font-weight: 600; color: #aaa; }
.info-value.ok { color: #00d4aa; }
.info-value.err { color: #ef4444; }
</style>
