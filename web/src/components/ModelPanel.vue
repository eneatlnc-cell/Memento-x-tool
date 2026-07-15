<template>
  <div class="model-panel">
    <h3 class="panel-title">模型状态</h3>

    <div class="section">
      <div class="section-header">
        <label class="label">自动下载</label>
        <span class="badge" :class="autoReady ? 'ok' : 'warn'">{{ autoReady ? '全部就绪' : '有缺失' }}</span>
      </div>
      <div v-if="loading" class="loading">加载中...</div>
      <div v-for="m in autoModels" :key="m.name" class="model-item">
        <span class="model-status" :class="m.ready ? 'ready' : 'missing'">{{ m.ready ? '✓' : '✗' }}</span>
        <span class="model-name">{{ m.name }}</span>
        <span class="model-size">{{ m.size }}</span>
      </div>
    </div>

    <div class="section">
      <div class="section-header">
        <label class="label">手动下载</label>
        <span class="badge" :class="manualReady ? 'ok' : 'warn'">{{ manualReady ? '全部就绪' : '需操作' }}</span>
      </div>
      <div v-for="m in manualModels" :key="m.name" class="model-item">
        <span class="model-status" :class="m.ready ? 'ready' : 'missing'">{{ m.ready ? '✓' : '✗' }}</span>
        <div class="model-info">
          <span class="model-name">{{ m.name }}</span>
          <span class="model-size">{{ m.size }}</span>
          <span v-if="!m.ready && m.reason" class="model-reason">{{ m.reason }}</span>
        </div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="section">
      <label class="label">模型目录</label>
      <code class="dir-path">{{ modelDir || '~/.memento/workspace/models/' }}</code>
    </div>

    <div class="section">
      <label class="label">已下载</label>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
      </div>
      <span class="progress-text">{{ downloadedGb }} GB / {{ totalGb }} GB</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'

const props = defineProps({ localUrl: String })

const loading = ref(true)
const autoModels = ref([])
const manualModels = ref([])
const autoReady = ref(false)
const manualReady = ref(false)
const modelDir = ref('')
const downloadedGb = ref(0)
const autoTotalGb = ref(0)
const manualTotalGb = ref(0)

const totalGb = computed(() => +(autoTotalGb.value + manualTotalGb.value).toFixed(1))
const progressPercent = computed(() => totalGb.value > 0 ? Math.min(100, (downloadedGb.value / totalGb.value) * 100) : 0)

async function loadStatus() {
  try {
    const resp = await fetch(props.localUrl + '/models/status', { signal: AbortSignal.timeout(5000) })
    const data = await resp.json()
    autoModels.value = data.auto || []
    manualModels.value = data.manual || []
    autoReady.value = data.auto_ready
    modelDir.value = data.model_dir || ''
    downloadedGb.value = data.downloaded_gb || 0
    autoTotalGb.value = data.auto_total_gb || 0
    manualTotalGb.value = data.manual_total_gb || 0

    // 检查手动模型是否全部就绪
    manualReady.value = (data.manual || []).every(m => m.ready)
  } catch {
    autoModels.value = []
    manualModels.value = []
  }
  loading.value = false
}

onMounted(() => {
  loadStatus()
  setInterval(loadStatus, 30000)
})
</script>

<style scoped>
.model-panel { padding: 16px; }
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
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.section-header .label { margin-bottom: 0; }
.badge {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
}
.badge.ok { background: #00d4aa15; color: #00d4aa; }
.badge.warn { background: #f59e0b15; color: #f59e0b; }
.loading { font-size: 12px; color: #555; padding: 8px 0; }
.model-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid #1a1a22;
}
.model-item:last-child { border-bottom: none; }
.model-status {
  font-size: 12px;
  font-weight: 700;
  width: 18px;
  text-align: center;
  flex-shrink: 0;
}
.model-status.ready { color: #00d4aa; }
.model-status.missing { color: #ef4444; }
.model-name { font-size: 12px; color: #ccc; flex: 1; }
.model-size { font-size: 11px; color: #555; flex-shrink: 0; }
.model-info { flex: 1; }
.model-reason {
  display: block;
  font-size: 10px;
  color: #f59e0b;
  margin-top: 2px;
}
.dir-path {
  font-size: 11px;
  background: #1a1a24;
  padding: 6px 8px;
  border-radius: 4px;
  color: #888;
  display: block;
  word-break: break-all;
}
.progress-bar {
  height: 6px;
  background: #1a1a24;
  border-radius: 3px;
  overflow: hidden;
  margin-top: 6px;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00d4aa, #6c5ce7);
  border-radius: 3px;
  transition: width 0.5s ease;
}
.progress-text {
  font-size: 11px;
  color: #666;
  margin-top: 4px;
  display: block;
}
.divider { height: 1px; background: #222; margin: 16px 0; }
</style>
