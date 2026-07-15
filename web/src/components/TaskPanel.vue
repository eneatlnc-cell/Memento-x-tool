<template>
  <div class="task-panel">
    <h3 class="panel-title">工作流任务</h3>

    <!-- 快速意图输入 -->
    <div class="section">
      <label class="label">意图输入</label>
      <textarea v-model="intent" class="textarea" rows="3" placeholder="描述你想做什么视频效果...&#10;例如: 把视频中的人物替换为卡通风格，保留背景不变"></textarea>
    </div>

    <div class="section">
      <button class="btn primary" @click="submitIntent" :disabled="submitting">
        {{ submitting ? '生成中...' : '生成工作流' }}
      </button>
    </div>

    <div v-if="workflowPreview" class="workflow-preview">
      <div class="preview-header">
        <span class="preview-label">工作流预览</span>
        <span class="preview-steps">{{ workflowPreview.total_steps || 0 }} 个节点</span>
      </div>
      <div class="steps-list">
        <div v-for="(step, i) in (workflowPreview.workflow?.steps || [])" :key="i" class="step-item">
          <span class="step-num">{{ i + 1 }}</span>
          <span class="step-name">{{ step.name || step.node || '节点 ' + (i+1) }}</span>
        </div>
      </div>
      <button class="btn primary" @click="dispatchWorkflow" :disabled="dispatching">
        {{ dispatching ? '下发中...' : '下发到本地执行' }}
      </button>
    </div>

    <div v-if="dispatchMsg" class="status-msg" :class="dispatchType">
      {{ dispatchMsg }}
    </div>

    <div class="divider"></div>

    <!-- 任务历史 -->
    <div class="section">
      <div class="section-header">
        <label class="label">任务历史</label>
        <button class="btn small" @click="loadTasks">刷新</button>
      </div>
      <div v-if="tasks.length === 0" class="empty">暂无任务</div>
      <div v-for="task in tasks" :key="task.task_id" class="task-card">
        <div class="task-header">
          <span class="task-id">#{{ (task.task_id || '').slice(-8) }}</span>
          <span class="task-status" :class="'status-' + (task.status || 'unknown')">{{ task.status || 'unknown' }}</span>
        </div>
        <div class="task-intent">{{ task.user_input || task.intent || '(无输入)' }}</div>
        <div v-if="task.result_url" class="task-result">
          <a :href="task.result_url" target="_blank" class="result-link">查看结果</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({ apiUrl: String })

const intent = ref('')
const submitting = ref(false)
const dispatching = ref(false)
const workflowPreview = ref(null)
const dispatchMsg = ref('')
const dispatchType = ref('info')
const tasks = ref([])

const token = () => localStorage.getItem('memento_token') || ''

async function submitIntent() {
  if (!intent.value.trim()) return
  submitting.value = true
  workflowPreview.value = null
  dispatchMsg.value = ''
  try {
    const resp = await fetch(props.apiUrl + '/workflow/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token()
      },
      body: JSON.stringify({ user_input: intent.value.trim() })
    })
    const data = await resp.json()
    if (data.workflow) {
      workflowPreview.value = data
    } else {
      dispatchMsg.value = '生成失败: ' + (data.detail || '未知错误')
      dispatchType.value = 'error'
    }
  } catch (e) {
    dispatchMsg.value = '请求失败: ' + e.message
    dispatchType.value = 'error'
  }
  submitting.value = false
}

async function dispatchWorkflow() {
  if (!workflowPreview.value) return
  dispatching.value = true
  dispatchMsg.value = ''
  try {
    const resp = await fetch(props.apiUrl + '/workflow/dispatch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + token()
      },
      body: JSON.stringify({
        user_input: intent.value.trim(),
        local_url: 'http://127.0.0.1:8188',
        priority: 'normal'
      })
    })
    const data = await resp.json()
    if (data.task_id) {
      dispatchMsg.value = '✅ 任务已下发: ' + data.task_id.slice(-8)
      dispatchType.value = 'success'
      loadTasks()
    } else {
      dispatchMsg.value = '下发失败: ' + (data.detail || data.message || '未知错误')
      dispatchType.value = 'error'
    }
  } catch (e) {
    dispatchMsg.value = '下发失败: ' + e.message
    dispatchType.value = 'error'
  }
  dispatching.value = false
}

async function loadTasks() {
  try {
    const resp = await fetch(props.apiUrl + '/workflow/tasks?limit=20', {
      headers: { 'Authorization': 'Bearer ' + token() }
    })
    const data = await resp.json()
    tasks.value = Array.isArray(data) ? data : (data.tasks || [])
  } catch {}
}
</script>

<style scoped>
.task-panel { padding: 16px; }
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
.textarea {
  width: 100%;
  padding: 8px 10px;
  background: #1a1a24;
  border: 1px solid #333;
  border-radius: 6px;
  color: #ddd;
  font-size: 13px;
  outline: none;
  resize: vertical;
  font-family: inherit;
  transition: border-color 0.15s;
}
.textarea:focus { border-color: #00d4aa; }
.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.15s;
}
.btn.primary { background: #00d4aa; color: #0a0a0f; }
.btn.primary:hover { background: #00e6b8; }
.btn.primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn.small { padding: 4px 10px; font-size: 11px; background: #222; color: #888; }
.btn.small:hover { background: #333; color: #ccc; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.section-header .label { margin-bottom: 0; }
.workflow-preview {
  background: #14141a;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 12px;
  margin-top: 12px;
}
.preview-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.preview-label { font-size: 12px; font-weight: 600; color: #00d4aa; }
.preview-steps { font-size: 11px; color: #666; }
.steps-list { margin-bottom: 12px; }
.step-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 12px;
}
.step-num {
  width: 20px;
  height: 20px;
  background: #00d4aa22;
  color: #00d4aa;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  flex-shrink: 0;
}
.step-name { color: #aaa; }
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
.empty { font-size: 12px; color: #555; padding: 16px 0; text-align: center; }
.task-card {
  background: #1a1a24;
  border: 1px solid #222;
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}
.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.task-id { font-size: 11px; color: #555; font-family: monospace; }
.task-status {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
  text-transform: uppercase;
}
.status-queued { background: #222; color: #888; }
.status-running { background: #00d4aa22; color: #00d4aa; }
.status-completed { background: #00d4aa15; color: #00d4aa; }
.status-failed { background: #ef444415; color: #ef4444; }
.status-unknown { background: #222; color: #666; }
.task-intent { font-size: 12px; color: #999; }
.task-result { margin-top: 6px; }
.result-link { font-size: 11px; color: #00d4aa; text-decoration: none; }
.result-link:hover { text-decoration: underline; }
</style>
