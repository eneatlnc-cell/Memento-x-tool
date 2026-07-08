# Memento 项目完整架构与上下文摘要

> 本文档是 Memento 项目的锚定记忆，系统重置后用于快速恢复上下文。
> 最后更新：2026-07-08

---

## 一、项目概述

Memento 是一个面向专业影视创作者的端到端视频编辑系统。

**核心理念**："原始底图不动，AI 生成 SVG 覆盖层，本地批量渲染成片。"

用户只需在 Web 端：选素材 → 选目标 → 点执行，系统自动完成从素材引用到成片输出的全过程。

---

## 二、仓库清单（4 个 GitHub 仓库）

| 仓库 | 定位 | 技术栈 | 状态 |
|------|------|--------|------|
| [Memento-X](https://github.com/eneatlnc-cell/Memento-X) | 云端 + 本地核心 | Python(FastAPI) + Electron | ✅ 核心就绪 |
| [Memento-Sol](https://github.com/eneatlnc-cell/Memento-Sol) | 手机端采集 | Kotlin + Jetpack Compose | ✅ 界面就绪 |
| [Memento-x-web](https://github.com/eneatlnc-cell/Memento-x-web) | Web 操作界面 | Next.js + TypeScript | ✅ 界面就绪 |
| [Memento-x-tool](https://github.com/eneatlnc-cell/Memento-x-tool) | 启动器 + 工具链 | Python | ❌ 待开发 |

---

## 三、四端职责划分

### 手机端 (Memento-Sol)
- 拍照/录视频/从相册选择
- 素材存在手机本地（不上传文件到云端）
- **仅上传元数据**（文件名、类型、时长）到云端
- 云端返回 `asset_id`，手机端保存
- 接收成片通知（FCM 推送）
- 预览/下载成片
- **不做任何 AI 推理，不做任何视频编辑**

### 云端 (Memento-X cloud)
- 账户系统：JWT 认证、注册/登录、配额管理
- 素材索引：存储 `asset_id` 与元数据的映射（不存储文件）
- 意图理解：将用户操作转换为结构化工作流 JSON
- 任务派发：将工作流 JSON 派发到用户本地 PC
- 状态推送：WebSocket 推送到 Web 端，FCM 推送到手机端
- **不存储任何视频文件**

### Web 端 (Memento-x-web)
- 任务中心：素材选择 → 目标选择 → 点击执行（三步完成）
- 素材库：展示手机端上传的素材列表
- 执行进度：WebSocket 实时显示任务状态
- 成片预览：视频预览 + 下载入口
- **用户不需要输入任何文字**

### 本地 PC 端 (Memento-X local + Memento-x-tool)
- 启动器：一键安装所有工具
- 接收任务：接收云端派发的工作流 JSON
- 素材解析：根据 `asset_id` 查找本地素材文件
- 工具执行：DAG 调度 → 调用工具链 → 批量渲染
- 成片输出：MP4 成片存在本地
- **所有工具和模型在本地运行，数据不出本地**

---

## 四、核心工作流（两步）

### ① 素材采集（手机端 → 云端）
```
用户拍照 → 素材存本地 → 上传元数据 → 云端生成 asset_id → 素材出现在 Web 端
```

### ② 素材替换（Web 端 → 云端 → 本地 PC → Web 端）
```
用户在 Web 端：选素材 → 选目标（人物/背景/物体） → 点执行
    ↓
Web 端生成标准化指令 {action, target, source_asset, target_asset}
    ↓
云端意图理解引擎 → 转换为工作流 JSON
    ↓
云端派发工作流 JSON 到本地 PC
    ↓
本地 PC 启动器：查 asset_id → 拆帧 → 生成 SVG → 批量渲染 → 合成 → 导出
    ↓
Web 端实时显示进度 → 成片预览 → 下载
```

---

## 五、API 端点清单

### 6.1 账号
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/account/register` | 注册 |
| POST | `/api/v1/account/login` | 登录 |
| POST | `/api/v1/account/refresh` | 刷新 Token |

### 6.2 素材
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/asset/metadata` | 上传素材元数据 → 返回 asset_id |
| GET | `/api/v1/asset/list` | 获取素材列表 |
| GET | `/api/v1/asset/{asset_id}` | 获取素材详情 |

### 6.3 工作流
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/workflow/generate` | 生成工作流 JSON |
| POST | `/api/v1/workflow/dispatch` | 派发到本地 PC |
| GET | `/api/v1/workflow/status/{id}` | 查询任务状态 |
| GET | `/api/v1/workflow/result/{id}` | 获取成片结果 |

### 6.4 通知
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/notification/register` | 注册 FCM Token |
| POST | `/api/v1/notification/push` | 推送通知 |

### 6.5 本地执行（Memento-x-tool）
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/local/execute` | 接收工作流 JSON → 执行 |
| GET | `/api/v1/local/status/{task_id}` | 查询任务状态 |
| GET | `/api/v1/local/result/{task_id}` | 获取任务结果 |
| DELETE | `/api/v1/local/cancel/{task_id}` | 取消任务 |

---

## 六、工具链（10 种 action）

| action | 用途 | 工具 |
|--------|------|------|
| `scene_edit` | SVG 场景编辑（提取元素结构化描述） | BiRefNet |
| `track` | 遮罩追踪 | SAM2 |
| `replace` | 主体替换 | ComfyUI |
| `composite` | 合成 | FFmpeg |
| `effect` | 特效 | ComfyUI |
| `color` | 调色 | FFmpeg |
| `subtitle` | 字幕 | HyperFrames |
| `render` | 渲染 | FFmpeg |
| `crop` | 裁剪 | FFmpeg |
| `export` | 导出 | FFmpeg |

---

## 七、设计原则（不可违背）

1. 原始素材永远只读，永不修改
2. 所有修改通过 SVG 覆盖层实现，覆盖层可逆、可编辑
3. AI 只做"翻译"，不做"理解"（素材由用户选择）
4. 云端不存储任何视频文件（零存储成本）
5. 所有工具在本地运行（零 API 成本）
6. 用户操作：选素材 → 选目标 → 点执行（三步完成）
7. Memento-x-web 是唯一操作界面
8. 手机端和启动器没有操作界面
9. 启动器完全由云端控制

---

## 八、当前开发状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Memento-Sol | ✅ 完成 | UI 就绪，架构对齐 |
| Memento-X cloud | ✅ 完成 | 账号/意图/派发/通知 API 就绪 |
| Memento-x-web | ✅ 完成 | 注册/登录/三步流程/视频预览就绪 |
| Memento-X local | ⚠️ 待完善 | 核心调度器就绪，工具链待接入 |
| Memento-x-tool | ❌ 未开始 | 启动器待开发（P0→P1→P2） |

---

## 九、Memento-x-tool 开发计划

### P0（启动器核心）
- `installer.py`：一键安装编排（detector + downloader）
- `service.py`：系统服务注册（macOS launchd / Linux systemd / Windows）

### P1（界面与注册）
- 系统托盘图标：状态指示灯（绿/黄/红）
- 云端注册：启动后向云端报备

### P2（运维增强）
- 结构化日志系统
- 远程日志拉取

---

## 十、关键技术决策

1. **术语统一**：`matting`/`抠图` → `scene_edit`/`SVG场景编辑`
2. **包名统一**：`com.myagent.app` → `com.memento.sol`
3. **仓库重命名**：`Memento-3.1.2` → `Memento-Sol`
4. **注册策略**：App 不提供注册，统一在 Web 端注册
5. **测试账号**：13800000001~3 / 验证码 888888
6. **素材采集**：手机端仅上传元数据，不上传文件
7. **残留清理**：已移除所有 MyAgent/Qwen3/OpenClaw/llama.cpp 引用

---

## 十一、目录结构约定

```
~/.memento/                    # 本地 PC 用户目录
├── tools/                     # 工具目录
│   ├── ffmpeg/
│   ├── birefnet/
│   ├── sam2/
│   ├── comfyui/
│   └── hyperframes/
├── assets/                    # 素材缓存（按 asset_id 组织）
├── workspace/                 # 工作目录
├── outputs/                   # 成片输出
└── logs/                      # 日志
```

---

## 十二、恢复指引

系统重置后，执行以下步骤恢复开发环境：

```bash
# 1. 克隆四个仓库
git clone https://github.com/eneatlnc-cell/Memento-X.git
git clone https://github.com/eneatlnc-cell/Memento-Sol.git
git clone https://github.com/eneatlnc-cell/Memento-x-web.git
git clone https://github.com/eneatlnc-cell/Memento-x-tool.git

# 2. 阅读本文档
cat Memento-X/CONTEXT.md
```

**本文档同时存在于四个仓库的根目录，内容完全一致。**