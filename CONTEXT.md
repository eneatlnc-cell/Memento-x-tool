# Memento 项目完整架构与上下文摘要

> 本文档是 Memento 项目的锚定记忆，系统重置后用于快速恢复上下文。
> 最后更新：2026-07-10
> 战略：v3.0 — 做减法，聚焦三大核心仓库，App 暂缓

---

## 一、项目概述

Memento 是一个面向专业影视创作者的端到端视频编辑系统。

**核心理念**："原始底图不动，AI 生成 SVG 覆盖层，本地批量渲染成片。"

用户只需在 Web 端：选素材 → 选目标 → 点执行，系统自动完成从素材引用到成片输出的全过程。

**界面哲学**：像用 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线。一个页面，三步操作，零配置。

---

## 二、仓库清单（4 个 GitHub 仓库）

| 仓库 | 定位 | 技术栈 | 状态 |
|------|------|--------|------|
| [Memento-X](https://github.com/eneatlnc-cell/Memento-X) | ★ 云端核心：JSON 调度 + 用户数据 | Python(FastAPI) | ✅ 核心就绪，ECS 已部署 |
| [Memento-x-web](https://github.com/eneatlnc-cell/Memento-x-web) | 网站操作界面（极简三步） | Next.js + TypeScript | ⚠️ 待改版 |
| [Memento-x-tool](https://github.com/eneatlnc-cell/Memento-x-tool) | 启动器 + 工具链镜像管理 | Python | ⚠️ Step 1-3 完成 |
| [Memento-Sol](https://github.com/eneatlnc-cell/Memento-Sol) | ~~手机端采集~~ | Kotlin + Jetpack Compose | 🔒 暂缓 |

---

## 三、三端职责划分

### 云端 (Memento-X) — ★ 唯一重心
- 账户系统：JWT 认证、注册/登录、配额管理
- 素材索引：存储 `asset_id` 与元数据的映射（不存储文件）
- 意图理解：将用户操作转换为结构化工作流 JSON
- 任务派发：将工作流 JSON 派发到用户本地 PC
- 状态推送：WebSocket 推送到 Web 端，FCM 推送到手机端
- **不存储任何图片/视频文件**
- **不存储任何媒体文件**
- **零媒体存储成本**

### Web 端 (Memento-x-web) — 极简界面
- 一个页面，三步操作：选素材 → 选目标 → 点执行
- 不暴露任何技术概念（没有"节点"、"管线"、"工作流"等词汇）
- 不显示任何 AI 模型名称
- 没有"设置"页面（默认配置即最优）
- 进度条 + 预估时间，不要技术日志

### 本地 PC 端 (Memento-x-tool) — 启动器
- 拉取工具链 Docker 镜像
- 启动容器 + 健康检查
- 接收任务 → 调度工具链 → 批量渲染
- 所有工具和模型在本地运行，数据不出本地

---

## 四、核心工作流（两步）

### ① 素材采集（Web 端直传）
```
用户上传素材 → 云端生成 asset_id → 元数据入库 → 素材出现在 Web 端素材库
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

## 五、渲染管线（8 阶段）

| 阶段 | 工具 | 输入 | 输出 |
|------|------|------|------|
| 1. 拆帧 | FFmpeg | H.264/HEVC 视频 | 30fps PNG 序列 |
| 2. 分割 | SAM2 点提示 | 帧 + 点击坐标 | 像素级 Mask |
| 3. 骨骼追踪 | MediaPipe Pose | 帧 + Mask | 33 关键点 3D 坐标 |
| 4. 插值 | Python 贝塞尔 | 离散坐标 | 30fps 连续轨迹 |
| 5. 矢量覆盖 | SVG 生成器 | Mask 轮廓 + 轨迹 | SVG 覆盖层 |
| 6. 融合 | ComfyUI 工作流 | 原始帧 + SVG + 坐标 | 合成帧 |
| 7. 光流传播 | HyperFrames (RIFE) | 关键帧 | 全部 30 帧 |
| 8. 合成 | FFmpeg | 30 帧序列 | MP4 成片 |

---

## 六、窗口化流处理（M12.5）

长视频处理机制，解决 GPU 显存限制：

1. **重叠切割**：视频按片段切分，重叠 7 帧
2. **上下文缓存**：骨骼坐标 + Mask 轮廓 + 光流矢量 + 色温 → JSON 写入 `~/.memento/context_buffer/`
3. **速度继承**：上片段重叠区速度作为下片段初始条件，权重衰减 0.75→0.5→0.25
4. **光流扭曲**：像素扭曲 + 边缘羽化混合（alpha 0.5）
5. **缓存滚动**：保留最后 10 帧，丢弃上片段缓存，恒定内存占用

---

## 七、API 端点清单

### 7.1 账号
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/account/register` | 注册 |
| POST | `/api/v1/account/login` | 登录 |
| POST | `/api/v1/account/refresh` | 刷新 Token |

### 7.2 素材
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/asset/metadata` | 上传素材元数据 → 返回 asset_id |
| POST | `/api/v1/asset/upload-url` | 获取预签名上传 URL |
| GET | `/api/v1/asset/list` | 获取素材列表 |
| GET | `/api/v1/asset/{asset_id}` | 获取素材详情 |

### 7.3 工作流
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/workflow/generate` | 生成工作流 JSON |
| POST | `/api/v1/workflow/dispatch` | 派发到本地 PC |
| GET | `/api/v1/workflow/status/{id}` | 查询任务状态 |
| GET | `/api/v1/workflow/result/{id}` | 获取成片结果 |
| GET | `/api/v1/workflow/tasks` | 列出用户任务 |
| POST | `/api/v1/workflow/local/register` | 本地引擎注册 |
| POST | `/api/v1/workflow/local/unregister` | 本地引擎注销 |
| POST | `/api/v1/workflow/local/heartbeat` | 本地引擎心跳 |
| GET | `/api/v1/workflow/local/status/{user_id}` | 查询本地引擎状态 |
| GET | `/api/v1/workflow/queue/status` | 查询调度器队列状态 |

### 7.4 通知
| 方法 | 端点 | 用途 |
|------|------|------|
| POST | `/api/v1/notification/register` | 注册 FCM Token |
| POST | `/api/v1/notification/push` | 推送通知 |

### 7.5 状态
| 方法 | 端点 | 用途 |
|------|------|------|
| WS | `/api/v1/status/ws` | WebSocket 实时状态推送 |
| POST | `/api/v1/status/report` | 本地引擎状态上报 |
| GET | `/api/v1/status/connections` | WebSocket 连接数 |
| GET | `/api/v1/status/local` | 本地引擎注册列表 |

---

## 八、工具链

| action | 用途 | 工具 |
|--------|------|------|
| `segment` | 点提示分割 | SAM2 |
| `track` | 骨骼追踪 | MediaPipe |
| `interpolate` | 轨迹插值 | Python 贝塞尔 |
| `overlay` | SVG 矢量覆盖层 | SVG 生成器 |
| `fuse` | 融合渲染 | ComfyUI |
| `propagate` | 光流传播 | HyperFrames (RIFE) |
| `composite` | 合成 | FFmpeg |
| `effect` | 特效 | ComfyUI |
| `color` | 调色 | FFmpeg |
| `render` | 渲染 | FFmpeg |
| `crop` | 裁剪 | FFmpeg |
| `export` | 导出 | FFmpeg |

---

## 九、设计原则（不可违背）

1. 原始素材永远只读，永不修改
2. 所有修改通过 SVG 覆盖层实现，覆盖层可逆、可编辑
3. AI 只做"翻译"，不做"理解"（素材由用户选择）
4. 云端不存储任何视频文件（零存储成本）
5. 所有工具在本地运行（零 API 成本）
6. 用户操作：选素材 → 选目标 → 点执行（三步完成）
7. Memento-x-web 是唯一操作界面
8. 界面极简：像 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线
9. 启动器完全由云端控制，没有操作界面
10. **工具链 = 不可变镜像**：SAM2/MediaPipe/ComfyUI/FFmpeg/RIFE 整体打包为 Docker 镜像，版本锁定，禁止自动更新

---

## 十、当前开发状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Memento-X cloud | ✅ 已部署 | ECS 118.31.189.101:8000，M8-M10 完成 |
| Memento-x-web | ⚠️ 待改版 | 框架就绪，需按极简哲学重新设计 |
| Memento-x-tool | ⚠️ Step 1-3 | 待 M11-M12 完成后对接 |
| Memento-Sol | 🔒 暂缓 | 非 MVP 瓶颈 |

---

## 十一、开发路线图（v3.0 战略）

### 阶段一：基础已就绪 ✅
- **M8 数据库**：PostgreSQL + Redis 已部署
- **M9 对象存储**：阿里云 OSS（boto3 S3 兼容模式）
- **M10 任务调度器**：asyncio.PriorityQueue + 心跳监控 + 自动重试
- **ECS 部署**：118.31.189.101:8000，systemd 开机自启

### 阶段二：核心闭环（M11-M12.5）— 约 4-6 周
- **M11 AI 推理引擎**：SAM2 点提示分割 + MediaPipe 骨骼追踪 + Python 贝塞尔插值器
- **M12 渲染管线**：SVG 覆盖层 + ComfyUI 工作流 + FFmpeg 拆帧合成 + HyperFrames 光流传播
- **M12.5 窗口化流处理**：重叠切割 + 上下文缓存 + 速度继承 + 光流扭曲 + 缓存滚动

### 阶段三：界面与分发（M13-M15）— 约 3-4 周
- **M13 极简界面**：Memento-x-web 改版，一个页面三步操作
- **M14 多任务并行**：本地队列 + GPU 显存调度
- **M15 工具链镜像**：Docker 构建 + Docker Hub 发布 + NSIS 启动器

### 阶段四：生产就绪（M16-M17）— 约 2-3 周
- **M16 部署运维**：Docker 化 + CI/CD + 监控
- **M17 通知系统**：FCM 接入 + 邮件通知

---

## 十二、里程碑目标

| 里程碑 | 目标 | 验证方式 |
|--------|------|----------|
| 阶段一 ✅ | 云端就绪 | API 正常注册/登录/上传/查询 |
| 阶段二 | 核心闭环 | 1 秒视频"点击替换"成片输出 |
| 阶段三 | 产品可用 | 用户通过 Web 界面完成完整编辑 |
| 阶段四 | 产品可发布 | 端到端全链路可用 |

---

## 十三、关键技术决策

1. **术语统一**：`matting`/`抠图` → `scene_edit`/`SVG场景编辑`
2. **包名统一**：`com.myagent.app` → `com.memento.sol`
3. **仓库重命名**：`Memento-3.1.2` → `Memento-Sol`
4. **注册策略**：App 不提供注册，统一在 Web 端注册
5. **测试账号**：13800000001~3 / 验证码 888888
6. **素材采集**：手机端仅上传元数据，不上传文件
7. **残留清理**：已移除所有 MyAgent/Qwen3/OpenClaw/llama.cpp 引用
8. **工具链分发**：Docker 镜像 + 版本锁定，禁止自动更新
9. **AIGC 定位**：AIGC 工具（Kling/Runway/Pika）是傻瓜式操作，Memento 的差异化在于**可控性**（坐标驱动确定性合成），而非操作难度
10. **战略调整**：App 暂缓，主攻三大核心仓库（Memento-X + x-web + x-tool）
11. **界面哲学**：像 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线

---

## 十四、ECS 部署信息

| 项目 | 值 |
|------|-----|
| 公网 IP | 118.31.189.101 |
| API 地址 | http://118.31.189.101:8000 |
| 系统 | Ubuntu 22.04，2核/2GB |
| Docker | postgres:16-alpine + redis:7-alpine |
| 代码路径 | `/opt/Memento-X/` |
| 服务名 | memento-api (systemd，开机自启) |

---

## 十五、目录结构约定

```
~/.memento/                    # 本地 PC 用户目录
├── tools/                     # 工具链 Docker 镜像内管理
├── assets/                    # 素材缓存（按 asset_id 组织）
├── workspace/                 # 工作目录
├── context_buffer/            # 窗口化流处理上下文缓存
├── outputs/                   # 成片输出
└── logs/                      # 日志
```

---

## 十六、恢复指引

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