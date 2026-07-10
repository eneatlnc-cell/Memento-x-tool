# Memento 项目完整架构与上下文摘要

> 本文档是 Memento 项目的锚定记忆，系统重置后用于快速恢复上下文。
> 最后更新：2026-07-10
> 战略：v3.1 — ComfyUI 魔改方案，三大核心仓库，App 暂缓

---

## 一、项目概述

Memento 是一个面向专业影视创作者的 AI 影视 VFX 系统。

**核心理念**："原始底图不动，AI 生成覆盖层，基于 ComfyUI 底层批量渲染成片。"

用户只需在网站：选素材 → 选目标 → 点执行，系统自动完成从素材引用到成片输出的全过程。

**界面哲学**：像用 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线。一个页面，三步操作，零配置。

**技术路线**：魔改 ComfyUI 前端骨架做网站（省去从零搭建 WebSocket/文件传输/队列管理），ComfyUI headless 做算力底座（省去从零适配 CUDA/显存/多卡）。

---

## 二、仓库清单（3 个活跃仓库）

| 仓库 | 定位 | 技术栈 | 部署方 | 状态 |
|------|------|--------|--------|------|
| [Memento-X](https://github.com/eneatlnc-cell/Memento-X) | ★ 云端 JSON 调度中枢 | Python(FastAPI) | 我们（ECS） | ✅ 已部署 |
| [Memento-x-web](https://github.com/eneatlnc-cell/Memento-x-web) | 网站操作界面（魔改 ComfyUI 前端） | HTML/JS（ComfyUI 前端骨架） | 我们（域名） | ⚠️ 待魔改 |
| [Memento-x-tool](https://github.com/eneatlnc-cell/Memento-x-tool) | 启动器（ComfyUI headless + custom_nodes） | Python + ComfyUI | 用户自己 | ⚠️ 待构建 |
| [Memento-Sol](https://github.com/eneatlnc-cell/Memento-Sol) | ~~手机端采集~~ | Kotlin + Jetpack Compose | - | 🔒 暂缓 |

---

## 三、三层架构与数据流

```
┌───────────────────────────────────────────────────────────────┐
│                    用户浏览器                                   │
│                  https://memento.xxx                            │
└──────────────────────┬────────────────────────────────────────┘
                       │ 访问网站
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Memento-x-web（网站 — 魔改 ComfyUI 前端骨架）                  │
│  • 保留：WebSocket 通信、文件上传/下载、队列管理、进度推送        │
│  • 移除：节点画布、连线编辑器、模型选择器                        │
│  • 替换为：极简三步界面（选素材 → 选目标 → 点执行）              │
│  • 部署：静态网站，挂载到域名                                   │
│  • 部署方：我们                                                │
└──────────────────────┬────────────────────────────────────────┘
                       │ 发送 JSON 指令 + Token
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Memento-X（云端 JSON 调度中枢）                                │
│  • 账户系统：JWT 认证、注册/登录、配额管理                       │
│  • 任务调度：意图理解 → 工作流 JSON 生成 → 派发到用户启动器      │
│  • 状态推送：WebSocket 透传到 Web 端                            │
│  • 硬性规则：不接收、不存储、不转发任何视频/图片/模型文件         │
│  • 只传输：文本、ID、JSON 指令、状态信息                         │
│  • 部署方：我们（ECS）                                         │
└──────────────────────┬────────────────────────────────────────┘
                       │ 派发工作流 JSON
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  Memento-x-tool（启动器 — ComfyUI headless + custom_nodes）     │
│  • 用户自己部署：本地 PC 或自己买的云端 GPU                      │
│  • 底层：ComfyUI headless（CUDA 适配、多卡显存管理、模型生命周期）│
│  • 业务：custom_nodes 中实现 Memento 9 节点管线                  │
│  • 文件：唯一存储层（素材缓存、中间产物、成品输出）                │
│  • 部署方：用户自己                                            │
└───────────────────────────────────────────────────────────────┘
```

**数据流总结**：`用户浏览器 → x-web(网站) → Memento-X(调度) → x-tool(用户自己的GPU启动器)`

**我们**：网站 + 云端中枢。**用户**：启动器。

---

## 四、核心工作流

```
用户在 Web 端：选素材 → 选目标（人物/背景/物体） → 点执行
    ↓
x-web 生成标准化指令 {action, target, source_asset, target_asset}
    ↓
Memento-X 意图理解引擎 → 转换为工作流 JSON
    ↓
Memento-X 派发到 x-tool 启动器
    ↓
x-tool（ComfyUI headless）运行 9 节点管线 → 渲染成片
    ↓
Web 端 WebSocket 实时显示进度 → 成片预览 → 下载
```

---

## 五、渲染管线（9 节点串行，不可调换顺序）

基于 Memento影视VFX系统 文档，升级为 ComfyUI custom_nodes 管线：

| 节点 | 工具 | 功能 |
|------|------|------|
| 1. 预处理 | FFmpeg | H.264/HEVC → 帧序列，分辨率适配 |
| 2. 时序分割 | SAM3-Large | 时序一致性分割，像素级 Mask |
| 3. 2D 骨骼 | MediaPipe | 33 关键点提取 |
| 4. 3D 归一化 | MotionBERT | 3D 人体姿态归一化，防抖动 |
| 5. 四通道编码 | QuadMask（自研） | 四通道特征编码，多人物深度分层 |
| 6. 影视级重绘 | LTX-Video 2.3 + IC-LoRA | 核心 AI 生成，内置姿态/掩码/深度控制，无需外挂 ControlNet |
| 7. 稠密光流 | RAFT | 时序矫正，亚像素级边缘稳定 |
| 8. 光影调色 | 分层调色融合 | 环境光/阴影/高光匹配 |
| 9. 4K 输出 | FFmpeg | 帧序列 → MP4/MOV，支持 EXR 分层 |

---

## 六、ComfyUI 改造方案

### 6.1 为什么选 ComfyUI 作为底层底座

1. 自带工业级显存管理（FP8/FP16 量化、防 OOM、防泄漏）
2. 自带多显卡自动兼容（4090/5090/A40 全部适配）
3. 自带节点生命周期、模型按需加载/卸载
4. 自带 WebSocket 进度推送、API 标准化
5. 可无头部署、极简资源占用
6. 可无限扩展自定义节点（完美承载全部拓展功能）
7. 不用自研底层（节省 2 个月底层开发排坑时间）

### 6.2 改造范围（最小侵入、最大扩展）

| 组件 | 做法 |
|------|------|
| ComfyUI 原生前端 | 关闭（headless），前端骨架魔改给 x-web |
| 原生节点 | 屏蔽无关节点，只保留基础设施 |
| 业务节点 | 全部放入 `custom_nodes/` 自定义节点 |
| 底层机制 | 全部保留（CUDA、显存、多卡、模型管理） |
| 拓展功能 | 新增多人物、道具、超分、补帧、EXR 节点 |

---

## 七、API 端点清单（Memento-X）

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
| POST | `/api/v1/workflow/dispatch` | 派发到启动器 |
| GET | `/api/v1/workflow/status/{id}` | 查询任务状态 |
| GET | `/api/v1/workflow/result/{id}` | 获取成片结果 |
| GET | `/api/v1/workflow/tasks` | 列出用户任务 |
| POST | `/api/v1/workflow/launcher/register` | 启动器注册（上线） |
| POST | `/api/v1/workflow/launcher/unregister` | 启动器注销（下线） |
| POST | `/api/v1/workflow/launcher/heartbeat` | 启动器心跳 |
| GET | `/api/v1/workflow/launcher/status/{user_id}` | 查询启动器状态 |
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
| POST | `/api/v1/status/report` | 启动器状态上报 |
| GET | `/api/v1/status/connections` | WebSocket 连接数 |
| GET | `/api/v1/status/launchers` | 启动器注册列表 |

---

## 八、设计原则（不可违背）

1. 原始素材永远只读，永不修改
2. 所有修改通过 AI 覆盖层实现，覆盖层可逆、可编辑
3. AI 只做"翻译"，不做"理解"（素材由用户选择）
4. 云端不存储任何视频/图片/模型文件（零存储成本）
5. **硬性规则**：不接收、不存储、不转发任何媒体文件
6. 用户操作：选素材 → 选目标 → 点执行（三步完成）
7. Memento-x-web 是唯一操作界面（网站，非安装包）
8. 界面极简：像 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线
9. 启动器完全由用户自己部署和运行，我们不提供云端 GPU 算力
10. **ComfyUI 仅作为底层壳**：不暴露原生前端，不暴露节点系统给用户
11. **工具链 = 不可变镜像**：ComfyUI + custom_nodes 整体打包为 Docker 镜像，版本锁定，禁止自动更新

---

## 九、当前开发状态

| 组件 | 部署方 | 状态 | 说明 |
|------|--------|------|------|
| Memento-X cloud | 我们（ECS） | ✅ 已部署 | ECS 118.31.189.101:8000，M8-M10 完成 |
| Memento-x-web | 我们（域名） | ⚠️ 待魔改 | 基于 ComfyUI 前端骨架改版 |
| Memento-x-tool | 用户自己 | ⚠️ 待构建 | ComfyUI headless + custom_nodes，用户自己部署在本地/云端GPU |
| Memento-Sol | - | 🔒 暂缓 | 非 MVP 瓶颈 |

---

## 十、开发路线图（v3.1 战略）

### 阶段一：基础已就绪 ✅
- **M8 数据库**：PostgreSQL + Redis 已部署
- **M9 对象存储**：阿里云 OSS（boto3 S3 兼容模式）
- **M10 任务调度器**：asyncio.PriorityQueue + 心跳监控 + 自动重试
- **ECS 部署**：118.31.189.101:8000，systemd 开机自启

### 阶段二：核心闭环（M11-M12.5）— 约 4-6 周
- **M11 算力引擎**：ComfyUI headless 部署 + SAM3/MediaPipe/MotionBERT/RAFT 模型集成
- **M12 渲染管线**：9 节点 custom_nodes 开发 + 管线联调
- **M12.5 窗口化流处理**：重叠切割 + 上下文缓存 + 速度继承 + 光流扭曲 + 缓存滚动

### 阶段三：界面与分发（M13-M15）— 约 3-4 周
- **M13 网站魔改**：ComfyUI 前端骨架 → 极简三步网站（移除画布，替换为三步操作）
- **M14 多任务并行**：启动器本地队列 + GPU 显存调度
- **M15 工具链镜像**：Docker 构建 + Docker Hub 发布 (mementoweb)

### 阶段四：生产就绪（M16-M17）— 约 2-3 周
- **M16 部署运维**：网站部署 + CI/CD + 监控
- **M17 通知系统**：FCM 接入 + 邮件通知

---

## 十一、里程碑目标

| 里程碑 | 目标 | 验证方式 |
|--------|------|----------|
| 阶段一 ✅ | 云端就绪 | API 正常注册/登录/上传/查询 |
| 阶段二 | 核心闭环 | 1 秒视频"点击替换"成片输出 |
| 阶段三 | 产品可用 | 用户通过网站完成完整编辑 |
| 阶段四 | 产品可发布 | 端到端全链路可用 |

---

## 十二、关键技术决策

1. **ComfyUI 前端魔改**：不从头开发 Web 界面，复用 ComfyUI 前端骨架（WebSocket、文件传输、队列、进度），移除画布，替换为三步极简界面
2. **ComfyUI 作为算力底座**：不自研底层 CUDA/显存/多卡适配，ComfyUI headless + custom_nodes 承载全部管线
3. **Memento-x-web 是网站**：不是 Electron 安装包，是纯 Web 网站，用户打开浏览器即可使用
4. **Memento-x-tool 是启动器**：用户自己部署在本地 PC 或自己买的云端 GPU，我们不提供算力，不建算力池
5. **术语统一**：`matting`/`抠图` → `scene_edit`/`场景编辑`
6. **注册策略**：统一在 Web 端注册
7. **测试账号**：13800000001~3 / 验证码 888888
8. **工具链分发**：Docker 镜像 + 版本锁定，禁止自动更新
9. **AIGC 定位**：差异化在于**可控性**（坐标驱动确定性合成），而非操作难度
10. **战略调整**：App 暂缓，主攻三大核心仓库
11. **界面哲学**：像 Instagram 滤镜一样简单，不像 ComfyUI 那样暴露管线
12. **管线升级**：SAM2→SAM3，RIFE→RAFT，新增 MotionBERT 3D 归一化 + QuadMask 四通道编码 + LTX-Video 2.3 + IC-LoRA 多特征融合渲染（内置姿态/掩码/深度控制，无需外挂 ControlNet）

---

## 十三、ECS 部署信息

| 项目 | 值 |
|------|-----|
| 公网 IP | 118.31.189.101 |
| API 地址 | http://118.31.189.101:8000 |
| 系统 | Ubuntu 22.04，2核/2GB |
| Docker | postgres:16-alpine + redis:7-alpine |
| 代码路径 | `/opt/Memento-X/` |
| 服务名 | memento-api (systemd，开机自启) |

---

## 十四、目录结构约定

```
~/.memento/                    # 启动器工作目录（用户 GPU 机器上）
├── assets/                    # 素材缓存（按 asset_id 组织）
├── workspace/                 # 工作目录
├── context_buffer/            # 窗口化流处理上下文缓存
├── outputs/                   # 成片输出
├── custom_nodes/              # Memento 管线节点
└── logs/                      # 日志
```

---

## 十五、恢复指引

系统重置后，执行以下步骤恢复开发环境：

```bash
# 1. 克隆三个仓库
git clone https://github.com/eneatlnc-cell/Memento-X.git
git clone https://github.com/eneatlnc-cell/Memento-x-web.git
git clone https://github.com/eneatlnc-cell/Memento-x-tool.git

# 2. 阅读本文档
cat Memento-X/CONTEXT.md
```

**本文档同时存在于三个仓库的根目录，内容完全一致。**