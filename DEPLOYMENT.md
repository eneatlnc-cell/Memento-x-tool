# Memento 完整部署教程

> 从零搭建三仓（云端 + 启动器 + 网站）的端到端部署指南。
> 适合首次部署，按顺序执行即可跑通。

---

## 〇、架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Memento 三仓架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐      ┌─────────────────┐      ┌────────────┐  │
│   │  x-web      │      │   Memento-X     │      │  x-tool    │  │
│   │  (网站)     │◄────►│   (云端)        │◄────►│  (启动器)   │  │
│   │  Vercel     │      │   ECS 服务器    │      │  用户 GPU 机│  │
│   │             │      │                 │      │             │  │
│   │  操作界面   │      │  意图理解       │      │  9节点管线  │  │
│   │  三步流程   │      │  账号/配额      │      │  ComfyUI    │  │
│   │  实时进度   │      │  任务派发       │      │  Docker容器 │  │
│   └─────────────┘      └─────────────────┘      └────────────┘  │
│         │                      │                      │         │
│         └──────────────────────┴──────────────────────┘         │
│                      HTTPS + WebSocket                          │
└─────────────────────────────────────────────────────────────────┘
```

**部署顺序**：云端 → 启动器 → 网站（后部署的依赖先部署的地址）

---

## 一、环境要求

### 1.1 云端服务器（Memento-X）

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 20 GB | 40 GB SSD |
| 带宽 | 3 Mbps | 5 Mbps |
| 系统 | Ubuntu 22.04 | Ubuntu 22.04 |
| 软件 | Python 3.11+、PostgreSQL 15+ | 同左 |
| GPU | **不需要** | 不需要 |

### 1.2 启动器机器（Memento-x-tool）

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 8GB 显存 | NVIDIA 24GB+ 显存 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 80 GB 空闲 | 150 GB SSD |
| 系统 | Windows 10/11 或 Ubuntu 22.04 | Windows 11 |
| 软件 | Docker Desktop + NVIDIA 驱动 | 同左 |
| 网络 | 能访问 hf-mirror.com | 同左 |

### 1.3 网站部署（Memento-x-web）

| 项目 | 要求 |
|------|------|
| 平台 | Vercel（推荐）或任意 Node.js 18+ 环境 |
| 费用 | Vercel 免费档够用 |
| 域名 | 可选（Vercel 默认提供 .vercel.app 域名） |

---

## 二、Phase 1 — 部署云端（Memento-X）

### 2.1 准备服务器

```bash
# SSH 登录你的 ECS 服务器
ssh root@你的服务器IP

# 更新系统
apt update && apt upgrade -y

# 安装基础工具
apt install -y git curl wget build-essential
```

### 2.2 安装 Python 3.12

```bash
# Ubuntu 22.04 默认是 3.10，需要装 3.12
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y python3.12 python3.12-venv python3.12-dev

# 验证
python3.12 --version  # 应显示 Python 3.12.x
```

> ⚠️ **不要用 Python 3.14**，pydantic-core 在 3.14 上编译会失败。用 3.12 最稳。

### 2.3 安装 PostgreSQL

```bash
# 安装
apt install -y postgresql postgresql-contrib

# 启动
systemctl enable postgresql
systemctl start postgresql

# 创建数据库和用户
sudo -u postgres psql <<EOF
CREATE USER memento WITH PASSWORD '你的强密码';
CREATE DATABASE memento OWNER memento;
GRANT ALL PRIVILEGES ON DATABASE memento TO memento;
EOF

# 验证
sudo -u postgres psql -c "\l" | grep memento
```

### 2.4 拉取代码 + 安装依赖

```bash
cd /opt
git clone https://github.com/eneatlnc-cell/Memento-X.git
cd Memento-X

# 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r cloud/requirements.txt
```

### 2.5 配置环境变量

```bash
cp cloud/.env.example cloud/.env
nano cloud/.env
```

`.env` 内容（按你的实际值填写）：

```ini
# 通义千问（必填）— 去 https://dashscope.console.aliyun.com 申请
DASHSCOPE_API_KEY=sk-your-dashscope-key

# 数据库（用刚才创建的）
DATABASE_URL=postgresql+asyncpg://memento:你的强密码@localhost:5432/memento

# JWT 密钥（必改！用随机字符串）
JWT_SECRET_KEY=用-python3 -c "import secrets; print(secrets.token_urlsafe(32))" 生成

# 服务
HOST=0.0.0.0
PORT=8000

# 配额
FREE_DAILY_QUOTA=10
PRO_DAILY_QUOTA=200
```

> ⚠️ `JWT_SECRET_KEY` 必须改，默认值 `change-me` 在生产环境是安全漏洞。

### 2.6 初始化数据库

```bash
cd /opt/Memento-X
source .venv/bin/activate

# 建表（生产环境后续应改用 alembic 迁移）
python -c "
import asyncio
from cloud.db.engine import init_db
asyncio.run(init_db())
print('数据库表创建完成')
"
```

### 2.7 启动云端服务

```bash
# 方式一：直接运行（测试用）
cd /opt/Memento-X
source .venv/bin/activate
python -m cloud.main

# 方式二：systemd 服务（生产用）
cat > /etc/systemd/system/memento-cloud.service <<EOF
[Unit]
Description=Memento Cloud API
After=network.target postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/Memento-X
ExecStart=/opt/Memento-X/.venv/bin/python -m cloud.main
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable memento-cloud
systemctl start memento-cloud

# 查看状态
systemctl status memento-cloud
journalctl -u memento-cloud -f  # 实时日志
```

### 2.8 开放防火墙端口

```bash
# 阿里云安全组放行 8000 端口（在控制台操作）
# 或本机防火墙：
ufw allow 8000/tcp
```

### 2.9 验证云端

```bash
# 在本地电脑测试
curl http://你的服务器IP:8000/docs
# 应返回 Swagger UI HTML 页面

curl http://你的服务器IP:8000/api/v1/account/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@memento.ai","password":"test123456"}'
# 应返回 {"user_id":"...","email":"..."}
```

✅ **云端部署完成**。记录下你的云端地址：`http://你的服务器IP:8000`

---

## 三、Phase 2 — 部署启动器（Memento-x-tool）

### 3.1 前置安装（Windows 示例）

#### 3.1.1 安装 NVIDIA 驱动
1. 下载：https://www.nvidia.com/download/
2. 安装后验证：命令行运行 `nvidia-smi`，能看到 GPU 信息

#### 3.1.2 安装 Docker Desktop
1. 下载：https://www.docker.com/products/docker-desktop/
2. 安装后**重启电脑**
3. 打开 Docker Desktop → Settings → Resources → 确认 GPU 支持（WSL2 backend）
4. 验证 GPU 直通：
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
   ```
   能看到 GPU 信息即可

#### 3.1.3 安装 Python
1. 下载 Python 3.11 或 3.12：https://www.python.org/downloads/
2. 安装时勾选 "Add Python to PATH"

### 3.2 拉取代码

```powershell
# Windows PowerShell
cd C:\
git clone https://github.com/eneatlnc-cell/Memento-x-tool.git
cd Memento-x-tool
```

### 3.3 安装启动器依赖

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r launcher/requirements.txt
```

### 3.4 模型下载（关键步骤）

模型总计约 **31 GB**，分自动和手动两类。

#### 3.4.1 创建模型目录

```powershell
mkdir $env:USERPROFILE\.memento\workspace\models\sam2
mkdir $env:USERPROFILE\.memento\workspace\models\iclora
mkdir $env:USERPROFILE\.memento\workspace\models\ltx
mkdir $env:USERPROFILE\.memento\workspace\models\pose
mkdir $env:USERPROFILE\.memento\workspace\models\raft
```

#### 3.4.2 手动下载 SAM2.1 权重（898 MB）

> Meta 美国 CDN，国内可能慢，建议用迅雷/IDM 多线程下载。

浏览器打开下载：
```
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

下载后放入：
```
C:\Users\你的用户名\.memento\workspace\models\sam2\sam2.1_hiera_large.pt
```

#### 3.4.3 手动下载 IC-LoRA Ingredients（1.31 GB）— 需要 HF Token

1. 注册 HuggingFace 账号：https://huggingface.co/join
2. 打开 https://hf-mirror.com/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
3. 点击 **"Agree and access repository"**
4. 打开 https://huggingface.co/settings/tokens
5. 点 **"New token"** → 选 **Read** → 创建
6. 复制 token（格式：`hf_xxxxxxxxxxxx`）
7. 保存到启动器目录的 `hf_token.txt`：
   ```powershell
   echo "hf_你的token" > hf_token.txt
   ```
8. 手动下载文件放入：
   ```
   C:\Users\你的用户名\.memento\workspace\models\iclora\ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors
   ```

#### 3.4.4 自动下载（启动器会自动完成以下模型）

| 模型 | 大小 | 来源 |
|------|------|------|
| LTX-2.3 FP8 主模型 | 29 GB | ModelScope CDN |
| IC-LoRA Union Control | 654 MB | hf-mirror |
| MotionBERT 姿态估计 | 162 MB | hf-mirror |
| RAFT 光流模型 | ~50 MB | torchvision 内置 |
| MediaPipe | ~30 MB | pip 包 |
| SAM2 源码 | ~100 MB | GitHub |

**这些不用手动操作**，启动器首次运行会自动下载（约 30-60 分钟）。

### 3.5 启动启动器

```powershell
# 获取 token（先在云端注册账号，用 curl 注册）
curl -X POST http://你的服务器IP:8000/api/v1/account/register `
  -H "Content-Type: application/json" `
  -d '{"email":"your@email.com","password":"yourpassword"}'

# 登录拿 token
curl -X POST http://你的服务器IP:8000/api/v1/account/login `
  -H "Content-Type: application/json" `
  -d '{"email":"your@email.com","password":"yourpassword"}'
# 返回 {"access_token":"eyJ...","token_type":"bearer"}

# 启动启动器
python launcher.py --token "eyJ你的token" --api-url "http://你的服务器IP:8000/api/v1"
```

首次启动会：
1. 检查 Docker 运行状态
2. 检查 GPU（< 8GB 报错）
3. 拉取 Docker 镜像（首次约 5 GB）
4. 自动下载缺失模型
5. 启动 ComfyUI 容器
6. 向云端注册 + 启动心跳
7. 系统托盘出现绿色图标

### 3.6 验证启动器

```powershell
# 检查本地启动器 API
curl http://127.0.0.1:8189/health
# 应返回 {"status":"healthy"}

# 检查 ComfyUI
curl http://127.0.0.1:8188/system_stats
# 应返回 ComfyUI 系统信息

# 检查模型状态
curl http://127.0.0.1:8189/models/status
# 所有模型 ready=true
```

✅ **启动器部署完成**。

---

## 四、Phase 3 — 部署网站（Memento-x-web）

### 4.1 方式一：Vercel 部署（推荐，免费）

1. Fork 仓库到你的 GitHub：https://github.com/eneatlnc-cell/Memento-x-web
2. 打开 https://vercel.com → 用 GitHub 登录
3. "New Project" → 选择你 fork 的仓库
4. 配置环境变量（在 Vercel 项目设置里）：
   ```
   NEXT_PUBLIC_API_BASE_URL = http://你的服务器IP:8000/api/v1
   NEXT_PUBLIC_WS_URL = ws://你的服务器IP:8000/api/v1/status/ws
   ```
5. Deploy
6. 部署完成后拿到 `https://你的项目.vercel.app`

### 4.2 方式二：本地开发运行

```bash
git clone https://github.com/eneatlnc-cell/Memento-x-web.git
cd Memento-x-web

# 配置环境变量
cp .env.example .env.local
nano .env.local
# 填入：
# NEXT_PUBLIC_API_BASE_URL=http://你的服务器IP:8000/api/v1
# NEXT_PUBLIC_WS_URL=ws://你的服务器IP:8000/api/v1/status/ws

npm install
npm run dev
# 打开 http://localhost:3000
```

### 4.3 验证网站

1. 浏览器打开网站地址
2. 注册账号 / 登录
3. 应看到三步操作界面（选素材 → 选目标 → 点执行）

✅ **网站部署完成**。

---

## 五、Phase 4 — 端到端联调

### 5.1 上传测试素材

目前网站还没有上传入口，用 curl 直接调云端 API 上传（后续会补上传 UI）：

```bash
# 用一个短视频测试（1-3 秒最佳）
curl -X POST http://你的服务器IP:8000/api/v1/workflow/dispatch \
  -H "Authorization: Bearer 你的JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "替换人物",
    "assets": [{"id": "test_video", "name": "test.mp4", "type": "video"}]
  }'
```

### 5.2 观察任务流

1. 网站上应看到任务进入"执行中"状态
2. 启动器机器上，ComfyUI 开始执行 9 节点管线
3. WebSocket 实时推送进度到网站
4. 完成后成片出现在结果区

### 5.3 排查链路

如果任务卡住，按这个顺序排查：

| 现象 | 排查点 |
|------|--------|
| 网站显示"无可用启动器" | 启动器没注册成功 → 检查启动器日志 + 云端 `/api/v1/workflow/local/status/你的user_id` |
| 任务派发后无响应 | 启动器收不到任务 → 检查 `local_url` 配置 + 启动器 8189 端口 |
| ComfyUI 报错 | 节点加载失败 → 看容器日志 `docker logs memento-tool` |
| 模型加载失败 | 路径不对 → 看下方"常见问题" |
| WebSocket 不通 | URL/token 错误 → 网站控制台看 WS 连接错误 |

---

## 六、常见问题

### Q1: MotionBERT 报 "模型不存在"

**根因**：Docker 容器没挂载模型目录（已在 commit `b833887` 修复）。

**解决**：
1. 确保拉取了最新代码：`git pull`
2. **重建容器**（不只是重启）：启动器会自动 stop + remove + start
3. 检查容器内是否有模型文件：
   ```powershell
   docker exec memento-tool ls /root/data/models/pose/
   # 应看到 motionbert_ft_h36m.pth
   ```

### Q2: 启动器提示 "Docker 未运行"

打开 Docker Desktop，等左下角显示 "Engine running" 绿色。

### Q3: 启动器提示 "GPU 不可用"

1. 确认 NVIDIA 驱动已装：`nvidia-smi`
2. 确认 Docker GPU 直通：
   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
   ```
3. Windows 用户确认 WSL2 backend 已启用

### Q4: LTX-2.3 下载慢/失败

LTX 主模型 29GB，优先走 ModelScope CDN。如果失败：
```bash
# 手动用 modelscope 下载
pip install modelscope
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Lightricks/LTX-2.3-fp8', local_dir='~/.memento/workspace/models/ltx')
"
```

### Q5: 显存不足 (CUDA out of memory)

- 最低 8GB 显存（LTX FP8 量化版）
- 4K 视频建议 24GB+
- 临时方案：降低视频分辨率到 1080p 或 720p

### Q6: IC-LoRA Ingredients 下载 403

没加 Token 或没同意协议。回到 3.4.3 步骤确认：
1. 已在 HF 页面点 "Agree and access repository"
2. `hf_token.txt` 文件内容是 `hf_` 开头的有效 token
3. token 有 Read 权限

### Q7: 网站登录后 WebSocket 连不上

检查：
1. `.env.local` 的 `NEXT_PUBLIC_WS_URL` 是否指向云端正确路径 `/api/v1/status/ws`
2. 云端 8000 端口是否开放
3. 浏览器控制台看具体 WS 错误

### Q8: 如何查看日志

| 组件 | 日志位置 |
|------|---------|
| 云端 | `journalctl -u memento-cloud -f` |
| 启动器 | `~/.memento/logs/launcher.log` |
| ComfyUI 容器 | `docker logs memento-tool -f` |
| 启动器 Web | http://127.0.0.1:8189/logs |

### Q9: 如何重启服务

```powershell
# 重启启动器（会重建容器）
# 右键托盘图标 → 退出 → 重新运行 python launcher.py

# 重启 ComfyUI 容器（不重启启动器）
curl -X POST http://127.0.0.1:8189/stop
curl -X POST http://127.0.0.1:8189/start

# 重启云端
systemctl restart memento-cloud
```

### Q10: 数据库连接失败

1. 确认 PostgreSQL 运行：`systemctl status postgresql`
2. 确认密码正确：`psql -U memento -h localhost -d memento`
3. 确认 `.env` 的 `DATABASE_URL` 格式：`postgresql+asyncpg://用户:密码@host:5432/库名`

---

## 七、生产环境加固清单

部署跑通后，上线前必做：

- [ ] `JWT_SECRET_KEY` 改为强随机值
- [ ] PostgreSQL 密码改为强密码
- [ ] 云端 CORS 配置改为白名单（不用 `*`）
- [ ] 配置 HTTPS（用 Nginx + Let's Encrypt）
- [ ] 数据库定期备份
- [ ] 云端日志收集（ELK 或云厂商日志服务）
- [ ] 启动器心跳超时告警
- [ ] 模型文件完整性校验（MD5）
- [ ] 网站域名 + SSL 证书

---

## 八、目录结构速查

### 云端（服务器）
```
/opt/Memento-X/
├── cloud/              # 主代码
├── schema/             # 工作流 JSON Schema
├── .venv/              # 虚拟环境
└── cloud/.env          # 配置
```

### 启动器（用户机）
```
~/Memento-x-tool/      # 代码
~/.memento/
├── config.json         # 启动器配置
├── logs/               # 日志
└── workspace/
    ├── models/         # 所有模型（挂载到容器 /root/data/models）
    ├── assets/         # 素材
    └── outputs/        # 成片
```

### 网站（Vercel/本地）
```
Memento-x-web/
├── app/                # Next.js 页面
├── components/         # UI 组件
├── lib/                # 工具库
└── .env.local          # 配置
```

---

## 九、获取帮助

- 查看各仓库的 `CONTEXT.md` 了解架构设计
- 查看 `SETUP_GUIDE.md` 了解启动器细节
- 查看 `MODEL_GUIDE.md` 了解模型清单
- 提 Issue：https://github.com/eneatlnc-cell

---

**部署成功标志**：网站注册账号 → 登录 → 看到三步界面 → 启动器在线 → 派发任务 → 看到 9 节点进度 → 成片输出。
