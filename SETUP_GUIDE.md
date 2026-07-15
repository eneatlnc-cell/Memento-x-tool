# Memento-X 工具链部署指南

## 系统架构速览

```
用户电脑                             云端
┌─────────────┐                    ┌──────────────────┐
│  Memento     │  JSON 工作流指令    │  Cloud Hub       │
│  启动器      │ ◄─────────────────→ │  memento.asia     │
│  (Docker)    │                    │  (意图理解+调度)   │
│  ComfyUI     │                    └──────────────────┘
│  + 9节点管线  │
└─────────────┘
```

**启动器 = Docker 容器管理 + 云端注册 + 本地 API 服务**，不需要你手动操作 Docker 命令行。

---

## 一、硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA 8GB 显存 | NVIDIA 24GB+ 显存 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 80 GB 空闲 | 150 GB SSD |
| 系统 | Windows 10/11 | Windows 11 |
| 依赖 | Docker Desktop + NVIDIA 驱动 | 同左 |

---

## 二、前置安装

### 2.1 安装 Docker Desktop

1. 下载：https://www.docker.com/products/docker-desktop/
2. 安装后**重启电脑**
3. 打开 Docker Desktop，确保状态为 "Engine running"

### 2.2 安装 NVIDIA 驱动

1. 下载：https://www.nvidia.com/download/
2. 安装后验证：命令行运行 `nvidia-smi`，能看到 GPU 信息即可

---

## 三、自动下载（启动器自动完成）

以下模型**启动器会自动下载**，无需手动操作：

| 模型 | 大小 | 来源 | 速度 |
|------|------|------|------|
| LTX-2.3 FP8 主模型 | 29 GB | ModelScope 国内 CDN | 快 |
| IC-LoRA Union Control | 654 MB | hf-mirror 镜像 | 快 |
| MotionBERT 姿态估计 | 162 MB | hf-mirror 镜像 | 快 |
| RAFT 光流模型 | 50 MB | PyTorch CDN | 快 |
| MediaPipe | pip 包 | 阿里云镜像 | 快 |
| SAM2 源码 | ~100 MB | GitHub | 中 |

**自动下载总计约 30 GB**，第一次启动需要等待 30-60 分钟（取决于网速）。

---

## 四、手动下载（需要你操作）

### 4.1 SAM2.1 权重（898 MB）

> **原因**：Meta 美国 CDN，国内下载较慢，建议手动下载后放入。

**步骤**：
1. 浏览器打开下载：
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   ```
2. 下载完成后，将文件放入：
   ```
   ~/.memento/workspace/models/sam2/sam2.1_hiera_large.pt
   ```
   （`~` 是你的用户目录，如 `C:\Users\你的用户名`）

### 4.2 IC-LoRA Ingredients（1.31 GB）— 需要 HF Token

> **原因**：Lightricks 要求同意社区许可证，需要 HuggingFace 账号。

**步骤**：
1. 打开 https://huggingface.co/join 注册账号（如已有则跳过）
2. 打开 https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
3. 点击页面上的 **"Agree and access repository"** 按钮
4. 打开 https://huggingface.co/settings/tokens
5. 点击 **"New token"** → 选择 **Read** 权限 → 创建
6. 复制生成的 token（格式：`hf_xxxxxxxxxxxx`）
7. 在启动器目录下创建文件 `hf_token.txt`，粘贴 token

### 4.3 MediaPipe 模型文件（首次运行自动触发）

> **原因**：MediaPipe 首次运行从 Google 服务器下载模型文件，国内可能被墙。

**解决**：启动器已配置国内镜像，无需额外操作。如遇到问题，运行：
```bash
export MEDIAPIPE_DOWNLOAD_MIRROR=https://storage.googleapis.com/mediapipe-models/
```

---

## 五、模型目录结构

启动器会在 `~/.memento/workspace/models/` 下创建以下结构：

```
~/.memento/workspace/models/
├── sam2/
│   └── sam2.1_hiera_large.pt          ← 手动放入
├── iclora/
│   ├── ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors  ← 自动
│   └── ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors       ← 手动
├── ltx/
│   └── ltx-2.3-22b-dev-fp8.safetensors  ← 自动
├── pose/
│   └── motionbert_ft_h36m.pth           ← 自动
└── raft/                                ← 自动
```

---

## 六、启动器使用

### 6.1 获取 Token

1. 打开 https://memento.asia 注册账号
2. 登录后，在"设置 → API Token"页面复制你的 Token

### 6.2 启动

**方式一：pip 安装后运行（推荐）**
```bash
pip install -r launcher/requirements.txt
python launcher.py --token 你的Token
```

**方式二：双击 exe（Windows 打包版）**
1. 下载 `memento-launcher.exe`
2. 双击运行
3. 首次运行会弹出配置窗口，填入 Token
4. 系统托盘出现 Memento 图标即启动成功

### 6.3 状态说明

| 托盘图标 | 状态 | 含义 |
|---------|------|------|
| 🟢 绿色 | online | 正常运行，已注册到云端 |
| 🟡 黄色 | installing | 正在下载模型/拉取镜像 |
| ⚪ 灰色 | idle | 已启动但未注册 |
| 🔴 红色 | error | 出错，查看日志 |

---

## 七、9 节点工作流

启动器自动管理以下 9 节点管线：

```
01_preprocess → 02_segment(SAM2.1) → 03_pose2d(MotionBERT) → 04_control(Union)
    → 05_crop → 06_ltx(LTX-2.3+Ingredients) → 07_raft → 08_fusion → 09_export
```

所有节点由 Docker 容器内的 ComfyUI 执行，通过 WebSocket 与 Cloud Hub 实时同步状态。

---

## 八、常见问题

### Q: 启动器提示 "Docker 未运行"
A: 打开 Docker Desktop，等待左下角显示 "Engine running" 绿色。

### Q: 启动器提示 "GPU 不可用"
A: 确认 NVIDIA 驱动已安装，命令行运行 `nvidia-smi` 测试。

### Q: 下载速度很慢
A: 启动器已配置国内镜像，如果某个模型特别慢，可以手动下载后放入对应目录。

### Q: 显存不足
A: 最低需要 8GB 显存。LTX-2.3 FP8 占用约 16GB，建议 24GB 以上。

### Q: 如何查看日志
A: 日志文件在 `~/.memento/logs/launcher.log`，右键托盘图标也可以查看。
