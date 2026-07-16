# Memento 工具链 — 模型部署与使用指南

## 概述

Memento 启动器内置了智能模型下载管理，将模型分为两类：

| 类型 | 说明 | 操作 |
|------|------|------|
| **自动下载** | 国内高速镜像，启动器自动完成 | 无需任何操作 |
| **手动下载** | 境外 CDN / 需要授权 | 按下方指引操作 |

---

## 一、自动下载模型（启动器自动完成）

启动器运行后会自动检查并下载以下模型，使用国内高速镜像源：

| 模型 | 大小 | 用途 | 下载源 |
|------|------|------|--------|
| MotionBERT 姿态估计 | 162 MB | 03_pose2d 节点 — 2D 人体姿态提取 | hf-mirror 镜像 |
| IC-LoRA Union Control | 654 MB | 04_control 节点 — 结构控制 LoRA | hf-mirror 镜像 |
| LTX-2.3 FP8 主模型 | 29 GB | 06_ltx 节点 — 视频生成核心模型 | ModelScope 国内 CDN 优先 |
| RAFT 光流模型 | ~50 MB | 07_raft 节点 — 光流估计 | PyTorch CDN |
| MediaPipe | ~30 MB | 人体/手势检测 | pip 阿里云镜像 |
| SAM2 源码 | ~100 MB | 02_segment 节点 — 视频分割 | GitHub |

**自动下载总计约 30 GB**，首次启动需等待 30-60 分钟（取决于网速）。

### 查看下载进度

启动器日志会实时显示下载进度：

```
[10:23:15] 模型(自动): 2/6 就绪 (30.1 GB)
[10:23:15]   待下载: MotionBERT 姿态估计, IC-LoRA Union Control, LTX-2.3 FP8 主模型, MediaPipe
[10:23:15] 启动模型自动下载...
[10:23:15]   下载: LTX-2.3 FP8 主模型 (29 GB)
[10:45:30]   [模型] LTX-2.3 FP8 主模型 ✓
[10:45:31]   下载: MotionBERT 姿态估计 (162 MB)
[10:45:35]   [模型] MotionBERT 姿态估计 ✓
```

也可以通过 API 查询：
```bash
curl http://127.0.0.1:8189/models/status
curl http://127.0.0.1:8189/models/progress
```

---

## 二、手动下载模型

以下模型因网络/授权原因需要你手动操作：

### 2.1 SAM2.1 权重（898 MB）

> **原因**：文件托管在 Meta 美国 CDN（`dl.fbaipublicfiles.com`），国内直连速度较慢（50 KB/s 以下），建议手动下载。

**步骤：**

1. 浏览器打开下载链接：
   ```
   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
   ```

2. 下载完成后，将文件放入以下目录：
   ```
   ~/.memento/workspace/models/sam2/sam2.1_hiera_large.pt
   ```
   > `~` 是你的用户目录。Windows 上通常是 `C:\Users\你的用户名`。

3. **验证**：确认文件存在即可，启动器会自动识别。

**命令行方式（macOS/Linux）：**
```bash
# 下载（如果网速还行）
wget -O ~/.memento/workspace/models/sam2/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# 或者从你电脑复制到 GPU 服务器
scp sam2.1_hiera_large.pt user@gpu-server:~/.memento/workspace/models/sam2/
```

---

### 2.2 IC-LoRA Ingredients（1.31 GB）— 需要 HuggingFace Token

> **原因**：Lightricks 要求同意社区许可证，仓库是 gated（受限访问），需要 HuggingFace 账号授权。

**步骤：**

#### 方式一：设置 Token 让启动器自动下载（推荐）

1. 注册 HuggingFace 账号：
   https://huggingface.co/join

2. 打开模型页面，点击 **"Agree and access repository"**：
   https://hf-mirror.com/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients

3. 创建 Access Token：
   https://huggingface.co/settings/tokens
   - 点击 "New token"
   - 类型选择 **Read**
   - 复制生成的 Token（格式：`hf_xxxxxxxxxxxxxxxxxxxx`）

4. 在启动器目录下创建 `hf_token.txt` 文件：
   ```
   ~/.memento/workspace/hf_token.txt
   ```
   内容就是你的 Token（一行，纯文本）。

5. 重启启动器，它会自动使用 Token 从 hf-mirror 下载 Ingredients。

#### 方式二：手动下载后放入

1. 完成方式一的步骤 1-3（获取 Token）

2. 使用命令行下载：
   ```bash
   export HF_TOKEN="hf_你的token"
   export HF_ENDPOINT=https://hf-mirror.com

   python3 -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download(
       'Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients',
       'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
       local_dir='~/.memento/workspace/models/iclora',
       local_dir_use_symlinks=False,
       token='$HF_TOKEN',
   )
   ```

3. 确认文件存在：
   ```
   ~/.memento/workspace/models/iclora/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors
   ```

---

### 2.3 MediaPipe 姿态模型（~15 MB）

> **原因**：MediaPipe 首次运行时会从 Google 服务器下载模型文件，国内可能被墙。

**解决方式：**

- **方式一**：启动器已配置国内镜像，通常不需要额外操作
- **方式二**：如果遇到下载失败，设置代理后重试
- **方式三**：手动下载 MediaPipe 模型缓存到 `~/.mediapipe/`

---

## 三、完整模型目录结构

启动器会在 `~/.memento/workspace/models/` 下创建以下结构：

```
~/.memento/workspace/models/
├── sam2/                                         ← SAM2.1 分割模型
│   └── sam2.1_hiera_large.pt                    [手动] 898 MB
│
├── iclora/                                       ← IC-LoRA 双 LoRA
│   ├── ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors  [自动] 654 MB
│   └── ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors       [手动] 1.31 GB
│
├── ltx/                                          ← LTX-2.3 主模型
│   └── ltx-2.3-22b-dev-fp8.safetensors          [自动] 29 GB
│
├── pose/                                         ← MotionBERT 姿态估计
│   └── motionbert_ft_h36m.pth                   [自动] 162 MB
│
├── raft/                                         ← RAFT 光流模型
│   └── (PyTorch 自动缓存)                        [自动] ~50 MB
│
└── mediapipe/                                    ← MediaPipe 模型
    └── (pip 安装 + 首次运行缓存)                  [自动] ~30 MB
```

**标记说明：** `[自动]` = 启动器自动下载 | `[手动]` = 需要你手动操作

---

## 四、如何将手动下载的文件放入工具链

### 核心原则

**`~/.memento/workspace/models/` 目录会被自动挂载到 Docker 容器内的 `/opt/models/`。**

所以你只需要把文件放到正确的位置，启动器会自动识别。

### 具体操作流程

#### 场景 A：你从浏览器下载了 SAM2.1 权重

```
1. 浏览器下载 → 得到 sam2.1_hiera_large.pt
2. 打开文件管理器，进入: C:\Users\你的用户名\.memento\workspace\models\sam2\
3. 把 sam2.1_hiera_large.pt 复制进去
4. 完成！启动器会自动识别
```

#### 场景 B：你从朋友那里拷贝了 IC-LoRA Ingredients

```
1. 拿到 ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors
2. 放入: C:\Users\你的用户名\.memento\workspace\models\iclora\
3. 完成！
```

#### 场景 C：你从 U 盘/移动硬盘迁移模型

```
1. 插入 U 盘
2. 复制所有模型文件到对应目录
3. 完整目录结构参考上方的"模型目录结构"
4. 启动启动器，它会自动跳过已有文件
```

#### 场景 D：GPU 云服务器部署

如果你在 GPU 云服务器上运行启动器，先把模型下载到本地电脑，再上传：

```bash
# 从本地电脑上传到 GPU 服务器
scp sam2.1_hiera_large.pt user@gpu-ip:~/.memento/workspace/models/sam2/
scp ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors user@gpu-ip:~/.memento/workspace/models/iclora/
```

### 验证模型是否就绪

**方法 1：通过启动器 API**
```bash
curl http://127.0.0.1:8189/models/status
```

返回示例：
```json
{
  "auto": [
    {"name": "LTX-2.3 FP8 主模型", "ready": true, "size": "29 GB"},
    {"name": "MotionBERT 姿态估计", "ready": true, "size": "162 MB"},
    ...
  ],
  "manual": [
    {"name": "SAM2.1 权重", "ready": true, "size": "898 MB"},
    {"name": "IC-LoRA Ingredients", "ready": false, "size": "1.31 GB"},
    ...
  ],
  "auto_ready": true,
  "downloaded_gb": 31.5
}
```

**方法 2：通过启动器日志**
```
[10:30:00] 模型(自动): 6/6 就绪 ✓
[10:30:00] 模型(手动): 1/2 就绪 — 需要手动下载: IC-LoRA Ingredients
```

**方法 3：通过 API 获取目录树**
```bash
curl http://127.0.0.1:8189/models/dir-tree
```

---

## 五、HF Token 高级用法

### 设置 Token（三种方式）

**方式 1：文件方式（推荐）**
```
~/.memento/workspace/hf_token.txt
```
内容：`hf_xxxxxxxxxxxxxxxxxxxx`

**方式 2：环境变量**
```bash
export HF_TOKEN="hf_你的token"
python launcher.py --token 你的MementoToken
```

**方式 3：通过 API 设置**
```bash
curl -X POST "http://127.0.0.1:8189/models/hf-token?token=hf_xxxxxxxxxxxx"
```

### Token 安全提示

- Token 仅存储在本地 `hf_token.txt`，不会上传到云端
- 创建 Token 时选择 **Read** 权限即可，不需要 Write
- 如果 Token 泄露，在 HuggingFace Settings 中撤销并重新创建

---

## 六、常见问题

### Q: 启动器提示 "模型(手动): 需要手动下载" 怎么办？
A: 这是正常的。手动下载的模型需要你按照上方指引操作。自动下载的模型启动器会自己搞定。

### Q: 自动下载失败了怎么办？
A: 启动器不会因为单个模型失败而中断。你可以：
1. 重启启动器，它会跳过已下载的，只下载缺失的
2. 通过 API 手动触发：`curl -X POST http://127.0.0.1:8189/models/download`
3. 手动下载失败的文件放入对应目录

### Q: 下载速度很慢？
A: 启动器已配置国内镜像：
- hf-mirror.com（HuggingFace 镜像）
- ModelScope 国内 CDN（LTX 大文件优先）
- 阿里云 pip 镜像

如果仍然慢，可以手动下载后放入对应目录。

### Q: 显存不足怎么办？
A: 最低需要 8GB 显存。LTX-2.3 FP8 版本占用约 16GB，推荐 24GB 以上显卡。

### Q: 模型文件可以放在其他盘吗？
A: 可以。修改 `~/.memento/config.json` 中的 `workspace` 路径：
```json
{
  "workspace": "D:/memento-data"
}
```
模型文件会从 `D:/memento-data/models/` 加载。

### Q: 如何查看完整日志？
A: 日志文件在 `~/.memento/logs/launcher.log`，或通过 API：
```bash
curl http://127.0.0.1:8189/logs
```

### Q: 所有模型都就绪后，多久能开始使用？
A: 启动器会自动拉取 Docker 镜像并启动容器。首次启动约 5-10 分钟（镜像拉取），之后每次启动约 30 秒。

---

## 七、快速检查清单

部署完成后，确认以下状态：

- [ ] Docker Desktop 运行中（Engine running）
- [ ] `nvidia-smi` 正常输出 GPU 信息
- [ ] 启动器托盘图标为绿色（在线）
- [ ] `curl http://127.0.0.1:8189/models/status` 中 auto_ready 为 true
- [ ] 手动下载的模型文件已放入对应目录
- [ ] `curl http://127.0.0.1:8189/health` 返回 healthy: true

全部就绪后，即可通过 Memento Web 端下发工作流任务！
