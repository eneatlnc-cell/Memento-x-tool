"""Memento Pipeline — 统一参数配置表

所有 9 节点可调参数的权威定义。每个参数包含:
  - id: 唯一标识符
  - node: 所属节点
  - default: 推荐默认值
  - range: 有效范围 [min, max]
  - step: UI 滑块步长
  - group: 分组 (basic / advanced / expert)
  - depends_on: 参数依赖关系 (例如 ingredients 启用时 num_inference_steps 默认值不同)
  - description: 中文说明
  - tooltip: UI 提示文案

WEB 端通过此配置表生成用户自定义参数面板。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class ParamGroup(str, Enum):
    """参数分组 — 控制 Web UI 折叠层级"""
    BASIC = "basic"        # 始终显示，普通用户可调
    ADVANCED = "advanced"  # 折叠在"高级设置"中
    EXPERT = "expert"      # 折叠在"专家设置"中，需确认


@dataclass
class ParamDef:
    """单个参数定义"""
    id: str                              # 唯一标识符
    node: str                            # 所属节点 (如 "01_preprocess")
    label: str                           # 中文显示名
    default: Any                         # 推荐默认值
    type: str                            # "float" | "int" | "bool" | "enum"
    group: ParamGroup = ParamGroup.BASIC
    range_min: Optional[float] = None    # 最小值
    range_max: Optional[float] = None    # 最大值
    step: Optional[float] = None         # 步长
    enum_values: Optional[List[str]] = None  # 枚举值列表
    depends_on: Optional[List[str]] = None   # 依赖的参数 ID 列表
    depends_defaults: Optional[Dict[str, Dict[Any, Any]]] = None  # 根据依赖值的默认值覆盖
    description: str = ""                # 中文说明
    tooltip: str = ""                    # UI 提示


# ════════════════════════════════════════════════════════════════
# 全管线参数定义
# ════════════════════════════════════════════════════════════════
PIPELINE_PARAMS: List[ParamDef] = [

    # ── 01 预处理 ──
    ParamDef(
        id="output_fps", node="01_preprocess", label="输出帧率",
        default=30, type="int", group=ParamGroup.BASIC,
        range_min=1, range_max=60, step=1,
        description="输出视频帧率，30fps 为推荐值，24fps 为电影感，60fps 需更多显存",
        tooltip="帧率越高动作越流畅，但管线处理时间成比例增加",
    ),
    ParamDef(
        id="max_resolution", node="01_preprocess", label="最大分辨率",
        default="1080p", type="enum",
        enum_values=["1080p", "4K", "8K"],
        description="输出分辨率上限，1080p 适合快速预览，4K 适合成品输出",
        tooltip="分辨率越高生成质量越好，但显存和推理时间成倍增长",
    ),

    # ── 02 SAM3 分割 ──
    ParamDef(
        id="score_threshold_detection", node="02_segment", label="检测置信度阈值",
        default=0.5, type="float", group=ParamGroup.ADVANCED,
        range_min=0.1, range_max=1.0, step=0.05,
        description="SAM3 检测器输出概率阈值，低于此值的检测结果将被丢弃",
        tooltip="降低可检测到更多目标（但可能误检），提高可减少误检（但可能漏检）",
    ),
    ParamDef(
        id="pred_iou_thresh", node="02_segment", label="掩码质量阈值",
        default=0.88, type="float", group=ParamGroup.ADVANCED,
        range_min=0.5, range_max=1.0, step=0.02,
        description="预测 IoU 阈值，低于此值的掩码认为质量不足而丢弃",
        tooltip="降低可保留更多掩码（但可能有低质量掩码），提高仅保留高质量掩码",
    ),
    ParamDef(
        id="stability_score_thresh", node="02_segment", label="掩码稳定性阈值",
        default=0.95, type="float", group=ParamGroup.ADVANCED,
        range_min=0.5, range_max=1.0, step=0.01,
        description="掩码对阈值变化的稳定性分数，低于此值丢弃",
        tooltip="提高可得到更稳定的掩码边缘，降低可保留更多细节",
    ),
    ParamDef(
        id="recondition_every_nth_frame", node="02_segment", label="重条件化间隔",
        default=16, type="int", group=ParamGroup.EXPERT,
        range_min=0, range_max=60, step=1,
        description="掩码重条件化频率（帧），0 表示禁用，用于防止追踪漂移",
        tooltip="值越小追踪越稳定但越慢，值越大追踪越快但可能漂移",
    ),

    # ── 03 MediaPipe 2D 姿态 ──
    ParamDef(
        id="model_complexity", node="03_pose2d", label="模型复杂度",
        default=2, type="int", group=ParamGroup.ADVANCED,
        range_min=0, range_max=2, step=1,
        description="MediaPipe 姿态模型复杂度: 0=轻量(最快), 1=平衡, 2=完整(最准)",
        tooltip="完整模型精度最高，适合对姿态准确性要求高的场景",
    ),
    ParamDef(
        id="min_detection_confidence", node="03_pose2d", label="检测最小置信度",
        default=0.5, type="float", group=ParamGroup.ADVANCED,
        range_min=0.1, range_max=1.0, step=0.05,
        description="人体检测模型最小置信度，低于此值视为检测失败",
        tooltip="远距离/遮挡场景可降至 0.3，误检多时可升至 0.7",
    ),
    ParamDef(
        id="min_tracking_confidence", node="03_pose2d", label="追踪最小置信度",
        default=0.5, type="float", group=ParamGroup.ADVANCED,
        range_min=0.1, range_max=1.0, step=0.05,
        description="关键点追踪最小置信度，低于此值触发重新检测",
        tooltip="提高可增加追踪鲁棒性，但可能增加延迟",
    ),
    ParamDef(
        id="heatmap_sigma", node="03_pose2d", label="热力图模糊半径",
        default=6.0, type="float", group=ParamGroup.EXPERT,
        range_min=1.0, range_max=20.0, step=1.0,
        description="骨骼热力图高斯模糊 sigma 值，控制热力图的扩散范围",
        tooltip="值越大热力图越模糊（控制更宽松），值越小越锐利（控制更严格）",
    ),

    # ── 04 MotionBERT 3D 姿态 ──
    # (推理时无可调参数，仅模型选择)
    ParamDef(
        id="motionbert_model", node="04_pose3d", label="MotionBERT 模型",
        default="full", type="enum", group=ParamGroup.EXPERT,
        enum_values=["full", "lite"],
        description="MotionBERT 完整版(162MB) vs Lite(61MB)，Lite 精度接近但更快",
        tooltip="完整版精度略高，Lite 版显存和速度更优",
    ),

    # ── 05 控制信号对齐 ──
    ParamDef(
        id="canny_low", node="05_align", label="Canny 低阈值",
        default=50, type="int", group=ParamGroup.ADVANCED,
        range_min=10, range_max=200, step=10,
        description="Canny 边缘检测低阈值，低于此值的边缘被忽略",
        tooltip="降低可检测更多边缘细节，提高仅保留强边缘",
    ),
    ParamDef(
        id="canny_high", node="05_align", label="Canny 高阈值",
        default=150, type="int", group=ParamGroup.ADVANCED,
        range_min=50, range_max=500, step=10,
        description="Canny 边缘检测高阈值，高于此值的边缘被保留",
        tooltip="降低可保留更多边缘，提高仅保留最显著边缘",
    ),
    ParamDef(
        id="temporal_smooth_window", node="05_align", label="时序平滑窗口",
        default=3, type="int", group=ParamGroup.EXPERT,
        range_min=1, range_max=11, step=2,
        description="时序平滑窗口大小（帧数），用于减少控制信号帧间抖动",
        tooltip="值越大越平滑但可能引入延迟感，值越小响应越快但可能抖动",
    ),

    # ── 06 LTX 局部重绘 ──
    ParamDef(
        id="control_mode", node="06_ltx", label="控制模式",
        default="pose+depth+canny", type="enum",
        enum_values=[
            "pose", "depth", "canny",
            "pose+depth", "pose+canny", "depth+canny",
            "pose+depth+canny",
        ],
        description="选择哪些控制信号约束生成，全选 = 最严格的控制",
        tooltip="pose 控制动作，depth 控制空间，canny 控制轮廓",
    ),
    ParamDef(
        id="control_strength", node="06_ltx", label="Union Control 强度",
        default=0.7, type="float", group=ParamGroup.BASIC,
        range_min=0.0, range_max=1.0, step=0.05,
        description="Union Control (depth+canny+pose) 控制强度，越高越严格遵循控制信号",
        tooltip="0.5=宽松(更具创造性)，0.7=推荐，0.9=严格(可能僵硬)",
    ),
    ParamDef(
        id="ingredients_enabled", node="06_ltx", label="启用 Ingredients",
        default=True, type="bool", group=ParamGroup.BASIC,
        description="启用 Ingredients Reference Sheet 角色外观一致性约束",
        tooltip="启用后角色面部/服装/体型更稳定，但推理时间增加",
    ),
    ParamDef(
        id="ingredients_strength", node="06_ltx", label="Ingredients 强度",
        default=1.4, type="float", group=ParamGroup.ADVANCED,
        range_min=0.0, range_max=2.0, step=0.05,
        depends_on=["ingredients_enabled"],
        depends_defaults={"ingredients_enabled": {False: 1.4}},
        description="Ingredients 控制强度，推荐 1.4，越高角色外观越严格匹配参考图",
        tooltip="1.0=标准，1.4=推荐(角色一致性最佳)，1.8=极严格(可能僵硬)",
    ),
    ParamDef(
        id="num_inference_steps", node="06_ltx", label="推理步数",
        default=30, type="int", group=ParamGroup.ADVANCED,
        range_min=5, range_max=50, step=1,
        depends_on=["ingredients_enabled"],
        depends_defaults={"ingredients_enabled": {True: 30, False: 8}},
        description="去噪步数，Ingredients 模式推荐 30，普通模式推荐 8",
        tooltip="步数越多质量越高但速度越慢，Ingredients 模式最低 20 步",
    ),
    ParamDef(
        id="guidance_scale", node="06_ltx", label="CFG 引导强度",
        default=4.0, type="float", group=ParamGroup.ADVANCED,
        range_min=1.0, range_max=15.0, step=0.5,
        description="分类器自由引导尺度，控制对 Prompt 的遵循度",
        tooltip="3-5=修复推荐，7-9=创意生成，>10=可能过拟合",
    ),
    ParamDef(
        id="stg_scale", node="06_ltx", label="STG 时空引导",
        default=3.0, type="float", group=ParamGroup.EXPERT,
        range_min=0.0, range_max=10.0, step=0.5,
        description="时空引导强度，控制视频帧间一致性",
        tooltip="3.0=推荐(动态场景)，1.5=静态场景，5.0=极强一致性",
    ),
    ParamDef(
        id="seed", node="06_ltx", label="随机种子",
        default=42, type="int", group=ParamGroup.ADVANCED,
        range_min=0, range_max=2**31 - 1, step=1,
        description="随机种子，相同种子产生相同结果",
        tooltip="设为 -1 使用随机种子，固定值用于复现结果",
    ),

    # ── 07 RAFT 光流矫正 ──
    ParamDef(
        id="raft_iters", node="07_raft", label="RAFT 迭代次数",
        default=20, type="int", group=ParamGroup.ADVANCED,
        range_min=4, range_max=32, step=1,
        description="RAFT 光流估计 GRU 迭代次数，越多越精确但越慢",
        tooltip="12=快速(精度略低)，20=推荐，32=最高精度",
    ),
    ParamDef(
        id="flow_diff_threshold", node="07_raft", label="光流差异阈值",
        default=2.0, type="float", group=ParamGroup.EXPERT,
        range_min=0.5, range_max=20.0, step=0.5,
        description="光流差异阈值，超过此值的像素判定为需矫正区域",
        tooltip="降低可矫正更多区域，提高仅矫正显著不一致区域",
    ),
    ParamDef(
        id="mask_blend_strength", node="07_raft", label="掩码混合强度",
        default=0.85, type="float", group=ParamGroup.EXPERT,
        range_min=0.0, range_max=1.0, step=0.05,
        description="光流矫正与原始帧的混合比例",
        tooltip="1.0=完全使用矫正帧，0.5=与原帧各半混合",
    ),
    ParamDef(
        id="temporal_weight", node="07_raft", label="时序一致权重",
        default=0.33, type="float", group=ParamGroup.EXPERT,
        range_min=0.0, range_max=1.0, step=0.01,
        description="帧间时序一致性权重，越高相邻帧越平滑",
        tooltip="0.33=推荐，0.5=更强平滑，0.0=无时序约束",
    ),

    # ── 08 分层光影融合 ──
    ParamDef(
        id="blend_alpha", node="08_fusion", label="融合透明度",
        default=0.7, type="float", group=ParamGroup.ADVANCED,
        range_min=0.0, range_max=1.0, step=0.05,
        description="前景与背景混合权重，Result = FG*alpha + BG*(1-alpha)",
        tooltip="1.0=完全使用生成前景，0.5=各半混合，0=完全使用原始背景",
    ),
    ParamDef(
        id="feather_radius", node="08_fusion", label="边缘羽化半径",
        default=10, type="int", group=ParamGroup.ADVANCED,
        range_min=0, range_max=30, step=1,
        description="掩码边缘高斯模糊羽化半径（像素），越大过渡越柔和",
        tooltip="0=硬边缘(可能有接缝)，5-15=推荐，30=极柔和过渡",
    ),
    ParamDef(
        id="shadow_strength", node="08_fusion", label="深度阴影强度",
        default=0.4, type="float", group=ParamGroup.ADVANCED,
        range_min=0.0, range_max=1.0, step=0.05,
        description="基于深度图的阴影合成强度，0=无阴影，1=最强阴影",
        tooltip="0.3-0.5=推荐(自然阴影)，0=无阴影(更亮)，>0.7=浓重阴影",
    ),
    ParamDef(
        id="color_match_strength", node="08_fusion", label="颜色匹配强度",
        default=0.6, type="float", group=ParamGroup.EXPERT,
        range_min=0.0, range_max=1.0, step=0.05,
        description="颜色分布匹配强度，1.0=完全匹配目标颜色分布",
        tooltip="0.5-0.8=推荐，1.0=完全匹配(可能过度)，0=不做颜色匹配",
    ),

    # ── 09 FFmpeg 成片 ──
    ParamDef(
        id="crf", node="09_composite", label="编码质量 (CRF)",
        default=18, type="int", group=ParamGroup.ADVANCED,
        range_min=0, range_max=51, step=1,
        description="H.264 编码 CRF 值，0=无损，18=视觉无损，23=默认，51=最差",
        tooltip="18=推荐(高质量)，23=标准，28=压缩(文件小但质量低)",
    ),
    ParamDef(
        id="preset", node="09_composite", label="编码预设",
        default="medium", type="enum", group=ParamGroup.EXPERT,
        enum_values=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        description="FFmpeg 编码速度预设，越慢文件越小质量越高",
        tooltip="medium=推荐，slow=更好压缩，ultrafast=最快但文件大",
    ),
]


# ════════════════════════════════════════════════════════════════
# 参数预设方案 (为不同场景提供一键配置)
# ════════════════════════════════════════════════════════════════
PRESETS: Dict[str, Dict[str, Any]] = {
    "快速预览": {
        "max_resolution": "1080p",
        "num_inference_steps": 8,
        "ingredients_enabled": False,
        "raft_iters": 12,
        "crf": 23,
        "preset": "veryfast",
        "description": "快速生成预览，质量较低但速度最快，适合快速迭代",
    },
    "标准质量": {
        "max_resolution": "1080p",
        "num_inference_steps": 30,
        "ingredients_enabled": True,
        "ingredients_strength": 1.4,
        "raft_iters": 20,
        "crf": 18,
        "preset": "medium",
        "description": "推荐默认配置，平衡质量与速度",
    },
    "高质量成品": {
        "max_resolution": "4K",
        "num_inference_steps": 50,
        "ingredients_enabled": True,
        "ingredients_strength": 1.6,
        "control_strength": 0.8,
        "raft_iters": 32,
        "crf": 15,
        "preset": "slow",
        "description": "最高质量输出，适合最终交付，速度较慢",
    },
}


# ════════════════════════════════════════════════════════════════
# 参数查找工具函数
# ════════════════════════════════════════════════════════════════
def get_param(param_id: str) -> Optional[ParamDef]:
    """按 ID 查找参数定义"""
    for p in PIPELINE_PARAMS:
        if p.id == param_id:
            return p
    return None


def get_params_by_node(node: str) -> List[ParamDef]:
    """按节点过滤参数"""
    return [p for p in PIPELINE_PARAMS if p.node == node]


def get_params_by_group(group: ParamGroup) -> List[ParamDef]:
    """按分组过滤参数"""
    return [p for p in PIPELINE_PARAMS if p.group == group]


def get_defaults() -> Dict[str, Any]:
    """获取所有参数默认值"""
    return {p.id: p.default for p in PIPELINE_PARAMS}


def apply_preset(preset_name: str) -> Dict[str, Any]:
    """应用预设方案，返回完整的参数覆盖 dict"""
    if preset_name not in PRESETS:
        raise ValueError(f"未知预设: {preset_name}, 可用: {list(PRESETS.keys())}")
    defaults = get_defaults()
    preset = PRESETS[preset_name]
    for k, v in preset.items():
        if k != "description" and k in defaults:
            defaults[k] = v
    return defaults


def resolve_dependencies(values: Dict[str, Any]) -> Dict[str, Any]:
    """根据依赖关系解析参数值（例如 ingredients 启用时自动调整推理步数）"""
    resolved = dict(values)
    for p in PIPELINE_PARAMS:
        if p.depends_on and p.depends_defaults:
            for dep_id, mapping in p.depends_defaults.items():
                dep_val = resolved.get(dep_id)
                if dep_val is not None and dep_val in mapping:
                    if p.id not in values or values[p.id] == p.default:
                        resolved[p.id] = mapping[dep_val]
    return resolved