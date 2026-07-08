"""
Memento-x-tool 一键安装器（Windows-only）

编排安装流程：
  1. 硬件检测 → 确定可用工具集
  2. 工具下载 → 下载 + 解压 + 校验
  3. 输出安装摘要

用法：
    python installer.py          # 安装所有推荐工具
    python installer.py --dry-run  # 仅检测，不下载
    python installer.py --tool ffmpeg  # 安装指定工具
"""
import argparse
import os
import sys

from detector import HardwareDetector
from downloader import ToolDownloader, TOOL_CATALOG


MEMENTO_ROOT = os.path.join(os.path.expanduser("~"), ".memento")


def install_tool(tool: str, downloader: ToolDownloader):
    """安装单个工具"""
    print(f"\n{'='*60}")
    print(f"  安装: {tool}")
    print(f"{'='*60}")
    result = downloader.download(tool)
    return result is not None


def main():
    parser = argparse.ArgumentParser(description="Memento-x-tool 一键安装器")
    parser.add_argument("--dry-run", action="store_true", help="仅检测硬件，不下载")
    parser.add_argument("--tool", type=str, help="仅安装指定工具")
    parser.add_argument("--tools-dir", type=str, default=None, help="工具安装目录")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════╗
    ║       Memento-x-tool 一键安装器               ║
    ║       Windows 版 · v1.0.0                     ║
    ╚══════════════════════════════════════════════╝
    """)

    # ── 步骤 1：硬件检测 ──
    print("[1/3] 检测硬件...\n")
    profile = HardwareDetector.detect()
    print(profile.summary())
    print()

    tools_dir = args.tools_dir or os.path.join(MEMENTO_ROOT, "tools")
    print(f"工具目录: {tools_dir}")
    print(f"推荐工具: {', '.join(profile.recommended_toolset)}")
    print()

    if args.dry_run:
        print("[DRY RUN] 仅检测，不下载。退出。")
        return

    # ── 步骤 2：下载工具 ──
    downloader = ToolDownloader(cache_dir=tools_dir)

    if args.tool:
        # 安装指定工具
        tools_to_install = [args.tool]
    else:
        # 安装推荐工具
        tools_to_install = profile.recommended_toolset

    print(f"[2/3] 安装工具 ({len(tools_to_install)} 个)...")

    results = {}
    for tool in tools_to_install:
        if tool not in TOOL_CATALOG:
            print(f"[WARN] 未知工具: {tool}，跳过")
            results[tool] = False
            continue
        results[tool] = install_tool(tool, downloader)

    # ── 步骤 3：安装摘要 ──
    print(f"\n{'='*60}")
    print("[3/3] 安装摘要")
    print(f"{'='*60}")

    success = []
    failed = []
    skipped = []

    for tool, ok in results.items():
        if ok:
            success.append(tool)
        elif TOOL_CATALOG.get(tool) and not TOOL_CATALOG[tool].download_url:
            skipped.append(tool)
        else:
            failed.append(tool)

    if success:
        print(f"\n  ✅ 成功 ({len(success)}): {', '.join(success)}")
    if failed:
        print(f"\n  ❌ 失败 ({len(failed)}): {', '.join(failed)}")
    if skipped:
        print(f"\n  ⏭  跳过 ({len(skipped)}): {', '.join(skipped)}（需手动安装）")

    print(f"\n工具目录: {tools_dir}")
    print(f"磁盘占用: {_get_dir_size(tools_dir):.1f} MB")

    if not failed:
        print("\n  🎉 安装完成！Memento-x-tool 已就绪。")
    else:
        print(f"\n  ⚠️  部分工具安装失败，请检查网络后重试。")
        sys.exit(1)


def _get_dir_size(path: str) -> float:
    """计算目录大小（MB）"""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


if __name__ == "__main__":
    main()