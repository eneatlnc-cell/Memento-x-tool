@echo off
REM ============================================================
REM Memento 启动器 打包脚本 (Windows)
REM 用法: 在项目根目录执行 build.bat
REM 输出: dist/memento-launcher.exe
REM ============================================================

echo [1/4] 安装依赖...
pip install -r launcher/requirements.txt
pip install pyinstaller

echo [2/4] 清理旧构建...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] 打包为 exe...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "memento-launcher" ^
    --add-data "launcher;launcher" ^
    --collect-submodules uvicorn ^
    --collect-submodules fastapi ^
    --hidden-import "pystray._win32" ^
    --hidden-import "PIL._imaging" ^
    --hidden-import "docker" ^
    --hidden-import "docker.errors" ^
    --hidden-import "docker.types" ^
    --hidden-import "huggingface_hub" ^
    --hidden-import "pydantic" ^
    --hidden-import "pydantic.deprecated.decorator" ^
    launcher/launcher_gui.py

echo [4/4] 完成！
echo.
echo 输出文件: dist/memento-launcher.exe
echo 文件大小:
dir dist\memento-launcher.exe
echo.
echo 发布步骤:
echo   1. 上传到 GitHub Releases
echo   2. 用户下载后双击运行
echo   3. 首次运行需要 Docker Desktop 和 NVIDIA 驱动
