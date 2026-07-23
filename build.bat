@echo off
REM ============================================================
REM Memento 启动器 打包脚本 (Windows)
REM 用法: 在项目根目录执行 build.bat
REM 输出: dist/memento-launcher.exe
REM 前提: Node.js 18+ (用于构建 Web 前端)
REM ============================================================

echo [1/4] 安装 Python 依赖...
pip install -r launcher/requirements.txt
pip install pyinstaller

echo [2/4] 构建 Web 前端...
cd web
call npm install
call npm run build
cd ..
if not exist "web\dist\index.html" (
    echo [警告] Web 前端构建失败，exe 将不包含 Web 界面
    echo         请确认已安装 Node.js 18+ 并重试
    echo         下载: https://nodejs.org/
)

echo [3/4] 打包为 exe（含 Web 前端）...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "memento-launcher" ^
    --add-data "launcher;launcher" ^
    --add-data "web\dist;web\dist" ^
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
    --hidden-import "fastapi.staticfiles" ^
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
echo   3. 浏览器打开 http://127.0.0.1:8189 即可使用
echo   4. 首次运行需要 Docker Desktop 和 NVIDIA 驱动
