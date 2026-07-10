@echo off
REM ============================================================
REM Memento 启动器 打包脚本 (Windows)
REM 输出: dist/memento-launcher.exe
REM ============================================================

echo [1/3] 安装依赖...
pip install -r requirements.txt
pip install pyinstaller

echo [2/3] 打包为 exe...
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "memento-launcher" ^
    --add-data "launcher;launcher" ^
    --hidden-import "uvicorn.logging" ^
    --hidden-import "uvicorn.loops.auto" ^
    --hidden-import "uvicorn.protocols.http.auto" ^
    --hidden-import "pystray._win32" ^
    --hidden-import "PIL._imaging" ^
    --hidden-import "docker" ^
    --hidden-import "docker.errors" ^
    --hidden-import "docker.types" ^
    launcher/launcher_gui.py

echo [3/3] 完成！
echo 输出: dist/memento-launcher.exe
dir dist\memento-launcher.exe