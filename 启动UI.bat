@echo off
chcp 65001 >nul
echo ========================================
echo    AI 实验室 - 项目进度 UI
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python，请先安装 Python 并添加到 PATH
    pause
    exit /b 1
)

echo [1/2] 扫描任务目录并生成数据...
python dashboard/generate.py

echo.
echo [2/2] 启动本地服务器...
echo 浏览器将自动打开: http://localhost:8899/dashboard/
echo 按 Ctrl+C 关闭服务器
echo.

cd /d "%~dp0"
start http://localhost:8899/dashboard/
python -m http.server 8899
