@echo off
chcp 65001
echo 系统运行环境初始化中...

if not exist ".venv" (
    echo 未检测到局部运行环境，正在自动构建...
    python -m venv .venv
)

echo 激活沙盒环境...
call .venv\Scripts\activate

echo 核对并装载系统依赖组件...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo 服务链路就绪，正在唤醒表现层控制台...
streamlit run app.py

pause