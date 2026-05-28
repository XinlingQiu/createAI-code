import os
import json
import base64
import random
import time
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_debug

set_debug(True)

# 尝试导入强化版知识构建器 (包含了 delete_single_file)
try:
    from rag_engine import get_rag_chain
    from knowledge_builder import build_local_vector_db, clear_vector_db, delete_single_file
except ImportError:
    st.error("⚠️ 找不到 rag_engine.py 或 knowledge_builder.py，请确保它们在同一目录下。")

# 解决 Matplotlib 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 页面基础配置 ---
st.set_page_config(page_title="AI 学习助手系统", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 🚨 全局 API KEY
# ==========================================
ALIYUN_API_KEY = "sk-2ee1e6f39a4c4efda8877758bb71345f"

# --- 深度定制的高级 CSS 样式 (UI 旗舰级升级版) ---
st.markdown("""
<style>
    /* ================= 主区域通用样式 ================= */
    .stApp { background-color: #f8fafc; }
    .main-title { font-size: 38px; font-weight: 800; color: #1e3a8a; margin-bottom: 8px; letter-spacing: 1px; }
    .sub-title { font-size: 16px; color: #64748b; margin-bottom: 24px; font-weight: 500; }
    .card { background-color: #ffffff; padding: 24px; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.04); margin-bottom: 20px; border: 1px solid #f1f5f9; transition: transform 0.2s ease; }
    .card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.08); }
    .section-title { font-size: 26px; font-weight: 800; color: #0f172a; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
    .metric-box { background: linear-gradient(135deg, #ffffff, #f8fafc); border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.02); }
    .metric-box p { margin: 0; color: #64748b; font-size: 15px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
    .metric-box h2 { margin: 8px 0 0 0; color: #0f172a; font-size: 28px; font-weight: 800; }

    /* ================= 登录页面样式 ================= */
    .login-container { display: flex; align-items: center; justify-content: center; height: 100vh; }
    .login-title { font-size: 42px; font-weight: 900; color: #1e40af; margin-bottom: 12px; text-align: center; letter-spacing: -0.5px; }
    .login-subtitle { font-size: 18px; color: #475569; margin-bottom: 40px; text-align: center; font-weight: 500; }
    .login-box { background-color: white; padding: 40px 30px; border-radius: 24px; box-shadow: 0 10px 40px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }
    .feature-item { margin-bottom: 18px; padding: 18px; border-left: 5px solid #3b82f6; background-color: #ffffff; border-radius: 0 12px 12px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.03); transition: all 0.3s ease;}
    .feature-item:hover { border-left: 5px solid #2563eb; background-color: #f8fafc; }
    .feature-item h4 { margin: 0 0 8px 0; color: #1e40af; font-size: 17px; font-weight: 700; }
    .feature-item p { margin: 0; color: #475569; font-size: 14px; line-height: 1.5; }

    /* ================= 答案与评测态样式 ================= */
    .correct-ans { color: #16a34a; font-weight: 800; padding: 4px 10px; background-color: #dcfce7; border-radius: 6px; }
    .wrong-ans { color: #dc2626; font-weight: 800; padding: 4px 10px; background-color: #fee2e2; border-radius: 6px; }
    .manual-ans { color: #ca8a04; font-weight: 800; padding: 4px 10px; background-color: #fef08a; border-radius: 6px; }
    .path-step { padding: 15px; border-left: 4px solid #8b5cf6; background: #f8fafc; margin-bottom: 10px; border-radius: 0 8px 8px 0; }
    .path-step-num { font-size: 18px; font-weight: bold; color: #8b5cf6; margin-right: 10px; }
    .path-step-title { font-size: 18px; font-weight: bold; color: #1e293b; }
    .file-item { display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #e2e8f0; }
    .file-item:last-child { border-bottom: none; }
    .danger-btn > button { background-color: #fee2e2; color: #dc2626; border: 1px solid #fca5a5; }
    .danger-btn > button:hover { background-color: #fecaca; color: #b91c1c; border: 1px solid #ef4444; }

    /* ================= 🚀 全新旗舰级侧边栏导航 (字体加大+分割线) ================= */
    [data-testid="stSidebar"] { 
        background-color: #0f172a !important; 
        border-right: none !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }

    /* 侧边栏顶部小标题 */
    [data-testid="stSidebar"] h3 {
        color: #94a3b8 !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase;
        margin-top: 15px;
        margin-bottom: 20px;
        padding-left: 12px;
    }

    /* 隐藏原生 Radio 小圆圈 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child { 
        display: none !important; 
    }

    /* 菜单项基础形态 + 分割线设计 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label { 
        padding: 16px 20px; 
        background-color: transparent; 
        border-radius: 0px; 
        border: none;
        /* ✨ 核心修改：增加底部横向分割线 */
        border-bottom: 1px solid rgba(255, 255, 255, 0.08); 
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
        cursor: pointer; 
        margin-bottom: 0px;
    }

    /* 菜单文字颜色与【加大字号】 */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p { 
        font-size: 22px !important; /* ✨ 核心修改：字号从 15px 提升至 18px */
        font-weight: 600 !important; 
        color: #94a3b8 !important;
        margin: 0; 
        transition: all 0.3s ease;
    }

    /* 悬停效果 (Hover) */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover { 
        background-color: rgba(255, 255, 255, 0.05); 
        transform: translateX(4px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.15); /* 悬停时分割线加亮 */
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover p { 
        color: #f1f5f9 !important; 
    }

    /* 极致高亮的选中态 (Active) */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] { 
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); 
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3); 
        border-radius: 8px; /* 选中时变为圆角卡片感 */
        margin: 4px 10px; /* 选中时向内收缩，增加立体感 */
        border-bottom: none; /* 选中态不需要分割线 */
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] p { 
        color: #ffffff !important; 
        font-weight: 800 !important; 
    }

    /* 侧边栏退出按钮 */
    [data-testid="stSidebar"] div.stButton > button { 
        background-color: rgba(239, 68, 68, 0.08); 
        color: #ef4444; 
        border: 1px solid rgba(239, 68, 68, 0.15); 
        border-radius: 12px; 
        font-weight: 600; 
        padding: 12px; 
        margin-top: 30px;
        transition: all 0.3s ease;
    }
    [data-testid="stSidebar"] div.stButton > button:hover { 
        background-color: #ef4444; 
        color: #ffffff; 
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4); 
    }
</style>
""", unsafe_allow_html=True)


# --- 动态数据生成器 ---
def init_learning_history():
    if "learning_history" not in st.session_state:
        history = []
        subjects = ["计算机基础 (408)", "高等数学(微积分)", "考研英语", "Python 编程与RAG实战"]
        actions = ["智能问答", "真题全真模拟", "知识图谱与路径规划", "错题复盘"]
        now = datetime.now()
        for i in range(6, 0, -1):
            past_date = now - timedelta(days=i)
            for _ in range(random.randint(1, 3)):
                history.append({
                    "date": past_date.strftime("%Y-%m-%d"),
                    "time_str": f"{random.randint(9, 22):02d}:{random.randint(10, 50):02d}",
                    "subject": random.choice(subjects),
                    "action": random.choice(actions),
                    "duration": random.choice([25, 30, 45, 60, 90])
                })
        history.append({
            "date": now.strftime("%Y-%m-%d"),
            "time_str": "09:00",
            "subject": "高等数学(微积分)",
            "action": "知识点网络结构拆解",
            "duration": 45
        })
        st.session_state.learning_history = history[::-1]


def add_study_record(subject, action, duration_mins):
    now = datetime.now()
    st.session_state.learning_history.insert(0, {
        "date": now.strftime("%Y-%m-%d"),
        "time_str": now.strftime("%H:%M"),
        "subject": subject,
        "action": action,
        "duration": duration_mins
    })


def navigate_to(page_name):
    st.session_state.page_nav = page_name


# --- 初始化全局状态 ---
init_learning_history()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = [{"role": "assistant",
                                                                             "content": "你好！我是你的 AI 学习助教。您可以提出具体的概念、公式推导请求，或者要求我解析错题。💡 提示：在提问后，您可以使用下方的【快捷指令】进行多轮追问。"}]
if "mistakes" not in st.session_state: st.session_state.mistakes = []
if "quiz_data" not in st.session_state: st.session_state.quiz_data = []
if "quiz_graded" not in st.session_state: st.session_state.quiz_graded = False
if "user_answers" not in st.session_state: st.session_state.user_answers = {}
if "page_nav" not in st.session_state: st.session_state.page_nav = "📊 首页与学习工作台"
if "learning_goal" not in st.session_state: st.session_state.learning_goal = "系统化掌握底层核心原理，冲刺高分"
if "current_course" not in st.session_state: st.session_state.current_course = "考研高等数学 / 计算机体系结构"
if "plan_progress" not in st.session_state: st.session_state.plan_progress = 68

USER_DB = {"admin": "123456", "student": "123456"}


@st.cache_resource
def init_engine():
    if os.path.exists("./chroma_db"):
        return get_rag_chain()
    return None


def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')


# --- 登录页面逻辑 ---
def render_login_page():
    st.markdown('<div class="login-title"> 全栈智学 · AI 多模态教育大模型平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">学习助手系统 | RAG · 多模态 · Agent代理 · 知识图谱 · 流式生成</div>',
                unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)

    col1, col_spacer, col2 = st.columns([1.2, 0.1, 1.8])
    with col1:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.subheader("🔑 身份认证入口")
        st.write("欢迎回来，请验证您的系统管理权限。")
        st.write("")
        with st.form("login_form"):
            username = st.text_input("系统账户 (User ID)")
            password = st.text_input("安全密码 (Password)", type="password")
            submit = st.form_submit_button("🚀 验证并进入系统", use_container_width=True)
            if submit:
                if username in USER_DB and USER_DB[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("权限拒绝：账户或密码效验失败！")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("💎 系统核心六大架构引擎")
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div class="feature-item"><h4>📊 实时可视化工作台</h4><p>实时追踪用户交互行为，生成可视化学习目标、快捷入口与学习状态大盘。</p></div>
            <div class="feature-item"><h4>🤖 智能问答与知识讲解</h4><p>挂载本地向量库，融合通用算力，支持多轮记忆、图示化讲解与课堂一键总结。</p></div>
            <div class="feature-item"><h4>👁️ Qwen-VL 多模态视觉</h4><p>接入先进的视觉语言模型，瞬间识别手写公式、图表架构，实现图像 OCR 答疑。</p></div>
            <div class="feature-item"><h4>📂 动态多模态知识容器</h4><p>支持 PDF/Word/视频 格式注入，支持细粒度文档剔除与向量数据库重载。</p></div>
            <div class="feature-item"><h4>✍️ 动态组卷与自动化批改</h4><p>支持多题型定制与全真模拟结构，集成客观题自动判卷与错因溯源诊断闭环。</p></div>
            <div class="feature-item"><h4>🧠 知识图谱与课程导航</h4><p>提取学科章节目录，渲染高频靶向动态网络拓扑图，并推荐依赖关联的学习路径。</p></div>
        </div>
        """, unsafe_allow_html=True)


# --- 主系统页面逻辑 ---
def render_main_system():
    qa_chain = init_engine()

    st.markdown('<div class="main-title">📘 全栈智学 AI 综合控制台</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-title">👨‍💻 访问节点：{st.session_state.username} ｜ 🌐 核心调度层状态：{"🟢 Chroma 向量库已挂载" if qa_chain else "🔴 本地库未挂载 (需注入语料)"}</div>',
        unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧭 核心模块导航")
        page = st.radio("", [
            "📊 首页与学习工作台",
            "💬 智能问答与知识讲解",
            "📷 多模态拍照答疑",
            "📂 知识库全生命周期管理",
            "✍️ 自动出题与批改",
            "🗓️ LLM 学习规划",
            "📌 AI 错题诊断本",
            "🧠 知识图谱与课程导航"
        ], key="page_nav")

        st.markdown("---")

        if st.button("🚪 安全退出系统", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # 1. 首页与学习工作台模块
    if page == "📊 首页与学习工作台":
        st.markdown('<div class="section-title">📊 学习概览与动态工作台</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e40af, #8b5cf6); padding: 24px; border-radius: 16px; color: white; margin-bottom: 20px; box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);">
            <h3 style="color: white; margin-top: 0; font-size: 24px; font-weight: 800;">🎯 核心学习目标：{st.session_state.learning_goal}</h3>
            <p style="font-size: 18px; margin-bottom: 12px; color: #e2e8f0;">当前聚焦课程：<strong style="color: white;">{st.session_state.current_course}</strong></p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 15px; font-weight: 600;">学习计划总完成度</span>
                <span style="font-size: 18px; font-weight: 800;">{st.session_state.plan_progress}%</span>
            </div>
            <div style="width: 100%; background-color: rgba(255,255,255,0.2); border-radius: 10px; height: 12px; overflow: hidden;">
                <div style="width: {st.session_state.plan_progress}%; background-color: #34d399; height: 100%; border-radius: 10px; transition: width 1s ease-in-out;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚡ 工作台快捷操作")
        col_q1, col_q2, col_q3, col_q4, col_q5, col_q6 = st.columns(6)
        with col_q1:
            st.button("🙋‍♂️ 我要提问", use_container_width=True, on_click=navigate_to, args=("💬 智能问答与知识讲解",))
        with col_q2:
            st.button("📖 知识讲解", use_container_width=True, on_click=navigate_to, args=("💬 智能问答与知识讲解",))
        with col_q3:
            st.button("🗓️ 生成计划", use_container_width=True, on_click=navigate_to, args=("🗓️ LLM 学习规划",))
        with col_q4:
            st.button("📝 出几道题", use_container_width=True, on_click=navigate_to, args=("✍️ 自动出题与批改",))
        with col_q5:
            st.button("🔍 错题分析", use_container_width=True, on_click=navigate_to, args=("📌 AI 错题诊断本",))
        with col_q6:
            st.button("🗺️ 图谱导航", use_container_width=True, on_click=navigate_to, args=("🧠 知识图谱与课程导航",))

        st.markdown("<br>", unsafe_allow_html=True)

        df = pd.DataFrame(st.session_state.learning_history)
        today_str = datetime.now().strftime("%Y-%m-%d")

        today_df = df[df["date"] == today_str]
        today_mins = today_df["duration"].sum() if not today_df.empty else 0
        today_hours = round(today_mins / 60, 1)

        past_7_days = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, -1, -1)]
        seven_days_df = df[df["date"].isin(past_7_days)]
        seven_days_mins = seven_days_df["duration"].sum() if not seven_days_df.empty else 0
        seven_days_hours = round(seven_days_mins / 60, 1)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><p>今日专注学习</p><h2>{today_hours} 小时</h2></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><p>近七日有效沉淀</p><h2>{seven_days_hours} 小时</h2></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><p>本周测试平均分</p><h2>86 分</h2></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(
                f'<div class="metric-box"><p>待复习薄弱错题</p><h2 style="color:#dc2626;">{len(st.session_state.mistakes)} 道</h2></div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns([2, 1])
        with col_chart1:
            st.markdown("#### 📈 过去 7 天学习热力图趋势")
            trend_data = []
            for d in past_7_days:
                day_df = df[df["date"] == d]
                mins = day_df["duration"].sum() if not day_df.empty else 0
                trend_data.append({"日期": d[5:], "时长(小时)": round(mins / 60, 1)})
            trend_df = pd.DataFrame(trend_data).set_index("日期")
            st.bar_chart(trend_df, color="#3b82f6", height=280)

        with col_chart2:
            st.markdown("#### 📚 今日学科重心分布")
            if not today_df.empty:
                subject_df = today_df.groupby("subject")["duration"].sum().reset_index()
                subject_df["时长(小时)"] = (subject_df["duration"] / 60).round(1)
                subject_df = subject_df.set_index("subject")[["时长(小时)"]]
                st.bar_chart(subject_df, color="#10b981", height=280)
            else:
                st.info("今日数据收集中...")

        st.markdown("---")
        st.markdown("#### 📝 最近练习情况与全链路行为追踪")
        display_df = df[["date", "time_str", "subject", "action", "duration"]].copy()
        display_df.columns = ["发生日期", "触发时间", "归属课程/学科", "学习操作/练习行为", "系统分配耗时(min)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 2. 智能问答与知识讲解
    elif page == "💬 智能问答与知识讲解":
        st.markdown('<div class="section-title">💬 智能问答与知识体系讲解中心</div>', unsafe_allow_html=True)
        tab_qa, tab_explain, tab_summary = st.tabs(["💬 自由问答与多轮追问", "📖 体系化知识点讲解", "📝 课堂智能总结"])

        with tab_qa:
            st.info("支持自然语言输入。您可以请求概念解释、公式推导、作业辅助或错题分析。")
            for item in st.session_state.chat_history:
                with st.chat_message(item["role"]):
                    st.write(item["content"])

            st.markdown(
                "<br><p style='color:#64748b; font-size:14px; margin-bottom:5px;'>⚡ 多轮对话快捷追问 (基于上下文)：</p>",
                unsafe_allow_html=True)
            col_qk1, col_qk2, col_qk3, col_qk4 = st.columns(4)
            quick_prompt = None
            if col_qk1.button("💡 再举一个例子",
                              use_container_width=True): quick_prompt = "请针对刚才讲的内容，再举一个与之不同的、贴近生活的具体例子。"
            if col_qk2.button("👶 请讲得更简单一点",
                              use_container_width=True): quick_prompt = "我觉得刚才的解释有点复杂，请用更通俗易懂、没有专业术语的话再讲一遍。"
            if col_qk3.button("🔢 用步骤推导方式解释",
                              use_container_width=True): quick_prompt = "请用严谨的步骤推导方式，一步一步地详细推导和解释刚才的知识点。"
            if col_qk4.button("📝 帮我总结重点",
                              use_container_width=True): quick_prompt = "请帮我把我们刚才讨论的所有核心重点总结提炼出来，用 Markdown 列表呈现。"

            user_input = st.chat_input("向助教提问概念解释、公式推导、错题分析等...")
            actual_input = user_input or quick_prompt

            if actual_input:
                st.session_state.chat_history.append({"role": "user", "content": actual_input})
                with st.chat_message("user"):
                    st.write(actual_input)
                with st.chat_message("assistant"):
                    if qa_chain:
                        try:
                            lc_history = []
                            recent_history = st.session_state.chat_history[1:-1][-6:]
                            for msg in recent_history:
                                if msg["role"] == "user":
                                    lc_history.append(HumanMessage(content=msg["content"]))
                                elif msg["role"] == "assistant":
                                    lc_history.append(AIMessage(content=msg["content"]))
                            chain_input = {"question": actual_input, "chat_history": lc_history}
                            answer = st.write_stream(qa_chain.stream(chain_input))
                            add_study_record("综合自主学习", "执行 RAG 多轮智能对话", 5)
                        except Exception as e:
                            answer = f"❌ 模型响应阻断: {e}"
                            st.write(answer)
                    else:
                        answer = "⚠️ 系统中断：知识库容器为空，请转至【📂 知识库全生命周期管理】上传核心语料。"
                        st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with tab_explain:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(
                "输入您想深入了解的特定知识点，AI 将从定义、原理、图示推导、举例到易错点，为您提供具有纯正教学属性的系统化长篇讲解。")
            explain_topic = st.text_input("📚 需要系统讲解的核心知识点：",
                                          placeholder="例如：极限的定义、支持向量机 (SVM)、动量守恒定律")

            if st.button("🚀 生成系统化知识讲解教案", type="primary"):
                if not explain_topic:
                    st.warning("⚠️ 请先输入要讲解的知识点！")
                else:
                    with st.spinner(f"🧠 名师 AI 正在为您构思【{explain_topic}】的系统级教案..."):
                        try:
                            explain_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                     temperature=0.4)
                            explain_prompt = ChatPromptTemplate.from_messages([
                                ("system", """你是一位拥有20年教学经验的顶级教授。请对用户提出的知识点进行极其系统、严谨、循序渐进的讲解。
                                必须严格包含以下 6 个模块：
                                1. 📌 **定义说明**：严谨但易懂的概念描述。
                                2. ⚙️ **核心原理**：核心思想。
                                3. 📊 **公式推导与图示描述**：如有公式请详细推导。
                                4. 💡 **应用举例**：极具代表性的实战或考研真题级别的例子。
                                5. 🛠️ **解题步骤推导**：标准化解题/应用步骤。
                                6. ⚠️ **易错点总结**：学生最容易踩坑的 2-3 个盲区。"""),
                                ("human", "请为我系统化讲解：{topic}")
                            ])
                            explain_chain = explain_prompt | explain_llm | StrOutputParser()

                            st.markdown("---")
                            st.success(f"✅ 【{explain_topic}】系统化讲解生成完毕：")
                            res_box = st.empty()
                            full_explanation = ""
                            for chunk in explain_chain.stream({"topic": explain_topic}):
                                full_explanation += chunk
                                res_box.markdown(full_explanation + "▌")
                            res_box.markdown(full_explanation)
                            add_study_record("体系化学习", f"知识点深度讲解: {explain_topic}", 25)
                        except Exception as e:
                            st.error(f"❌ 讲解生成失败: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_summary:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(
                "点击下方按钮，AI 将提取您在本系统中刚才进行的全部聊天问答对话（Memory），自动生成专属的“课堂总结笔记”。")

            if st.button("📑 一键生成今日课堂总结笔记", type="primary"):
                if len(st.session_state.chat_history) <= 2:
                    st.warning("⚠️ 当前对话历史太少，请先去【自由问答】模块与 AI 讨论一些知识点，再来生成总结吧！")
                else:
                    with st.spinner("🤖 正在回溯您的所有对话历史并提炼核心价值..."):
                        try:
                            history_text = "\n".join(
                                [f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[1:]])
                            summary_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                     temperature=0.3)
                            summary_prompt = ChatPromptTemplate.from_messages([
                                ("system", """你是一位严谨的学管师。请根据用户提供的【历史对话记录】，自动生成一份结构化的课堂总结。
                                必须严格包含：1. 🎯 本节核心概念 2. 🔑 重点方法与推导 3. 💣 常见误区/盲区 4. 📅 课后巩固建议。"""),
                                ("human", "【历史对话记录】：\n{history}")
                            ])
                            summary_chain = summary_prompt | summary_llm | StrOutputParser()

                            st.markdown("---")
                            st.success("✅ 课堂总结生成完毕：")
                            res_box = st.empty()
                            full_summary = ""
                            for chunk in summary_chain.stream({"history": history_text[:8000]}):
                                full_summary += chunk
                                res_box.markdown(full_summary + "▌")
                            res_box.markdown(full_summary)
                            add_study_record("学情复盘", "生成全局课堂总结", 10)
                        except Exception as e:
                            st.error(f"❌ 总结生成失败: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # 3. 拍照答疑
    elif page == "📷 多模态拍照答疑":
        st.markdown('<div class="section-title">📷 Qwen-VL 视觉智能解析</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["📤 本地图像注入", "📸 实时捕获解析"])
        img_file = None
        with tab1:
            img_file_upload = st.file_uploader("支持摄入复杂图表、手写公式等非结构化图像格式 (PNG/JPG)",
                                               type=["png", "jpg", "jpeg"])
            if img_file_upload: img_file = img_file_upload
        with tab2:
            img_file_camera = st.camera_input("使用终端设备镜头进行数据捕获")
            if img_file_camera: img_file = img_file_camera
        if img_file:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col_img, col_form = st.columns([1, 1.5])
                with col_img:
                    st.image(img_file, caption="待解构输入源", use_column_width=True)
                with col_form:
                    user_question = st.text_area("请设定您的视觉处理指令 (Prompt)：",
                                                 placeholder="例如：详细解答这道题，并给出推导过程。")
                    if st.button("✨ 唤醒多模态解析算力", type="primary", use_container_width=True):
                        if not user_question:
                            st.warning("⚠️ 指令不能为空！")
                        else:
                            with st.spinner("👀 跨模态处理通道已建立，视觉张量提取中..."):
                                try:
                                    base64_image = encode_image(img_file)
                                    from langchain_openai import ChatOpenAI
                                    vision_llm = ChatOpenAI(model="qwen-vl-max", api_key=ALIYUN_API_KEY,
                                                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                            temperature=0.2)
                                    message = HumanMessage(content=[{"type": "text", "text": user_question},
                                                                    {"type": "image_url", "image_url": {
                                                                        "url": f"data:image/jpeg;base64,{base64_image}"}}])
                                    st.success("✅ 图像解析穿透完成，结果回传如下：")
                                    res_box = st.empty()
                                    full_reply = ""
                                    for chunk in vision_llm.stream([message]):
                                        full_reply += chunk.content
                                        res_box.markdown(full_reply + "▌")
                                    res_box.markdown(full_reply)
                                    add_study_record("图像非结构化提取", "多模态大模型视觉解构", 15)
                                except Exception as e:
                                    st.error(f"❌ 视觉模块连接失败: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

    # 4. 知识库全生命周期管理
    elif page == "📂 知识库全生命周期管理":
        st.markdown('<div class="section-title">📂 多模态语料摄入与向量池管理</div>', unsafe_allow_html=True)

        tab_upload, tab_list, tab_manage = st.tabs(["⬆️ 语料上传与热更新", "📋 挂载文件精准管理", "☢️ 危险操作区"])

        with tab_upload:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.info(
                "💡 引擎已升级：当前支持注入 **PDF讲义、Word文档(.docx)、纯文本(.txt) 以及 视频文件(.mp4)** 格式的内容。")
            uploaded_file = st.file_uploader("拖拽学术文件或多媒体至此区域进行自动切块与 Embedding",
                                             type=["pdf", "docx", "txt", "mp4", "avi"])

            if uploaded_file is not None:
                if st.button("🚀 启动语义入库与系统无感热重载", type="primary"):
                    with st.spinner("🔄 数据清洗、切块处理及大模型向量转化运算中..."):
                        try:
                            upload_dir = "uploads"
                            if not os.path.exists(upload_dir): os.makedirs(upload_dir)
                            file_path = os.path.join(upload_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            build_local_vector_db(file_path)
                            st.cache_resource.clear()

                            st.success(
                                f"✅ 多模态数据包 【{uploaded_file.name}】 已成功灌入 Chroma 本地池，RAG 引擎热重载完毕！")
                            st.balloons()
                            add_study_record("系统底座构建", f"挂载新语料: {uploaded_file.name}", 5)
                        except Exception as e:
                            st.error(f"❌ 向量管道写入中断：{e}")
            st.markdown('</div>', unsafe_allow_html=True)

        # 定点精准删除文件列表
        with tab_list:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📋 本地专属知识库文件清单")
            st.write("您可以从这里精确定位并删除不需要的语料文件，系统将同步抹除向量空间中对应的切片特征。")

            upload_dir = "uploads"
            if os.path.exists(upload_dir):
                files = os.listdir(upload_dir)
                if files:
                    for f in files:
                        st.markdown('<div class="file-item">', unsafe_allow_html=True)
                        col_f1, col_f2 = st.columns([5, 1])
                        with col_f1:
                            icon = "📄"
                            if f.endswith(".mp4") or f.endswith(".avi"):
                                icon = "🎬"
                            elif f.endswith(".docx"):
                                icon = "📝"
                            st.markdown(f"{icon} **{f}**")
                        with col_f2:
                            if st.button("🗑️ 剥离", key=f"del_{f}"):
                                with st.spinner("正在从 Chroma 向量数据库中剥离切片特征..."):
                                    delete_single_file(f)
                                    st.success(f"已成功删除 {f}")
                                    time.sleep(1)
                                    st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("📦 当前知识库为空，请前往上传界面注入语料。")
            else:
                st.info("📦 当前知识库为空，请前往上传界面注入语料。")
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_manage:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ 危险操作区")
            st.write(
                "执行此操作将**物理抹除**系统本地的 ChromaDB 向量数据库文件夹，并重置 RAG 检索对话引擎的所有挂载记忆。此操作不可逆，请谨慎操作。")

            confirm_delete = st.checkbox("我确认要销毁当前所有的本地专属知识语料。")

            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("☢️ 格式化并清空本地知识库"):
                if not confirm_delete:
                    st.warning("⚠️ 请先勾选上方的确认框。")
                else:
                    with st.spinner("正在执行底层磁盘物理擦除..."):
                        try:
                            clear_vector_db()
                            st.cache_resource.clear()

                            st.success("✅ 物理擦除指令执行完毕！本地知识库已归零，RAG 引擎恢复出厂状态。")
                            add_study_record("系统管理员操作", "全局格式化知识数据库", 1)
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 擦除失败，权限受限或文件被占用: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


    # 5. 自动出题与批改
    elif page == "✍️ 自动出题与批改":
        st.markdown('<div class="section-title">✍️ 结构化组卷与智能自动化批改</div>', unsafe_allow_html=True)
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                quiz_subject = st.selectbox("学科领域", ["计算机考研大纲 (408)", "考研高等数学(微积分)", "大学物理",
                                                         "编程算法体系"])
            with c2:
                quiz_topic = st.text_input("靶向知识边界", placeholder="例：函数的连续性与间断点、微积分基本定理")
            with c3:
                quiz_diff = st.selectbox("难度阈值设定",
                                         ["A级 - 基础概念扫盲", "B级 - 核心原理穿透", "S级 - 综合压轴突破"])
            st.markdown("---")
            quiz_mode = st.radio("组卷策略选择", ["📝 细粒度专项练习", "🎓 高仿真全真模拟卷"], horizontal=True)

            total_count = 0
            if quiz_mode == "📝 细粒度专项练习":
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1:
                    count_choice = st.number_input("单项选择题", min_value=0, max_value=15, value=3)
                with col_q2:
                    count_fill = st.number_input("填空题", min_value=0, max_value=10, value=1)
                with col_q3:
                    count_essay = st.number_input("解答题(推导题)", min_value=0, max_value=5, value=1)
                with col_q4:
                    count_code = st.number_input("算法实现", min_value=0, max_value=5, value=0)

                total_count = count_choice + count_fill + count_essay + count_code
                quantity_instruction = f"生成约束：严格生成 {total_count} 题。分布：选择 {count_choice} 题，填空 {count_fill} 题，解答 {count_essay} 题，编程 {count_code} 题。\n"
            else:
                total_count = 1
                quantity_instruction = "组卷规则设定为【综合全真比例】。要求总题量处于 8-12 题的高效测试区间。"

            if st.button("📝 执行组卷引擎调度", type="primary"):
                if not quiz_topic:
                    st.warning("⚠️ 请求拦截：请提供靶向知识点。")
                elif total_count == 0:
                    st.warning("⚠️ 请求拦截：题量参数不能全为空。")
                else:
                    with st.spinner("🤖 正在执行题目检索与组装重构..."):
                        try:
                            st.session_state.quiz_data = []
                            st.session_state.user_answers = {}
                            st.session_state.quiz_graded = False

                            quiz_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                  temperature=0.3)
                            quiz_prompt = ChatPromptTemplate.from_messages([
                                ("system", """你担任架构级教育测验调度中枢。请依据请求生成测试数据流。
                                {quantity_instruction}
                                输出协议：返回格式必须是绝对纯净的 JSON 数组，无任何代码块前缀。
                                键值规范：[ {{"type": "选择题", "question": "问题", "options": ["A. x", "B. y", "C. z", "D. w"], "answer": "A", "analysis": "解析"}} ]"""),
                                ("human", "领域：{subject}\n锚点：{topic}\n难度：{diff}")
                            ])
                            quiz_chain = quiz_prompt | quiz_llm | StrOutputParser()
                            raw_quiz = quiz_chain.invoke(
                                {"subject": quiz_subject, "topic": quiz_topic, "diff": quiz_diff,
                                 "quantity_instruction": quantity_instruction})

                            st.session_state.quiz_data = json.loads(
                                raw_quiz.replace("```json", "").replace("```", "").strip())
                            st.success(f"✅ 数据包回传成功。载入有效试题 {len(st.session_state.quiz_data)} 项。")
                            add_study_record(quiz_subject, "执行结构化系统智能测评", 45)
                        except Exception as e:
                            st.error(f"❌ JSON 序列化崩溃: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.quiz_data:
            st.markdown("### 📝 前端测评沙盒")
            with st.form("quiz_paper"):
                for i, q in enumerate(st.session_state.quiz_data):
                    q_type = q.get('type', '选择题')
                    st.markdown(f"**第 {i + 1} 节点 【{q_type}】**：{q['question']}")
                    if q_type == "选择题":
                        user_ans = st.radio("选项：", q.get('options', ['A', 'B', 'C', 'D']), key=f"q_{i}",
                                            disabled=st.session_state.quiz_graded)
                        st.session_state.user_answers[i] = user_ans[0] if user_ans else ""
                    elif q_type == "填空题":
                        st.session_state.user_answers[i] = st.text_input("填空：", key=f"q_{i}",
                                                                         disabled=st.session_state.quiz_graded)
                    else:
                        st.session_state.user_answers[i] = st.text_area("详细解答：", key=f"q_{i}", height=150,
                                                                        disabled=st.session_state.quiz_graded)
                    st.markdown("---")

                if not st.session_state.quiz_graded:
                    if st.form_submit_button("✅ 提交沙盒数据并执行批改算法", type="primary"):
                        st.session_state.quiz_graded = True
                        st.rerun()
                else:
                    st.form_submit_button("🔒 数据已封存", disabled=True)

            if st.session_state.quiz_graded:
                st.markdown("### 📊 自动化判卷流转结果")
                score = 0;
                obj_total = 0
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    q_type = q.get('type', '选择题')
                    correct_ans = str(q['answer']).strip()
                    user_ans = str(st.session_state.user_answers.get(i, "未反馈")).strip()

                    if q_type in ["选择题", "填空题"]:
                        obj_total += 1
                        is_correct = (correct_ans.upper() == user_ans.upper()) if q_type == "选择题" else (
                                correct_ans in user_ans)
                        if is_correct:
                            score += 1
                            st.markdown(f"**节点 {i + 1}**：<span class='correct-ans'>✅ 逻辑命中</span>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"**节点 {i + 1}**：<span class='wrong-ans'>❌ 逻辑偏离</span>",
                                        unsafe_allow_html=True)
                            st.markdown(f"传入：{user_ans} | 基准：**{correct_ans}**")
                            if not any(m['title'] == q['question'] for m in st.session_state.mistakes):
                                st.session_state.mistakes.insert(0, {"title": q['question'], "subject": "系统测评",
                                                                     "reason": "节点验证失败", "user_answer": user_ans,
                                                                     "correct_answer": correct_ans, "status": "待诊断",
                                                                     "ai_analysis": q['analysis']})
                    else:
                        st.markdown(f"**节点 {i + 1}**：<span class='manual-ans'>⚠️ 需人类核验</span>",
                                    unsafe_allow_html=True)
                        st.markdown(f"**传入**：\n```text\n{user_ans}\n```\n**基准**：\n{correct_ans}")
                    st.markdown(f"**💡 AI 深度分析**：\n{q['analysis']}")
                    st.markdown("</div>", unsafe_allow_html=True)

    # 6. LLM 学习规划
    elif page == "🗓️ LLM 学习规划":
        st.markdown('<div class="section-title">🗓️ 目标导向型学习路径生成器</div>', unsafe_allow_html=True)
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                goal = st.text_input("🎯 设定学习锚点", placeholder="比如：考研数学微积分全通")
                subject = st.selectbox("📚 专业方向", ["考研高等数学", "考研专业课(408)", "考研英语"])
            with col2:
                time_per_day = st.slider("⏱️ 每日资源分配 (时)", 1, 8, 2)
                style = st.selectbox("💡 执行流派设定", ["渐进式底层铺垫", "高并发实战刷题", "考前魔鬼冲刺压缩包"])

            if st.button("🚀 呼叫 AI 规划算力", type="primary"):
                if not goal:
                    st.warning("⚠️ 空指针：目标域不可为空！")
                else:
                    with st.spinner("🧠 Agent 代理正在推演时间轴与学习切片..."):
                        try:
                            planner_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                                     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                     temperature=0.5)
                            plan_prompt = ChatPromptTemplate.from_messages([
                                ("system", "你作为系统的最高教育架构师。请输出专业 Markdown 学习计划。"),
                                ("human",
                                 "核心赛道：{subject}\n里程碑：{goal}\n资源额度：日均 {time_per_day} 小时\n推进策略：{style}\n要求包含：1. 破局点分析 2. 时间粒度表 3. 避坑指南")
                            ])
                            plan_chain = plan_prompt | planner_llm | StrOutputParser()
                            st.success("✅ 规划蓝图已出炉！")
                            st.markdown("---")
                            res_box = st.empty()
                            full_plan = ""
                            for chunk in plan_chain.stream(
                                    {"subject": subject, "goal": goal, "time_per_day": time_per_day, "style": style}):
                                full_plan += chunk
                                res_box.markdown(full_plan + "▌")
                            res_box.markdown(full_plan)
                            add_study_record(subject, "请求智能学习规划推演", 10)
                        except Exception as e:
                            st.error(f"❌ 链路挂起: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    # 7. AI 错题诊断本
    elif page == "📌 AI 错题诊断本":
        st.markdown('<div class="section-title">📌 智能靶向诊断与错因溯源池</div>', unsafe_allow_html=True)
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        tab1, tab2 = st.tabs(["📚 待处理异常记录池", "➕ 手工录入异常数据"])

        with tab1:
            if not st.session_state.mistakes:
                st.info("🎉 暂无告警记录，系统运转一切正常！")
            else:
                for i, m in enumerate(st.session_state.mistakes):
                    with st.expander(f"[{m['subject']}] {m['title']} - 节点状态: {m['status']}", expanded=(i == 0)):
                        st.markdown(f"**❌ 拦截传入：** {m.get('user_answer', 'Null')}")
                        st.markdown(f"**✅ 系统基准：** {m.get('correct_answer', 'Null')}")
                        if "ai_analysis" in m:
                            st.success("🤖 深度下钻报告：")
                            st.markdown(m["ai_analysis"])
                        else:
                            if st.button("🤖 发起 AI 深度溯源与变形推演", key=f"analyze_btn_{i}"):
                                with st.spinner("AI 正在重构关联..."):
                                    try:
                                        diagnostic_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                                                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                                                    temperature=0.4)
                                        diag_prompt = ChatPromptTemplate.from_messages([
                                            ("system",
                                             "你现在是底层逻辑推演系统。指出逻辑崩塌根源，并出一道相似变形题(附提示)。"),
                                            ("human",
                                             "学科：{subject}\n题目：{title}\n触发输入：{user_answer}\n正确值：{correct_answer}\n请执行溯源：")
                                        ])
                                        diag_chain = diag_prompt | diagnostic_llm | StrOutputParser()
                                        analysis_result = st.write_stream(diag_chain.stream(
                                            {"subject": m['subject'], "title": m['title'],
                                             "user_answer": m.get('user_answer', '无'),
                                             "correct_answer": m.get('correct_answer', '无')}))
                                        st.session_state.mistakes[i]["ai_analysis"] = analysis_result
                                        st.session_state.mistakes[i]["status"] = "诊断完毕"
                                        add_study_record(m['subject'], "发起 AI 靶向错题修复", 20)
                                    except Exception as e:
                                        st.error(f"❌ 诊断断开: {e}")

        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            with st.form("add_mistake_form", clear_on_submit=True):
                m_title = st.text_input("捕获点标识", placeholder="输入错误实体特征")
                m_subject = st.selectbox("学科大类", ["考研数学(微积分)", "系统架构层", "英语解析层", "基础编程层"])
                m_user_ans = st.text_area("当时产生的污染数据：")
                m_correct_ans = st.text_area("官方文档验证基准：")
                if st.form_submit_button("💾 commit 提单", type="primary") and m_title:
                    st.session_state.mistakes.insert(0, {"title": m_title, "subject": m_subject, "reason": "人工挂起",
                                                         "user_answer": m_user_ans, "correct_answer": m_correct_ans,
                                                         "status": "待测"})
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # 8. 知识图谱与课程导航
    elif page == "🧠 知识图谱与课程导航":
        st.markdown('<div class="section-title">🧠 知识图谱与多维课程导航引擎</div>', unsafe_allow_html=True)
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input("🔍 设定需要拆解的课程或综合知识点：",
                                  placeholder="例如：考研高等数学(微积分)、计算机体系结构")
        with col2:
            st.write("<br>", unsafe_allow_html=True)
            generate_btn = st.button("🚀 唤醒全局解析管线", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

        if generate_btn:
            if not topic:
                st.warning("⚠️ 拦截请求：知识簇参数为空！")
            else:
                with st.spinner(f"⚙️ 系统正在深度解构【{topic}】体系边界，约 30 秒..."):
                    try:
                        nav_llm = ChatOpenAI(model="qwen-max", api_key=ALIYUN_API_KEY,
                                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                                             temperature=0.1)
                        nav_prompt = ChatPromptTemplate.from_messages([
                            ("system", """你担任系统的顶级课程总监与知识图谱架构师。
                            输出 JSON 要求(勿包含 Markdown 代码块)：
                            {{
                                "course_chapters": [{{"chapter": "章名", "desc": "简介"}}],
                                "nodes": [{{"id": "节点名", "attribute": "高频考点|易错点|先修知识点|常规节点"}}],
                                "edges": [["节点A", "节点B"]],
                                "learning_path": [{{"step": 1, "topic": "节点", "reason": "原因"}}],
                                "summary": "全局深度总结"
                            }}"""),
                            ("human", "解构目标：{topic}")
                        ])
                        nav_chain = nav_prompt | nav_llm | StrOutputParser()
                        nav_data = json.loads(
                            nav_chain.invoke({"topic": topic}).replace("```json", "").replace("```", "").strip())
                        st.success(f"✅ 【{topic}】全维度课程导航矩阵生成完毕！")

                        tab_nav_ch, tab_nav_graph, tab_nav_path = st.tabs(
                            ["📖 课程宏观章节导航", "🧠 动态靶向知识图谱", "🗺️ 依赖路径智能推荐"])

                        with tab_nav_ch:
                            for ch in nav_data.get("course_chapters", []):
                                with st.expander(f"📌 {ch.get('chapter', '未知章节')}"): st.write(ch.get('desc', ''))

                        with tab_nav_graph:
                            nodes_info = nav_data.get("nodes", [])
                            edges_info = nav_data.get("edges", [])
                            if nodes_info and edges_info:
                                G = nx.DiGraph()
                                color_map = {}
                                for n in nodes_info:
                                    n_id = n.get("id");
                                    attr = n.get("attribute", "")
                                    G.add_node(n_id)
                                    color_map[
                                        n_id] = "#ef4444" if "高频" in attr or "考点" in attr else "#f59e0b" if "易错" in attr or "盲区" in attr else "#10b981" if "先修" in attr or "基础" in attr else "#3b82f6"
                                G.add_edges_from(edges_info)
                                fig, ax = plt.subplots(figsize=(12, 7))
                                nx.draw_networkx_nodes(G, pos=nx.spring_layout(G, k=0.9, seed=42), node_size=2000,
                                                       node_color=[color_map.get(n, "#3b82f6") for n in G.nodes()],
                                                       edgecolors="#e2e8f0", linewidths=1.5, ax=ax)
                                nx.draw_networkx_edges(G, pos=nx.spring_layout(G, k=0.9, seed=42), ax=ax, arrows=True,
                                                       edge_color="#cbd5e1", width=1.5, arrowsize=18,
                                                       connectionstyle='arc3, rad = 0.15')
                                nx.draw_networkx_labels(G, pos=nx.spring_layout(G, k=0.9, seed=42), font_size=11,
                                                        font_weight="bold", font_family="SimHei", font_color="white",
                                                        ax=ax)
                                ax.legend(handles=[mpatches.Patch(color='#ef4444', label='高频重点考点'),
                                                   mpatches.Patch(color='#f59e0b', label='常见易错盲区'),
                                                   mpatches.Patch(color='#10b981', label='底层先修知识'),
                                                   mpatches.Patch(color='#3b82f6', label='体系常规节点')],
                                          loc='upper right', prop={'family': 'SimHei', 'size': 10})
                                ax.axis("off");
                                st.pyplot(fig)

                        with tab_nav_path:
                            for item in nav_data.get("learning_path", []):
                                st.markdown(
                                    f'<div class="path-step"><span class="path-step-num">Step {item.get("step")}</span><span class="path-step-title">{item.get("topic")}</span><p style="margin: 8px 0 0 0; color: #475569; font-size: 14px;"><strong>执行理由</strong>：{item.get("reason")}</p></div>',
                                    unsafe_allow_html=True)
                            st.markdown("---");
                            st.markdown(f"### 📖 架构师总结批注");
                            st.markdown('<div class="card">', unsafe_allow_html=True);
                            st.markdown(nav_data.get("summary", ""));
                            st.markdown('</div>', unsafe_allow_html=True)
                            add_study_record("课程架构导航", f"执行图谱解构: {topic}", 20)
                    except Exception as e:
                        st.error(f"❌ 渲染引擎错误: {e}")


if not st.session_state.logged_in:
    render_login_page()
else:
    render_main_system()