import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agent_workflow import build_graph

# Resolve .env relative to this script so it works regardless of the
# directory Streamlit is launched from.
load_dotenv(Path(__file__).parent / ".env")

AVAILABLE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]
DEFAULT_MODEL_INDEX = 0  # gemini-2.5-flash-lite


# ---------------------------------------------------------------------------
# LLM interface — Phase 3: LangGraph multi-agent with Function Calling
# ---------------------------------------------------------------------------

def run_agent_graph(history: list, model: str, dev_mode: bool, api_key: str) -> list[str]:
    if dev_mode:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        return [
            f"**[開發測試模式]** 收到訊息：「{last_user}」\n\n"
            "（未呼叫 API，此為模擬回覆。關閉開發測試模式後將使用真實模型。）"
        ]

    if not api_key:
        return ["⚠️ 找不到有效的 API Key，請在 .env 設定或於側邊欄輸入自訂金鑰。"]

    try:
        lc_messages = []
        for m in history:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        graph = build_graph(api_key, model)
        result = graph.invoke({"messages": lc_messages, "intent": ""})

        msgs = result["messages"]
        # Only search for the broadcast in messages added THIS turn (after the
        # last HumanMessage). Searching from the start would return a stale
        # broadcast from a previous turn stored in the conversation history.
        last_human_idx = next(
            (len(msgs) - 1 - i for i, m in enumerate(reversed(msgs)) if isinstance(m, HumanMessage)),
            -1,
        )
        current_turn_msgs = msgs[last_human_idx + 1:]
        broadcast = next(
            (m for m in current_turn_msgs if isinstance(m, AIMessage) and m.content.startswith("🚦")),
            None,
        )
        final = msgs[-1].content
        if broadcast and broadcast.content != final:
            return [broadcast.content, final]
        return [final]
    except Exception as e:
        return [f"⚠️ Agent 執行發生錯誤：{e}"]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="中華電信智能客服系統",
    page_icon="📡",
    layout="wide",
)

st.title("📡 中華電信智能客服系統 (Multi-Agent PoC)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── API Key 來源 ──────────────────────────────────────────────────────────
    st.header("🔑 API Key 來源")
    key_source = st.radio(
        "選擇金鑰來源",
        options=["系統預設 (.env)", "自訂輸入 (BYOK)"],
        label_visibility="collapsed",
    )
    if key_source == "自訂輸入 (BYOK)":
        custom_api_key = st.text_input("請輸入您的 Gemini API Key", type="password")
    else:
        custom_api_key = ""

    # Determine the active key; show a warning when BYOK is chosen but empty.
    if key_source == "系統預設 (.env)":
        active_api_key = os.getenv("GEMINI_API_KEY", "")
    else:
        active_api_key = custom_api_key

    st.divider()
    st.header("🤖 系統狀態")
    st.info(
        "目前系統運行 1 個路由中樞與 3 個專業 Agent：\n\n"
        "- 🚦 **Router (總機)** — 負責意圖判斷與精準派單\n"
        "- 🛠️ **Tech (維修)** — 專職網路與設備故障排除\n"
        "- 💰 **Sales (業務)** — 處理方案升級與帳單諮詢\n"
        "- 🛡️ **General (客服)** — 處理非業務範圍之閒聊與邊界防護"
    )
    st.caption("⚠️ 備註：若遇到 429 錯誤，請從下方切換其他模型，或開啟開發測試模式。")

    st.divider()
    dev_mode = st.checkbox("🧪 開發測試模式（不呼叫 API）", value=False)
    selected_model = st.selectbox(
        "🧠 選擇 AI 運算模型",
        options=AVAILABLE_MODELS,
        index=DEFAULT_MODEL_INDEX,
        disabled=dev_mode,
    )

    st.divider()
    st.subheader("💬 情境模擬")

    if st.button("📍 模擬：信義區網路斷線", use_container_width=True):
        st.session_state.prefill = "我住在信義區，網路突然斷線了，已經斷了快一個小時，請問怎麼辦？"

    if st.button("📶 模擬：我想了解 5G 升級方案", use_container_width=True):
        st.session_state.prefill = "我目前是 4G 499 吃到飽的用戶，想了解升級 5G 吃到飽方案的費用和優惠。"

    if st.button("💰 模擬：這期帳單費用異常", use_container_width=True):
        st.session_state.prefill = "我這個月的電信帳單怎麼暴增了 500 塊？可以幫我查一下嗎？"

    if st.button("🤖 模擬：與業務無關之閒聊", use_container_width=True):
        st.session_state.prefill = "請問今天台北的天氣如何？還有幫我訂一張去台中的高鐵票。"

    if st.button("🔥 模擬：混合意圖 (網路慢+想換約)", use_container_width=True):
        st.session_state.prefill = "我家網路最近晚上都好慢，一直轉圈圈。我是不是該直接升級成 5G 吃到飽比較快？你們現在有什麼優惠？"

    st.divider()
    st.subheader("🧪 Phase 3 實戰測試")

    if st.button("🛠️ 實戰：斷線排障與自動派工", use_container_width=True):
        st.session_state.prefill = "我住在信義區，網路突然斷線了，已經斷了快一個小時，請問怎麼辦？"

    if st.button("📊 實戰：查詢合約與 5G 升級", use_container_width=True):
        st.session_state.prefill = "我的身分證是 A123456789，可以幫我查一下我現在的合約嗎？還有我想升級 5G 有推薦的嗎？"

    st.divider()
    st.caption(f"Phase 3 PoC — LangGraph Multi-Agent + Function Calling\n模型：{'開發測試模式' if dev_mode else selected_model}")

    if not dev_mode and key_source == "自訂輸入 (BYOK)" and not active_api_key:
        st.warning("請輸入有效的 API Key 才能呼叫真實模型。")

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "您好！我是中華電信智能客服，請問有什麼可以協助您的？",
        }
    ]

if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle sidebar simulation buttons ─────────────────────────────────────────
# Buttons set prefill; we move it to pending_input and rerun so the chat input
# renders with the correct default before the user can interact.
if st.session_state.prefill:
    st.session_state.pending_input = st.session_state.pop("prefill")
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("請輸入您的問題…")

# Consume any pending input injected by the simulation buttons.
if not user_input and "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Agent 處理中…"):
        replies = run_agent_graph(st.session_state.messages, selected_model, dev_mode, active_api_key)

    for reply in replies:
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
