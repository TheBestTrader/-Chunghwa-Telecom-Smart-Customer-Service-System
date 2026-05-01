from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tools import (
    check_network_status,
    create_repair_ticket,
    query_user_plan,
    recommend_5g_plan,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # set by router_node; read by route_intent


_ROUTER_BROADCASTS = {
    "tech": (
        "🚦 [Router 總機] 偵測到您的問題與「網路設備或故障排除」相關，"
        "正在為您轉接給 🛠️ Tech Agent (技術部)..."
    ),
    "sales": (
        "🚦 [Router 總機] 偵測到您的需求與「資費升級、帳單或合約」相關，"
        "正在為您轉接給 💰 Sales Agent (業務部)..."
    ),
    "general": "🚦 [Router 總機] 正在為您轉接給 🛡️ General Agent (一般客服)...",
}

_TECH_TOOLS = [check_network_status, create_repair_ticket]
_SALES_TOOLS = [query_user_plan, recommend_5g_plan]


def get_llm(api_key: str, model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


def _strip_broadcasts(messages: list) -> list:
    """Remove router broadcast AIMessages so they don't pollute agent context."""
    broadcast_texts = set(_ROUTER_BROADCASTS.values())
    return [
        m for m in messages
        if not (isinstance(m, AIMessage) and m.content in broadcast_texts)
    ]


def build_graph(api_key: str, model_name: str):
    llm = get_llm(api_key, model_name)

    # ── Nodes ────────────────────────────────────────────────────────────────

    def router_node(state: AgentState) -> dict:
        last_content = state["messages"][-1].content
        prompt = (
            "你是一個電信客服路由系統。請判斷以下用戶訊息屬於哪個類別，"
            "只回覆一個英文單字，絕對不得包含其他文字、標點或說明。\n\n"
            "分類規則：\n"
            "- 回覆 'tech'：問題與網路斷線、連線速度慢、設備異常、亮紅燈、無法上網、排障有關。\n"
            "- 回覆 'sales'：問題與方案升級、續約、攜碼、資費查詢、帳單費用、金額異常、帳單暴增有關。"
            "只要涉及費用或帳單，即使同時提到網路問題，仍優先回覆 'sales'。\n"
            "- 回覆 'general'：問題與電信業務完全無關（例如：訂車票、問天氣、一般閒聊）。\n\n"
            f"用戶訊息：{last_content}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip().lower().strip("'\"")
        if intent not in ("tech", "sales", "general"):
            intent = "general"
        broadcast = AIMessage(content=_ROUTER_BROADCASTS[intent])
        return {"messages": [broadcast], "intent": intent}

    def tech_node(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "你是中華電信技術客服 (Tech Agent)，負責處理斷線、設備異常。"
            "請先使用工具查詢網路狀態或建立維修工單，再根據查詢結果以專業簡潔的語氣協助用戶排障。"
            "若用戶未提供地點，請以「信義區」作為預設地點。若用戶未提供 user_id，請以「U001」作為預設值。"
            "回覆請使用繁體中文。"
        ))
        llm_with_tools = llm.bind_tools(_TECH_TOOLS)
        response = llm_with_tools.invoke([system] + _strip_broadcasts(state["messages"]))
        return {"messages": [response]}

    def sales_node(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "你是中華電信業務客服 (Sales Agent)，負責處理資費升級、續約。"
            "請先使用工具查詢用戶目前方案，再推薦適合的升級方案，以親切的語氣說明優惠內容。"
            "若用戶未提供 user_id，請以「U001」作為預設值。"
            "回覆請使用繁體中文。"
        ))
        llm_with_tools = llm.bind_tools(_SALES_TOOLS)
        response = llm_with_tools.invoke([system] + _strip_broadcasts(state["messages"]))
        return {"messages": [response]}

    def general_node(state: AgentState) -> dict:
        system = SystemMessage(content=(
            "你是中華電信智能客服，負責回答各類一般性問題。"
            "請以繁體中文、親切專業的語氣回應。"
        ))
        response = llm.invoke([system] + _strip_broadcasts(state["messages"]))
        return {"messages": [response]}

    # ── Routing functions ─────────────────────────────────────────────────────

    def route_intent(state: AgentState) -> str:
        return state["intent"]

    def tech_should_use_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tech_tools"
        return END

    def sales_should_use_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "sales_tools"
        return END

    # ── Graph assembly ────────────────────────────────────────────────────────

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("tech", tech_node)
    graph.add_node("sales", sales_node)
    graph.add_node("general", general_node)
    graph.add_node("tech_tools", ToolNode(_TECH_TOOLS))
    graph.add_node("sales_tools", ToolNode(_SALES_TOOLS))

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_intent,
        {"tech": "tech", "sales": "sales", "general": "general"},
    )

    # Tech agent loop: tech → (tool_calls?) → tech_tools → tech → END
    graph.add_conditional_edges(
        "tech",
        tech_should_use_tools,
        {"tech_tools": "tech_tools", END: END},
    )
    graph.add_edge("tech_tools", "tech")

    # Sales agent loop: sales → (tool_calls?) → sales_tools → sales → END
    graph.add_conditional_edges(
        "sales",
        sales_should_use_tools,
        {"sales_tools": "sales_tools", END: END},
    )
    graph.add_edge("sales_tools", "sales")

    graph.add_edge("general", END)

    return graph.compile()
