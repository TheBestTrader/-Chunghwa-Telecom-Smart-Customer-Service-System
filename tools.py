from langchain_core.tools import tool


@tool
def check_network_status(location: str) -> dict:
    """查詢指定地點的網路機房狀態與障礙資訊。"""
    return {
        "location": location,
        "機房狀態": "異常",
        "影響範圍": f"{location}周邊 500 公尺",
        "預計恢復時間": "2 小時內",
        "事件代碼": "INC-20240501-007",
    }


@tool
def create_repair_ticket(user_id: str, location: str, issue: str) -> dict:
    """為用戶建立維修工單並派遣技術人員到場處理。"""
    return {
        "工單編號": f"TKT-{user_id}-9981",
        "派工狀態": "已派工 — 維修人員預計 1 小時內到場",
        "負責技術員": "林小明 (0912-345-678)",
        "問題描述": issue,
        "地點": location,
    }


@tool
def query_user_plan(user_id: str) -> dict:
    """查詢用戶目前的電信資費方案與合約資訊。"""
    return {
        "user_id": user_id,
        "目前方案": "4G 499 吃到飽",
        "合約到期": "2024-07-31",
        "剩餘合約": "3 個月",
        "月租費": "NT$ 499",
    }


@tool
def recommend_5g_plan(current_plan: str) -> dict:
    """根據用戶目前方案，推薦最適合的 5G 升級方案與優惠活動。"""
    return {
        "推薦方案": "5G 999 無限暢用專案",
        "月租費": "NT$ 999",
        "速率": "下載最高 2Gbps / 上傳最高 100Mbps",
        "優惠": "現在升級享 3 個月 NT$ 799 優惠價，加贈 5G 智慧手機 12 期零利率",
        "目前方案": current_plan,
        "合約": "綁約 24 個月",
    }
