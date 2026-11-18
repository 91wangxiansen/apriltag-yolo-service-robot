from fastmcp import FastMCP
import sys
import logging
import json
import os
import requests

# 日志配置
logger = logging.getLogger("FallDetectionMCP")

# Windows 编码修复
if sys.platform == 'win32':
    sys.stderr.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')

# ===================== 配置 =====================
STATUS_FILE = os.environ.get("ROBOT_STATUS_FILE", "robot_status.json")
WEBHOOK_URL = os.environ.get("WECHAT_WEBHOOK_URL")

# 创建 MCP 服务
mcp = FastMCP("FallDetectionRobot")

# 防重复报警标记
_already_sent_fall = False

# ===================== 工具1：获取实时跌倒/抬手状态 =====================
@mcp.tool()
def get_robot_alert_status() -> dict:
    """
    获取机器人实时状态：是否跌倒、是否抬手、时间戳
    小智AI可以定时调用这个接口来监控报警
    """
    global _already_sent_fall

    try:
        if not os.path.exists(STATUS_FILE):
            return {"fall_detected": False, "hand_raised": False, "timestamp": ""}

        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            status = json.load(f)

        fall = status.get("fall_detected", False)
        hand = status.get("hand_raised", False)
        ts = status.get("timestamp", "")

        return {
            "fall_detected": fall,
            "hand_raised": hand,
            "timestamp": ts
        }

    except Exception as e:
        return {"fall_detected": False, "hand_raised": False, "error": str(e)}

# ===================== 工具2：发送微信报警消息 =====================
@mcp.tool()
def send_wechat_alert(message: str) -> dict:
    """
    发送跌倒报警到企业微信群
    """
    if not WEBHOOK_URL:
        return {"success": False, "error": "请先配置 WECHAT_WEBHOOK_URL"}

    data = {
        "msgtype": "text",
        "text": {
            "content": message
        }
    }
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(WEBHOOK_URL, data=json.dumps(data), headers=headers)
        return {"success": True, "response": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ===================== 启动 MCP 服务 =====================
if __name__ == "__main__":
    logger.info("✅ 小智AI跌倒检测MCP服务已启动")
    mcp.run(transport="stdio")