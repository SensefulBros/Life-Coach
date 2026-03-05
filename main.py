"""
life_coach_final.py
====================
Life Coach Agent with 3 tools:
  1. Web Search    — 조언/팁 검색
  2. File Search   — 개인 목표 문서 참조
  3. Image Generation — 비전보드 / 동기부여 이미지 생성

Run:
    streamlit run life_coach_final.py
"""

import os
import asyncio
import base64
import dotenv
import streamlit as st
from openai import OpenAI
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
)

dotenv.load_dotenv()

# ============================================================
# Page setup
# ============================================================
st.set_page_config(page_title="🌱 Life Coach", page_icon="🌱", layout="centered")

# ============================================================
# Constants
# ============================================================
VECTOR_STORE_ID = os.getenv(
    "VECTOR_STORE_ID",
    "",  # fallback — replace with your ID if not using .env
)
SESSION_DB = "life_coach_memory.db"
SESSION_ID = "life-coach-session"

if not VECTOR_STORE_ID:
    st.error("⚠️ VECTOR_STORE_ID가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

openai_client = OpenAI()

# ============================================================
# Agent — created once, cached across reruns
# ============================================================
COACH_INSTRUCTIONS = """You are a Life Coach named "코치 하나" (Coach Hana).
You are warm, encouraging, and action-oriented.

YOU HAVE 3 TOOLS:

1. 📂 File Search — The user's personal goals, journal entries, and progress data.
   → Use this FIRST when they ask about their goals, progress, or journal.
   → Quote specific numbers, dates, and entries from their document.

2. 🔍 Web Search — Search the internet for evidence-based advice.
   → Use this to find current tips, scientific research, and best practices.
   → Always cite what you found to build trust.

3. 🎨 Image Generation — Create motivational images and vision boards.
   → Use this when the user asks for a vision board, celebration image, or motivational poster.
   → Also use this proactively when the user achieves a goal — surprise them with a congratulatory image!
   → Create vivid, inspiring images with warm colors and positive energy.

DECISION LOGIC:
- "목표 어때?" / "진행 상황" → File Search first, then Web Search for tips
- "팁 알려줘" / "방법" → Web Search first
- "비전보드" / "이미지" / "축하" / goal achieved → Image Generation
- Complex requests → combine multiple tools in sequence

STYLE:
- Respond in Korean always.
- Celebrate every win, no matter how small.
- When the user is behind on a goal, reframe it positively.
- Give concrete 3-step action plans, not vague advice.
- Use emojis naturally but not excessively.
- End with one follow-up question to keep the conversation going."""


@st.cache_resource
def build_agent():
    return Agent(
        name="Coach Hana",
        instructions=COACH_INSTRUCTIONS,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=5,
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,
                }
            ),
        ],
        model="gpt-4o-mini",
    )


coach = build_agent()

# ============================================================
# Session — SQLite-backed persistent memory
# ============================================================
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(SESSION_ID, SESSION_DB)

memory = st.session_state["session"]

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🌱 코치 하나")
    st.caption("당신의 AI 라이프 코치")
    st.divider()

    st.markdown("### 🛠️ 도구 현황")
    st.markdown(
        """
| 도구 | 상태 |
|------|------|
| 🔍 웹 검색 | ✅ 활성 |
| 📂 파일 검색 | ✅ 활성 |
| 🎨 이미지 생성 | ✅ 활성 |
"""
    )
    
    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        asyncio.run(memory.clear_session())
        st.rerun()

# ============================================================
# Helpers — render chat history from SQLiteSession
# ============================================================


async def render_saved_history():
    """Read all items from SQLiteSession and display them."""
    items = await memory.get_items()
    for item in items:
        # User messages
        if item.get("role") == "user":
            with st.chat_message("user"):
                content = item.get("content", "")
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "input_text":
                                st.write(part.get("text", ""))
                            elif "image_url" in part:
                                st.image(part["image_url"])

        # Assistant final text messages
        elif item.get("role") == "assistant" and item.get("type") == "message":
            with st.chat_message("assistant"):
                content_parts = item.get("content", [])
                for part in content_parts:
                    if isinstance(part, dict) and "text" in part:
                        st.write(part["text"].replace("$", r"\$"))

        # Tool activity indicators
        elif item.get("type") == "web_search_call":
            with st.chat_message("assistant"):
                st.info("🔍 웹에서 정보를 검색했습니다")

        elif item.get("type") == "file_search_call":
            with st.chat_message("assistant"):
                st.info("📂 목표 문서를 확인했습니다")

        elif item.get("type") == "image_generation_call":
            with st.chat_message("assistant"):
                try:
                    img_bytes = base64.b64decode(item["result"])
                    st.image(img_bytes, caption="🎨 코치 하나가 만든 이미지")
                except Exception:
                    st.caption("🎨 이미지가 생성되었습니다")


asyncio.run(render_saved_history())


# ============================================================
# Helpers — streaming status labels
# ============================================================
STREAM_LABELS = {
    # Web search
    "response.web_search_call.in_progress": ("🔍 웹 검색 시작...", "running"),
    "response.web_search_call.searching": ("🔍 검색 중...", "running"),
    "response.web_search_call.completed": ("✅ 웹 검색 완료", "complete"),
    # File search
    "response.file_search_call.in_progress": ("📂 목표 문서 검색 시작...", "running"),
    "response.file_search_call.searching": ("📂 문서 검색 중...", "running"),
    "response.file_search_call.completed": ("✅ 문서 검색 완료", "complete"),
    # Image generation
    "response.image_generation_call.in_progress": ("🎨 이미지 생성 준비...", "running"),
    "response.image_generation_call.generating": ("🎨 이미지 그리는 중...", "running"),
    "response.image_generation_call.partial_image": ("🎨 이미지 완성 중...", "running"),
    # Done
    "response.completed": ("✅ 응답 완료", "complete"),
}


# ============================================================
# Core — stream the agent's response
# ============================================================


async def stream_coach_response(user_message: str):
    """Run the agent in streaming mode and render output live."""
    with st.chat_message("assistant"):
        status_box = st.status("🌱 코치 하나가 생각하는 중...", expanded=False)
        image_area = st.empty()
        text_area = st.empty()

        accumulated_text = ""

        stream = Runner.run_streamed(
            coach,
            user_message,
            session=memory,
        )

        async for event in stream.stream_events():
            if event.type != "raw_response_event":
                continue

            event_name = event.data.type

            # Update status indicator
            if event_name in STREAM_LABELS:
                label, state = STREAM_LABELS[event_name]
                status_box.update(label=label, state=state)

            # Stream text tokens
            if event_name == "response.output_text.delta":
                accumulated_text += event.data.delta
                text_area.write(accumulated_text.replace("$", r"\$"))

            # Stream generated image (partial → final)
            elif event_name == "response.image_generation_call.partial_image":
                try:
                    img_bytes = base64.b64decode(event.data.partial_image_b64)
                    image_area.image(img_bytes, caption="🎨 코치 하나가 만든 이미지")
                except Exception:
                    pass


# ============================================================
# Title
# ============================================================
st.title("🌱 Life Coach Agent")
st.caption("목표 점검 · 맞춤 조언 · 비전보드 이미지까지 — 코치 하나에게 물어보세요!")

# ============================================================
# Chat input — handle sidebar button prompts + typed input
# ============================================================
# Check if a sidebar button was pressed
sidebar_prompt = st.session_state.pop("sidebar_prompt", None)

typed_input = st.chat_input("코치 하나에게 메시지를 보내세요...")

active_message = sidebar_prompt or typed_input

if active_message:
    # Show user message
    with st.chat_message("user"):
        st.write(active_message)

    # Run the agent
    asyncio.run(stream_coach_response(active_message))