import dotenv
dotenv.load_dotenv()

from openai import OpenAI
import asyncio
import base64
import time
import streamlit as st
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    WebSearchTool,
    FileSearchTool,
    ImageGenerationTool,
)

client = OpenAI()

VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"

LIFE_COACH_INSTRUCTIONS = """
You are a Life Coach Agent. Default language: Korean (unless the user writes in English).

You have exactly these tools:
- Web Search Tool: Use to find advice/tips, motivational content, quotes, and up-to-date best practices.
- File Search Tool: Use to reference the user's goals, plans, journals, and personal documents (vector store).
- Image Generation Tool: Use to create (1) goal-based vision boards, (2) motivational posters with custom messages, (3) visual progress representations.

CRITICAL: Your final answer MUST ALWAYS include these 3 sections as bullet lists (even if a tool was not used, write '- (이번 요청에서는 사용하지 않음)'):

### 🔍 Web Search (조언/팁 검색)
- ...

### 🗂️ File Search (목표 문서 참조)
- ...

### 🎨 Image Generation (비전보드/포스터/진행 시각화)
- ...

TOOL ORCHESTRATION RULES:
1) For vision board / motivational poster / progress visualization:
   - First: File Search (extract goals/themes/progress data from docs)
   - Second: Web Search (collect 3–7 relevant tips/quotes; keep it short)
   - Third: Image Generation (generate an image based on the extracted goals + selected tips/quotes)
2) For pure “조언/팁” request: Web Search only is fine, but still print the 3 sections.
3) For requests referencing “내 목표/일기/문서”: File Search must be used.

IMAGE PROMPT STYLE GUIDELINES (Korean text in the image):
- Vision board: clean collage layout (3x2 or 2x2), each panel matches a theme (health, finance, learning, relationships, travel, career). Minimal, modern aesthetic. Short Korean keywords only.
- Motivational poster: bold Korean headline + smaller subline, strong typography, simple background, high contrast, no clutter.
- Progress visualization: infographic style (progress bar, milestones, simple icons), clear numbers, Korean labels.

Safety/Privacy:
- Do not reveal overly sensitive personal info from documents; paraphrase goals.
- If the user’s prompt is missing key specifics (dates, counts), make reasonable assumptions and proceed.
"""

def map_role(role: str) -> str:
    # streamlit chat_message roles are typically "human"/"ai"
    if role in ("user", "human"):
        return "human"
    return "ai"

def poll_vector_store_file(vector_store_id: str, vs_file_id: str, status_container=None, timeout_s: int = 60):
    start = time.time()
    while True:
        vs_file = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=vs_file_id,
        )
        state = getattr(vs_file, "status", None)
        if status_container:
            status_container.update(label=f"⏳ Indexing file... ({state})")
        if state == "completed":
            return True
        if state == "failed":
            return False
        if time.time() - start > timeout_s:
            return False
        time.sleep(0.5)

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions=LIFE_COACH_INSTRUCTIONS,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
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
    )

agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",
        "life-coach-memory.db",
    )
session = st.session_state["session"]

async def paint_history():
    messages = await session.get_items()

    for message in messages:
        # Render chat messages
        if "role" in message:
            with st.chat_message(map_role(message["role"])):
                content = message.get("content")

                # user might be str or list parts
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "image_url" in part:
                            st.image(part["image_url"])
                        elif isinstance(part, dict) and part.get("type") == "output_text":
                            st.write(part.get("text", ""))

                # assistant message format in your session sometimes:
                if message.get("type") == "message":
                    try:
                        st.write(message["content"][0]["text"].replace("$", "\$"))
                    except Exception:
                        pass

        # Render tool call breadcrumbs / outputs
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Web Search 사용")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ File Search 사용")
            elif message_type == "image_generation_call":
                # some sessions store final base64 in "result"
                b64 = message.get("result")
                if b64:
                    image = base64.b64decode(b64)
                    with st.chat_message("ai"):
                        st.image(image)

asyncio.run(paint_history())

def update_status(status_container, event_type: str):
    status_messages = {
        "response.web_search_call.in_progress": ("🔍 웹 검색 시작...", "running"),
        "response.web_search_call.searching": ("🔍 웹 검색 중...", "running"),
        "response.web_search_call.completed": ("✅ 웹 검색 완료.", "complete"),

        "response.file_search_call.in_progress": ("🗂️ 파일 검색 시작...", "running"),
        "response.file_search_call.searching": ("🗂️ 파일 검색 중...", "running"),
        "response.file_search_call.completed": ("✅ 파일 검색 완료.", "complete"),

        "response.image_generation_call.in_progress": ("🎨 이미지 생성 중...", "running"),
        "response.image_generation_call.generating": ("🎨 이미지 생성 중...", "running"),
        "response.image_generation_call.completed": ("✅ 이미지 생성 완료.", "complete"),

        "response.completed": (" ", "complete"),
    }
    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)

async def run_agent(message: str):
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        image_placeholder = st.empty()
        text_placeholder = st.empty()
        response = ""

        stream = Runner.run_streamed(
            agent,
            message,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                etype = event.data.type
                update_status(status_container, etype)

                if etype == "response.output_text.delta":
                    response += event.data.delta
                    text_placeholder.write(response.replace("$", "\$"))

                # partial image
                if etype == "response.image_generation_call.partial_image":
                    img_b64 = getattr(event.data, "partial_image_b64", None)
                    if img_b64:
                        image = base64.b64decode(img_b64)
                        image_placeholder.image(image)

                # final image fallback (schema differs by sdk version, so we guard)
                if etype == "response.image_generation_call.completed":
                    final_b64 = getattr(event.data, "result", None) or getattr(event.data, "image_b64", None)
                    if final_b64:
                        image = base64.b64decode(final_b64)
                        image_placeholder.image(image)

prompt = st.chat_input(
    "Life Coach에게 메시지를 입력하세요 (파일 업로드 가능)",
    accept_file=True,
    file_type=["txt", "jpg", "jpeg", "png"],
)

if prompt:
    for file in prompt.files:
        if file.type.startswith("text/"):
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching to vector store...")

                    vs_file = client.vector_stores.files.create(
                        vector_store_id=VECTOR_STORE_ID,
                        file_id=uploaded_file.id,
                    )

                    ok = poll_vector_store_file(VECTOR_STORE_ID, vs_file.id, status_container=status)
                    if ok:
                        status.update(label="✅ File uploaded + indexed", state="complete")
                    else:
                        status.update(label="⚠️ File uploaded but indexing not confirmed (try again)", state="error")

        elif file.type.startswith("image/"):
            with st.status("⏳ Uploading image...") as status:
                file_bytes = file.getvalue()
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"

                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")

            with st.chat_message("human"):
                st.image(data_uri)

    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)
        asyncio.run(run_agent(prompt.text))

with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
    st.write(asyncio.run(session.get_items()))