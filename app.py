
import streamlit as st
import time
import json
import re
from datetime import datetime
from pathlib import Path

# ============ BACKEND AGENT ============
from rag.lang_graph_rag import call_agent
# ========================================

# ============ CONFIGURATION ============
CONVERSATIONS_DIR = "saved_conversations"
Path(CONVERSATIONS_DIR).mkdir(exist_ok=True)

# ============ UTILITIES ============
def truncate_text(text, max_length=35):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def timestamp_to_seconds(timestamp: str) -> int:
    """Chuyển HH:MM:SS hoặc MM:SS sang seconds"""
    try:
        parts = list(map(int, timestamp.split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except:
        pass
    return 0

def response_to_display_text(response) -> str:
    """Convert response thành plain text"""
    if isinstance(response, dict):
        text = response.get('text', '')
        clean_text = re.sub(r'\[(\d+)\]', r'[\1]', text)
        return clean_text
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def render_response(response):
    """Universal renderer"""
    if isinstance(response, dict):
        response_type = response.get('type', 'unknown')
        text = response.get('text', '')
        video_urls = response.get('video_url', [])
        titles = response.get('title', [])
        start_timestamps = response.get('start_timestamp', [])
        end_timestamps = response.get('end_timestamp', [])
        confidences = response.get('confidence', [])
        
        if video_urls:
            def replace_citation(match):
                index = int(match.group(1))
                if index < len(video_urls):
                    url = video_urls[index]
                    title = titles[index] if index < len(titles) else f"Video {index}"
                    start = start_timestamps[index] if index < len(start_timestamps) else "00:00:00"
                    seconds = timestamp_to_seconds(start)
                    video_link = f"{url}&t={seconds}" if '?' in url else f"{url}?t={seconds}"
                    return f'<a href="{video_link}" target="_blank" style="color: #1E88E5; font-weight: bold; text-decoration: none; border-bottom: 1px dotted #1E88E5;" title="{title} - {start}">[{index}]</a>'
                return match.group(0)
            formatted_text = re.sub(r'\[(\d+)\]', replace_citation, text)
        else:
            formatted_text = text
        
        st.markdown(formatted_text, unsafe_allow_html=True)
        
        if video_urls and response_type == "rag":
            st.markdown("---")
            st.markdown("### 📺 Nguồn tham khảo:")
            for i, url in enumerate(video_urls):
                title = titles[i] if i < len(titles) else f"Video {i}"
                start = start_timestamps[i] if i < len(start_timestamps) else "00:00:00"
                end = end_timestamps[i] if i < len(end_timestamps) else start
                confidence = confidences[i] if i < len(confidences) else "unknown"
                seconds = timestamp_to_seconds(start)
                video_link = f"{url}&t={seconds}" if '?' in url else f"{url}?t={seconds}"
                conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🟠', 'zero': '🔴'}.get(confidence, '⚪')
                st.markdown(f"**{i}.** {conf_emoji} [{title}]({video_link}) ⏱️ `{start}` → `{end}`")
    
    elif isinstance(response, str):
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Unknown response format: {type(response)}")

# ============ SAVE/LOAD FUNCTIONS ============
def save_conversation(convo_id: str):
    """Lưu conversation ra file JSON (auto-save)"""
    try:
        convo = st.session_state.conversations[convo_id]
        filename = f"{CONVERSATIONS_DIR}/{convo_id}.json"
        
        # Convert messages to serializable format
        serializable_messages = []
        for msg in convo["messages"]:
            content = msg["content"]
            if isinstance(content, dict):
                serializable_messages.append({"role": msg["role"], "content": content})
            else:
                serializable_messages.append({"role": msg["role"], "content": str(content)})
        
        data = {
            "id": convo_id,
            "title": convo["title"],
            "messages": serializable_messages,
            "created_at": convo.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        return False

def load_all_conversations():
    """Load tất cả conversations từ folder"""
    conversations = {}
    if Path(CONVERSATIONS_DIR).exists():
        for file in Path(CONVERSATIONS_DIR).glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    convo_id = data["id"]
                    conversations[convo_id] = {
                        "title": data["title"],
                        "messages": data["messages"],
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at")
                    }
            except:
                pass
    return conversations

def delete_conversation(convo_id: str):
    """Xóa conversation"""
    try:
        # Xóa file
        filename = f"{CONVERSATIONS_DIR}/{convo_id}.json"
        if Path(filename).exists():
            Path(filename).unlink()
        
        # Xóa khỏi session state
        if convo_id in st.session_state.conversations:
            del st.session_state.conversations[convo_id]
        
        # Reset current ID nếu đang active
        if st.session_state.current_conversation_id == convo_id:
            remaining_convos = list(st.session_state.conversations.keys())
            if remaining_convos:
                st.session_state.current_conversation_id = remaining_convos[-1]
            else:
                create_new_conversation()
        
        return True
    except:
        return False

def reset_conversation(convo_id: str):
    """Reset conversation về trạng thái ban đầu"""
    try:
        st.session_state.conversations[convo_id] = {
            "title": "Cuộc trò chuyện mới",
            "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
            "created_at": datetime.now().isoformat()
        }
        # Save after reset
        save_conversation(convo_id)
        return True
    except:
        return False

# ============ SESSION MANAGEMENT ============
def create_new_conversation():
    """Tạo conversation mới"""
    convo_id = f"chat_{int(time.time())}"
    st.session_state.conversations[convo_id] = {
        "title": "Cuộc trò chuyện mới",
        "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_conversation_id = convo_id
    # Auto-save new conversation
    save_conversation(convo_id)

def set_current_conversation(convo_id):
    """Switch conversation"""
    st.session_state.current_conversation_id = convo_id

# ============ SETUP PAGE ============
st.set_page_config(
    page_title="PUQ Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 PUQ Q&A")

# ============ INITIALIZE SESSION STATE ============
if "conversations" not in st.session_state:
    # Load từ disk
    st.session_state.conversations = load_all_conversations()

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if not st.session_state.conversations:
    create_new_conversation()

# ============ SIDEBAR ============
with st.sidebar:
    st.title("💬 Cuộc trò chuyện")
    
    # New conversation button
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        create_new_conversation()
        st.rerun()
    
    st.divider()
    
    # Search box
    search_query = st.text_input("🔍 Tìm kiếm", placeholder="Nhập từ khóa...")
    
    st.subheader("Gần đây")
    
    convo_ids = list(st.session_state.conversations.keys())
    
    # Filter by search
    if search_query:
        filtered_ids = [
            cid for cid in convo_ids
            if search_query.lower() in st.session_state.conversations[cid]["title"].lower()
        ]
    else:
        filtered_ids = convo_ids
    
    # Display conversations
    for convo_id in reversed(filtered_ids):
        convo = st.session_state.conversations[convo_id]
        title = convo["title"]
        is_active = (convo_id == st.session_state.current_conversation_id)
        
        # Conversation item với delete/reset
        col1, col2, col3 = st.columns([7, 1.5, 1.5])
        
        with col1:
            if st.button(
                title,
                key=f"select_{convo_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                set_current_conversation(convo_id)
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{convo_id}", help="Xóa"):
                if delete_conversation(convo_id):
                    st.rerun()
        
        with col3:
            if st.button("🔄", key=f"reset_{convo_id}", help="Reset"):
                if reset_conversation(convo_id):
                    st.rerun()

# ============ MAIN CHAT AREA ============
current_id = st.session_state.current_conversation_id

if current_id and current_id in st.session_state.conversations:
    current_convo = st.session_state.conversations[current_id]
    messages = current_convo["messages"]
    
    # Display messages
    for message in messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                render_response(content)
            else:
                st.markdown(content)
    
    # User input
    if prompt := st.chat_input("Nhắn tin..."):
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Update title if new conversation
        if current_convo["title"] == "Cuộc trò chuyện mới":
            current_convo["title"] = truncate_text(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare chat history
        chat_history = []
        for m in messages:
            content = m["content"]
            if isinstance(content, dict):
                content = response_to_display_text(content)
            chat_history.append({"role": m["role"], "content": content})
        
        # Call agent
        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."):
                try:
                    response = call_agent(chat_history)
                    render_response(response)
                    messages.append({"role": "assistant", "content": response})
                    
                    # Auto-save after each message
                    save_conversation(current_id)
                    
                except Exception as e:
                    error_msg = f"⚠️ Có lỗi xảy ra: {str(e)}"
                    st.error(error_msg)
                    messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

else:
    st.info("👈 Vui lòng chọn hoặc tạo cuộc trò chuyện từ thanh bên.")