"""
FastAPI Backend Server for RAG Q&A System
- Handles RAG inference via /chat endpoint
- Stores conversation history in MongoDB
- Supports ~50 concurrent users
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
 
import logging

# RAG agent import
from rag.lang_graph_rag import call_agent



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PUQ Q&A Backend", version="1.0.0")

# CORS middleware (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory conversation store
conversations_store = {}

# ============ MODELS ============
class Message(BaseModel):
    role: str
    content: Any  # can be str or dict (response object)

class ChatRequest(BaseModel):
    conversation_id: str
    messages: List[Message]
    user_message: str

class ChatResponse(BaseModel):
    conversation_id: str
    response: Any
    updated_at: str

class ConversationCreate(BaseModel):
    title: str = "Cuộc trò chuyện mới"

class ConversationResponse(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str

class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int





# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return {"message": "PUQ Q&A Backend API", "status": "running"}










# ============ IN-MEMORY CHAT ENDPOINT ============
from uuid import uuid4

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - invokes RAG and stores in memory
    """
    try:
        # 1. Retrieve conversation from memory
        doc = conversations_store.get(request.conversation_id)
        if not doc:
            # Create new conversation if not found
            now = datetime.now().isoformat()
            doc = {
                "conversation_id": request.conversation_id,
                "title": "Cuộc trò chuyện mới",
                "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
                "created_at": now,
                "updated_at": now
            }
            conversations_store[request.conversation_id] = doc
        # 2. Append user message to history
        messages = doc.get("messages", [])
        messages.append({"role": "user", "content": request.user_message})
        # 3. Prepare chat history for agent (convert content to plain text if needed)
        chat_history = []
        for m in messages:
            content = m["content"]
            if isinstance(content, dict):
                content = content.get("text", str(content))
            chat_history.append({"role": m["role"], "content": content})
        # 4. Call RAG agent
        response = call_agent(chat_history)
        # 5. Append assistant response
        messages.append({"role": "assistant", "content": response})
        # 6. Update title if it's still default
        title = doc["title"]
        if title == "Cuộc trò chuyện mới" and len(messages) > 1:
            first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            if isinstance(first_user_msg, str):
                title = first_user_msg[:35] + ("..." if len(first_user_msg) > 35 else "")
        # 7. Save to memory
        now = datetime.now().isoformat()
        doc["title"] = title
        doc["messages"] = messages
        doc["updated_at"] = now
        conversations_store[request.conversation_id] = doc
        return ChatResponse(
            conversation_id=request.conversation_id,
            response=response,
            updated_at=now
        )
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")



# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
