
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from generation.llm_model import get_llm
from langchain_core.prompts import PromptTemplate
import json  # <<< THÊM
import re    # <<< THÊM
# -------------------------
# RAG chain
# -------------------------
from data_loader.file_loader import Loader
from vector_store.vectorstore import VectorDB
from rag.offline_rag import Offline_RAG
from retriever.reranking import CrossEncoderReranker
from retriever.keyword_search import BM25KeywordSearch
from retriever.hybrid_search import HybridSearch

import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
def build_rag_chain():
    llm = get_llm()
    vector_db = VectorDB()
    vector_retriever = vector_db.get_retriever()
    documents = vector_db.get_documents()
    bm25_search = BM25KeywordSearch(documents).get_retriever()
    hybrid_search = HybridSearch(bm25_search, vector_retriever).get_retriever()
    print("Loading reranker model...")
    reranker = CrossEncoderReranker(device=device)
    print("Reranker model loaded.")
    rag_chain = Offline_RAG(llm, hybrid_search, reranker)
    return rag_chain.get_chain()

rag_chain = build_rag_chain()

# -------------------------
# Tool Prompt + Retrieve tool
# -------------------------
toolPrompt = PromptTemplate.from_template("""
Bạn là trợ lý AI cho hệ thống hỏi đáp môn học.
Với bất kỳ câu hỏi nào về machine learning, deep learning, AI sinh tạo hoặc các kiến thức liên quan, bạn ***phải sử dụng công cụ `Retrieve`*** để lấy thông tin chính xác.
Khi sử dụng công cụ `Retrieve`, hãy trích xuất một truy vấn rõ ràng, ngắn gọn từ câu hỏi của người dùng và lịch sử chat. Truy vấn này sẽ được đặt vào tham số `query` của tool.
Với các câu hỏi khác hoặc tương tác thông thường, bạn có thể trả lời trực tiếp.
\\n Người dùng là sinh viên Việt Nam, hãy trả lời hoàn toàn bằng tiếng Việt.
\\n Đây là lịch sử chat: {chat_history}
""")


class Retrieve(BaseModel):
    query: str = Field(description="should be a search query")

llm_agent = get_llm()
agent = llm_agent.bind_tools([Retrieve])
agent_chain = toolPrompt | agent

# -------------------------
# LangGraph State
# -------------------------
class State(BaseModel):
    chat_history: List[dict]
    agent_output: dict = {}
    response: str = ""

# -------------------------
# Node — Agent quyết định (tool hoặc direct)
# -------------------------
def node_agent(state: State):
    chat_hist_str = "\n".join([f"{m['role']} : {m['content']}" for m in state.chat_history])
    resp = agent_chain.invoke({"chat_history": chat_hist_str})

    # chuyển AIMessage sang dict
    if hasattr(resp, "content"):
        resp_dict = {"content": resp.content, "tool_calls": getattr(resp, "tool_calls", [])}
    else:
        resp_dict = resp

    return {"agent_output": resp_dict} # -> format : {"content": ..., "tool_calls": [...]}

# -------------------------
# CONDITIONAL — Xem Agent có gọi tool không
# -------------------------
def route_decision(state: State):
    out = state.agent_output
    if "tool_calls" in out and out["tool_calls"]:
        return "rag"
    return "direct"

# -------------------------
# Node 3A — Direct Answer
# -------------------------
def node_direct_answer(state: State):
    direct_text = state.agent_output["content"]
    
    # Return dict format giống RAG, nhưng không có sources
    return {
        "response": {
            "text": direct_text,
            "video_url": [],
            "title": [],
            "start_timestamp": [],
            "end_timestamp": [],
            "confidence": [],
            "type": "direct"  # ← Flag để phân biệt
        }
    }

# -------------------------
# Node 3B — RAG Answer
# -------------------------
# -------------------------
# -------------------------
# -------------------------
# Helper Function: Convert timestamp (M:SS or H:M:SS) to total seconds
# -------------------------
def timestamp_to_seconds(timestamp: str) -> int:
    """Chuyển đổi chuỗi timestamp (ví dụ: '0:01:28') thành tổng số giây."""
    parts = list(map(int, timestamp.split(':')))
    
    seconds = 0
    if len(parts) == 3: # H:M:S
        seconds += parts[0] * 3600
        seconds += parts[1] * 60
        seconds += parts[2]
    elif len(parts) == 2: # M:S
        seconds += parts[0] * 60
        seconds += parts[1]
    return seconds

# -------------------------
# Node 3B — RAG Answer (FIX CUỐI CÙNG CHO VẤN ĐỀ GẮN LINK/INDEX)
# -------------------------
def node_rag_answer(state: State):
    tool_call = state.agent_output["tool_calls"][0]
    query = tool_call["args"]["query"]
    
    rag_result = rag_chain.invoke(query)
    
    if hasattr(rag_result, 'content'):
        raw_content = rag_result.content
        print("Raw content from RAG:", raw_content)  # Chỉ in 200 chars đầu
    else:
        raw_content = str(rag_result)

    try:
        
        import re
        
        # ✅ Tìm JSON block trong response
        # Pattern 1: Tìm ```json ... ```
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            print("✅ Found JSON in markdown block")
        else:
            # Pattern 2: Tìm JSON object trực tiếp (không có markdown)
            json_match = re.search(r'(\{.*\})', raw_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                print("✅ Found JSON object")
            else:
                # Pattern 3: Toàn bộ content là JSON
                json_str = raw_content.strip()
        
        # Parse JSON
        data = json.loads(json_str)
        print("✅ JSON parsed successfully")
        
        data["type"] = "rag"
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(f"Attempted to parse: {json_str if 'json_str' in locals() else raw_content}")
        
        data = {
            "text": f"Xin lỗi, có lỗi xảy ra: {e}",
            "video_url": [],
            "title": [],
            "filename": [], 
            "start_timestamp": [],
            "end_timestamp": [],
            "confidence": [],
            "type": "error"
        }
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        print("có lỗi rùi")
        data = {
            "text": f"Xin lỗi, có lỗi xảy ra: {e}",
            "video_url": [],
            "title": [],
            "filename": [],
            "start_timestamp": [],
            "end_timestamp": [],
            "confidence": [],
            "type": "error"
        }

    return {"response": data}


# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(State)

graph.add_node("agent", node_agent)
graph.add_node("direct", node_direct_answer)
graph.add_node("rag", node_rag_answer)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    route_decision,
    {
        "direct": "direct",
        "rag": "rag"
    }
)
graph.add_edge("direct", END)
graph.add_edge("rag", END)

workflow = graph.compile()

# -------------------------
# Hàm chính để export (Được app.py import)
# -------------------------
def call_agent(chat_history: List[dict]) -> str:
    """
    Chạy LangGraph workflow với lịch sử chat.
    """
    # Khởi tạo trạng thái ban đầu
    initial_state = State(chat_history=chat_history)
    
    # Chạy workflow và nhận trạng thái cuối cùng
    final_state = workflow.invoke(initial_state)
    
    return final_state["response"]

# -------------------------
# Test run (chỉ chạy khi chạy file trực tiếp)
# -------------------------
if __name__ == "__main__":
    from IPython.display import display, Image
    import pprint

    # Dữ liệu test 1: Hỏi về chủ đề RAG (Nên gọi tool)
    chat_history_rag = [ 
                        {"role": "user", "content": "naive bayes là gì?"}
                       ]
    print("--- TEST 1: RAG Question ---")
    out_rag = workflow.invoke({"chat_history": chat_history_rag})
    print("Final Response:", out_rag["response"])
    print("---" * 10)

    # Dữ liệu test 2: Hỏi về chủ đề thông thường (Nên trả lời trực tiếp)
    chat_history_direct = [
                        {"role": "user", "content": "loss diffusion gồm các thành phần nào"} 
                       ]
    print("--- TEST 2: Direct Question ---")
    out_direct = workflow.invoke({"chat_history": chat_history_direct})
    print("Final Response:", out_direct["response"])
    print("---" * 10)
    
    
    # --- Code kiểm thử thủ công bị lỗi NameError ---
    # *Đây là các dòng bạn cần XÓA hoặc BỎ QUA trong file gốc của bạn*
    # state = State(chat_history=chat_history)
    # # Node Agent
    # # Chạy node agent
    # agent_out = node_agent(state)
    # state.agent_output = agent_out["agent_output"]
    # # Kiểm tra xem agent có gọi tool không
    # if state.agent_output.get("tool_calls"):
    #     tool_call = state.agent_output["tool_calls"][0]
    #     query = tool_call["args"]["query"]
    #     print("Query mà agent sẽ gửi cho RAG:", query)
    # else:
    #     print("Agent không gọi tool, trả lời trực tiếp:", state.agent_output.get("content"))