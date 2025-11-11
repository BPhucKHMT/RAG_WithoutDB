from pydantic import BaseModel, Field

from rag.file_loader import Loader
from rag.vectorstore import VectorDB
from rag.offline_rag import Offline_RAG
from rag.reranking import CrossEncoderReranker
from rag.llm_model import get_llm


def build_rag_chain():
    llm = get_llm()
    vector_db = VectorDB()
    retriever = vector_db.get_retriever()
    reranker = CrossEncoderReranker()
    rag_chain = Offline_RAG(llm, retriever, reranker)
    return rag_chain.get_chain()

if __name__ == "__main__":
    rag_chain = build_rag_chain()
    response = rag_chain.invoke("Tại sao việc tự huấn luyện mô hình CLIP từ đầu là khó khả thi?")
    print(response)
