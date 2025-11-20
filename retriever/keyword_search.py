from langchain.schema import Document
from typing import List
from langchain_community.retrievers import BM25Retriever

''' 
thực thi bm25: 
 vector_db  = VectorDB()
 documents = vector_db.get_documents()
 BM25_search = BM25KeywordSearch(documents)
'''
class BM25KeywordSearch:
    def __init__(self, documents: List[Document], k: int = 40):
        self.retriever = BM25Retriever.from_documents(documents, k=k)

    def get_retriever(self):
        return self.retriever


