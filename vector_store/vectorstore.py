from typing import Union
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db =Chroma,
                 embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device": "cpu"}),
    )->None:
        self.vector_db=vector_db
        self.embedding=embedding
        self.db= self._build_db(documents) if documents else self._get_db() 
    def _build_db(self,documents):
        db = Chroma(embedding_function=self.embedding,
                                        persist_directory="database")
        db.add_documents(documents)
        return db
    
    def _get_db(self): # lấy ra db đã có sẵn nếu không truyền vào documents
        db = Chroma(embedding_function=self.embedding,
                                        persist_directory="database")
        return db
    
    def add_documents(self, documents): # thêm tài liệu
        return self.db.add_documents(documents)
    
    def get_documents(self): # lấy document để truyền vào bm25 search
        raw = self.db.get(include=["documents", "metadatas"])
        docs = []

        for content, metadata in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(
                page_content=content,
                metadata={
                    "filename": metadata.get("filename", ""),
                    "video_url": metadata.get("video_url", ""),
                    "start_timestamp": metadata.get("start_timestamp", "")
                }
            ))
            return docs
        
            
    def get_retriever(self,
                      search_type: str="mmr",
                      search_kwargs: dict={
                          "k": 40, "fetch_k": 80, "lambda_mult": 0.3
                      }):
        retriever=self.db.as_retriever(search_type=search_type,
                                        search_kwargs=search_kwargs)
        return retriever
    
if __name__ == "__main__":
    import os
    print(os.getcwd())