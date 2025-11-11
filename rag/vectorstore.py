from typing import Union
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorDB:
    def __init__(self,
                 documents=None,
                 vector_db =Chroma,
                 embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3",model_kwargs={"device": "cuda"}),
    )->None:
        self.vector_db=vector_db
        self.embedding=embedding
        self.db= self._build_db(documents) if documents else self._get_db()
    def _build_db(self,documents):
        db=Chroma.from_documents(documents=documents,
                                         embedding=self.embedding(),persist_directory="../database")
        return db
    def _get_db(self):
        db = Chroma(embedding_function=self.embedding,
                                        persist_directory="database")
        return db
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