from langchain.retrievers import EnsembleRetriever

''' 
thực thi ensemble retriever: 
 vector_db  = VectorDB()
 documents = vector_db.get_documents()
 BM25_search = BM25KeywordSearch(documents)

 keyword_retriever = BM25_search.get_retriever()
 vector_retriever = vector_db.get_retriever()

 ==> hybrid_search = HybridSearch(vector_retriever, keyword_retriever)

'''

class HybridSearch:
    def __init__(self,
                 vector_retriever,
                 keyword_retriever):
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5]  # điều chỉnh trọng số theo nhu cầu
        )

    def get_retriever(self):
        return self.retriever