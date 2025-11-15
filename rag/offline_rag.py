from pydantic  import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap,RunnableLambda
from typing import List

class VideoAnswer(BaseModel):
    text: str = Field(description="Câu trả lời tóm tắt trong 3 câu")
    filename: List[str] = Field(description="Tên file transcript gốc")
    video_url: List[str] = Field(description="URL của video gốc")
    start_timestamp: List[str] = Field(description="Thời điểm bắt đầu (format: HH:MM:SS)")
    end_timestamp: List[str] = Field(description="Thời điểm kết thúc (format: HH:MM:SS)")
    confidence: List[str] = Field(description="Độ tin cậy: zero/low/medium/high")

parser = JsonOutputParser(pydantic_object=VideoAnswer)

class Offline_RAG:
    def __init__(self, llm, retriever, reranker)-> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
        Dựa vào transcript sau, trả lời câu hỏi của người dùng bằng tiếng Việt.Phần tóm tắt nội dung thì nên tóm tắt trong 3 câu, 
        dựa vào các đoạn transcript được cung cấp và chỉ ra đoạn video chứa thông tin đó và các đoạn video liên quan khác (nếu có)  (video url, thời điểm bắt đầu và kết thúc).
        Đồng thời làm mượt lại nội dung tóm tắt đó
        Khi trích dẫn thông tin, **luôn sử dụng đúng [Video URL] và [Start] từ doc chứa nội dung đó**.
        Nếu không biết câu trả lời thì cứ trả lời là tôi không biết và độ tin cậy là zero
        Nếu câu hỏi không liên quan đến nội dung video thì trả lời tôi chỉ được huấn luyện trả lời các câu hỏi liên quan đến nội dung video và độ tin cậy là zero
        Không bịa ra thông tin không có căn cứ, không trả lời sai format
        Nếu bạn cực kỳ chắc chắn về câu trả lời, hãy đặt độ tin cậy là high. Nếu bạn khá chắc chắn, hãy đặt độ tin cậy là medium. Nếu bạn không chắc chắn về câu trả lời, hãy đặt độ tin cậy là low.
        Định dạng đầu ra phải tuân theo JSON schema sau:
        {format_instructions}
        Transcript:
        {context}

        Câu hỏi: {question}
        \nAnswer:               
        """)
        self.retriever = retriever
        self.reranker = reranker


    def format_doc(self, docs,*args, **kwargs):
        formatted = []
        for doc in docs:
            url = doc.metadata.get("video_url", "")
            filename = doc.metadata.get("filename", "")
            start = doc.metadata.get("start_timestamp", "")
            end = doc.metadata.get("end_timestamp", "")
            content = doc.page_content
            formatted.append(f"""[Video URL]: {url}
                                [Filename]: {filename}
                                [Start]: {start}
                                [End]: {end}
                                [Content]: {content}""")
        return "\n\n".join(formatted)
    
    def rerank_with_query(self, docs_and_query):
        docs, query = docs_and_query
        return self.reranker.rerank(docs, query)
    
    # Hàm lấy context để đưa vào prompt 
    def get_context(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        reranked = self.reranker.rerank(docs, query)
        return self.format_doc(reranked)
    
    def get_chain(self):
        return (
            {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(self.get_context),
        }
        | self.prompt.partial(format_instructions=parser.get_format_instructions())
        | self.llm
        )
