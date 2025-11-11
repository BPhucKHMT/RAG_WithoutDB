import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.BAD_HINTS = ("Cảm ơn các bạn đã xem", "đăng ký kênh", "subscribe", "like và share")

    @torch.no_grad()
    def batch_scores(self, query: str, texts: List[str], batch_size: int = 16, max_len: int = 512) -> List[float]:
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer([query]*len(batch), batch, padding=True, truncation=True,
                                    max_length=max_len, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits.squeeze(-1)
            scores.extend(logits.tolist())
        return scores

    def rerank(self, docs, query: str, top_k: int = 10) -> List:
        texts = [d.page_content for d in docs]
        scores = self.batch_scores(query, texts)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        final_docs = []
        for d, s in ranked:
            if all(h.lower() not in d.page_content.lower() for h in self.BAD_HINTS):
                final_docs.append(d)
            if len(final_docs) >= top_k:
                break
        return final_docs
