from typing import List, Union, Dict
import glob
import json
import os
import re
from tqdm import tqdm
import multiprocessing

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
from preprocess.preprocess import correct_spelling

load_dotenv()
gptkey = os.getenv("myAPIKey")
# ==========================================================
def get_num_cpu() -> int:
    """Trả về số CPU khả dụng."""
    return multiprocessing.cpu_count()


def parse_transcript(file_path: str) -> tuple[str, list[dict], str]:
    """Đọc file transcript, tách từng dòng thành block có start-end-text, sau đó sửa chính tả."""
    full_text = ""
    position_map = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line or "[âm nhạc]" in line.lower():
                continue
            lines.append(line)

    # Ghép lại toàn bộ transcript
    text_block = "\n".join(lines)
    corrected_text = correct_spelling(text_block)  # gọi LLM sửa chính tả

    # Duyệt lại từng dòng corrected
    for line in corrected_text.split("\n"):
        match = re.match(r"(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+),\s*(.+)", line)
        if match:
            start, end, text = match.groups()
            pos = len(full_text)
            full_text += text + " "
            position_map.append({
                "start": start,
                "end": end,
                "text": text,
                "pos_start": pos,
                "pos_end": len(full_text)
            })

    filename = os.path.basename(file_path).replace(".txt", "")
    return full_text.strip(), position_map, filename

def map_metadata(metadata_path: str, filename: str) -> tuple[Union[str, None], Union[str, None]]:
    """Đọc file metadata và ánh xạ filename sang title và url."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    metadata = metadata_list["videos"]
    title, url = next(
        ((item["title"], item["url"]) for item in metadata if item["video_id"] == filename),
        (None, None)
    )
    return title, url


def load_transcript(file_path: str, metadata_path: str) -> dict:
    """Load 1 file transcript và ánh xạ metadata."""
    full_text, position_map, filename = parse_transcript(file_path)
    title, url = map_metadata(metadata_path, filename)
    return {
        "full_text": full_text,
        "position_map": position_map,
        "filename": filename,
        "title": title,
        "url": url
    }


class BaseLoader:
    """Base class cho các loader."""
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs):
        raise NotImplementedError("Subclasses phải implement __call__.")


class TranscriptLoader(BaseLoader):
    """Loader cho transcript .txt."""
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, txt_files: List[str], metadata_path: str, workers: int = 1) -> List[dict]:
        num_processes = min(self.num_processes, workers)
        data = []

        with multiprocessing.Pool(processes=num_processes) as pool:
            total_files = len(txt_files)
            args = [(path, metadata_path) for path in txt_files]
            with tqdm(total=total_files, desc="Loading Transcripts", unit="file") as pbar:
                for result in pool.starmap(load_transcript, args):
                    data.append(result)
                    pbar.update(1)
        return data


class TranscriptChunker:
    """Chunk văn bản bằng SemanticChunker."""
    def __init__(self, open_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=open_api_key
        )
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
            min_chunk_size=300,
            add_start_index=True,
            buffer_size=1
        )

    def __call__(self, documents: List[dict], output_dir: str) -> List[dict]:
        all_chunks = []
        for item in documents:
            full_text = item["full_text"]
            position_map = item["position_map"]
            filename = item["filename"]
            title = item["title"]
            url = item["url"]

            chunks = self.splitter.create_documents(
                texts=[full_text],
                metadatas=[{
                    "video_url": url,
                    "filename": filename,
                    "title": title
                }]
            )
            for i, chunk in enumerate(chunks):
                start_index = chunk.metadata.pop("start_index")
                end_index = start_index + len(chunk.page_content)
                matched_ts = [
                    pos for pos in position_map
                    if not (pos["pos_end"] < start_index or pos["pos_start"] > end_index)
                ]
                if matched_ts:
                    chunk.metadata["start_timestamp"] = matched_ts[0]["start"]
                    chunk.metadata["end_timestamp"] = matched_ts[-1]["end"]
                else:
                    chunk.metadata["start_timestamp"] = None
                    chunk.metadata["end_timestamp"] = None
                chunk.metadata["chunk_id"] = i
            all_chunks.extend(chunks)

        output_path = os.path.join(output_dir, "semantic_chunks.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([
                {"page_content": chunk.page_content, "metadata": chunk.metadata}
                for chunk in all_chunks
            ], f, ensure_ascii=False, indent=4)
        print(f"Saved {len(all_chunks)} chunks to {output_path}")
        return all_chunks


class Loader:
    """Class chính: Load + Chunk transcript."""
    def __init__(self, open_api_key: str, split_kwargs: dict = None) -> None:
        self.transcript_loader = TranscriptLoader()
        self.chunker = TranscriptChunker(open_api_key=open_api_key)
        self.split_kwargs = split_kwargs or {}

    def load(self, txt_files: Union[str, List[str]], metadata_path: str, workers: int = 1):
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        docs = self.transcript_loader(txt_files, metadata_path=metadata_path, workers=workers)
        return docs

    def load_dir(self, transcript_dir: str, metadata_path: str, output_dir: str, workers: int = 1):
        txt_files = glob.glob(os.path.join(transcript_dir, "*.txt"))
        assert len(txt_files) > 0, "Không tìm thấy file .txt nào trong thư mục."
        docs = self.load(txt_files, metadata_path, workers=workers)
        chunks = self.chunker(docs, output_dir)
        return chunks

if __name__ == "__main__":
    transcript_dir = "data/test/"
    metadata_path = "data/metadata.json"
    output_dir = "data/test_chunk/"
    loader = Loader(open_api_key=gptkey)
    chunks = loader.load_dir(transcript_dir, metadata_path, output_dir, workers=4)
    print(f"Loaded and chunked {len(chunks)} transcripts.")
