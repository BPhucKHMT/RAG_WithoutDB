from typing import List, Union, Dict
import glob
import json
import os
import re
from tqdm import tqdm
import multiprocessing


from dotenv import load_dotenv

from vector_store.vectorstore import VectorDB # để kiểm tra các file đã chunk chưa
from text_splitters.semantic import TranscriptChunker # để chunk transcript

# ==========================================================
'''
Cấu trúc thư mục transcript
data
|-- playlist1
|    |-- transcripts
|    |    |-- raw_transcript
|    |    |--processed_transcript --> target folder  để lấy các transcript ra chunk
|    |metadata.json --> file metadata chung cho cả playlist
|-- playlist2
|    |-- transcripts
|        |-- raw_transcript
|        |--processed_transcript --> target folder  để lấy các transcript ra chunk
|    |metadata.json --> file metadata chung cho cả playlist
'''
#=========================================================
# cần hàm kiểm tra xem trước đây đã load playlist đó chưa, nếu rồi thì thôi không load lại
load_dotenv()
gptkey = os.getenv("myAPIKey")

# các đường dẫn
root_data_dir = "data_test/"
transcript_dir = "transcripts/processed_transcripts/"
metadata_dir = "metadata.json"
output_dir = "chunks/"
# ==========================================================


def get_num_cpu() -> int:
    """Trả về số CPU khả dụng."""
    return multiprocessing.cpu_count()

def parse_transcript(file_path: str) -> tuple[str, list[dict], str]:
        """Đọc file transcript, tách từng dòng thành block có start-end-text"""
        try: 
            full_text = ""
            position_map = []  # lưu vị trí start của mỗi đoạn text trong full_text

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "[âm nhạc]" in line.lower():
                        continue
                    filename = os.path.basename(file_path).replace(".txt", "") # lấy file name
                    match = re.match(r"(\d+:\d+:\d+)\s*-\s*(\d+:\d+:\d+),\s*(.+)", line)
                    if match:
                        start, end, text = match.groups()
                        pos = len(full_text)
                        full_text += text + " "
                        position_map.append({
                            "start": start,
                            "end": end,
                            "text": text,
                            "pos_start": pos, # vị trí bắt đầu của đoạn text trong full_text
                            "pos_end": len(full_text) # vị trí kết thúc của đoạn text trong full_text
                        })
        except Exception as e:
            print(f"Lỗi khi phân tích file {file_path}: {e}")
            return "", [], ""

        return full_text.strip(), position_map, filename

def map_metadata(metadata_path: str, filename: str) -> tuple[Union[str, None], Union[str, None]]:
    """Đọc file metadata và ánh xạ filename sang title và url."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        playlist_name = metadata_list["playlist_id"] # id playlist
        metadata = metadata_list["videos"]
        title, url = next(
            ((item["title"], item["url"]) for item in metadata if item["video_id"] == filename),
            (None, None)
        )
    except Exception as e:
        print(f"Lỗi khi đọc metadata từ {metadata_path}: {e}")
        return None, None, None
    return title, url, playlist_name


def load_transcript(file_path: str, metadata_path: str) -> dict:
    """Load 1 file transcript và ánh xạ metadata."""
    full_text, position_map, filename = parse_transcript(file_path)
    title, url, playlist_name = map_metadata(metadata_path, filename)
    return {
        "full_text": full_text,
        "position_map": position_map,
        "playlist" : playlist_name, 
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

    def __call__(self, txt_files: List[str], metadata_path: str, workers: int = 2) -> List[dict]:
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




class Loader:
    """Class chính: Load + Chunk transcript."""
    def __init__(self, open_api_key: str, split_kwargs: dict = None) -> None:
        self.transcript_loader = TranscriptLoader()
        self.chunker = TranscriptChunker(open_api_key=open_api_key)
        self.split_kwargs = split_kwargs or {}
        self.db = VectorDB().db  # khởi tạo vector db để kiểm tra các file đã chunk

    def load(self, txt_files: Union[str, List[str]], metadata_path: str, workers: int = 1):
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        docs = self.transcript_loader(txt_files, metadata_path=metadata_path, workers=workers)
        return docs
    
    def get_filename_already_chunks(self, chroma_db) -> set: # lấy ra các filename đã được chunk
        try:
            collection = chroma_db._collection

            results = collection.get(
                limit=1000000  # giả sử không có quá 1 triệu chunk
            )
            filenames = set()
            metadatas  = results.get("metadatas", [])
            filenames = {meta.get("filename") for meta in metadatas if meta and "filename" in meta}
            return filenames
        except Exception as e:
            print(f"Lỗi khi lấy filenames từ vector DB: {e}")
            return set()


    def load_dir(self, root_data_dir: str, transcript_dir: str, metadata_dir: str, output_dir: str, workers: int = 1):
        all_chunks = []
        try:
            filename_already_chunked =  self.get_filename_already_chunks(chroma_db=self.db)  # truyền db nếu cần kiểm tra
        except Exception as e:
            print(f"Không thể kết nối đến vector DB: {e}")
            filename_already_chunked = set()

        for playlist_folder in os.listdir(root_data_dir):
            try:
                playlist_path = os.path.join(root_data_dir, playlist_folder) # data/playlist1
                output_path = os.path.join(output_dir, playlist_folder) # data/chunks/playlist1
                os.makedirs(output_path, exist_ok=True) #tạo output folder nếu chưa có

                metadata_path = os.path.join(playlist_path, metadata_dir) # data/playlist1/metadata.json
                transcript_path = os.path.join(playlist_path, transcript_dir) # data/playlist1/transcripts/processed_transcript

                pattern =  os.path.join(transcript_path, "*.txt")
                txt_list = glob.glob(pattern)
                print(f"Tìm thấy {len(txt_list)} file transcript trong playlist {playlist_folder}.")
                txt_files = [f for f in txt_list if os.path.basename(f).replace(".txt", "") not in filename_already_chunked]
                print(f"Có {len(txt_files)} file cần được load và chunk trong playlist {playlist_folder}.")
                # kiểm tra nếu tất cả các file đã được chunk
                assert len(txt_files) > 0, "Tất cả các transcript đã được chunk"

                docs = self.load(txt_files, metadata_path, workers=workers)

                chunks = self.chunker(docs, output_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Lỗi khi xử lý playlist {playlist_folder}: {e}")

        return all_chunks
        


if __name__ == "__main__":
    loader = Loader(open_api_key=gptkey)
    chunks = loader.load_dir(root_data_dir=root_data_dir, transcript_dir=transcript_dir, metadata_dir=metadata_dir, output_dir=output_dir, workers=2)
    print(f"Loaded and chunked {len(chunks)} transcripts.")
    
