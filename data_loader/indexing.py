from vector_store.vectorstore import VectorDB
from data_loader.file_loader import Loader
import os
from dotenv import load_dotenv
load_dotenv()

gptkey = os.getenv("myAPIKey")

# các đường dẫn
root_data_dir = "data/"
transcript_dir = "processed_transcripts/"
metadata_dir = "metadata.json"
output_dir = "chunks/"

def index_data():
    vector_db = VectorDB().db
    loader = Loader(open_api_key=gptkey, vector_db=vector_db)
    chunks = loader.load_dir(
        root_data_dir=root_data_dir,
        transcript_dir=transcript_dir,
        metadata_dir=metadata_dir,
        output_dir=output_dir,
        workers=2
    )
    print('Đang index tài liệu vào vector database...')
    vector_db.add_documents(chunks)
    print(f"Indexed {len(chunks)} documents into the vector database.")

if __name__ == "__main__":
    index_data()