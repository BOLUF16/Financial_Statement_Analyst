import os, json
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, ServiceContext, StorageContext
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("----")
from dotenv import load_dotenv
load_dotenv


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
print(GROQ_API_KEY)

models = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

def upload_dir(dir):

    from llama_index.core.node_parser import TokenTextSplitter

    print("uploading...")
    documents = SimpleDirectoryReader(dir).load_data()

    Settings.text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=50)
    index = VectorStoreIndex.from_documents(documents)

    return index

def qa_engine(query: str, index, llm_client):
    query_engine = index.as_query_engine(llm = llm_client, similarity_top_k = 5)
    response = query_engine.query(query)

    return response