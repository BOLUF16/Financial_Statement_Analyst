import tempfile, traceback, asyncio
from typing import List, Literal, Any
from fastapi import FastAPI, Request, UploadFile, Depends
from fastapi.responses import PlainTextResponse,StreamingResponse
from src.utils.helpers import *
from src.main import *
from src.exception.operationshandler import userops_logger
from dotenv import load_dotenv
import pymongo

load_dotenv()
MONGO_URI = os.environ["MONGO_URI"] # Retrieve MongoDB URI from environment variables
app = FastAPI() # Initialize FastAPI app

# Health check endpoint to verify if the API is running successfully
@app.get('/healthz')
async def health():
   
    return {
        "application": "Simple LLM API",
        "message": "running successfully"
    }

# Endpoint to upload and process files
@app.post('/upload')
async def process(
    files: List[UploadFile] = None,):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:# Create a temporary directory to store uploaded files during processing
            _uploaded = await upload_files(files, temp_dir)
            userops_logger.info(
                f"""
                User Request:
                -----log response-----
                User data: {_uploaded}
                """
            )
            if _uploaded["status_code"] == 200:# If files were uploaded successfully
                documents = SimpleDirectoryReader(temp_dir).load_data()# Load the files as documents
                
                # Initialize MongoDB client and connect to the specified database and collection
                mongodb_client = pymongo.MongoClient(MONGO_URI)
                store = MongoDBAtlasVectorSearch(mongodb_client = mongodb_client,db_name="Financial_statement",collection_name="fin_document")
                
                # Create a storage context with the MongoDB vector store
                storage_context = StorageContext.from_defaults(vector_store=store)
                try:
                 # Create a VectorStoreIndex from the loaded documents and store it in MongoDB
                 VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, show_progress=True)
                 return {
                     "detail": "Embeddings generated succesfully",
                     "status_code": 200
                 }
                except Exception as e:
                    print(str({e}))                  
            else:
                return _uploaded
    except Exception as e:
         print(traceback.format_exc())
         return {
            "detail": f"Could not process documents: {str(e)}",
            "status_code": 500
        }

    
    
if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)