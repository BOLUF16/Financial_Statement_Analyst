import os,sys
from werkzeug.utils import secure_filename
from src.exception.operationshandler import system_logger
# from src.main import *
from llama_index.core import VectorStoreIndex
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
import os
from langchain_mongodb import MongoDBAtlasVectorSearch as LangChainMongoDBVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from src.utils.helpers import *
from src.exception.operationshandler import userops_logger, llmresponse_logger
from dotenv import load_dotenv
import pymongo

load_dotenv()
MONGO_URI = os.environ["MONGO_URI"]

allowed_files = ["txt", "csv", "json", "pdf", "doc", "docx", "pptx"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_files

def file_checks(files):
    if not files:
        return {
            "detail": "No file found",
            "status_code": 400
        }
    
    for file in files:
        if not file and file.filenames == "":
            return {
                "detail": "No file found",
                "status_code": 400
            }
        
        if not allowed_file(file.filename):
            print(file.filename)
            return {
                "detail": f"File format not supported. use any of {allowed_files}",
                "status_code": 415
            }
    return {
        "detail" : "success",
        "status_code": 200
    }

async def upload_files(files, temp_dir):

    checks = file_checks(files)
    if checks["status_code"] == 200:
        try:
            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)

                file_obj = await file.read()

                with open(file_path, "wb") as buffer:
                    buffer.write(file_obj)

            return {
                "detail" : "Upload completed",
                "status_code": 200
            }
        except Exception as e:
            message = f"An error occured during upload: {e}"
            system_logger.error(
                message,
                exc_info=1
            )
            raise Customexception(e,sys)

class Customexception(Exception):

    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
         return "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        self.file_name, self.lineno, str(self.error_message))


    

