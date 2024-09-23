from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import os
from langchain_mongodb import MongoDBAtlasVectorSearch
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

MONGO_URI = os.environ["MONGO_URI"]
HF_kEY = os.environ["HF_kEY"] # Hugging Face API key for embeddings model
parse_output = StrOutputParser()  #Initialize an output parser for parsing LLM outputs
app = FastAPI()


# POST endpoint to generate chat responses using an LLM
@app.post('/generate')
async def generate_chat(
    request: Request # Request object to handle incoming user queries
):
    
    # Parse the incoming request data
    query = await request.json()

    userops_logger.info(
          f"""
          User Request:
          -----log prompt-----
          User data: {query}
          """
    )
    model = query["model"]  # Model name for the LLM
    temperature = query["temperature"] # Temperature parameter for controlling LLM output randomness
    session_id = query["session_id"] # Session ID for tracking user interactions

        # Initialize the LLM (ChatGroq) based on the provided model and temperature    
    llm = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=1024
        )
 
    # Set up the embedding model using Hugging Face Inference API for text embedding
    embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_kEY,
            model_name="BAAI/bge-small-en-v1.5"
        )
    
    # Initialize MongoDB-based vector store for retrieving document embeddings
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=MONGO_URI,
            namespace="Financial_statement" + "." + "fin_document",
            embedding=embeddings,
            index_name = "vector_index"
        )

     # Configure the retriever to retrieve the top 5 most similar documents based on embeddings  
    embeddings_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        
    # Function to define the initial prompt template for interacting with the LLM
    def get_prompt_template() -> ChatPromptTemplate:
                        prompt = """
                        You are a financial analysis assistant specialized in retrieving, analyzing, and summarizing financial information from documents such as balance sheets, income statements, and cash flow statements. You can perform tasks such as:
                            - Extracting key financial metrics (e.g., revenue, net income, assets, liabilities).
                            - Calculating financial ratios (e.g., profitability, liquidity, debt ratios).
                            - Identifying trends or anomalies across multiple periods.
                            - Providing insights on the financial health of companies.
                            - Summarizing sections of financial statements to answer specific queries.
                        When asked to perform calculations or generate reports, use the relevant financial data from the uploaded documents. Always provide accurate, context-relevant answers to assist investment analysts in making informed decisions.
                        """

                        chat_desc = "Your name is Johnny Sins."
                        system_prompt = "".join([chat_desc, prompt])

                        question_prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", system_prompt),
                                MessagesPlaceholder(variable_name="history"),
                                ("human", "{question}")
                            ]
                        )

                        
                        question = question_prompt | llm | parse_output
                        return RunnablePassthrough.assign(
                            context=question | embeddings_retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
                        )

      # Define the prompt template for retrieval-augmented generation (RAG)   
    def get_rag_template() -> ChatPromptTemplate:
        rag_system_prompt = """Answer the question based only on the following context: {context}"""
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rag_system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ]
        )

        retrieval_chain = get_prompt_template()
        rag_chain = retrieval_chain | rag_prompt | llm | parse_output
        return rag_chain
    
    # Function to fetch message history for a given session from MongoDB
    def get_message_history(session_id: str):
     return MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=MONGO_URI,
        database_name="Financial_statement",
        collection_name="chat_histories",
    )
    
     # Function to create an LLM with message history functionality for better user interactions
    def get_chat_llm_with_history():
        prompt_template = get_rag_template()
        return RunnableWithMessageHistory(prompt_template, get_message_history, input_messages_key="question", history_messages_key="history")

     # Interact with the LLM, passing the user's question and maintaining session history
    def interact_with_llm(user_question: str):
        chat_llm_with_history = get_chat_llm_with_history()
        config = {"configurable": {"session_id": session_id}}
        return chat_llm_with_history.invoke({"question": user_question}, config=config)
    
    # Generate the response by invoking the LLM interaction function
    response = interact_with_llm(query["question"])

     # Log the response from the LLM using a custom logger 
    llmresponse_logger.info(
          f"""
          LLM Response:
          -----log response-----
          Response: {response}
          """
    )
    
    # Return the generated response as a streamed response
    return StreamingResponse(content=response)
    
        
    
if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)
    