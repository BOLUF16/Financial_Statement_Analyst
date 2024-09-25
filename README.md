# Finacial Statement Assistant

## Documentation
This project is a Streamlit-based web application designed to assist investment analysts and finance professionals by providing an interactive platform to query and extract information from financial documents, such as income statements, balance sheets, and cash flow statements. The application simplifies the process of analyzing large volumes of financial data by allowing users to upload multiple documents, ask specific questions, and receive insightful responses based on the content of the documents

## Tech Stack
- Streamlit: For creating the interactive web app.
- Llamaindex: For reading the document, generating and storing the embeddings in a MongoDB database.
- Langchain: For adding memory to our rag and storing it in a MongoDB database.
- Fastapi: For the backend of the streamlit app.
- Pymongo: For connecting to my MongoDB uri.
- dotenv: To manage environment variables.

## Technical Walkthrough
### Important Note:
You'll need a Mongodb account for this to work.
- First, register for a [MongoDB Atlas account](https://www.google.com/url?q=https%3A%2F%2Fwww.mongodb.com%2Fcloud%2Fatlas%2Fregister). For existing users, sign into MongoDB Atlas.
- [Follow the instructions](https://www.google.com/url?q=https%3A%2F%2Fwww.mongodb.com%2Fdocs%2Fatlas%2Ftutorial%2Fdeploy-free-tier-cluster%2F). Select Atlas UI as the procedure to deploy your first cluster.
- Create the database: `Financial_statement`.
- Within the database `Financial_statement`, create the collection `fin_document`.
- Create a [vector search index](https://www.google.com/url?q=https%3A%2F%2Fwww.mongodb.com%2Fdocs%2Fatlas%2Fatlas-vector-search%2Fcreate-index%2F%23procedure%2F) named vector_index for the ‘listings_reviews’ collection. This index enables the RAG application to retrieve records as additional context to supplement user queries via vector search. Below is the JSON definition of the data collection vector search index.
  Your vector search index created on MongoDB Atlas should look like below:
  ```
  {
      "fields": [
        {
          "numDimensions": 384, #the dimension of the embedding model
          "path": "embedding",
          "similarity": "cosine",
          "type": "vector"
    }
  ]
  }
Follow MongoDB’s [steps to get the connection](https://www.google.com/url?q=https%3A%2F%2Fwww.mongodb.com%2Fdocs%2Fmanual%2Freference%2Fconnection-string%2F) string from the Atlas UI(which i refer to in my code as MONGO_URI). After setting up the database and obtaining the Atlas cluster connection URI, securely store the URI within your development environment.

### Running the App Locally:
* #### Clone the Repository:
   ```
    git clone <repository-url>
    cd <repository-directory>
* #### Create a Virtual Environment:
   ```
   # Open command prompt and activate conda base enviroment
    # - skip this step if your Command prompt opens in a base environment, indicated with `(base)` at the start of your command line

    # run this command for Anaconda
    "C:\ProgramData\Anaconda3\Scripts\activate.bat"

    # run this for Miniconda
    "C:\ProgramData\Miniconda3\Scripts\activate.bat"

    # If the above doesn't work, then you probably did User installation of Anaconda or Miniconda
    # - Navigate to the specific path it was installed in, and copy the full path to `activate.bat` file, then run it in your terminal.

    # Now that your command line is in the base environment, create a new virtual environment
    conda create -n your_desired_venv_name python=3.9

    # activate the environment
    conda activate your_desired_venv_name
* #### install Dependencies:
  `pip install -r requirements.txt`
* #### Run the Streamlit Application:
   `streamlit run streamlit.py`
* #### Run the Fastapi:
   `uvicorn src.app:app --host 127.0.0.1 --port 5000 --reload`

## Future Improvements
* Adding tools for additional functionality
* refining the system prompt to improve the accuracy and relevancy in querying
* optimizing the workflow to speed up the response time

