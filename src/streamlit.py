import streamlit as st
import requests

# List of available models for user to select
models = [
    # "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

# Backend URLs for file upload and chat generation services
BACKEND_URL_1 = "http://127.0.0.1:5000" # API for file upload
BACKEND_URL_2 = "http://127.0.0.1:9000"  # API for chat generation

# Function to upload files to the backend
def upload_files(files):
    url = f"{BACKEND_URL_1}/upload"
    files_data = [("files", file) for file in files]
    response = requests.post(url, files=files_data)
    return response.json()

# Function to generate chat response by interacting with backend
def generate_chat(question, model, temperature, session_id):
    url = f"{BACKEND_URL_2}/generate"
    payload = {
        "question": question,
        "model": model,
        "temperature": temperature,
        "session_id": session_id
        
    }
    response = requests.post(url, json=payload)
    return response.text

def main():
    # Title of the app
    st.markdown("<h1 style='text-align: center;'>Financial Document Assistant</h1>", unsafe_allow_html=True)
    
    # Centered image in a column layout
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
     st.image("https://th.bing.com/th/id/OIP.O8mZ0XXXbskfUKnSfRV11gHaEi?rs=1&pid=ImgDetMain", use_column_width=True)

    # Initialize session state to store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Initialize an empty chat history

    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        # Dropdown to select the model
        model = st.selectbox("Select model", models)  
         # Slider for selecting temperature (controls randomness in LLM responses)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
         # Text input for session ID (used to track user conversations)
        session_id = st.text_input("Enter session id")
        
         # Section for document upload
        st.header("Document Upload")
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True) # File uploader for multiple files
        if st.button("Process Documents"):# Button to trigger file upload process
            if uploaded_files : # Check if files are uploaded
                with st.spinner("Processing documents..."):# Show spinner while processing
                    result = upload_files(uploaded_files)# Call upload_files function to process files
                    if result.get("status_code") == 200:
                        st.success(result.get("detail", "Documents processed successfully"))
                    else:
                        st.error(result.get("detail", "Error processing documents"))
            else:
                st.warning("Please upload files") #Show warning if no files are uploaded
        
         # Display chat history in the sidebar
        st.header("Chat History")
        response_container = st.container()
        with response_container:
             # Loop through chat history and display in reverse order (latest first)
            for chat in reversed(st.session_state.chat_history):
                st.markdown(f"""
                <div style="background-color: #000000; padding:10px; border-radius:5px;">
                    <p style="font-family:Arial, sans-serif;"><strong>You:</strong> {chat['human']}</p>
                    <p style="font-family:Arial, sans-serif;"><strong>Bot:</strong> {chat['Ai']}</p>
                </div>
                """, unsafe_allow_html=True)
    
     # Main content section for chat interaction
    st.header("Chat Generation")
    question = st.text_input("Enter your question")
    

    if st.button("Generate Response"):# Button to trigger chat generation
        if question:# Ensure that a question is provided
            with st.spinner("Generating response..."): # Show spinner while generating response
                response = generate_chat(question, model, temperature, session_id) # Call the generate_chat function
                st.write("Response:", response)# Display the response
                 # Append the interaction to the chat history stored in session state
                st.session_state.chat_history.append({'human': question, 'Ai': response})
                
        else:
            st.warning("Please enter a question and ensure you have processed documents for the project") # Warn user if no question is entered
    

if __name__ == "__main__":
    main()