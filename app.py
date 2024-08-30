# Import necessary libraries
import streamlit as st
import openai
from brain import get_index_for_mdx
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Set the title for the Streamlit app
st.title("Keploy RAG Chatbot")

# Azure OpenAI setup
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Function to get all MDX files from the docs directory and its subdirectories
def get_mdx_files(directory):
    mdx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                mdx_files.append(os.path.join(root, file))
    return mdx_files

# Cached function to create a vectordb for the provided MDX files
@st.cache_resource
def create_vectordb(files, filenames):
    with st.spinner("Creating vectordb..."):
        vectordb = get_index_for_mdx(files, filenames, openai.api_key)
    return vectordb

# Load MDX files from the docs folder and its subdirectories
docs_folder = os.path.join(os.getcwd(), "docs")
mdx_file_paths = get_mdx_files(docs_folder)


prompt_template = """
You are a knowledgeable assistant for Keploy, an API testing and mocking tool. Your role is to provide accurate, concise, and helpful information based on the Keploy documentation.

Instructions:
1. Answer questions directly and concisely.
2. Use information only from the provided context.
3. If the question can't be answered from the context, say "I don't have enough information to answer that question."
4. Cite sources by mentioning the filename at the end of relevant sentences, e.g., (source: filename.md).
5. If asked about code, provide explanations and examples when possible.
6. For multi-step processes, use numbered lists.
7. Highlight important terms or concepts using bold text.
8. If the user asks about a topic not related to Keploy, politely redirect them to Keploy-related questions.

Context from Keploy documentation:
{context}

Remember, your goal is to help users understand and use Keploy effectively.
"""

if mdx_file_paths:
    mdx_files = [open(f, "rb").read() for f in mdx_file_paths]
    mdx_file_names = [os.path.basename(f) for f in mdx_file_paths]
    st.session_state["vectordb"] = create_vectordb(mdx_files, mdx_file_names)

    # Create a conversational chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False, output_key="answer")
    llm = AzureChatOpenAI(
        azure_endpoint="https://chatsupportsys5416848984.cognitiveservices.azure.com/openai/deployments/chatbot-ai/chat/completions?api-version=2023-03-15-preview",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name="chatbot-ai",
        temperature=0.7
    )
    st.session_state["conversation_chain"] = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state["vectordb"].as_retriever(),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
else:
    st.error("No MDX files found in the docs folder.")
    st.stop()




# Initialize the chat history in session state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get the current prompt from the session state or set a default value
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

# Display previous chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get the user's question using Streamlit's chat input
question = st.chat_input("Ask anything")

# Handle the user's question
if question:
    vectordb = st.session_state.get("vectordb", None)
    if vectordb is None:
        st.error("No vectordb found in session state.")
        st.stop()
    
    search_results = vectordb.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in search_results])
    prompt_with_context = prompt_template.format(context=context)

    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state["conversation_chain"].stream({"question": question}):
            if "answer" in chunk:
                full_response += chunk["answer"]
                response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        # Display source information
        if 'source_documents' in chunk:
            st.write("Sources:")
            for doc in chunk['source_documents']:
                st.write(f"- {doc.metadata['source']}")

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

    # Rerun only when a new question is asked
    st.rerun()