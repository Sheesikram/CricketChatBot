import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time
#tavily api
# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please configure it in your environment.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Cricket Live | ESPN Style",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Initialization
if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")

    #st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:1b")
    st.session_state.loader = WebBaseLoader("https://www.espncricinfo.com/")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Custom CSS for ESPN-like theme
st.markdown(
    """
    <style>
    .main { background-color: #ffffff; color: #000000; font-family: Arial, sans-serif; }
    .title { font-size: 32px; font-weight: bold; color: #d40a0a; text-align: center; margin-bottom: 20px; }
    .sidebar .sidebar-content { background-color: #333333; color: #ffffff; }
    .live-scores { background-color: #f8f9fa; border: 2px solid #d40a0a; border-radius: 10px; padding: 15px; margin-bottom: 20px; }
    .stButton button { background-color: #d40a0a; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "ChatBot", "Live Scores", "Schedule", "Stats"])

# Home Page
if menu == "Home":
    st.markdown("<div class='title'>🏏 Welcome to Shees CricketTalks Dashboard</div>", unsafe_allow_html=True)
    st.image("ore.PNG", width=400)

# ChatBot Page
elif menu == "ChatBot":
    st.title("Shees CricketTalks Chatbot")
    
    llm = ChatGroq(groq_api_key=groq_api_key, model="mixtral-8x7b-32768")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question:
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
    
    
    # Initialize chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Chat Input
    user_input = st.text_input("Ask your cricket-related questions:")
    if user_input:
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_input})
        st.write("Response time:", time.process_time() - start_time)
        st.write(response["answer"])

        # Similarity Search Results
        with st.expander("Relevant Context from Documents"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Live Scores Pageu
elif menu == "Live Scores":
    st.markdown("<h3>🔴 Live Scores</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='live-scores'>
            <h4>India vs Australia - 2nd ODI</h4>
            <p>India: 284/6 (50 Overs)</p>
            <p>Australia: 250/8 (47 Overs)</p>
            <p><strong>India Won by 34 Runs</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Schedule Page
elif menu == "Schedule":
    st.markdown("<h3>📅 Upcoming Matches</h3>", unsafe_allow_html=True)
    st.table(
        {
            "Match": ["India vs England", "Australia vs South Africa", "Pakistan vs Sri Lanka"],
            "Date": ["2024-11-27", "2024-11-28", "2024-11-29"],
            "Time": ["14:00", "16:00", "18:00"],
        }
    )

# Stats Page
elif menu == "Stats":
    st.markdown("<h3>📊 Player Stats</h3>", unsafe_allow_html=True)
    st.bar_chart(
        {
            "Virat Kohli": [75],
            "Babar Azam": [72],
            "Steve Smith": [70],
            "Joe Root": [68],
        }
    )
