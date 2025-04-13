import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# Optional Langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Set page config (streamlit theme toggle is enabled)
st.set_page_config(page_title="Kumaa Chatbot", page_icon="🤖", layout="wide")

# Add user interface
st.title("🤖 Welcome to Kumaa — Your AI Assistant")

# User input box
input_text = st.text_input("💭 Ask me anything...")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input handling
if input_text:
    # Add user message to session history
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    # LangChain prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ])

    # LLM and chain setup
    llm = Ollama(model="gemma:2b")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    with st.chat_message("assistant"):
        try:
            response = chain.invoke({"question": input_text})
            st.markdown(response)
            # Store assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
