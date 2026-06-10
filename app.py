import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# Page configuration
st.set_page_config(
    page_title="Acrux Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom premium styling (CSS)
st.markdown("""
<style>
    /* Premium Styling and Dark Theme Feel */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        background: linear-gradient(90deg, #8A2387, #E94057, #F27121);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .chat-container {
        border-radius: 16px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 2px 18px;
        margin-bottom: 10px;
        text-align: right;
        display: inline-block;
        float: right;
        clear: both;
        max-width: 80%;
    }
    
    .bot-msg {
        background: rgba(255, 255, 255, 0.08);
        color: #f1f1f1;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        margin-bottom: 10px;
        display: inline-block;
        float: left;
        clear: both;
        max-width: 80%;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .refusal-msg {
        border-left: 4px solid #E94057;
        background: rgba(233, 64, 87, 0.1);
    }
    
    .brand-title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .brand-subtitle {
        text-align: center;
        color: #888;
        font-size: 14px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="brand-title">ACRUX DYNAMICS</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">HR Help Desk — Intelligent Policy Assistant</div>', unsafe_allow_html=True)

# Load secrets and setup env
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

# Initialize session state for LLM and Vector Store
@st.cache_resource
def get_vectorstore_and_llm():
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Load FAISS index
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = None
        
    # Initialize LLM
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0.1, max_tokens=512)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=0.1, max_tokens=512)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.1, max_output_tokens=512)
    else:
        llm = None
        
    return vectorstore, llm

vectorstore, llm = get_vectorstore_and_llm()

if vectorstore is None or llm is None:
    st.error("Error: FAISS Vectorstore or LLM not initialized. Make sure you run the starter notebook first!")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Setup RAG and Guardrails
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful HR help desk assistant for Acrux Dynamics.
Answer the question based ONLY on the following retrieved context.
If the context does not contain the answer, state: "I don't have that information in the documentation."
Do not make up facts or assume anything not directly mentioned in the context.

Context:
{context}"""),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a guardrail classifier for an HR Help Desk chatbot at Acrux Dynamics.
Your task is to classify whether the user's question is IN-SCOPE or OUT-OF-SCOPE.

IN-SCOPE queries:
- Questions about Acrux Dynamics HR policies, leaves, holidays, benefits, insurance, salary, payroll, performance reviews, PIP, WFH (work from home) arrangements, ESOPs, recruitment/hiring process, or general workplace queries for employees.

OUT-OF-SCOPE queries:
- Questions about other companies (like Zoho, Freshworks, Salesforce, etc.).
- Questions about technical product features (like AcruxCRM vs Salesforce).
- Financial performance or revenue of Acrux Dynamics.
- General questions unrelated to HR policies (e.g., cooking, programming, news, math).

Respond with exactly one word: "IN-SCOPE" or "OUT-OF-SCOPE". Do not explain.
"""),
    ("human", "{question}")
])

REFUSAL_MESSAGE = "I am sorry, but I can only assist with questions related to Acrux Dynamics HR policies and procedures."

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@traceable(name="rag_chain_app")
def rag_chain(question: str):
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)

@traceable(name="ask_bot_app")
def ask_bot(question: str):
    classification_chain = OOS_PROMPT | llm | StrOutputParser()
    cls = classification_chain.invoke({"question": question}).strip().upper()
    if "OUT-OF-SCOPE" in cls or "IN-SCOPE" not in cls:
        return REFUSAL_MESSAGE
    try:
        return rag_chain(question)
    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar configuration / metadata info
with st.sidebar:
    st.markdown("### 🏢 Acrux Dynamics")
    st.markdown("Welcome to the internal policy chatbot portal.")
    st.markdown("---")
    st.markdown("**Example Questions:**")
    st.markdown("- *How many maternity leave weeks am I entitled to?*")
    st.markdown("- *What is the carry forward limit for Earned Leaves?*")
    st.markdown("- *When is the payroll cut-off date?*")
    st.markdown("- *What is L4 Grade Compensation?*")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about Acrux Dynamics HR policies..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Retrieving policies..."):
            response = ask_bot(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})