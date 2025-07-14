import streamlit as st
from extract_text import extract_text_from_pdf 
from qa_pipeline import build_qa_index
from summarizer import summarize_text
from ask_me import answer_question
from challenge_me import run_challenge_mode

# === Streamlit Config ===
st.set_page_config(page_title="Smart Research Assistant AI", layout="wide")

# === Styling ===
st.markdown("""
    <style>
    html, body, .main {
        background-color: #F5F7FA;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 4rem;
    }
    .section-summary {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .section-summary h4 {
        color: #000000;
    }
    .section-ask {
        background-color: #6A8ACD;;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .section-challenge {
        background-color: #C2798B;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background-color: #6C63FF;
        color: white;
        border: none;
        padding: 0.5rem 1.1rem;
        border-radius: 10px;
        font-weight: 500;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4B47F5;
        transform: scale(1.03);
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #333333;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .header-box {
        background-color: #1F2937;
        color: white;
        padding: 1rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .header-box h2 {
        color: #90cdf4;
    }
    a {
        color: #90cdf4;
        text-decoration: none;
    }
    a:hover {
        color: #63b3ed;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# === Top Header ===
st.markdown("""
<div class='header-box'>
    <h2>📄 Intelligent Research Companion</h2>
    <p>Streamline your document understanding with AI-powered summarization, question answering, and logical reasoning.</p>
    <p>👨‍💻 <strong>Harshit Srivastava</strong> &nbsp;&nbsp;
    📧 <a href="mailto:harshitsrivastava170@gmail.com" target="_blank">harshitsrivastava170@gmail.com</a> &nbsp;|&nbsp;        
    <a href="https://github.com/Harshitsri170" target="_blank">🐙 GitHub</a> &nbsp;|&nbsp;
    <a href="https://www.linkedin.com/in/harshit-srivastava-ai" target="_blank">💼 LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)

# === Upload PDF/TXT ===
uploaded_file = st.file_uploader("📁 Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.clear()
        st.session_state.last_uploaded_filename = uploaded_file.name

    if "retriever" not in st.session_state or st.session_state.retriever is None:
        try:
            with st.spinner("⏳ Extracting and summarizing document..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                st.session_state.raw_text = raw_text
                summary = summarize_text(raw_text[:5000])
                st.session_state.summary = summary
                retriever, vector_store, num_chunks = build_qa_index(raw_text)
                st.session_state.retriever = retriever
                st.session_state.vector_store = vector_store
        except Exception as e:
            st.error(f"Failed to process file: {e}")


# === Auto-Generated Summary ===
if "summary" in st.session_state and st.session_state.summary:
    st.markdown("<div class='section-summary'><h4>📌 Auto-Generated Summary:</h4>", unsafe_allow_html=True)
    st.markdown(f"<p>{st.session_state.summary}</p></div>", unsafe_allow_html=True)


# === Main 2 Sections: Ask & Challenge ===
if "retriever" in st.session_state and st.session_state.retriever:
    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.markdown("<div class='section-ask'><h4>💬 Ask Anything (Q&A)</h4>", unsafe_allow_html=True)
            user_query = st.text_input("Ask a question from the document:")
            if user_query:
                with st.spinner("Generating answer..."):
                    answer, docs = answer_question(user_query, st.session_state.retriever)
                st.markdown(f"**Answer:**\n\n{answer}")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append((user_query, answer))
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.container():
            st.markdown("<div class='section-challenge'><h4>🧩 Challenge Me (Quiz)</h4>", unsafe_allow_html=True)
            run_challenge_mode(st.session_state.raw_text)
            st.markdown("</div>", unsafe_allow_html=True)


# === Chat History ===
if "chat_history" in st.session_state and st.session_state.chat_history:
    st.markdown("---")
    st.subheader("📜 Chat History")
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**Ans:** {a}")
