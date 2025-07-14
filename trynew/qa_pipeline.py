from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model/tokenizer once (faster + cached)
model_name = "google/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable GPU if available
device = 0 if torch.cuda.is_available() else -1

# HuggingFace pipeline setup
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=False,
    temperature=0.3,
    device=device
)

# LangChain wrapper
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# Embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Text chunker
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

# Build vector store retriever
def build_qa_index(raw_text):
    try:
        # Step 1: Chunk the document
        texts = text_splitter.split_text(raw_text)
        if not texts:
            raise ValueError("No text chunks were created. Check document content.")

        # Step 2: Create FAISS vector store
        vector_store = FAISS.from_texts(texts, embedding_model)

        # Step 3: Build retriever (top 4 results)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        return retriever, vector_store, len(texts)
    except Exception as e:
        print(f"[ERROR] build_qa_index (FAISS): {e}")
        raise