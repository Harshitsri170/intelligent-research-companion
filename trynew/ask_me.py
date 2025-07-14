# ask_me.py
from qa_pipeline import llm
import concurrent.futures

def answer_question(user_query, retriever):
    try:
        docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([doc.page_content for doc in docs])
        if not context.strip():
            return "No relevant content found in the document.", docs

        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(llm.invoke, prompt)
            answer = future.result(timeout=30)  # 30-second limit
            return answer, docs
    except concurrent.futures.TimeoutError:
        return "Timed out while generating response.", []
    except Exception as e:
        return f"Failed to answer the question: {e}", []