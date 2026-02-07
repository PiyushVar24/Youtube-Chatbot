import streamlit as st
import requests
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# Streamlit config
st.set_page_config(page_title="YouTube RAG Chat", layout="centered")
st.title("🎥 Chat with YouTube Video")
st.write("Ask questions and get answers **only from the video content**.")


# Helpers
def fetch_transcript(video_url):
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "youtube-transcript3.p.rapidapi.com",
    }
    params = {"url": video_url, "lang": "en"}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        raise ValueError("Transcript API failed")

    data = response.json()
    transcript = data.get("transcript", [])

    if not transcript:
        raise ValueError("Transcript not available")

    return " ".join([item["text"] for item in transcript])


def build_rag_chain(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(k=4)

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know."

Context:
{context}

Question: {question}
""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# UI
video_url = st.text_input(
    "🔗 Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=xxxxx",
)

if st.button("📄 Process Video"):
    if not video_url:
        st.warning("Please enter a YouTube URL")
    else:
        with st.spinner("Fetching transcript & building index..."):
            try:
                transcript_text = fetch_transcript(video_url)
                st.session_state.rag_chain = build_rag_chain(transcript_text)
                st.success("Video processed successfully! 🎉")
            except Exception as e:
                st.error(str(e))


# Chat Section
if "rag_chain" in st.session_state:
    question = st.text_input("💬 Ask a question about the video")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = st.session_state.rag_chain.invoke(question)
            st.markdown("### ✅ Answer")
            st.write(answer)
