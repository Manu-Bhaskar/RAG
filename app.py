import streamlit as st
import os
import shutil
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ---------------- CONFIG ----------------
DATA_PATH = "./data/docs"
DB_PATH = "./vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

LOG_FILE = "qa_bleu_log.json"

def compute_bleu(reference_text, generated_text):
    smoothie = SmoothingFunction().method4
    reference_tokens = reference_text.split()
    generated_tokens = generated_text.split()

    return sentence_bleu(
        [reference_tokens],
        generated_tokens,
        smoothing_function=smoothie
    )

def save_result(question, answer, bleu):
    entry = {
        "question": question,
        "answer": answer,
        "bleu_score": round(bleu, 4)
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def add_in_batches(db, docs, batch_size=1000):
    for i in range(0, len(docs), batch_size):
        db.add_documents(docs[i:i+batch_size])

os.makedirs(DATA_PATH, exist_ok=True)

st.set_page_config(page_title="Local CRAG Chat", layout="wide")
st.title("🧠 Local Corrective RAG Assistant")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Documents")

    files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if files:
        for f in files:
            with open(os.path.join(DATA_PATH, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("Uploaded!")

    if st.button("➕ Add to Vector Database"):

        st.cache_resource.clear()  # release any open DB handles

        with st.spinner("Adding new documents..."):

            embedding = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL
            )

            # Load or create DB
            if os.path.exists(DB_PATH):
                db = Chroma(
                    persist_directory=DB_PATH,
                    embedding_function=embedding
                )
            else:
                db = Chroma.from_documents(
                    documents=[],
                    embedding=embedding,
                    persist_directory=DB_PATH
                )

            loaders = [
                DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader),
                DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader),
            ]

            docs = []
            for loader in loaders:
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(docs)

            # 🔥 Add instead of rebuild
            add_in_batches(db, chunks)

        st.success("New documents added to vector DB!")
        st.rerun()

    # ---- Show DB size ----
    if os.path.exists(DB_PATH):
        embedding = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding
        )

        count = db._collection.count()
        st.info(f"📊 Vector entries: {count}")
    else:
        st.info("📊 Vector entries: 0")



# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_chain():
    embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model=LLM_MODEL)

    grade_prompt = ChatPromptTemplate.from_template("""
Given the retrieved context and the question, decide if the context is useful.

Context:
{context}

Question:
{question}

Respond only with:
RELEVANT or NOT_RELEVANT
""")

    answer_prompt = ChatPromptTemplate.from_template("""
Use the following context ONLY as reference material.

Explain the answer clearly but concisely.
Do not quote the text directly unless necessary.

Context:
{context}

Question: {question}
""")

    fallback_prompt = ChatPromptTemplate.from_template("""
The provided documents do not contain enough information.

Politely say that you don’t have sufficient context to answer accurately.
Question: {question}
""")

    def grade_context(inputs):
        result = llm.invoke(
            grade_prompt.format(
                context=inputs["context"],
                question=inputs["question"]
            )
        )
        return "RELEVANT" in result.upper()

    def final_answer(inputs):
        if inputs["relevant"]:
            return llm.invoke(
                answer_prompt.format(
                    context=inputs["context"],
                    question=inputs["question"]
                )
            )
        else:
            return llm.invoke(
                fallback_prompt.format(question=inputs["question"])
            )

    crag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(lambda x: {
            "context": x["context"],
            "question": x["question"],
            "relevant": grade_context(x)
        })
        | RunnableLambda(final_answer)
    )

    return crag_chain


chain = load_chain() if os.path.exists(DB_PATH) else None

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Ask your question..."):

    if not chain:
        st.warning("Please build the vector database first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Run CRAG
            answer = chain.invoke(prompt)

            # Retrieve context again for BLEU reference
            embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
            db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            docs = retriever.invoke(prompt)

            reference_text = " ".join([d.page_content for d in docs])

            bleu = compute_bleu(reference_text, answer)

            save_result(prompt, answer, bleu)

            st.markdown(answer)
            st.caption(f"📊 BLEU Score: {round(bleu,4)}")


    st.session_state.messages.append({"role": "assistant", "content": answer})
