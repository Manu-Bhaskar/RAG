from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

DB_PATH = "./vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"

embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model=LLM_MODEL)

# --- Prompt to judge relevance ---
grade_prompt = ChatPromptTemplate.from_template("""
Given the retrieved context and the question, decide if the context is useful.

Context:
{context}

Question:
{question}

Respond only with:
RELEVANT or NOT_RELEVANT
""")

# --- Normal RAG answer prompt ---
answer_prompt = ChatPromptTemplate.from_template("""
Answer ONLY using this context:

{context}

Question: {question}
""")

# --- Fallback prompt when context is weak ---
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


# --- Chat loop ---
print("CRAG ready (local only). Type exit to quit.\n")

while True:
    q = input("User: ")
    if q.lower() in ["exit", "quit"]:
        break

    answer = crag_chain.invoke(q)
    print("\nAI:", answer)
    print("-"*40)
