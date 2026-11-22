
from fastapi import FastAPI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

loader = PyPDFLoader("dell_latitude_5420_troubleshooting_demo.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = splitter.split_documents(pages)

embeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",api_key = GEMINI_API_KEY)
db = FAISS.from_documents(chunks,embeddings)
llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_template("""
You are a laptop troubleshooting assistant.

Answer using only the context provided below.

If the answer is not available in the context,say:
"I could not find that information in the manual."

Context
{context}

Question
{question}

Give a clear, easy-to-understand answer.
""")

def ask_question(question):
    # Step 1: Search FAISS for relevant text/chunks
    results = db.similarity_search(question, k=4)  # 4 relevant chunks
    # Step 2: Combine the retrieved text
    context = "\n\n".join(
        doc.page_content for doc in results
    )
    # Step 3: Prepate the prompt
    messages = prompt.format_messages(
        context = context,
        question = question
    )
    # Step 4: Ask the LLM
    response = llm.invoke(messages)
    # Step 5: Display the answer
    print("ANSWER:\n")
    print(response.content)
    # Step 6: Show the retrieved source pages
    print("\nSOURCES:\n")
    for doc in results:
        print("Page:",doc.metadata.get("page",0)+1)
        print(doc.page_content[:200])
        print("-"*40)
    return response.content

app = FastAPI(title="Laptop Troubleshooting API")

@app.get("/")
def home():
    return {"message":"LAPTOP RAG API is running"}
@app.get("/ask")
def ask(question:str):
    return {
        "question": question,
        "answer" : ask_question(question)
    }
