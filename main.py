from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

# Load the embedding model and vectorstore
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)

# Load the FAISS vectorstore
embedding_vectorestore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = embedding_vectorestore.as_retriever(search_type="similarity")

# Initialize the LLM
llm = OllamaLLM(model="llama3.2:1b")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.post("/query/")
async def query_rag(request: QueryRequest):
    try:
        result = qa.invoke(request.query)
        return {"query": request.query, "result": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
