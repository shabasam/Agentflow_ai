from fastapi import FastAPI
from pydantic import BaseModel

from agents.llm import get_llm
from rag.ingest import load_and_split
from rag.vector_store import create_vector_store
from rag.retriever import get_retriever
from agents.graph import build_graph



app = FastAPI()
llm = get_llm()

chunks = load_and_split("knowledge/company_reports.txt")
vectorstore = create_vector_store(chunks)
retriever = get_retriever(vectorstore)
graph = build_graph()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
         
         print("TYPE:", type(retriever))
         
         relevant_docs = retriever.invoke(query.question)
         context = "\n".join([doc.page_content for doc in relevant_docs])

         result = graph.invoke({
            "question": query.question,
            "context": context,
            "plan": "",
            "answer": ""
            
         })

         try:
            structured_output = json.loads(result["answer"])
         except:
            structured_output = {
            "error": "Invalid JSON output from model",
            "raw_output": result["answer"]
             }

         return structured_output