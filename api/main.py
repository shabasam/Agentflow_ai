from fastapi import FastAPI
from pydantic import BaseModel

from agents.llm import get_llm
from rag.ingest import load_and_split_folder
from rag.vector_store import create_or_load_vector_store
from rag.retriever import get_retriever
from agents.graph import build_graph
import json
import time
import re

app = FastAPI()
llm = get_llm()
MIN_CONFIDENCE = 0.6

chunks = load_and_split_folder("knowledge")
vectorstore = create_or_load_vector_store(chunks)
retriever = get_retriever(vectorstore)
graph = build_graph()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
         
         start_time = time.time()
         
         relevant_docs = retriever.invoke(query.question)
         context = "\n".join([doc.page_content for doc in relevant_docs])

         result = graph.invoke({
            "question": query.question,
            "context": context,
            "plan": "",
            "answer": ""
            
         })

         end_time = time.time()

         latency = round(end_time - start_time, 3)

         try:

            raw_text = result["answer"]

            match = re.search(r"\{.*\}", raw_text, re.DOTALL)

            if not match:
                 raise ValueError("no jason found")
            
            json_string = match.group()

            structured_output = json.loads(json_string)

            structured_output["latency_seconds"] = latency
            
            confidence = structured_output.get("confidence", 0)

            if confidence<MIN_CONFIDENCE:
                 structured_output["warning"] = "Low confidence response. Review recommended."

            return structured_output     
         except Exception:

            return {
            "error": "Invalid JSON output from model",
            "raw_output": result["answer"],
            "latency_seconds": latency

             }

        