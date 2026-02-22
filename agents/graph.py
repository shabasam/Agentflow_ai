from typing import TypedDict
from langgraph.graph import StateGraph
from agents.llm import get_llm
from pydantic import BaseModel
from typing import List

class FinalOutput(BaseModel):
    summary: str
    risks: List[str]
    recommendation: str
    confidence: float

class AgentState(TypedDict):
    question:str
    plan:str
    context:str
    answer:str

llm = get_llm()


def planner_node(state: AgentState):
    prompt = f"""
    Break down the following task into ashort plan:

    Task:
    {state['question']}
    """

    response = llm.invoke(prompt)

    return{
        "plan": response.content
    }

def answer_node(state: AgentState):

    prompt = f"""
    You are a business analyst.

    Use the context below to answer.

    Return your response STRICTLY in JSON format like this:

    {{
        "summary": "...",
        "risks": ["...", "..."],
        "recommendation": "...",
        "confidence": 0.0
    }}

    Context:
    {state['context']}

    Question:
    {state['question']}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


def critc_node(state: AgentState):
    prompt= f"""
    You are a strict reviewer.

    Check if the answer is supoorted by the context.

    Context:
    {state['context']}

    If the answer is well_supported, return it as is.
    If not, improve it using only the context.
    """

    response = llm.invoke(prompt)
    
    return{
        "answer": response.content
    }


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("answer", answer_node)
    graph.add_node("critic", critc_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "answer")
    graph.add_edge("answer", "critic")

    graph.set_finish_point("answer")

    return graph.compile()

