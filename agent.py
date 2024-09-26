from langgraph.graph import StateGraph, START, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated


def get_weather(location: str):
    """Get The Weather Of Country"""
    if location.lower() == "pakistan":
        return {"location": "Pakistan", "temperature": 40}
    elif location.lower() == "germany":
        return {"location": "Germany", "temperature": 19}
    elif location.lower() == "canada":
        return {"location": "Canada", "temperature": 20}


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm_with_tools = llm.bind_tools([get_weather])


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(MessagesState)


def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder.add_node("assistant", tool_calling_llm)
builder.add_node("tools", ToolNode([get_weather]))


builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()
