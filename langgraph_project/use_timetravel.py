from typing import Annotated
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TONGYI_API_KEY")
base_url = os.getenv("TONGYI_BASE_URL")
tavily_api_key = os.getenv("Travil_API_KEY")
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)


tool = TavilySearch(max_results=2,api_key=tavily_api_key)

tools = [tool]

llm = ChatOpenAI(
    model = "qwen-plus",
    api_key = api_key,
    base_url = base_url
)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state:START):
    return {"message":[llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools",tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)



config = {"configurable":{"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
