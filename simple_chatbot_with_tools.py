from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
import random
import json

from langchain_core.messages import ToolMessage

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("openai:gpt-4.1-nano")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

#tools
@tool
def get_bank_account_number(user_id: str):
    "Get the user's bank account number."
    return str(random.randint(1000000000, 9999999999))

@tool
def get_student_grade(student_id: str):
    "Get the student's grade."
    return random.choice(['A', 'B', 'C', 'D', 'F'])

tools = [get_bank_account_number, get_student_grade]

graph_builder = StateGraph(State)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

old_messages = []
config = {"configurable": {"thread_id": "1"}}
for _ in range(10):  # You can change 3 to any number of iterations you want
    user_input = input("User: ")
    if old_messages:
        state = graph.invoke({"messages": old_messages + [{"role": "user", "content": user_input}]},config=config)
    else:
        state = graph.invoke({"messages": [{"role": "user", "content": user_input}]},config=config)
    print(state)
    old_messages = state["messages"]