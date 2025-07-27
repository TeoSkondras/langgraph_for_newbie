from typing import Annotated, Literal
from pydantic import BaseModel, Field

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("openai:gpt-4.1-nano")

class MessageClassifer(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message needs an emotional or logical response."
    )

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifer)

    result = classifier_llm.invoke(
        [{"role": "system", "content": "Classify the message if it needs an emotional or logical response."},
         {"role": "user", "content": last_message.content}]
    )
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}

def therapist_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system", "content": "You are a therapist. Respond to the user's message with empathy and understanding."},
        {"role": "user", "content": last_message.content}
    ]
    response = llm.invoke(messages)
    # Extract string content if response is a message object
    content = response.content
    return {"messages": [{"role": "assistant", "content": content}]}

def logical_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system", "content": "You are a logical agent. Respond to the user's message with logic and reasoning."},
        {"role": "user", "content": last_message.content}
    ]
    response = llm.invoke(messages)
    # Extract string content if response is a message object
    content = response.content
    return {"messages": [{"role": "assistant", "content": content}]}

graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist_agent", therapist_agent)
graph_builder.add_node("logical_agent", logical_agent)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges("router", 
                                    lambda state: state.get("next"),
                                    {"therapist": "therapist_agent", 
                                     "logical": "logical_agent"})
graph_builder.add_edge("therapist_agent", END)
graph_builder.add_edge("logical_agent", END)

graph = graph_builder.compile()

def run_chatbot():
    state = {"messages": [], "message_type": None, "next": None}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chatbot.")
            break

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_input}]
        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Chatbot: {last_message.content}")
            print(state)

if __name__ == "__main__":
    run_chatbot()