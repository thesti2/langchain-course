from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from nodes import run_agent_reasoning, tool_node
import os

load_dotenv()  # Load environment variables from .env file

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
# flow.set_entry_point(AGENT_REASON)                  not really required. this is the old way
flow.add_node(ACT, tool_node)

flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow_diagram.png")

if __name__ == "__main__":
    # Your main code logic here
    print("This script is being run directly.")
    res = app.invoke({"messages": [HumanMessage(content="What is the temparature in Tokyo? List the temperature and then use a tool to triple the temperature value.")]})

    print(res["messages"][LAST].content)
