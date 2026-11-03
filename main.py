from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

import os

load_dotenv()


tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# @tool
# def search(query: str) -> str:
#     """
#     A Tool that searches the internet
#     Args:
#         query (str): The search query
#     Returns:
#         str: The search results
#     """
#     print(f"Search for {query}")
#     return tavily_client.search(query=query)


llm = ChatOpenAI(
    model_name="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(llm, tools=[search])



def main():
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 jobs advertisement for AI engineer in Singapore")]})
    print(result)


if __name__ == "__main__":
    main()