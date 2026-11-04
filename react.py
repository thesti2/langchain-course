from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

import os

load_dotenv()  # Load environment variables from .env file


@tool
def triple(num: float) -> float:
    """
    Triples the value of the input number.
    param num: a number to triple the value
    return: the tripled value of the input number
    """
    return num * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="openrouter/free", 
                 temperature=0, 
                 base_url="https://openrouter.ai/api/v1",
                 api_key=os.environ.get("OPENROUTER_API_KEY")).bind_tools(tools)


