import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()


print("Initializing Components...")

# get the model from OPENROUTER provider
embeddings = OpenAIEmbeddings(
                            api_key=os.environ.get("OPENROUTER_API_KEY"),
                            base_url="https://openrouter.ai/api/v1",
                            model="openai/text-embedding-ada-002",
                            )
llm = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
)

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

# configuration: search top 3 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based on only on the following context:
    {context}

    Question: {question}
    Provide a detailed answer
    """
)

# join all documents from vector search into 1 string
def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain wihtout LCEL.
    manually retrieves documents, formats them and generates a response.

    Limitations:
    - manual step-by-step execution
    - no built-in streaming
    - no async support
    """
    docs = retriever.invoke(query)

    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter[str]("question") | retriever | format_docs 
        )        
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving")

    query = "What is pinecone in machine learning?"

    # option 0: raw searching without RAG
    print("\n" + "=" * 70)
    print("Implementation 0: RAW LLM invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)


    # option 1: use implementation without LCEL
    print("\n" + "=" * 70)
    print("Implementation 1: Without LCEL")
    print("=" * 70)
    result_no_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_no_lcel)

    # option 2: use implementation with LCEL 
    print("\n" + "=" * 70)
    print("Implementation 1: With LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke("question": query)
    print("\nAnswer:")
    print(result_with_lcel)