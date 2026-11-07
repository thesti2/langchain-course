import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


if __name__ == '__main__':
    print("Ingesting...")

    # max_characters is a big number in order to be able to load the whole document at once
    loader = UnstructuredLoader(file_path="./mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)
    document = loader.load()

    print("splitting...")
    # chunk splitter can be very complex (many parameters), this sample show a bacis approach
    # chunk size need a sweetsport. Too big it will be large to fit into the LLM context (too many context), Too small than the content may not have good enough semantic context 
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"created {len(texts)} chunks")

    # get the model from OPENROUTER provider
    embeddings = OpenAIEmbeddings(
                                openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                                openai_api_base="https://openrouter.ai/api/v1",
                                model="openai/text-embedding-ada-002",
                                )

    
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
    print("Finish")