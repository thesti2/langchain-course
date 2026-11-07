import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
name = os.environ["INDEX_NAME"]

pc.create_index(
    name=name,
    dimension=1536,  # text-embedding-ada-002 produces 1536-dim vectors
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
print(pc.describe_index(name))