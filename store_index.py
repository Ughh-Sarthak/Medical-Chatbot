from dotenv import load_dotenv
from src.config import INDEX_NAME

import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Step 1: Load & Process Documents
extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Step 2: Load Embeddings
embeddings = download_hugging_face_embeddings()
dimension = 384  # all-MiniLM-L6-v2

# Step 3: Initialize Pinecone v5
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medibot"

# Get existing indexes safely (v5 style)
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index already exists: {index_name}")

# Step 4: Connect to Index
index = pc.Index(index_name)

# Step 5: Store Documents in Vector DB
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Indexing completed successfully.")