from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your Pinecone index
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Query Pinecone to fetch an embedding
query_response = pinecone_index.fetch(["Exercises Routine_chunk_1"])  # Use a stored ID from your data

# Print the vector (embedding)
print(query_response)
