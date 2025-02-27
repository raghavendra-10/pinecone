from flask import Flask, request, jsonify
import boto3
from pinecone import Pinecone
import openai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF processing
import docx  # python-docx for DOCX files
import pandas as pd 
import json 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS
import tiktoken
import csv


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# AWS S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to Pinecone index
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

@app.route("/")

def home():
    return "Multi-Tenant File Upload and Embedding API is Running"

# Upload file to S3
def upload_to_s3(file, org_id):
    """
    Uploads a file to S3 in an organization-specific folder.
    """
    file_key = f"{org_id}/{file.filename}"
    s3_client.upload_fileobj(file, BUCKET_NAME, file_key)
    file_url = f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{file_key}"
    return file_url, file_key

# Extract text from PDF
def extract_text_from_file(file_path, file_type):
    """
    Extracts text from different file types: PDF, DOCX, TXT, JSON.
    """
    text = ""

    if file_type == "pdf":
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text("text") + "\n"

    elif file_type == "docx":
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)  # Attempt to parse JSON
                if isinstance(data, dict) or isinstance(data, list):
                    text = json.dumps(data, indent=2)  # Convert structured JSON to text
                elif isinstance(data, str):  # ✅ Handle plain-string JSON
                    text = data  
                else:
                    text = str(data)  # Convert other types to string
            except json.JSONDecodeError:
                f.seek(0)
                text = f.read()
    elif file_type == "csv":
        try:
            text_lines = []
            for chunk in pd.read_csv(file_path, chunksize=1000, encoding="utf-8"):
                for row in chunk.itertuples(index=False):
                    text_lines.append(" | ".join(map(str, row)))
                    if len(text_lines) >= 100000:
                        break
            text = "\n".join(text_lines)
            if not text.strip():
                raise ValueError("Extracted text from CSV is empty.")
        except Exception as e:
            text = f"Error processing CSV: {str(e)}"
    
    return text


# text to be embedded in Pinecone
# Embed Text in Pinecone
# Generate OpenAI Embeddings
def generate_openai_embeddings(text_chunks):
    """Generate embeddings in batches, ensuring each chunk is <= 8192 tokens."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    tokenizer = tiktoken.get_encoding("cl100k_base")

    embeddings = []
    batch_size = 50
    filtered_chunks = [chunk for chunk in text_chunks if len(tokenizer.encode(chunk)) <= 8192]

    if not filtered_chunks:
        print("⚠️ No valid text to embed. Skipping OpenAI call.")
        return []

    try:
        for i in range(0, len(filtered_chunks), batch_size):
            batch = filtered_chunks[i : i + batch_size]
            response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
            embeddings.extend([embedding.embedding for embedding in response.data])
    except openai.BadRequestError as e:
        print(f"❌ OpenAI Embeddings API Error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected OpenAI Error: {e}")

    return embeddings

#module to process text and store embeddings in Pinecone
# Process Text and Store Embeddings in Pinecone
def process_text_and_store_embeddings(text, org_id, doc_id):
    """Chunks text, generates embeddings in batches, and stores them in Pinecone."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    batch_size = 50
    valid_chunks = [chunk for chunk in text_splitter.split_text(text) if len(tokenizer.encode(chunk)) <= 8192]

    print(f"✅ Processing {len(valid_chunks)} valid chunks for embedding.")
    total_uploaded = 0

    for i in range(0, len(valid_chunks), batch_size):
        batch_chunks = valid_chunks[i : i + batch_size]
        batch_embeddings = generate_openai_embeddings(batch_chunks)

        if not batch_embeddings:
            print("⚠️ Skipping batch due to empty embeddings.")
            continue

        pinecone_vectors = [
            {"id": f"{doc_id}_chunk_{i+j}", "values": batch_embeddings[j], "metadata": {"org_id": org_id, "text": batch_chunks[j]}}
            for j in range(len(batch_embeddings))
        ]

        if pinecone_vectors:
            try:
                pinecone_index.upsert(vectors=pinecone_vectors)
                total_uploaded += len(batch_embeddings)
            except Exception as e:
                print(f"❌ Pinecone Batch Upload Error: {str(e)}")

    print(f"✅ Successfully uploaded {total_uploaded} chunks to Pinecone.")
    return total_uploaded


# routes for the project 
# Flask API for File Upload
@app.route("/upload", methods=["POST"])
def upload_file():
    """
    API Endpoint to upload files and store embeddings.
    """
    if "file" not in request.files or "org_id" not in request.form:
        return jsonify({"error": "File and Org ID required"}), 400

    file = request.files["file"]
    org_id = request.form["org_id"]
    file_ext = file.filename.split(".")[-1].lower()  # Extract file extension
    allowed_types = ["pdf", "docx", "txt", "json","csv"]  # Add more types if needed

    if file_ext not in allowed_types:
        return jsonify({"error": "Unsupported file type!"}), 400
    doc_id = os.path.splitext(file.filename)[0]

    # ✅ Read file into memory before using it (prevents closure issues)
    file_bytes = file.read()

    # ✅ Upload file to S3 using in-memory bytes
    file.seek(0)  # Reset file pointer before passing it to upload function
    file_url, file_key = upload_to_s3(file, org_id)

    # ✅ Save file locally for text extraction
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_bytes)  # Write the in-memory file to disk

    # ✅ Extract text from the saved file
    extracted_text = extract_text_from_file(temp_path, file_ext)

    # ✅ Process embeddings
    chunk_count = process_text_and_store_embeddings(extracted_text, org_id, doc_id)

    # ✅ Cleanup temp file
    os.remove(temp_path)

    return jsonify(
        {
            "message": "File uploaded successfully",
            "file_url": file_url,
            "doc_id": doc_id,
            "chunks_stored": chunk_count,
        }
    ), 200
    
@app.route("/retrieve", methods=["GET"])
def retrieve_relevant_chunks():
    """
    API to retrieve relevant document chunks from Pinecone using a query.
    """
    query_text = request.args.get("query")
    org_id = request.args.get("org_id")  # Ensure multi-tenancy

    if not query_text or not org_id:
        return jsonify({"error": "Query and Org ID are required"}), 400

    # Generate embedding for the query using OpenAI
    query_embedding = generate_openai_embeddings([query_text])[0]  # Extract first embedding

    # Search Pinecone for relevant matches
    search_results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,  # Retrieve top 3 results
        include_metadata=True,
        filter={"org_id": org_id}  # Ensure results are limited to the same org
    )

    # Format results
    retrieved_chunks = [
        {"id": match["id"], "score": match["score"], "text": match["metadata"]["text"]}
        for match in search_results["matches"]
    ]

    return jsonify({"query": query_text, "retrieved_chunks": retrieved_chunks}), 200


@app.route("/answer", methods=["GET"])
def get_answer():
    """
    API to retrieve relevant document chunks from Pinecone and generate an answer using OpenAI GPT.
    """
    query_text = request.args.get("query")
    org_id = request.args.get("org_id")  # Multi-tenancy enforcement

    if not query_text or not org_id:
        return jsonify({"error": "Query and Org ID are required"}), 400

    # Generate embedding for the query using OpenAI
    query_embedding = generate_openai_embeddings([query_text])[0]  # Extract first embedding

    # Retrieve relevant matches from Pinecone
    search_results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,  # Retrieve top 3 results
        include_metadata=True,
        filter={"org_id": org_id}  # Restrict search to the same org
    )

    # Extract relevant chunks
    retrieved_chunks = [
        match["metadata"]["text"] for match in search_results["matches"]
    ]

    # If no relevant documents found
    if not retrieved_chunks:
        return jsonify({"query": query_text, "answer": "No relevant information found.", "retrieved_chunks": []}), 200

    # Format retrieved chunks into a single context
    context = "\n\n".join(retrieved_chunks)

    # Generate answer using OpenAI GPT
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Context: {context}\n\nUser Query: {query_text}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4 or GPT-3.5
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=200
    )

    # Extract the response text
    answer = response.choices[0].message.content.strip()

    return jsonify({"query": query_text, "answer": answer, "retrieved_chunks": retrieved_chunks}), 200

@app.route("/delete", methods=["DELETE"])
def delete_document():
    """
    API to delete a document and its embeddings.
    """
    doc_id = request.args.get("doc_id")
    org_id = request.args.get("org_id")  # Ensure multi-tenancy

    if not doc_id or not org_id:
        return jsonify({"error": "Document ID and Org ID are required"}), 400

    try:
        # ✅ 1. List all files in the org's folder to find the correct file type
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{org_id}/")

        # Find the matching file based on doc_id (ignoring extensions)
        matching_files = [
            obj["Key"] for obj in response.get("Contents", [])
            if obj["Key"].startswith(f"{org_id}/{doc_id}")
        ]

        if not matching_files:
            return jsonify({"error": "File not found in S3."}), 404

        # ✅ 2. Delete file(s) from S3
        for file_key in matching_files:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=file_key)

        # ✅ 3. Delete embeddings from Pinecone
        pinecone_index.delete(ids=[f"{doc_id}_chunk_{i}" for i in range(1000)])  # Adjust chunk range if needed

        return jsonify({"message": "Document and embeddings deleted successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/delete", methods=["DELETE"])
# def delete_document():
#     """
#     API to delete a document and all its embeddings from S3 & Pinecone.
#     """
#     doc_id = request.args.get("doc_id")
#     org_id = request.args.get("org_id")  # Ensure multi-tenancy

#     if not doc_id or not org_id:
#         return jsonify({"error": "Document ID and Org ID are required"}), 400

#     try:
#         # ✅ 1. Find the correct file type from S3
#         response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{org_id}/")

#         matching_files = [
#             obj["Key"] for obj in response.get("Contents", [])
#             if obj["Key"].startswith(f"{org_id}/{doc_id}")
#         ]

#         if not matching_files:
#             return jsonify({"error": "File not found in S3."}), 404

#         # ✅ 2. Delete file(s) from S3
#         for file_key in matching_files:
#             s3_client.delete_object(Bucket=BUCKET_NAME, Key=file_key)

#         # ✅ 3. Retrieve all stored vector IDs for this document from Pinecone
#         query_filter = {"org_id": org_id}  # Filter by organization
#         vector_data = pinecone_index.query(
#             namespace="",
#             top_k=1000,  # Fetch all relevant vectors
#             filter=query_filter
#         )

#         # Extract vector IDs related to this document
#         vector_ids_to_delete = [match["id"] for match in vector_data["matches"] if doc_id in match["id"]]

#         # ✅ 4. Delete all matching embeddings from Pinecone
#         if vector_ids_to_delete:
#             pinecone_index.delete(ids=vector_ids_to_delete)

#         return jsonify({"message": "Document and all embeddings deleted successfully."}), 200

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route("/fetch", methods=["GET"])
def fetch_uploaded_files():
    """
    API to fetch all uploaded files and their embedded documents.
    """
    org_id = request.args.get("org_id")  # Multi-tenancy

    if not org_id:
        return jsonify({"error": "Org ID is required"}), 400

    try:
        # 🔹 1. Fetch List of Files from S3
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{org_id}/")
        files = [
            {
                "file_name": obj["Key"].split("/")[-1],
                "file_url": f"https://{BUCKET_NAME}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{obj['Key']}"
            }
            for obj in response.get("Contents", [])
        ]

        # 🔹 2. Fetch Embedded Documents from Pinecone
        pinecone_results = pinecone_index.describe_index_stats()
        total_vectors = pinecone_results.get("total_vector_count", 0)

        # 🔹 3. Return JSON Response
        return jsonify({"org_id": org_id, "total_uploaded_files": len(files), "files": files, "total_vectors": total_vectors}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Flask App
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
