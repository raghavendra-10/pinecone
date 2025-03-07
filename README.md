# **Multi-Tenant File Upload & Retrieval API**
🚀 **A Flask-based API for uploading, storing, retrieving, and deleting documents with embeddings in Pinecone.**  
Supports **AWS S3** for file storage and **OpenAI** for embedding generation.

---

## **📁 Project Structure**
```
/project_root/
│── app.py                 # Main Flask App
│── /services/             # Backend Services
│   │── embeddings.py       # OpenAI Embedding Logic
│   │── pinecone_store.py   # Pinecone Database Operations
│   │── s3_storage.py       # AWS S3 Operations
│   │── text_processing.py  # File Extraction and Processing
│── /routes/               # API Route Handlers
│   │── upload_routes.py    # File Upload & Embedding API
│   │── query_routes.py     # Retrieval & Answer API
│   │── delete_routes.py    # Deletion API
│── requirements.txt       # Required Python Dependencies
│── .env                   # Environment Variables (API Keys, Configs)
│── README.md              # Project Documentation
```

---

## **🛠️ Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo-name.git
cd project_root
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Environment Variables**
Create a **`.env`** file and add your API keys:
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_s3_bucket

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name

OPENAI_API_KEY=your_openai_api_key
```

### **4️⃣ Run the Flask App**
```bash
python app.py
```
The API will start at **http://0.0.0.0:8080**

---

## **🚀 API Endpoints**
### **1️⃣ File Upload & Embeddings**
📌 **Upload a file & store embeddings**
```http
POST /upload
```
**Request Form Data**
| Parameter  | Type  | Description |
|------------|--------|--------------|
| `file`     | File   | PDF, DOCX, TXT, JSON, or CSV file |
| `org_id`   | String | Organization ID |

📌 **Response**
```json
{
  "message": "File uploaded successfully",
  "file_url": "https://s3.amazonaws.com/bucket/file.pdf",
  "doc_id": "file",
  "chunks_stored": 50
}
```

---

### **2️⃣ Retrieve Relevant Chunks**
📌 **Retrieve the top relevant document chunks**
```http
GET /retrieve
```
**Query Parameters**
| Parameter  | Type  | Description |
|------------|--------|--------------|
| `query`    | String | Search query |
| `org_id`   | String | Organization ID |

📌 **Response**
```json
{
  "query": "What is in the document?",
  "retrieved_chunks": [
    { "id": "file_chunk_1", "score": 0.85, "text": "Relevant document text..." }
  ]
}
```

---

### **3️⃣ Generate Answer from Documents**
📌 **Retrieve relevant data & generate an AI-generated answer**
```http
GET /answer
```
**Query Parameters**
| Parameter  | Type  | Description |
|------------|--------|--------------|
| `query`    | String | Search query |
| `org_id`   | String | Organization ID |

📌 **Response**
```json
{
  "query": "What is in the document?",
  "answer": "This document contains details about...",
  "retrieved_chunks": [ "Relevant text..." ]
}
```

---

### **4️⃣ Delete Files & Embeddings**
📌 **Delete a document and its embeddings**
```http
DELETE /delete
```
**Query Parameters**
| Parameter  | Type  | Description |
|------------|--------|--------------|
| `doc_id`   | String | Document ID |
| `org_id`   | String | Organization ID |

📌 **Response**
```json
{
  "message": "File and embeddings deleted"
}
```

---

## **📜 Modules Breakdown**
### **📂 `/services/` (Backend Services)**
| File              | Description |
|-------------------|------------|
| `embeddings.py`   | Handles OpenAI embedding generation |
| `pinecone_store.py` | Manages storing/retrieving embeddings in Pinecone |
| `s3_storage.py`   | Handles AWS S3 file upload & deletion |
| `text_processing.py` | Extracts text from PDF, DOCX, TXT, JSON, CSV |

### **📂 `/routes/` (API Routes)**
| File              | Description |
|-------------------|------------|
| `upload_routes.py` | API for uploading files and embedding storage |
| `query_routes.py`  | API for retrieving & answering queries |
| `delete_routes.py` | API for deleting documents & embeddings |

---

## **🔗 Additional Notes**
- Uses **Pinecone** for vector search.
- Uses **OpenAI** for text embeddings.
- Supports **AWS S3** for cloud storage.
- Modular and scalable **Flask-based microservices**.

---

## **📌 Next Steps**
✅ Add **BM25 + Cohere reranking** for hybrid search.  
✅ Deploy using **Docker + AWS Lambda**.  
✅ Add authentication with **JWT Tokens**.

---

🚀 **Ready to use this API?** Just start **`python app.py`** and enjoy seamless document retrieval! 🎉

