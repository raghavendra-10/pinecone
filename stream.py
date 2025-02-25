import streamlit as st
import requests

# Backend URL (Update if running on a server)
BASE_URL = "https://ainew-371840213392.us-central1.run.app"

# 🌟 1️⃣ Organization Input
st.title("Multi-Tenant Document RAG System")

st.sidebar.header("Setup Organization")
org_id = st.sidebar.text_input("Enter Organization ID:", value="test_org")

if not org_id:
    st.warning("Please enter an Organization ID to proceed.")
    st.stop()

st.sidebar.success(f"Using Organization: **{org_id}**")

# 🌟 2️⃣ File Upload Section
st.header("📂 Upload Documents")
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf","txt","docx","json"])

if uploaded_file:
    with st.spinner("Uploading and processing file..."):
        files = {"file": uploaded_file}
        data = {"org_id": org_id}

        response = requests.post(f"{BASE_URL}/upload", files=files, data=data)

        if response.status_code == 200:
            st.success("File uploaded successfully!")
            st.json(response.json())
        else:
            st.error("Upload failed.")
            st.json(response.json())

# 🌟 3️⃣ Fetch Uploaded Documents
st.header("📜 Uploaded Documents")

if st.button("Fetch Documents"):
    response = requests.get(f"{BASE_URL}/fetch?org_id={org_id}")
    
    if response.status_code == 200:
        data = response.json()
        st.write(f"**Total Files:** {data['total_uploaded_files']}")
        for file in data["files"]:
            st.write(f"📄 **{file['file_name']}**")
            st.write(f"🔗 [View File]({file['file_url']})")
    else:
        st.error("Failed to fetch documents.")

# 🌟 4️⃣ Chat Interface for Querying Documents
st.header("💬 Chat with Documents")
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Retrieving answer..."):
        response = requests.get(f"{BASE_URL}/answer?query={query}&org_id={org_id}")

        if response.status_code == 200:
            data = response.json()
            st.subheader("🤖 AI Response:")
            st.write(data["answer"])

            st.subheader("📜 Retrieved Chunks:")
            for chunk in data["retrieved_chunks"]:
                st.write(f"🔹 {chunk}")

        else:
            st.error("Failed to retrieve answer.")
            st.json(response.json())

# 🌟 5️⃣ Delete Files Section
st.header("🗑️ Delete Uploaded Documents")

# Fetch uploaded documents again to get file IDs
response = requests.get(f"{BASE_URL}/fetch?org_id={org_id}")

if response.status_code == 200:
    files_data = response.json()["files"]

    if files_data:
        file_to_delete = st.selectbox("Select a file to delete:", files_data, format_func=lambda x: x["file_name"])

    if st.button("Delete Selected File"):
        doc_id = ".".join(file_to_delete["file_name"].split(".")[:-1])  # Remove extension
        response = requests.delete(f"{BASE_URL}/delete?doc_id={doc_id}&org_id={org_id}")

        if response.status_code == 200:
            st.success("File deleted successfully!")
        else:
            st.error("Deletion failed.")
            st.json(response.json())

    else:
        st.warning("No files uploaded yet.")
else:
    st.error("Failed to fetch files for deletion.")
