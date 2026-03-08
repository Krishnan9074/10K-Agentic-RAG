"""
Streamlit-based web file upload service
Supports txt / pdf
Includes batch processing + progress bar
"""

import streamlit as st
import time
import filetype
import pdfplumber
from knowledge_base import KnowledgeBaseService

MAX_FILE_SIZE_MB = 10


st.title("Information Upload System")

uploader_file = st.file_uploader(
    "Please upload a text or PDF file:",
    type=["txt", "pdf"],
    accept_multiple_files=False,
)

# Initialize session state
if "counter" not in st.session_state:
    st.session_state["counter"] = 0
if uploader_file is not None:

    if "service" not in st.session_state:
        st.session_state["service"] = KnowledgeBaseService()


def extract_text_from_file(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name

    # TXT — no magic bytes; validate by attempting UTF-8 decode
    if file_name.endswith(".txt"):
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("File is not valid UTF-8 text.")

    # PDF — verify actual magic bytes before parsing
    elif file_name.endswith(".pdf"):
        kind = filetype.guess(file_bytes)
        if kind is None or kind.mime != "application/pdf":
            raise ValueError("File content does not match a valid PDF.")
        import io
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    else:
        raise ValueError("Unsupported file type")



if uploader_file is not None:

    file_name = uploader_file.name
    file_size_kb = uploader_file.size / 1024

    # Reject files that exceed the size limit
    if uploader_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    st.write(f"File name: {file_name}")
    st.write(f"File size: {file_size_kb:.2f} KB")

    try:
        text = extract_text_from_file(uploader_file)

        if not text.strip():
            st.error("No readable text found in file.")
            st.stop()

        st.success("Text extracted successfully.")
        st.write("Preview (first 1000 characters):")
        st.write(text[:1000])

        st.session_state["counter"] += 1

        with st.spinner("Processing and uploading to knowledge base..."):

            # Split into chunks
            chunks = st.session_state["service"].spliter.split_text(text)
            total_chunks = len(chunks)

            progress = st.progress(0)
            batch_size = 50

            metadata = {
                "source": file_name,
                "operator": "admin"
            }

            # Write in batches
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]

                st.session_state["service"].chroma.add_texts(
                    batch,
                    metadatas=[metadata for _ in batch]
                )

                progress.progress(min((i + batch_size) / total_chunks, 1.0))
                time.sleep(0.1)

            st.success("Upload completed!")

        st.write(f"You have uploaded {st.session_state['counter']} files.")

    except Exception as e:
        st.error(f"Error: {str(e)}")