from dotenv import load_dotenv
load_dotenv()

md5_path = "./md5.text"

#Chroma
collection_name = "rag"
persist_directory = "./chroma_db"


#spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n","\n","。","？","?","！","!"]
max_split_char_number = 1000

similarity_threshold = 2

# Local embedding model via fastembed (no API key needed)
embedding_model_name = "BAAI/bge-small-en-v1.5"

# Groq chat model
chat_model_name = "llama-3.3-70b-versatile"



session_con = {
        "configurable":{
            "session_id":"user_002"
        }
    }