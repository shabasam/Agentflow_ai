import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "vector_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


def create_or_load_vector_store(chunks):

    embeddings = get_embeddings()    

    if os.path.exists(INDEX_PATH):
        print("loading existing FAISS index..")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("creating new FAISS index..")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_PATH)

    return vectorstore
