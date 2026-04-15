from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os 
from pypdf import PdfReader

embedding = OllamaEmbeddings(model="all-minilm:l6-v2")
db_path = "./chrome_langchain_db"
add_documents = not os.path.exists(db_path)

# Leer PDF
reader = PdfReader(r"C:\Users\karol\Documents\proyecto_rag_cugdl\La ciudad de los recuerdos.pdf")
texto = ""

if add_documents:
    for page in reader.pages:
        texto += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(texto)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                'source': "La ciudad de los recuerdos",
                'chunk_id': i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=documents,
        persist_directory=db_path,
        embedding=embedding
    )

else:
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embedding
    )

retriever = vector_store.as_retriever(
    search_kwargs={'k': 5}
)