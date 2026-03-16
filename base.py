from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os 
from pypdf import PdfReader

embedding = OllamaEmbeddings(model="qwen3-embedding:0.6b")
db_path = "./chrome_langchain_db"
add_documents = not os.path.exists(db_path)


#Prepearing the data from PDF uwu 

reader = PdfReader(r"C:\Users\karol\Documents\proyecto_rag_cugdl\La ciudad de los recuerdos (2).pdf")
texto = ""
if add_documents:
    for page in reader.pages:
        texto += page.extract_text()

    def dividir_texto(texto, chunk_size=200, overlap=50):

        palabras = texto.split()
        chunks = []
        for i in range(0, len(palabras), chunk_size - overlap):
            chunk = palabras[i:i + chunk_size]
            chunk_texto = " ".join(chunk)
            chunks.append(chunk_texto)
        return chunks
    chunks = dividir_texto(texto)
    def agregar_documents(chunks):
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, 
                metadata={'source': "La ciudad de los recuerdos",
                'chunk_id' : i
                }
            )
            documents.append(doc)
        return documents

    documents = agregar_documents(chunks=chunks)

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
    search_kwargs = {'k': 5}
)
