from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from base import retriever


model = OllamaLLM(model="qwen3:4b")

template = """"
Eres un asistente que brinda apoyo y orientación como respuesta en cada una de las solicitudes 
hechas por el usuario, manteniendo tus respuestas dentro del margen de restricciones establecido.
Tu tarea es darle respuestas claras al usuario, manteniendo una tonalidad media-formal en 
todo momento, siguiendo los puntos establecidos en las secciones de tono, estructura de 
mensajes, usuario y restricciones.

Aquí está la información relevante: {informacion}

Aquí está la pregunta: {pregunta}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


"""
while True:
    print("*"*100)
    question = input("Escribe tu pregunta: (Presiona q para salir)")
    if question == "q":
        break
    informacion_docs = retriever.invoke(question)
    informacion = "\n\n".join(
        [doc.page_content for doc in informacion_docs]
    )
    result = chain.invoke({"informacion": informacion, "pregunta": question})
    print(result)

    """

def ejecutar_rag(pregunta):
    docs = retriever.invoke(pregunta)

    contexts = [
    f"""
    Description:
    {doc.page_content}
    """
    for doc in docs
]
    contexts_text = "\n\n".join(contexts)

    result = chain.invoke({
        "informacion": contexts_text,
        "pregunta" : pregunta
    })
    
    answer = result

    return {
        "question": pregunta,
        "answer": answer,
        "contexts": contexts
    }