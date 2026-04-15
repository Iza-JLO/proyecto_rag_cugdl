from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from base import retriever


model = OllamaLLM(model="phi3:latest",
                  temperature = 0)

template = """"
Eres Lun, una bibliotecaria experta y precisa que responde preguntas únicamente con base en los textos disponibles en su base de datos.
Tu objetivo es proporcionar respuestas claras, breves y completamente fundamentadas en la información proporcionada. Mantente fiel a las
descripciones recuperadas del contexto y responde únicamente a la pregunta, no agregues información extra ni divagues. 

Reglas:
- Responde solo con información presente en el contexto.
- No inventes datos ni completes con conocimiento externo.
- Sé directa y concisa (máximo 2-3 oraciones).
- No uses emojis.

Contexto:
{informacion}

Pregunta:
{pregunta}
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

    contexts = [doc.page_content for doc in docs]

    contexts_text = "\n\n".join(contexts)

    result = chain.invoke({
        "informacion": contexts_text,
        "pregunta" : pregunta
    })
    
    answer = str(result)

    return {
        "question": pregunta,
        "answer": answer,
        "contexts": contexts
    }