from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("El script sí se ejecutó")

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

result = chain.invoke({"informacion": [], "pregunta": "De que está hecha la ciudad?"})
print(result)