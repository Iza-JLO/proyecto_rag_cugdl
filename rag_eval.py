from ragas import RunConfig
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from main import ejecutar_rag
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="phi3:latest",
                 temperature=0,
                 max_tokens = 100)
embedding = OllamaEmbeddings(model="all-minilm:l6-v2")

questions = [
    "¿Cómo luce el vestido de la chica?",
    "¿Cómo son los ojos de la chica?"
]

ground_truths = [
    "ataviada en un vestido corto de una oscuridad que no dejaba escapar ni el más diminuto rayo de luz",
    " carentes de pupila, mostraban una infinidad de estrellas que nadaban en un púrpura brillante."
]

dataset = []

for q,g_t in zip(questions, ground_truths):
    print(f"Ejecutando pregunta {q}:")
    result = ejecutar_rag(q)
    answer = result["answer"] if isinstance(result["answer"], str) else str(result["answer"])
    print(f"Respuesta: {result['answer']}")

    dataset.append({
        "question": q,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": g_t
    })

print(f"Base de datos finalizada. Ejecutando evaluaciones...")

dataset_evaluation = Dataset.from_list(dataset)
resultados = evaluate(
    dataset_evaluation,
    metrics=[
        answer_relevancy,
        context_recall
    ],
    llm=llm,
    embeddings=embedding,
    run_config=RunConfig(
        max_workers=1,
        timeout=10000
    )
)

print(f"Resultados: \n{resultados}")