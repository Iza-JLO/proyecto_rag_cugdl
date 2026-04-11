import ragas
from datasets import Dataset
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import evaluate
from main import ejecutar_rag

questions = [
    "¿Cómo luce la chica del cuento?",
    "¿Cómo son los ojos de la chica?"
]

ground_truths = [
    "De una belleza indescriptible. Ataviada en un vestido corto de una oscuridad que no dejaba escapar ni el más diminuto rayo de luz",
    "Carentes de pupila, mostraban una infinidad de estrellas que nadaban en un púrpura brillante"
]

dataset = []

for q,g_t in zip(questions, ground_truths):
    print(f"Ejecutando pregunta {q}:")
    result = ejecutar_rag(q)

    dataset.append({
        "question": q,
        "answer": result["answer"],
        "contexts": result["contexts"],
        "ground_truth": g_t
    })

dataset_evaluation = Dataset.from_list(dataset)
results = evaluate(dataset=dataset_evaluation, metrics=[Faithfulness(), AnswerRelevancy()])