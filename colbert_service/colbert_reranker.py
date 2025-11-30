from pylate import models, rank

print("Importing ColBert model...")
model = models.ColBERT(model_name_or_path='colbert_service\output\colbert-ruBert-v1\checkpoint-192')
print("Import of ColBert model successfull")

def rerank(query: str, documents: list[str]):
  documents_ids = [list(range(len(documents)))]
  queries_embeddings = model.encode([query], is_query=True,)
  documents_embeddings = model.encode([documents], is_query=False,)
  reranked = rank.rerank(
      documents_ids=documents_ids,
      queries_embeddings=queries_embeddings,
      documents_embeddings=documents_embeddings,
  )
  ranked_info = reranked[0]
  ranked_ids = [item['id'] for item in ranked_info]

  return ranked_ids