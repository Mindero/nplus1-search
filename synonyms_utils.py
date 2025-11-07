import pandas as pd
from gensim.models import FastText
from normalize_utils import clean_text, lemmatize_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = "news_w2v_extra.model"
CSV_PATH = "news_data_extra.csv"

BERT_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def build_model():
  df = pd.read_csv(CSV_PATH)
  texts = (
        df[['title', 'subtitle', 'text']]
        .fillna('')
        .agg(' '.join, axis=1)
    )
  corpus = [lemmatize_text(clean_text(text)) for text in texts if text.strip()]
  print("Начинаем обучать модель")
  model = FastText(
    sentences=corpus,
    vector_size=300,
    window=5,
    min_count=4,
    workers=4,
    sg=0,
    epochs=10
  )
  model.save(MODEL_PATH)
  print(f"Модель сохранена в {MODEL_PATH}")

def load_model() -> FastText:
  return FastText.load(MODEL_PATH)

def bert_similarity(word1: str, word2: str) -> float:
    """
    Вычисляет косинусную близость между двумя словами через BERT.
    """
    embeddings = BERT_MODEL.encode([word1, word2], convert_to_numpy=True)
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def expand_with_model(query_tokens: list[str], model: FastText, topn: int = 2, bert_threshold: float = 0.85) -> list[str]:
  """
  Расширяет список слов по модели Word2Vec.
  Возвращает исходные токены + найденные синонимы.
  """
  expanded = list(query_tokens)
  seen = set(query_tokens)

  for token in query_tokens:
    similar = [
        w for w, sim in model.wv.most_similar(token, topn=topn)
        if sim > 0.8 and w not in seen and bert_similarity(token, w) >= bert_threshold
    ]
    expanded.extend(similar)
    seen.update(similar)

  return expanded

if __name__ == '__main__':
  build_model()