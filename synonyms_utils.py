import pandas as pd
from gensim.models import Word2Vec
from normalize_utils import clean_text, lemmatize_text

MODEL_PATH = "news_w2v.model"
CSV_PATH = "news_data.csv"

def build_model():
  df = pd.read_csv(CSV_PATH)
  texts = df[['title', 'subtitle', 'text']].fillna('').agg(' '.join, axis=1)
  corpus = [lemmatize_text(clean_text(text)).split() for text in texts if text.strip()]

  model = Word2Vec(
      sentences=corpus,
      vector_size=150,
      window=5,
      min_count=2,
      workers=4,
      sg=1
  )
  model.save(MODEL_PATH)
  print(f"Модель сохранена в {MODEL_PATH}")

def load_model() -> Word2Vec:
  return Word2Vec.load(MODEL_PATH)

def expand_with_model(query_tokens: list[str], model: Word2Vec, topn: int = 3) -> list[str]:
  """
  Расширяет список слов по модели Word2Vec.
  Возвращает исходные токены + найденные синонимы.
  """
  expanded = set(query_tokens)
  for token in query_tokens:
      if token in model.wv:
          similar = [w for w, sim in model.wv.most_similar(token, topn=topn) if sim > 0.7]
          expanded.update(similar)
  return list(expanded)

if __name__ == '__main__':
  build_model()