import pandas as pd
from gensim.models import FastText
from normalize_utils import clean_text, lemmatize_text
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

MODEL_PATH = "news_w2v_extra.model"
CSV_PATH = "news_data_extra.csv"

MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)


def mean_pooling(model_output, attention_mask):
    """
    Среднее (mean) пулирование эмбеддингов токенов с учетом attention_mask.
    """
    token_embeddings = model_output[0]  # (batch, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_sentence_embedding(texts: list[str]):
    """
    Возвращает sentence embeddings для списка строк.
    """
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, max_length=64, return_tensors='pt'
    )
    with torch.no_grad():
        model_output = bert_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings.cpu().numpy()


def build_model():
    df = pd.read_csv(CSV_PATH)
    texts = (
        df[['title', 'subtitle', 'text']]
        .fillna('')
        .agg(' '.join, axis=1)
    )
    corpus = [lemmatize_text(clean_text(text)) for text in texts if text.strip()]
    print("Начинаем обучать модель FastText...")
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
    Вычисляет косинусную близость между двумя словами через SBERT (ai-forever/sbert_large_nlu_ru).
    """
    embeddings = get_sentence_embedding([word1, word2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def expand_with_model(query_tokens: list[str], model: FastText, topn: int = 2, bert_threshold: float = 0.75) -> list[str]:
    """
    Расширяет список слов по модели Word2Vec.
    Возвращает исходные токены + найденные синонимы.
    """
    expanded = list(query_tokens)
    seen = set(query_tokens)

    for token in query_tokens:
        similar = [
            w for w, sim in model.wv.most_similar(token, topn=topn)
            if sim > 0.7 and w not in seen and bert_similarity(token, w) >= bert_threshold
        ]
        expanded.extend(similar)
        seen.update(similar)

    return expanded


if __name__ == '__main__':
    # build_model()
    print(bert_similarity(word1='премия', word2='награда'))
