from elasticsearch import Elasticsearch
from normalize_utils import lemmatize_text, clean_text
from synonyms_utils import load_model, expand_with_model, get_sentence_embedding
from spellcheker_utils import correct_query, load_spellchecker
# from weight_utils import get_token_weights
import pandas as pd

# Настройки
ES_URL = "http://localhost:9200"
INDEX_NAME = "news_extra_extra3"
DEFAULT_SIZE = 10  # количество документов по умолчанию

es = Elasticsearch(ES_URL)
model = load_model() # модель для извлечения синонимов
spellchecker = load_spellchecker()

def search_news(query: str, size: int = DEFAULT_SIZE):
    """
    query: поисковая строка
    size: количество документов в выдаче
    return: список найденных документов
    """
    query_tokens = lemmatize_text(clean_text(query))

    query_tokens = [correct_query(word, spellchecker) for word in query_tokens]

    expanded_tokens = expand_with_model(query_tokens=query_tokens, model=model)
    expanded_query = " ".join(expanded_tokens)

    print(f"Расширенный запрос: {expanded_query}\n")

    query_vector = get_sentence_embedding([expanded_query])[0].tolist()
    body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "field": "content_vector",
                            "query_vector": query_vector,
                            "k": 2 * size,
                            "num_candidates": 5 * size
                        }
                    },
                    {
                        "multi_match": {
                            "query": expanded_query,
                            "fields": ["title^3", "subtitle^2", "text"],
                            "type": "most_fields"
                        }
                    },
                    
                ],
            }
        },
        "_source": ["true_title", "url"]
    }

    response = es.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    return hits

def print_results(hits):
    if not hits:
        print("\nНичего не найдено.")
        return

    rows = []
    print(f"\n Найдено документов: {len(hits)}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        title = src.get("true_title", "Без названия")
        url = src.get("url", "")
        # author = src.get("author", "Неизвестен")
        # date = src.get("date", "—")
        # tags = ", ".join(src.get("tags", []))

        src = hit["_source"]
        print(f"{i}. Новость: {title}")
        # print(f"   Автор: {author}")
        # print(f"   Дата: {date}")
        # print(f"   Теги: {', '.join(tags)}")

        rows.append({
            "№": i,
            "Название": title,
            "URL": url,
            # "Автор": author,
            # "Дата": date,
            # "Теги": tags,
        })
    df = pd.DataFrame(rows)
    df.to_csv("result-query.csv", index=False)

def main():
    print("Введите запрос для поиска. Для выхода напишите 'exit'.\n")

    while True:
        query = input("Ваш запрос: ").strip()
        if query.lower() == 'exit':
            print("Выход из программы.")
            break

        size_input = input("Сколько результатов показать? (Enter = 10): ").strip()
        size = int(size_input) if size_input.isdigit() else DEFAULT_SIZE

        hits = search_news(query, size=size)
        print_results(hits)

if __name__ == "__main__":
    main()
