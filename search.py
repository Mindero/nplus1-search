from elasticsearch import Elasticsearch
from normalize_utils import lemmatize_text, clean_text
from synonyms_utils import load_model, expand_with_model
from spellcheker_utils import correct_query, load_spellchecker

# Настройки
ES_URL = "http://localhost:9200"
INDEX_NAME = "news"
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
    corrected = correct_query(query, spellchecker)
    if corrected != query:
        print(f"Исправлено: «{query}» -> «{corrected}»")
    query = corrected

    query_tokens = lemmatize_text(clean_text(query))

    expanded_tokens = expand_with_model(query_tokens=query_tokens, model=model)
    expanded_query = " ".join(expanded_tokens)

    print(f"Расширенный запрос: {expanded_query}\n")

    body = {     
        "query": {
            "multi_match": {
                "query": expanded_query,
                "fields": ["title^3", "subtitle^2", "text"],  # вес полей
                "fuzziness": "AUTO"  # допускаем опечатки
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "text": {}
            }
        },
        "size": size
    }

    response = es.search(index=INDEX_NAME, body=body)
    hits = response["hits"]["hits"]
    return hits

def print_results(hits):
    if not hits:
        print("\nНичего не найдено.")
        return

    print(f"\n Найдено документов: {len(hits)}\n")
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        print(f"{i}. Новость: {src.get('true_title', 'Без названия')}")
        print(f"   Автор: {src.get('author', 'Неизвестен')}")
        print(f"   Дата: {src.get('date', '—')}")
        print(f"   Сложность: {src.get('difficulty', '—')}")
        print(f"   Теги: {', '.join(src.get('tags', []))}")

        if "highlight" in hit:
            snippet = " ".join(hit["highlight"].get("text", [])[:1])
            print(f"   Фрагмент: {snippet}")

        print(f"   URL: {src.get('url', '')}")
        print("-" * 100)

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
