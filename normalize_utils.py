import re
import nltk
from datetime import datetime
from ast import literal_eval
from pymorphy3 import MorphAnalyzer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

morph = MorphAnalyzer()
stop_words = set(nltk.corpus.stopwords.words("russian"))

def normalize_date(date_str: str) -> str | None:
    for fmt in ("%d.%m.%y", "%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None

def normalize_difficulty(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def normalize_tags(tags_str: str):
    if not tags_str:
        return []
    try:
        tags = literal_eval(tags_str)
        if isinstance(tags, list):
            return [str(tag).strip().lower() for tag in tags]
        return [str(tags).strip().lower()]
    except Exception:
        return [tags_str.strip().lower()]

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # Убираем HTML
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text: str) -> list[str]:
    text = clean_text(text)
    tokens = nltk.word_tokenize(text, language="russian")
    lemmas = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return lemmas
