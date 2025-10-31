import pandas as pd
from symspellpy import SymSpell
from normalize_utils import clean_text, lemmatize_text

MODEL_PATH = "spell_dictionary.txt"


def build_dictionary(csv_path: str, dictionary_path: str = MODEL_PATH):
  df = pd.read_csv(csv_path)
  text_columns = ["title", "subtitle", "text"]

  # Объединяем все текстовые поля в один столбец
  df["full_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

  all_tokens = []
  for text in df["full_text"]:
      clean_tokens = lemmatize_text(clean_text(text))
      all_tokens.extend(clean_tokens)

  freq = pd.Series(all_tokens).value_counts()

  freq.to_csv(dictionary_path, sep=" ", header=False)


def load_spellchecker(dictionary_path: str = MODEL_PATH) -> SymSpell:
  max_edit_distance_dictionary = 2
  prefix_length = 7
  sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

  sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
  return sym_spell


def correct_query(query: str, spellchecker: SymSpell) -> str:
  """
  Исправляет опечатки в пользовательском запросе.
  """
  suggestions = spellchecker.lookup_compound(query, max_edit_distance=2)
  return suggestions[0].term if suggestions else query


if __name__ == '__main__':
  build_dictionary(csv_path="news_data.csv")