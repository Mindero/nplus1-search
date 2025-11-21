import numpy as np
import joblib
from synonyms_utils import get_sentence_embedding

model_lr_path = 'model_lr.pkl'

def load_model_lr():
    return joblib.load(model_lr_path)

def build_model_lr_data(query, answer):
    query = get_sentence_embedding([query])[0].tolist()
    answer = get_sentence_embedding([answer])[0].tolist()
    return np.multiply(query, answer)

def rerank_results_with_lr(query: str, hits: list, model_lr):
    rescored = []

    for hit in hits:
        doc = hit["_source"]
        answer_text = doc.get("true_title", "")

        # построить вектор для модели
        X_vec = build_model_lr_data(query, answer_text)

        # предсказать вероятность класса 1
        prob = float(model_lr.predict_proba([X_vec])[0][1])

        # добавить в документ
        hit["model_score"] = prob
        rescored.append(hit)

    # сортировка по убыванию вероятности
    rescored = sorted(rescored, key=lambda x: x["model_score"], reverse=True)

    return rescored


if __name__ == '__main__':
    X = np.load("X_combined_embeddings.npy")
    y = np.load("y_labels.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(np.unique(y))
    print(np.bincount(y))
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_lr = LogisticRegression(max_iter=10000, class_weight='balanced',  penalty='l1', solver='liblinear')
    model_lr.fit(X_train, y_train)

    y_pred = model_lr.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(model_lr, model_lr_path)
