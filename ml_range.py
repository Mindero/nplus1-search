from synonyms_utils import get_sentence_embedding
import numpy as np
import pandas as pd

df = pd.read_csv('query_news_dataset_combined.csv') 

X_vectors = []
X_combined_vectors = []
y = []
for i, row in df.iterrows():
  if (i % 1000 == 0):
    print(i)
  query = row["query"]
  answer = row["answer"]
  label = int(row["relevance"])
  query = get_sentence_embedding([query])[0].tolist()
  answer = get_sentence_embedding([answer])[0].tolist()
  dot_score = np.dot(query, answer)
  combined = np.multiply(query, answer)
  X_vectors.append([dot_score])
  X_combined_vectors.append(combined)
  y.append(label)
  # break

X = np.array(X_vectors, dtype=np.float32)
X_combined = np.array(X_combined_vectors)
y = np.array(y, dtype=np.int32)

print("X shape:", X.shape)
print("X_combined shape:", X_combined.shape)
print("y shape:", y.shape)

np.save("X_embeddings.npy", X)
np.save("X_combined_embeddings.npy", X_combined)
np.save("y_labels.npy", y)