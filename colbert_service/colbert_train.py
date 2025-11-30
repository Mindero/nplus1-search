import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import pandas as pd
import random, os
from pylate import evaluation, losses, models, utils

def build_triplets():
  if os.path.exists("./colbert_service/triplets.csv"):
    print("triplets.csv already exists — skipping generation.")
    return
  print("triplets.csv not found — generating...")
  df = pd.read_csv("query_news_dataset_combined.csv")

  triplets = []

  grouped = df.groupby("query")

  for query, group in grouped:
      positives = group[group["relevance"] == 1]["answer"].tolist()
      negatives = group[group["relevance"] == 0]["answer"].tolist()

      if len(positives) == 0 or len(negatives) == 0:
          continue

      for pos in positives:
          neg = random.choice(negatives)
          triplets.append({
              "query": query,
              "positive": pos,
              "negative": neg
          })

  triplets_df = pd.DataFrame(triplets)
  triplets_df.to_csv("./colbert_service/triplets.csv", index=False)  

model_name = "ai-forever/ruBert-base"
run_name = "colbert-ruBert-v1"
output_dir = f"./colbert_service/output/{run_name}"
if __name__ == '__main__':
  build_triplets()
  batch_size = 32
  num_train_epochs = 2


  model = models.ColBERT(
      model_name_or_path=model_name
  )

  # model = torch.compile(model, mode="reduce-overhead")
  dataset = load_dataset("csv", data_files="triplets.csv")["train"]
  print("load dataset finish")
  splits = dataset.train_test_split(test_size=0.05)
  train_dataset = splits["train"]
  eval_dataset = splits["test"]

  train_loss = losses.Contrastive(model=model)

  dev_evaluator = evaluation.ColBERTTripletEvaluator(
      anchors=eval_dataset["query"],
      positives=eval_dataset["positive"],
      negatives=eval_dataset["negative"],
  )

  args = SentenceTransformerTrainingArguments(
      output_dir=output_dir,
      num_train_epochs=num_train_epochs,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      fp16=False,   # CPU -> False
      bf16=False,
      run_name=run_name,
      learning_rate=3e-6,
      dataloader_num_workers=4,
  )

  trainer = SentenceTransformerTrainer(
      model=model,
      args=args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      loss=train_loss,
      evaluator=dev_evaluator,
      data_collator=utils.ColBERTCollator(model.tokenize),
  )

  print("Going to train")
  trainer.train()
  print("Train finished")
  trainer.save_model(output_dir + "/test")
