from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader
import json
import os

# 1. Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = "../output/hinglish-model-v1"
BATCH_SIZE = 16
NUM_EPOCHS = 3
DATA_FILE = "../data/hinglish_dataset.json"

def train():
    # 2. Check if data exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run generate_data.py first.")
        return

    # 3. Load Data
    print("Loading dataset...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    train_examples = []
    for item in raw_data:
        # We create pairs: (Hinglish, English)
        # The model learns that these two should be close in vector space
        train_examples.append(InputExample(texts=[item['hinglish'], item['english']]))

    print(f"Loaded {len(train_examples)} training pairs.")

    # 4. Initialize Model
    print(f"Loading base model: {MODEL_NAME}...")
    word_embedding_model = models.Transformer(MODEL_NAME)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 5. Prepare DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    # 6. Define Loss
    # MultipleNegativesRankingLoss is great for (query, positive) pairs
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 7. Train
    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )

    print(f"Training complete! Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    train()
