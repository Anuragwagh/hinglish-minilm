# Hinglish-MiniLM-v1

This is a **Hinglish (Hindi-English)** sentence embedding model.
It maps "Hinglish" queries (e.g., *"Paise transfer karne hai"*) to their English semantic equivalents (e.g., *"Money Transfer"*).

It is designed for Indian developers building **Search Engines**, **Chatbots**, or **Recommendation Systems** for India.

## 🚀 Quick Start

First, install the library:
```bash
pip install sentence-transformers
```

### 1. Basic Usage (The "Hello World")
This shows how the model converts text into numbers (vectors).

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('anuragwagh0/hinglish-minilm-v1')

# Encode sentences
sentences = ["Mujhe loan chahiye", "I want a loan"]
embeddings = model.encode(sentences)

print(embeddings.shape)
# Output: (2, 384) -> Two sentences, each is a vector of size 384
```

### 2. Real World Example: Building a Semantic Router
This is how you use the model to "route" user queries to the right function in your app.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('anuragwagh0/hinglish-minilm-v1')

# 1. Define your App's Capabilities (The "Targets")
app_actions = [
    "Check Account Balance",
    "Transfer Money",
    "Call Customer Support"
]

# 2. Encode your actions (Do this once on startup)
action_vectors = model.encode(app_actions)

# 3. Simulate a User Query
user_query = "Bhai paise bhejne the"  # Hinglish Input
query_vector = model.encode(user_query)

# 4. Find the best match
scores = util.cos_sim(query_vector, action_vectors)[0]
best_match_idx = scores.argmax()
best_action = app_actions[best_match_idx]

print(f"User said: '{user_query}'")
print(f"Bot Action: {best_action}")
# Output: Bot Action: Transfer Money
```

## 🛠 Use Cases
1.  **Customer Support Chatbots**: Understand "Order kab aayega?" -> "Track Order".
2.  **E-Commerce Search**: Understand "Saste joote" -> "Cheap Shoes".
3.  **Content Recommendation**: Match Hinglish comments to English video tags.

## 📊 Performance
- **Base Model**: `all-MiniLM-L6-v2` (Lightweight, ~80MB)
- **Language**: Hindi + English (Code-Mixed)
- **Dimensions**: 384

## 👨‍💻 Training
Fine-tuned using `sentence-transformers` on a synthetic dataset of banking, tech support, and casual conversation pairs.
