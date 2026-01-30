from sentence_transformers import SentenceTransformer, util

# Load our freshly trained model
model_path = "../output/hinglish-model-v1"
model = SentenceTransformer(model_path)

# Test Queries
queries = [
    "Mujhe loan chahiye",
    "Account kaise kholu?",
    "Net nahi chal raha"
]

# Database of English FAQs (The "Target")
english_faqs = [
    "I want to apply for a personal loan",
    "How to open a new bank account",
    "Internet connection is down",
    "Where is the nearest branch?",
    "What is the interest rate?"
]

# Encode everything
query_embeddings = model.encode(queries)
faq_embeddings = model.encode(english_faqs)

print(f"Model loaded from: {model_path}\n")

for i, query in enumerate(queries):
    print(f"Query (Hinglish): '{query}'")
    
    # Find closest match
    scores = util.cos_sim(query_embeddings[i], faq_embeddings)[0]
    best_score_idx = scores.argmax()
    best_score = scores[best_score_idx]
    best_match = english_faqs[best_score_idx]
    
    print(f"Best Match (English): '{best_match}' (Score: {best_score:.4f})")
    print("-" * 30)
