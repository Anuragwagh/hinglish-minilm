from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("../output/hinglish-model-v1")

# The "Menu" of what our app can do (Fixed targets)
app_actions = [
    "I want to apply for a personal loan",
    "I want to transfer money",
    "Internet is not working"
]

# Queries that are NOT in the training data (Unseen variations)
# The model has never seen these exact sentences.
unseen_queries = [
    "Loan ki sakht zarurat hai",       # "I am in dire need of a loan"
    "Kisi ko paise bhejne the",        # "Had to send money to someone"
    "Wifi chal hi nahi raha",          # "Wifi is not working at all"
    "Net bilkul dead hai"              # "Net is completely dead"
]

print("="*50)
print("   GENERALIZATION TEST (UNSEEN QUERIES)")
print("="*50)

encoded_actions = model.encode(app_actions)

for query in unseen_queries:
    query_vec = model.encode(query)
    scores = util.cos_sim(query_vec, encoded_actions)[0]
    best_idx = scores.argmax()
    best_action = app_actions[best_idx]
    
    print(f"User typed:   '{query}'")
    print(f"Model mapped: '{best_action}'")
    print(f"Confidence:   {scores[best_idx]:.4f}")
    print("-" * 30)
