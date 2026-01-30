from sentence_transformers import SentenceTransformer, util
import os

# Load the model
model_path = "../output/hinglish-model-v1"
if not os.path.exists(model_path):
    print("Model not found! Please run train.py first.")
    exit()

print("Loading model... (this may take a few seconds)")
model = SentenceTransformer(model_path)

# A small database of English intents/FAQs
english_faqs = [
    "I want to apply for a personal loan",
    "How to open a new bank account",
    "My internet connection is not working",
    "Where is the nearest branch?",
    "What is the interest rate?",
    "I want to return my order",
    "How do I reset my password?",
    "Contact customer support",
    "Check my account balance",
    "I want to transfer money",
    "What are you doing?",
    "Talk to me",
    "How are you?",
    "The Taj Mahal is in Agra",
    "Narendra Modi is the Prime Minister",
    "Python is a programming language"
]

print("\nEncoding database...")
faq_embeddings = model.encode(english_faqs)

print("\n" + "="*50)
print("   HINGLISH SEMANTIC SEARCH DEMO")
print("="*50)
print("Type ANY Hindi/Hinglish sentence.")
print("If I have a matching 'Concept' in my database, I will find it.")
print("-" * 50)

while True:
    query = input("\nEnter Query: ")
    if query.lower() in ['exit', 'quit']:
        break
        
    query_embedding = model.encode(query)
    
    # Find the top match
    scores = util.cos_sim(query_embedding, faq_embeddings)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx].item()
    best_match = english_faqs[best_idx]
    
    print(f"Input: '{query}'")
    
    if best_score < 0.25:
        print(f"❌ No good match found (Best: {best_score:.4f})")
        print(f"   (Closest guess was: '{best_match}')")
    else:
        print(f"✅ Match Found! (Score: {best_score:.4f})")
        print(f"   Concept: {best_match}")
