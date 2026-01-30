from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the model
model = SentenceTransformer("../output/hinglish-model-v1")

print("="*60)
print("   WHAT IS INSIDE THE MODEL? (THE VECTORS)")
print("="*60)
print("The model converts text into 384 numbers (a Vector).")
print("These numbers represent the 'Meaning'.\n")

# 1. Let's look at the raw numbers for "Paise"
word_1 = "Paise"
vec_1 = model.encode(word_1)

print(f"Input: '{word_1}'")
print(f"Output (First 10 numbers of 384): {vec_1[:10]}")
print("... [374 more numbers] ...\n")

# 2. Let's look at "Money"
word_2 = "Money"
vec_2 = model.encode(word_2)

print(f"Input: '{word_2}'")
print(f"Output (First 10 numbers of 384): {vec_2[:10]}")
print("... [374 more numbers] ...\n")

# 3. Compare them
similarity = util.cos_sim(vec_1, vec_2)[0][0]
print(f"Similarity Score between '{word_1}' and '{word_2}': {similarity:.4f}")
print("(High score means the model knows they mean the same thing)\n")

# 4. Compare with something different like "Banana"
word_3 = "Banana"
vec_3 = model.encode(word_3)
similarity_wrong = util.cos_sim(vec_1, vec_3)[0][0]

print(f"Similarity Score between '{word_1}' and '{word_3}': {similarity_wrong:.4f}")
print("(Low score means the model knows they are different)")
