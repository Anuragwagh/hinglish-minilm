from huggingface_hub import HfApi, create_repo, upload_folder
import os

# ==========================================
# CONFIGURATION
# ==========================================
HF_USERNAME = input("Enter your Hugging Face Username: ")
MODEL_NAME = "hinglish-minilm-v1"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
TOKEN = input("Enter your Hugging Face Write Token: ")

# ==========================================
# 1. CREATE REPO
# ==========================================
print(f"\nCreating repository: {REPO_ID}...")
try:
    create_repo(repo_id=REPO_ID, token=TOKEN, exist_ok=True)
    print("✅ Repository created (or already exists).")
except Exception as e:
    print(f"❌ Error creating repo: {e}")
    exit()

# ==========================================
# 2. UPLOAD MODEL FILES
# ==========================================
print("\nUploading Model Files (from ./output)...")
api = HfApi()
api.upload_folder(
    folder_path="./output/hinglish-model-v1",
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)
print("✅ Model files uploaded.")

# ==========================================
# 3. UPLOAD README (Documentation)
# ==========================================
print("\nUploading README.md...")
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)
print("✅ README uploaded.")

print("\n" + "="*50)
print(f"🎉 SUCCESS! Your model is live at:")
print(f"https://huggingface.co/{REPO_ID}")
print("="*50)
