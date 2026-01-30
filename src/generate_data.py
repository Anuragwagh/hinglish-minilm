import json
import random

# Templates for generating synthetic Hinglish data
# We map specific Hindi words/phrases to English equivalents and mix them into templates.

domains = {
    "finance": [
        {"h": "loan", "e": "loan"},
        {"h": "account", "e": "account"},
        {"h": "paise", "e": "money"},
        {"h": "balance", "e": "balance"},
        {"h": "transfer", "e": "transfer"},
        {"h": "credit card", "e": "credit card"},
        {"h": "statement", "e": "statement"},
        {"h": "interest rate", "e": "interest rate"},
    ],
    "ecommerce": [
        {"h": "order", "e": "order"},
        {"h": "return", "e": "return"},
        {"h": "refund", "e": "refund"},
        {"h": "delivery", "e": "delivery"},
        {"h": "track", "e": "track"},
        {"h": "buy", "e": "buy"},
        {"h": "price", "e": "price"},
        {"h": "discount", "e": "discount"},
    ],
    "tech": [
        {"h": "password", "e": "password"},
        {"h": "login", "e": "login"},
        {"h": "reset", "e": "reset"},
        {"h": "install", "e": "install"},
        {"h": "update", "e": "update"},
        {"h": "error", "e": "error"},
        {"h": "screen", "e": "screen"},
        {"h": "keyboard", "e": "keyboard"},
    ],
    "conversation": [
        {"h": "baat", "e": "talk"},
        {"h": "madad", "e": "help"},
        {"h": "kya", "e": "what"},
        {"h": "kaise", "e": "how"},
        {"h": "hello", "e": "hello"},
        {"h": "hi", "e": "hi"},
    ]
}

# Sentence templates: {h_phrase} will be replaced by Hinglish, {e_phrase} by English
templates = [
    {
        "h": "Mujhe {h_phrase} chahiye", 
        "e": "I want {e_phrase}"
    },
    {
        "h": "Mera {h_phrase} kaam nahi kar raha", 
        "e": "My {e_phrase} is not working"
    },
    {
        "h": "{h_phrase} kaise check karu?", 
        "e": "How do I check {e_phrase}?"
    },
    {
        "h": "{h_phrase} kab aayega?", 
        "e": "When will the {e_phrase} arrive?"
    },
    {
        "h": "Kya main {h_phrase} kar sakta hu?", 
        "e": "Can I {e_phrase}?"
    },
    {
        "h": "{h_phrase} kitne ka hai?", 
        "e": "How much is the {e_phrase}?"
    },
    {
        "h": "Mujhe {h_phrase} me help chahiye", 
        "e": "I need help with {e_phrase}"
    },
    {
        "h": "{h_phrase} change karna hai", 
        "e": "I want to change {e_phrase}"
    },
    {
        "h": "{h_phrase} karna hai", 
        "e": "I want to do {e_phrase}"
    },
    {
        "h": "Mujhe {h_phrase} karni hai", 
        "e": "I want to {e_phrase}"
    },
    {
        "h": "Mujhe {h_phrase} karna hai", 
        "e": "I want to {e_phrase}"
    },
    {
        "h": "Tum {h_phrase} kar rahe ho?", 
        "e": "Are you {e_phrase}ing?"
    },
     {
        "h": "{h_phrase} band ho gaya", 
        "e": "{e_phrase} has stopped"
    },
    {
        "h": "Naya {h_phrase} lena hai", 
        "e": "I want to get a new {e_phrase}"
    },
    {
        "h": "{h_phrase} kaam nahi kar raha hai",
        "e": "{e_phrase} is not working"
    }
]

def generate_dataset(num_samples=1000):
    data = []
    
    # Flatten domain words
    all_words = []
    for category in domains.values():
        all_words.extend(category)

    print(f"Generating data with {len(all_words)} keywords and {len(templates)} templates...")

    for _ in range(num_samples):
        # Pick a random word pair
        word_pair = random.choice(all_words)
        # Pick a random template
        template = random.choice(templates)
        
        # Construct sentences
        hinglish_sent = template["h"].format(h_phrase=word_pair["h"])
        english_sent = template["e"].format(e_phrase=word_pair["e"])
        
        data.append({
            "hinglish": hinglish_sent,
            "english": english_sent
        })
    
    # Add some hardcoded specific examples for variety
    hardcoded = [
        {"hinglish": "Bhai light chali gayi", "english": "Brother, the electricity is gone"},
        {"hinglish": "Net nahi chal raha", "english": "Internet is not working"},
        {"hinglish": "Phone uthao", "english": "Pick up the phone"},
        {"hinglish": "Gadi kharab ho gayi", "english": "The car broke down"},
        {"hinglish": "Bhook lagi hai", "english": "I am hungry"},
        {"hinglish": "Paani khatam ho gaya", "english": "Water is finished"},
        {"hinglish": "Station kaha hai?", "english": "Where is the station?"},
        {"hinglish": "Time kya hua hai?", "english": "What is the time?"},
        {"hinglish": "Kal milte hai", "english": "Let's meet tomorrow"},
        {"hinglish": "Bahut mehenga hai", "english": "It is very expensive"},
        {"hinglish": "Paise transfer karne hai", "english": "I want to transfer money"},
        {"hinglish": "Balance check karna hai", "english": "I want to check my balance"},
        {"hinglish": "Loan apply karna hai", "english": "I want to apply for a loan"},
        {"hinglish": "Customer care se baat karni hai", "english": "I want to talk to customer care"},
        {"hinglish": "Order return karna hai", "english": "I want to return my order"},
        {"hinglish": "Kya kar rahe ho?", "english": "What are you doing?"},
        {"hinglish": "Kya kr rahe ho", "english": "What are you doing?"},
        {"hinglish": "Ky karto a", "english": "What are you doing?"},
        {"hinglish": "Mujhse baat karo", "english": "Talk to me"},
        {"hinglish": "Kaise ho?", "english": "How are you?"},
        {"hinglish": "Tum kaun ho?", "english": "Who are you?"}
    ]
    
    data.extend(hardcoded)
    
    return data

if __name__ == "__main__":
    dataset = generate_dataset(200) # Generate 200 samples for this demo
    
    output_file = "../data/hinglish_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated {len(dataset)} pairs in '{output_file}'")
