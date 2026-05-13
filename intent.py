import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# =====================================================
# Device
# =====================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================
# Load Models
# =====================================================
t5_model_name = "google/flan-t5-large"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)

embed_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name).to(device)


# =====================================================
# Embedding function (mask-aware mean pooling)
# =====================================================
def get_embedding(texts, batch_size=64):

    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = embed_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = embed_model(**inputs)
            token_embeddings = outputs.last_hidden_state

            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * mask).sum(1)
            counts = mask.sum(1)

            emb = (summed / counts).cpu().numpy()

        all_embeddings.append(emb)

    all_embeddings = np.vstack(all_embeddings)

    if single:
        return all_embeddings[0]
    return all_embeddings


# =====================================================
# Load data
# =====================================================
data_train = pd.read_csv('../../datasets/Dunnhumby/data_train_des.csv')
data_test = pd.read_csv('../../datasets/Dunnhumby/data_test_des.csv')


# =====================================================
# Build user → items mapping
# =====================================================
user_to_item_test_dict = {}

for user_id, content in data_test.groupby(['user_id']):
    user_to_item_test_dict[user_id] = {
        "items": content['item_id'].values,
        "descriptions": content['SUB_COMMODITY_DESC'].values
    }


# =====================================================
# Generate basket intent text
# =====================================================
def generate_shopping_intents_per_user(baskets, user_map, batch_size=8, max_length=250):

    all_intents = []

    for i in tqdm(range(0, len(baskets), batch_size), desc="Generate basket intents"):

        batch = baskets[i:i+batch_size]

        inputs = t5_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = t5_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length
            )

        decoded = [t5_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        all_intents.extend(decoded)

    return all_intents


# =====================================================
# Build T5 inputs (basket-level)
# =====================================================
basket_inputs = []
user_item_map = []

print("\nPreparing basket prompts...")

for user, item_data in tqdm(user_to_item_test_dict.items()):

    items = item_data["items"]
    descriptions = item_data["descriptions"]


    for i in range(len(items)):
        given_desc = np.delete(descriptions, i)
        # 购物篮

        # 版本5
        # prompt = (
        #     f"Given the basket items {given_desc}, write a brief shopping intent for the user and the situation that might lead them to buy these items together."
        # )
  
       
        # 版本9
        # prompt = (
        #     f"Given the basket items: {given_desc}, "
        #     "infer the user's underlying shopping intent and the real-life scenarios "
        #     "that explain why these items are purchased together. "
        #     "Describe the concrete situations and how the items function together to support these goals. "
        #     "Focus on motivations and behaviors rather than listing products."
        # )

        # 版本10
        prompt = (
          f"Given the basket items: {given_desc}, "
          "infer the user's underlying shopping intent and the real-life scenarios "
          "that explain why these items are purchased together. "
)

        basket_inputs.append(prompt)
        user_item_map.append(user)
# =====================================================
# Generate basket intent text
# =====================================================
shopping_intents = generate_shopping_intents_per_user(
    basket_inputs,
    user_item_map
)


# =====================================================
# Save intent text → CSV (NEW)
# =====================================================
intent_df = pd.DataFrame({
    "user_id": user_item_map,
    "intent_text": shopping_intents
})

intent_csv_path = '../../datasets/Dunnhumby/basket_intent_texts_10.csv'
intent_df.to_csv(intent_csv_path, index=False)

print(f"Saved intent texts → {intent_csv_path}")
