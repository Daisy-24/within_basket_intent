import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =====================================================
# Load Models
# =====================================================
t5_model_name = "google/flan-t5-small"
t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)

embed_model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name).to(device)


# =====================================================
# Embedding function 
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

        #1
        # prompt = (
        #                 f"Generate a specific shopping intent or multiple scenarios (if have) for a user with items {given_desc} "
        #                 "in their basket. Explain why the user might select these items, considering various scenarios such as "
        #                 "preparation, household needs, or special events."
        #             )
        # 5
        prompt = (
            f"Given the basket items {given_desc}, write a brief shopping intent for the user and the situation that might lead them to buy these items together."
        )
        #6
        # prompt = (
        #     "You are a shopping behavior analyst. "
        #     "Infer the overall shopping intent. "
        #     "Describe the life scenario or goal. "
        #     "Do NOT list products.\n"
        #     f"Products: {given_desc}"
        # )
        # 7
        # prompt = ("You are a shopping behavior analyst. A customer adds the following products to the basket.\n"
        #           "Infer the overall shopping intent. \n"
        #           "Summarize: the possible shopping goal, the life scenario or activity, what the customer is preparing for. \n"
        #           "Do NOT describe items one by one. Do NOT list products. \n"
        #           "Provide a concise scenario-level description (1-2 sentences).\n"
        #           f"Products: {given_desc}"
        # )
        # 8
    #     prompt = (
    #     "Describe the purchase intent in one sentence using the format:\n"
    #     "'Customers buy this product to [goal] in order to [benefit] during [scenario].'\n"
    #     f"Product: {given_desc}"
    # )
        # 9

        # prompt = (
        #     f"Given the basket items: {given_desc}, "
        #     "infer the user's underlying shopping intent and the real-life scenarios "
        #     "that explain why these items are purchased together. "
        #     "Describe the concrete situations and how the items function together to support these goals. "
        #     "Focus on motivations and behaviors rather than listing products."
        # )
        # 10
#         prompt = (
#           f"Given the basket items: {given_desc}, "
#           "infer the user's underlying shopping intent and the real-life scenarios "
#           "that explain why these items are purchased together. "
# )

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
# Save intent text
# =====================================================
# intent_df = pd.DataFrame({
#     "user_id": user_item_map,
#     "intent_text": shopping_intents
# })
#
# intent_csv_path = '../../datasets/Dunnhumby/basket_intent_texts_10.csv'
# intent_df.to_csv(intent_csv_path, index=False)
#
# print(f"Saved intent texts → {intent_csv_path}")


# =====================================================
# Convert to embedding
# =====================================================
shopping_intent_embeddings = get_embedding(shopping_intents)


# =====================================================
# Build list
# =====================================================
basket_intent_list = []

for uid, emb in zip(user_item_map, shopping_intent_embeddings):
    basket_intent_list.append({
        "user_id": uid,
        "shopping_intent_embedding": emb
    })


basket_pickle_path = '../../datasets/Dunnhumby/basket_embeddings_384_t5_small.pkl'

with open(basket_pickle_path, 'wb') as f:
    pickle.dump(basket_intent_list, f)

print(f"Saved basket embeddings → {basket_pickle_path}")


# =====================================================
# Item-level intent
# =====================================================


print("\nGenerating item intents...")

unique_items_sorted = pd.concat([
    data_train[['item_id', 'SUB_COMMODITY_DESC']],
    data_test[['item_id', 'SUB_COMMODITY_DESC']]
]).drop_duplicates(subset='item_id').sort_values(by='item_id')


def generate_item_intent_embeddings(items_df):

    item_list = []

    for _, row in tqdm(items_df.iterrows(), total=len(items_df)):

        stock_code = row['item_id']
        description = row['SUB_COMMODITY_DESC']
        # version 1
        # prompt= (
        #     f"Generate a specific shopping intent or multiple scenarios (if have) for a user with item {description} "
        #     "in their basket. Explain why the user might select the item, considering various scenarios such as "
        #     "preparation, household needs, or special events." )
        # 
        # 5
        prompt = (
            f"Given the  item {description}, write a brief shopping intent for the user."
        )
        #6
        # prompt = (
        #     "You are an expert retail analyst. "
        #     "Infer the underlying purchase motivation and usage scenario.\n"
        #     f"Product: {description}"
        # )
        # 版本7
        # prompt = (
        #     "You are an expert retail analyst. Given the product title and description,\n"
        #     "infer the underlying purchase intent. Describe: 1) the main function of the product,\n"
        #     "2) the typical usage scenario, 3) the customer need or problem it solves.\n"
        #     "Do NOT repeat the product name. Do NOT list attributes.\n"
        #     "Focus on the shopping motivation and usage purpose (1-2 sentences).\n"
        #     f"Product: {description}"
        # )
        # 8
        # prompt = (
        #     "Describe the purchase intent in one sentence using the format:\n"
        #     "'Customers buy this product to [goal] in order to [benefit] during [scenario].'\n"
        #     f"Product: {description}"
        # )
        # 9
        # prompt = (
        #     f"Given the product title and description: {description}, "
        #     "infer the underlying purchase intent and explain why a customer would buy this product. "
        #     "Describe: "
        #     "the main purpose or function, "
        #     "specific real-life usage scenarios, "
        #     "the user’s goals or problems it solves. "
        #     "Focus on motivations and situations rather than product attributes. "
        #     "Do NOT repeat the product name or list specifications."
        # )

        # 10
        # prompt = (
        #     f"Given the product title and description: {description}, "
        #     "infer the underlying purchase intent and explain why a customer would buy this product. "
        # )
        inputs = t5_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = t5_model.generate(inputs["input_ids"], max_length=200)

        intent_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        emb = get_embedding(intent_text)

        item_list.append({
            "StockCode": stock_code,
            "embedding": emb
        })

    return item_list


item_intent_list = generate_item_intent_embeddings(unique_items_sorted)


item_pickle_path = '../../datasets/Dunnhumby/item_intent_embeddings_384_t5_small.pkl'

with open(item_pickle_path, 'wb') as f:
    pickle.dump(item_intent_list, f)

print(f"Saved item embeddings → {item_pickle_path}")


print("\nAll intents generated successfully ✓")




