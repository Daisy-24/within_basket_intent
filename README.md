## File Description
- `intent.py` Intent feature construction
- `main.py` Model building and testing

## Usage
Please download the corresponding dataset first, modify the relevant paths, 
then run intent.py followed by main.py.

## Dependencies
- Python 3.11
- PyTorch 2.5.1 (CUDA 11.8)
- transformers 5.2.0
- numpy 1.25.2
- pandas 2.3.3
- scikit-learn 1.8.0
- tqdm 4.67.3
- cupy

## Prompt for Basket Intent Generation

Style 1: Generate a specific shopping intent or multiple scenarios (if have) for a user with items {descriptions} in their basket. Explain why the user might select these items, consider-ing various scenarios such as “preparation, household needs, or special events.

Style 2: Generate a shopping intent or multiple scenarios (if have) for a user whose basket contains {descriptions}. Describe possible real-life scenarios or motivations behind choosing each item — for example, daily household needs, event preparation, or other special events. Make the reasoning natural and human-like.

Style 3: Generate a detailed shopping intent or multiple scenarios (if have) for a user whose basket contains {descriptions}. Describe possible real-life scenarios or motivations behind choosing each item — for example, daily household needs, event preparation, or other special events. Make the reasoning natural and human-like.

Style 4: The user’s basket contains {descriptions}. Generate several possible shopping scenar-ios and corresponding user intentions — e.g., daily routines, event preparation, or special plans. For each scenario, briefly explain how the items relate to the user’s motivation.

Style 5: Given the basket items {descriptions}, write a brief shopping intent for the user and the situation that might lead them to buy these items together.

Style 6: You are a shopping behavior analyst. Infer the overall shopping intent. Describe the life scenario or goal. Do NOT list products. Products: {descriptions}.

Style 7: You are a shopping behavior analyst. A customer adds the following products to the basket. Infer the overall shopping intent. Summarize: the possible shopping goal, the life sce-nario or activity, what the customer is preparing for. “Do NOT describe items one by one. Do NOT list products. Provide a concise scenario-level description (1-2 sentences). Products: {de-scriptions}.       

Style 8: Describe the purchase intent in one sentence using the format: Customers buy this product to [goal] in order to [benefit] during [scenario]. Product:{descriptions}.

Style 9: Given the basket items: {descriptions}, infer the user’s underlying shopping intent and the real-life scenarios that explain why these items are purchased together. Describe the concrete situations and how the items function together to support these goals. Focus on moti-vations and behaviors rather than listing products.       
Style 10: Given the basket items: { descriptions }, infer the user’s underlying shopping intent and the real-life scenarios that explain why these items are purchased together. 
