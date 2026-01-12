import gradio as gr
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load the Model (pointing to your local 'model' folder)
model = SentenceTransformer('./model')

# 2. Load the Index
index = faiss.read_index("my_amazon_index.faiss")

# 3. Load the IDs
item_ids = np.load("item_ids.npy")

# 4. Load the Lookup Dictionary
with open("product_lookup.pkl", "rb") as f:
    product_lookup = pickle.load(f)

# --- Search Logic ---
def search_products(query):
    k = 20  # Number of results to show
    
    # Encode Query
    query_vector = model.encode([query], normalize_embeddings=True)
    
    # Search Index
    D, I = index.search(query_vector, k)
    
    # Format Results for Gradio Gallery
    gallery_data = []
    
    for i in range(k):
        idx = I[0][i]
        score = float(D[0][i])
        product_id = item_ids[idx]
        
        # Retrieve Details
        details = product_lookup.get(product_id, {"title": "Unknown Product", "images": []})
        title = f"{details['title'][:60]}..."
        label = f"{title}\n(Match: {int(score*100)}%)"
        
        # Retrieve Image
        img_url = "https://via.placeholder.com/200?text=No+Image"
        if details['images']:
             # detailed check to handle different metadata formats
             img_data = details['images'][0]
             if isinstance(img_data, dict):
                 img_url = img_data.get('large', img_data.get('thumb', img_url))
             elif isinstance(img_data, str):
                 img_url = img_data
        
        # Append (Image, Caption) tuple
        gallery_data.append((img_url, label))
        
    return gallery_data

# --- Build & Launch Interface ---
with gr.Blocks(theme=gr.themes.Soft(), title="Bridging Language and Items for Explainable RecSys: A Scalable Semantic Search Framework using Contrastive Fine-Tuning of MPNet and FAISS") as demo:
    gr.Markdown("# Amazon Fashion Search Engine")
    
    with gr.Row():
        inp = gr.Textbox(placeholder="Search for 'red floral summer dress'...", label="Search Query", scale=4)
        btn = gr.Button("Search", variant="primary", scale=1)
    
    gallery = gr.Gallery(label="Recommendations", columns=4, height="auto", object_fit="contain")
    
    # Triggers
    btn.click(fn=search_products, inputs=inp, outputs=gallery)
    inp.submit(fn=search_products, inputs=inp, outputs=gallery)

demo.launch()