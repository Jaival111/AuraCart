import polars as pl
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pickle

# Load and clean data
df = pl.read_csv("temp.csv", ignore_errors=True)

desc_embeddings = pickle.load(open("product_description_embeddings.pkl", "rb"))

# Extract descriptions as list
# descriptions = df.select("desc").to_series().to_list()

# Load model
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

# Batch encode descriptions
# batch_size = 32
# desc_embeddings = []

# for i in tqdm(range(0, len(descriptions), batch_size)):
#     batch_texts = descriptions[i:i+batch_size]
#     batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
#     desc_embeddings.extend(batch_embeddings.cpu())

# Save embeddings
# with open("product_description_embeddings.pkl", "wb") as f:
#     pickle.dump(desc_embeddings, f)

# Get query
query = "I want a durable travel bag with spinner wheels under $200"
query_embedding = model.encode(query, convert_to_tensor=True)

# Convert list of embeddings to tensor
desc_embeddings_tensor = torch.stack(desc_embeddings).to(query_embedding.device)

# Compute cosine similarity
similarities = util.cos_sim(desc_embeddings_tensor, query_embedding)

# Add similarity scores to the dataframe
df = df.with_columns([
    pl.Series(name="score", values=similarities.cpu().squeeze().tolist())
])

# Get top 5 results
top_results = df.sort("score", descending=True).head(5)

# Print top results
for row in top_results.iter_rows(named=True):
    print(f"🎒 {row['title']}")
    print(f"💰 Price: ${row['price']}")
    print(f"⭐ Rating: {row['stars']}")
