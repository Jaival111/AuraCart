from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes.authentication import authenticator
from routes.cart import router as cart_router
from config.database import get_cart_count, add_newsletter_subscriber
from routes.authentication import get_user, userdb
from pinecone import Pinecone
import polars as pl
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
from jose import jwt
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI()
app.include_router(authenticator)
app.include_router(cart_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

df = pd.read_csv("amazon.csv")
data = pd.read_csv("amazon_cleaned.csv")

# Recommend Products
def recommend_by_cluster(product_id, df, top_n=8):
    cluster_id = data[data['product_id'] == product_id]['embeddings'].values[0]
    similar_products = df[data['embeddings'] == cluster_id]
    return similar_products[similar_products['product_id'] != product_id].head(top_n)


def product_by_category(category):
    return df[df['category'] == category]


def get_products_by_categories():
    # Get unique categories by splitting the category string
    all_categories = ["Electronics", "Fashion", "Home", "Living", "Beauty", "Sports", "Toys", "Books", "Games", "Jewelry"]
    
    # Get top 6 categories with most products
    category_counts = {}
    for category in all_categories:
        category_counts[category] = len(df[df['category'].str.contains(category, na=False)])
    
    category_products = []
    for category, _ in category_counts.items():
        category_df = df[df['category'].str.contains(category, na=False)]
        products = category_df.to_dict(orient="records")
        category_products.append({
            'category': category,
            'products': len(products)
        })
    
    return category_products


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
data['search_text'] = data['product_name'] + " " + data['tags']
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")
index = pc.Index("auracart-product-search")
model = SentenceTransformer('all-mpnet-base-v2')

def semantic_search(query, top_n=5):
    query_vector = model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_n, include_metadata=True)
    return results['matches']


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    token = request.cookies.get("access_token")
    user = None
    cart_count = 0
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username:
            user = get_user(userdb, username)
            cart_count = get_cart_count(username)

    num_of_products = 20
    chunk_size = 4

    df["rating"] = df["rating"].apply(lambda x: float(x))
    top_products = df[df["rating"] >= 4]
    random_top_products = top_products.sample(n=num_of_products).to_dict(orient="records")
    data_top_products = [random_top_products[i:i+chunk_size] for i in range(0, len(random_top_products), chunk_size)]
    
    category_count = {
        'electronics': "150K+",
        'fashion': "270K+",
        'home': "370K+",
        'beauty': "60K+",
        'sports': "20K+",
        'stationery': "10K+",
        'toys': "190K+",
        'others': "280K+"
    }

    return templates.TemplateResponse("home.html", {
        "request": request, 
        "top_products": data_top_products,
        "user": user,
        "cart_count": cart_count,
        "category_count": category_count
    })


@app.post('/search', response_class=HTMLResponse)
async def search(request: Request, search: str = Form(...)):
    try:
        token = request.cookies.get("access_token")
        user = None
        cart_count = 0
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
                cart_count = get_cart_count(username)
    except:
        user = None
        cart_count = 0

    search_query = search.strip().lower()
    
    if not search_query:
        return templates.TemplateResponse("search.html", {
            "request": request, 
            "search_results": [], 
            "message": "Please enter a search query.",
            "user": user,
            "cart_count": cart_count
        })

    search_results = semantic_search(search_query, top_n=10)
    search_results = [result['metadata'] for result in search_results]

    return templates.TemplateResponse("search.html", {
        "request": request, 
        "search_results": search_results, 
        "message": "No results found." if not search_results else "",
        "user": user,
        "cart_count": cart_count
    })


@app.get('/product-{product_id}', response_class=HTMLResponse)
async def product(product_id: str, request: Request):
    try:
        token = request.cookies.get("access_token")
        user = None
        cart_count = 0
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
                cart_count = get_cart_count(username)
    except:
        user = None
        cart_count = 0

    chunk_size = 4
    product = df[df["product_id"] == product_id].to_dict(orient="records")[0]
    
    # Get recommended products using the new cluster-based recommendation
    recommended_products_df = recommend_by_cluster(product_id, df)
    recommended_products = recommended_products_df.to_dict(orient="records")
    recommended_products = [recommended_products[i:i+chunk_size] for i in range(0, len(recommended_products), chunk_size)]

    return templates.TemplateResponse("product.html", {
        "request": request, 
        "product": product, 
        "recommended_products": recommended_products,
        "user": user,
        "cart_count": cart_count
    })

###################
###################
###################
###################
###################


products_data = pd.read_csv("amazon_products_cleaned.csv")


def get_top_products(category: str):
    num_of_products = 20
    chunk_size = 4
    category_df = products_data[products_data['category_id'] == category]
    
    # Try to get products with stars >= 4
    high_rated_products = category_df[category_df['stars'] >= 4]
    
    # If there are no high-rated products, use all products in the category
    if len(high_rated_products) == 0:
        top_products = category_df.sample(n=min(num_of_products, len(category_df))).to_dict(orient="records")
    else:
        top_products = high_rated_products.sample(n=min(num_of_products, len(high_rated_products))).to_dict(orient="records")
    
    # Ensure we have at least one product
    if not top_products:
        # If no products found, return an empty list
        return []
    
    # Group products into chunks for the carousel
    top_products = [top_products[i:i+chunk_size] for i in range(0, len(top_products), chunk_size)]
    return top_products


def get_best_sellers(category: str):
    num_of_products = 20
    chunk_size = 4
    category_df = products_data[products_data['category_id'] == category]
    
    # Try to get best sellers
    best_seller_products = category_df[category_df['isBestSeller'] == True]
    
    # If there are no best sellers, use all products in the category
    if len(best_seller_products) == 0:
        best_sellers = category_df.sample(n=min(num_of_products, len(category_df))).to_dict(orient="records")
    else:
        best_sellers = best_seller_products.sample(n=min(num_of_products, len(best_seller_products))).to_dict(orient="records")
    
    # Ensure we have at least one product
    if not best_sellers:
        # If no products found, return an empty list
        return []
    
    # Group products into chunks for the carousel
    best_sellers = [best_sellers[i:i+chunk_size] for i in range(0, len(best_sellers), chunk_size)]
    return best_sellers


def recommend_by_cluster_category(product_id, df, top_n=10):
    cluster_id = df[df['asin'] == product_id]['embeddings'].values[0]
    similar_products = df[df['embeddings'] == cluster_id]
    return similar_products[similar_products['asin'] != product_id][['asin', 'title', 'price']].head(top_n)


@app.get('/category/{category_}', response_class=HTMLResponse)
async def products_by_category(category_: str, request: Request):
    try:
        token = request.cookies.get("access_token")
        user = None
        cart_count = 0
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
                cart_count = get_cart_count(username)
    except:
        user = None
        cart_count = 0

    top_products = get_top_products(category_)
    best_sellers = get_best_sellers(category_)

    return templates.TemplateResponse("category.html", {
        "request": request,
        "category": category_,
        "top_products": top_products,
        "best_sellers": best_sellers,
        "user": user,
        "cart_count": cart_count
    })


@app.get('/p-{product_id}', response_class=HTMLResponse)
async def product_by_category(product_id: str, request: Request):
    try:
        token = request.cookies.get("access_token")
        user = None
        cart_count = 0
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
                cart_count = get_cart_count(username)
    except:
        user = None
        cart_count = 0

    chunk_size = 4
    product = products_data[products_data["asin"] == product_id].to_dict(orient="records")[0]
    
    # Get recommended products using the new cluster-based recommendation
    recommended_products_df = recommend_by_cluster_category(product_id, products_data)
    recommended_products = recommended_products_df.to_dict(orient="records")
    recommended_products = [recommended_products[i:i+chunk_size] for i in range(0, len(recommended_products), chunk_size)]

    return templates.TemplateResponse("product2.html", {
        "request": request, 
        "product": product, 
        "recommended_products": recommended_products,
        "user": user,
        "cart_count": cart_count
    })


@app.post('/subscribe-newsletter')
async def subscribe_newsletter(request: Request):
    try:
        # Try to get JSON data first
        try:
            data = await request.json()
            email = data.get('email')
        except:
            # If JSON parsing fails, try form data
            form_data = await request.form()
            email = form_data.get('email')
        
        if not email:
            print("Error: Email is required")
            return JSONResponse(
                status_code=400,
                content={"message": "Email is required"}
            )
            
        print(f"Attempting to subscribe email: {email}")
        success = add_newsletter_subscriber(email)
        
        if success:
            print(f"Successfully subscribed email: {email}")
            return JSONResponse(
                status_code=200,
                content={"message": "Successfully subscribed to newsletter!"}
            )
        else:
            print(f"Email already subscribed: {email}")
            return JSONResponse(
                status_code=400,
                content={"message": "Email already subscribed"}
            )
            
    except Exception as e:
        print(f"Error in newsletter subscription: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

# Load chatbot model and data
chatbot_df = pl.read_csv("temp.csv", ignore_errors=True)
# chatbot_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
desc_embeddings = pickle.load(open("product_description_embeddings.pkl", "rb"))

@app.get('/chatbot', response_class=HTMLResponse)
async def chatbot_page(request: Request):
    try:
        token = request.cookies.get("access_token")
        user = None
        cart_count = 0
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
                cart_count = get_cart_count(username)
    except:
        user = None
        cart_count = 0

    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "user": user,
        "cart_count": cart_count
    })

@app.post('/chatbot/search')
async def chatbot_search(request: Request, query: str = Form(...)):
    try:
        token = request.cookies.get("access_token")
        user = None
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(userdb, username)
    except:
        user = None

    # Process the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    desc_embeddings_tensor = torch.stack(desc_embeddings).to(query_embedding.device)
    similarities = util.cos_sim(desc_embeddings_tensor, query_embedding)

    # Add scores to dataframe
    df_with_scores = chatbot_df.with_columns([
        pl.Series(name="score", values=similarities.cpu().squeeze().tolist())
    ])

    # Get top 5 results
    top_results = df_with_scores.sort("score", descending=True).head(10)
    
    # Format results
    results = []
    for row in top_results.iter_rows(named=True):
        results.append({
            "title": row['title'],
            "price": row['price'],
            "rating": row['stars'],
            "image_url": row.get('image_url', '/static/default-product.png'),  # Use default image if no image URL
            "link": f"/p-{row['asin']}"  # Using the existing product endpoint
        })

    return JSONResponse(content={"results": results})