from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes.authentication import authenticator
from routes.cart import router as cart_router
from config.database import get_cart_count

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from jose import jwt
import os
from dotenv import load_dotenv
from routes.authentication import get_user, userdb

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

# Recommend Products
data = pd.read_csv("amazon_cleaned.csv")
# cv = CountVectorizer(max_features=5000, stop_words='english')
# cv.fit_transform(data['tags'])
# vectors = cv.transform(data['tags']).toarray()
# similarity = cosine_similarity(vectors)


def recommend_by_cluster(product_id, df, top_n=8):
    cluster_id = data[data['product_id'] == product_id]['embeddings'].values[0]
    similar_products = df[data['embeddings'] == cluster_id]
    return similar_products[similar_products['product_id'] != product_id].head(top_n)


def recommend(product):
    product_index = df[df['product_name'] == product].index[0]
    distance = similarity[product_index]
    product_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:9]
    return product_list


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


@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
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

        num_of_products = 20
        chunk_size = 4

        df["rating"] = df["rating"].apply(lambda x: float(x))
        top_products = df[df["rating"] >= 4]
        random_top_products = top_products.sample(n=num_of_products).to_dict(orient="records")
        data_top_products = [random_top_products[i:i+chunk_size] for i in range(0, len(random_top_products), chunk_size)]
        
        # category_products = get_products_by_categories()

        return templates.TemplateResponse("home.html", {
            "request": request, 
            "top_products": data_top_products,
            "user": user,
            "cart_count": cart_count
        })
    except:
        return templates.TemplateResponse("home.html", {
            "request": request, 
            "top_products": data_top_products,
            "user": None,
            "cart_count": 0
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

    search_results = df[df["product_name"].str.contains(search_query, case=False, na=False)].to_dict(orient="records")

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
