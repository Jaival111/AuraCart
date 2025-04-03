from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

df = pd.read_csv("amazon.csv")

# Recommend Products
data = pd.read_csv("amazon_cleaned.csv")
cv = CountVectorizer(max_features=5000, stop_words='english')
cv.fit_transform(data['tags'])
vectors = cv.transform(data['tags']).toarray()
similarity = cosine_similarity(vectors)
def recommend(product):
    product_index = df[df['product_name'] == product].index[0]
    distance = similarity[product_index]
    product_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:9]
    return product_list

@app.get('/', response_class=HTMLResponse)
async def home(request: Request):
    num_of_products = 20
    chunk_size = 4

    df["rating"] = df["rating"].apply(lambda x: float(x))
    top_products = df[df["rating"] >= 4]
    random_top_products = top_products.sample(n=num_of_products).to_dict(orient="records")
    data_top_products = [random_top_products[i:i+chunk_size] for i in range(0, len(random_top_products), chunk_size)]

    return templates.TemplateResponse("home.html", {"request": request, "top_products": data_top_products})

@app.post('/', response_class=HTMLResponse)
async def search(request: Request, search: str = Form(...)):
    search_query = search.strip().lower()
    
    if not search_query:
        return templates.TemplateResponse("search.html", {"request": request, "search_results": [], "message": "Please enter a search query.", "search_query": ""})

    search_results = df[df["product_name"].str.contains(search_query, case=False, na=False)].to_dict(orient="records")

    return templates.TemplateResponse("search.html", {"request": request, "search_results": search_results, "message": "No results found." if not search_results else "", "search_query": search})

@app.get('/product-{product_id}', response_class=HTMLResponse)
async def product(product_id: str, request: Request):
    chunk_size = 4
    product = df[df["product_id"] == product_id].to_dict(orient="records")[0]
    recommended_product_indices = recommend(product["product_name"])
    recommended_products = df.iloc[[i[0] for i in recommended_product_indices]].to_dict(orient="records")
    recommended_products = [recommended_products[i:i+chunk_size] for i in range(0, len(recommended_products), chunk_size)]

    return templates.TemplateResponse("product.html", {"request": request, "product": product, "recommended_products": recommended_products})

@app.get('/login', response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get('/signup', response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})