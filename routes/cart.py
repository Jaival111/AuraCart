from fastapi import APIRouter, Request, HTTPException, status, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from config.database import cartdb, get_cart, get_cart_count, userdb
from routes.authentication import get_user
import pandas as pd
import os
from dotenv import load_dotenv
import re

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter()
templates = Jinja2Templates(directory="templates")

df = pd.read_csv("amazon.csv")

def clean_price(price_str):
    """
    Clean price string by removing currency symbol and converting to float.
    
    Args:
        price_str (str): Price string that may contain currency symbol
        
    Returns:
        float: Cleaned price value
    """
    if isinstance(price_str, str):
        # Remove any non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.]', '', price_str)
        return float(cleaned) if cleaned else 0.0
    return float(price_str)

@router.post("/add-to-cart/{product_id}")
async def add_item_to_cart(request: Request, product_id: str):
    try:
        token = request.cookies.get("access_token")
        if not token or not token.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        # Get product details
        product = df[df['product_id'] == product_id].iloc[0]
        if product.empty:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
        
        # Add to cart
        cart = get_cart(username)
        if not cart:
            cart = {"user_id": username, "items": []}
        
        # Check if product already in cart
        for item in cart["items"]:
            if item["product_id"] == product_id:
                item["quantity"] += 1
                break
        else:
            cart["items"].append({
                "product_id": product_id,
                "quantity": 1
            })
        
        # Update cart in database
        cartdb.update_one(
            {"user_id": username},
            {"$set": {"items": cart["items"]}},
            upsert=True
        )
        
        return {"status": "success", "message": "Item added to cart"}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/remove-from-cart/{product_id}")
async def remove_item_from_cart(request: Request, product_id: str):
    try:
        token = request.cookies.get("access_token")
        if not token or not token.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        cart = get_cart(username)
        if not cart:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart not found")
        
        # Remove item from cart
        cart["items"] = [item for item in cart["items"] if item["product_id"] != product_id]
        
        # Update cart in database
        cartdb.update_one(
            {"user_id": username},
            {"$set": {"items": cart["items"]}}
        )
        
        return {"status": "success", "message": "Item removed from cart"}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/cart", response_class=HTMLResponse)
async def view_cart(request: Request):
    try:
        token = request.cookies.get("access_token")
        cart_items = []
        cart_count = 0
        user = None
        
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username = payload.get("sub")
                if username:
                    user = get_user(userdb, username)
                    cart = get_cart(username)
                    if cart:
                        for item in cart["items"]:
                            product = df[df['product_id'] == item["product_id"]].iloc[0]
                            if not product.empty:
                                cart_items.append({
                                    "product_id": item["product_id"],
                                    "product_name": product["product_name"],
                                    "img_link": product["img_link"],
                                    "discounted_price": clean_price(product["discounted_price"]),
                                    "quantity": item["quantity"]
                                })
                        cart_count = get_cart_count(username)
            except JWTError:
                pass
        
        return templates.TemplateResponse("cart.html", {
            "request": request,
            "cart_items": cart_items,
            "user": user,
            "cart_count": cart_count
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/update-cart/{product_id}")
async def update_cart_quantity(request: Request, product_id: str, quantity: int = Query(..., ge=1)):
    try:
        token = request.cookies.get("access_token")
        if not token or not token.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        
        cart = get_cart(username)
        if not cart:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart not found")
        
        # Update the quantity for the specified product
        for item in cart["items"]:
            if item["product_id"] == product_id:
                item["quantity"] = quantity
                break
        
        # Update the cart in the database
        cartdb.update_one(
            {"user_id": username},
            {"$set": {"items": cart["items"]}}
        )
        
        return {"status": "success", "message": "Cart updated successfully"}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 