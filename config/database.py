from pymongo import MongoClient
from dotenv import load_dotenv
import os
from typing import Optional, Dict, List

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)

db = client.get_database("AuraCart")

userdb = db["userdata"]
cartdb = db["carts"]


def get_cart(user_id: str) -> Optional[Dict]:
    """
    Get cart data for a user.
    
    Args:
        user_id (str): The user ID to look up
        
    Returns:
        Optional[Dict]: Cart data if found, None otherwise
    """
    cart = cartdb.find_one({"user_id": user_id})
    if cart:
        # Remove the _id field as it's not JSON serializable
        cart.pop('_id', None)
        return cart
    return None

def add_to_cart(user_id, product_id, quantity=1):
    cart = get_cart(user_id)
    if not cart:
        cart = {
            "user_id": user_id,
            "items": []
        }
        cartdb.insert_one(cart)
    
    # Check if product already exists in cart
    for item in cart["items"]:
        if item["product_id"] == product_id:
            item["quantity"] += quantity
            cartdb.update_one(
                {"user_id": user_id},
                {"$set": {"items": cart["items"]}}
            )
            return True
    
    # Add new product to cart
    cart["items"].append({
        "product_id": product_id,
        "quantity": quantity
    })
    cartdb.update_one(
        {"user_id": user_id},
        {"$set": {"items": cart["items"]}},
        upsert=True
    )
    return True

def remove_from_cart(user_id, product_id):
    cart = get_cart(user_id)
    if cart:
        cart["items"] = [item for item in cart["items"] if item["product_id"] != product_id]
        cartdb.update_one(
            {"user_id": user_id},
            {"$set": {"items": cart["items"]}}
        )
    return True

def get_cart_count(user_id: str) -> int:
    """
    Get the total number of items in a user's cart.
    
    Args:
        user_id (str): The user ID to look up
        
    Returns:
        int: Total number of items in the cart
    """
    cart = get_cart(user_id)
    if not cart or not cart.get("items"):
        return 0
    return sum(item.get("quantity", 0) for item in cart["items"])

