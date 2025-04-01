from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)

db = client.get_database("AuraCart")

userdb = db["userdata"]

