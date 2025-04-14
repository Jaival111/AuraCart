from fastapi import Request, APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from config.database import userdb, get_cart_count, get_cart
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

authenticator = APIRouter()
authenticator.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hashed(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    user_data = db.find_one({"username": username})
    if user_data:
        return UserInDB(**user_data)
    return None


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(userdb, token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@authenticator.get('/login', response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@authenticator.get('/signup', response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


def create_user(db, username: str, email: str, password: str):
    existing_user = db.find_one({"username": username})
    if existing_user:
        return False
    
    hashed_password = get_password_hashed(password)
    user_data = {
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False
    }
    db.insert_one(user_data)
    return True


@authenticator.post("/")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(userdb, form_data.username, form_data.password)
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Incorrect username or password"}
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Get the same data as server.py
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
        'jewelry': "40K+"
    }

    response = templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user": user,
            "top_products": data_top_products,
            "cart_count": get_cart_count(user.username),
            "category_count": category_count
        }
    )
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return response


@authenticator.post("/signup")
async def register_user(request: Request):
    form_data = await request.form()
    username = form_data.get("username")
    email = form_data.get("email")
    password = form_data.get("password")
    
    if not username or not email or not password:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "All fields are required"}
        )
    
    if create_user(userdb, username, email, password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "message": "User created successfully. Please login."}
        )
    else:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Username already exists"}
        )


df = pd.read_csv("amazon.csv")

@authenticator.get('/', response_class=HTMLResponse)
async def home(request: Request):
    # Get top-rated products for all users
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
        'jewelry': "40K+"
    }

    try:
        token = request.cookies.get("access_token")
        if token and token.startswith("Bearer "):
            token = token.split(" ")[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username = payload.get("sub")
                if username:
                    # Get complete user data from database
                    user = get_user(userdb, username)
                    cart_count = get_cart_count(username)
                    
                    if user:
                        return templates.TemplateResponse("home.html", {
                            "request": request, 
                            "user": user, 
                            "top_products": data_top_products,
                            "cart_count": cart_count,
                            "category_count": category_count
                        })
            except JWTError:
                pass
    except Exception as e:
        print(f"Error in home route: {str(e)}")
    
    return templates.TemplateResponse("home.html", {
        "request": request, 
        "user": None,
        "top_products": data_top_products,
        "cart_count": 0,
        "category_count": category_count
    })


@authenticator.get('/logout')
async def logout(request: Request):
    response = templates.TemplateResponse(
        "login.html",
        {"request": request, "message": "You have been successfully logged out."}
    )
    response.delete_cookie(key="access_token")
    return response