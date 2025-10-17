import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv, set_key
import bcrypt

# Импортируем db_manager и модели из файла database
# Это позволит избежать циклического импорта с main.py
from database import db_manager, UserInDB

# --- ИНИЦИАЛИЗАЦИЯ ROUTER ---
# Вместо app = FastAPI() мы используем APIRouter для группировки эндпоинтов
router = APIRouter(
    tags=["Authentication"]  # Группируем эндпоинты в документации Swagger
)

# --- МОДЕЛИ ДАННЫХ (Pydantic) ---
class User(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- НАСТРОЙКА БЕЗОПАСНОСТИ ---
def setup_secret_key():
    """Проверяет/генерирует SECRET_KEY и сохраняет его в .env."""
    dotenv_path = find_dotenv()
    if not dotenv_path:
        with open('.env', 'w') as f: pass
        dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        new_key = secrets.token_hex(32)
        set_key(dotenv_path, "SECRET_KEY", new_key)
        print("✓ Новый SECRET_KEY сгенерирован и сохранен в .env.")
        return new_key
    else:
        print("✓ SECRET_KEY успешно загружен из .env.")
        return secret_key

SECRET_KEY = setup_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ БЕЗОПАСНОСТИ ---
def verify_password(plain_password, hashed_password):
    password_byte = plain_password.encode('utf-8')
    hashed_password_byte = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte, hashed_password_byte)

def get_password_hash(password):
    password_byte = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_byte, salt)
    return hashed_password.decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- ФУНКЦИЯ-ЗАВИСИМОСТЬ ДЛЯ ПОЛУЧЕНИЯ ТЕКУЩЕГО ПОЛЬЗОВАТЕЛЯ ---
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
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
    
    user = db_manager.get_user_by_username(username=token_data.username)
    
    if user is None:
        raise credentials_exception
    return user

# --- ЭНДПОИНТЫ АУТЕНТИФИКАЦИИ ---
    
@router.post("/register")
async def register_user(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Регистрирует нового пользователя."""
    if db_manager.get_user_by_username(form_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(form_data.password)
    db_manager.create_user(form_data.username, hashed_password)
    
    return {"message": f"User '{form_data.username}' registered successfully"}

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """Проверяет логин/пароль и возвращает JWT токен."""
    user = db_manager.get_user_by_username(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}