import sqlite3
import json
import uuid
from typing import Optional, List, Dict, Any

# Этот импорт нужен для тайп-хинтинга, он не вызовет циклической зависимости
from pydantic import BaseModel

# Определяем модели прямо здесь, чтобы избежать циклических импортов
# Это более надежная структура, чем импортировать из main
class FavoriteFieldCreate(BaseModel):
    name: str
    bbox: List[float]

class UserInDB(BaseModel):
    username: str
    hashed_password: str


class DatabaseManager:
    """
    Класс для управления всеми операциями с базой данных SQLite.
    """
    def __init__(self, db_path: str = "favorites_and_users.db"):
        """
        Инициализирует менеджер базы данных.
        
        :param db_path: Путь к файлу базы данных SQLite.
        """
        self.db_path = db_path
        self._initialize_tables()

    def _get_connection(self):
        """Создает и возвращает соединение с базой данных."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по имени
        return conn

    def _initialize_tables(self):
        """Создает таблицы users и favorites, если они еще не существуют."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 1. Таблица пользователей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL
                    )
                """)
                # 2. Таблица избранных полей
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS favorites (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        bbox_json TEXT NOT NULL,
                        user_id INTEGER NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                """)
                conn.commit()
            print(f"✓ База данных '{self.db_path}' успешно инициализирована.")
        except Exception as e:
            print(f"❗️ Ошибка при инициализации базы данных: {e}")

    # --- Методы для работы с пользователями ---

    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Находит пользователя по имени и возвращает его данные."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, hashed_password FROM users WHERE username = ?", (username,))
            user_data = cursor.fetchone()
            if user_data:
                return UserInDB(username=user_data['username'], hashed_password=user_data['hashed_password'])
        return None

    def create_user(self, username: str, hashed_password: str) -> None:
        """Создает нового пользователя в базе данных."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                (username, hashed_password)
            )
            conn.commit()

    # --- Методы для работы с избранными полями ---

    def get_favorites_for_user(self, username: str) -> List[Dict[str, Any]]:
        """Возвращает список избранных полей для указанного пользователя."""
        favorites_list = []
        with self._get_connection() as conn:
            query = """
                SELECT f.id, f.name, f.bbox_json FROM favorites f
                JOIN users u ON f.user_id = u.id
                WHERE u.username = ? ORDER BY f.name
            """
            cursor = conn.cursor()
            rows = cursor.execute(query, (username,)).fetchall()
            for row in rows:
                favorites_list.append({
                    "id": row['id'],
                    "name": row['name'],
                    "bbox": json.loads(row['bbox_json'])
                })
        return favorites_list

    def create_favorite_for_user(self, username: str, field: FavoriteFieldCreate) -> Dict[str, Any]:
        """Добавляет новое избранное поле для пользователя и возвращает его."""
        new_id = str(uuid.uuid4())
        bbox_as_json = json.dumps(field.bbox)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            user_id = cursor.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()['id']
            cursor.execute(
                "INSERT INTO favorites (id, name, bbox_json, user_id) VALUES (?, ?, ?, ?)",
                (new_id, field.name, bbox_as_json, user_id)
            )
            conn.commit()
            
        return {"id": new_id, "name": field.name, "bbox": field.bbox}
    def delete_favorite_for_user(self, username: str, field_id: str) -> bool:
        """
        Удаляет избранное поле по его ID для указанного пользователя.
        
        Возвращает True, если удаление прошло успешно (была найдена и удалена 1 запись),
        и False в противном случае (поле не найдено или не принадлежит пользователю).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Сначала получаем user_id по имени пользователя
            user_row = cursor.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not user_row:
                return False  # Пользователь не найден
            
            user_id = user_row['id']
            
            # Выполняем удаление, проверяя и field_id, и user_id
            result = cursor.execute(
                "DELETE FROM favorites WHERE id = ? AND user_id = ?",
                (field_id, user_id)
            )
            conn.commit()
            
            # cursor.rowcount вернет количество удаленных строк.
            # Для успешного удаления оно должно быть равно 1.
            return result.rowcount > 0
    
db_manager = DatabaseManager()