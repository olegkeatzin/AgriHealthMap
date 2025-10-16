import requests
import random
import string
import time

# --- НАСТРОЙКИ ---
BASE_URL = "http://127.0.0.1:8000"

def generate_random_string(length=8):
    """Генерирует случайную строку для создания уникальных имен пользователей."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def print_status(message, success=True):
    """Красиво выводит статус операции."""
    if success:
        print(f"✅ SUCCESS: {message}")
    else:
        print(f"❌ FAILURE: {message}")

def run_tests():
    """Основная функция для запуска всех тестов."""
    print("--- Запуск API тестов ---")
    
    # Создаем уникальные данные для двух пользователей, чтобы тесты не пересекались при повторных запусках
    user1_username = f"testuser_{generate_random_string()}"
    user1_password = "password123"
    
    user2_username = f"testuser_{generate_random_string()}"
    user2_password = "password456"

    user1_token = None
    user2_token = None

    # Используем сессию для удобства
    session = requests.Session()

    try:
        # --- ТЕСТ 1: РЕГИСТРАЦИЯ ---
        print("\n--- Сценарий 1: Регистрация пользователей ---")
        # Регистрация user1
        reg_response1 = session.post(f"{BASE_URL}/register", data={"username": user1_username, "password": user1_password})
        if reg_response1.status_code == 200:
            print_status(f"Пользователь '{user1_username}' успешно зарегистрирован.")
        else:
            print_status(f"Ошибка регистрации '{user1_username}'. Статус: {reg_response1.status_code}, Ответ: {reg_response1.text}", success=False)
            return

        # Попытка повторной регистрации user1 (ожидаем ошибку)
        reg_response1_fail = session.post(f"{BASE_URL}/register", data={"username": user1_username, "password": user1_password})
        if reg_response1_fail.status_code == 400:
            print_status("Повторная регистрация существующего пользователя корректно заблокирована.")
        else:
            print_status(f"Тест на повторную регистрацию провален. Статус: {reg_response1_fail.status_code}", success=False)
            return

        # Регистрация user2
        reg_response2 = session.post(f"{BASE_URL}/register", data={"username": user2_username, "password": user2_password})
        if reg_response2.status_code == 200:
            print_status(f"Пользователь '{user2_username}' успешно зарегистрирован.")
        else:
            print_status(f"Ошибка регистрации '{user2_username}'. Статус: {reg_response2.status_code}, Ответ: {reg_response2.text}", success=False)
            return

        # --- ТЕСТ 2: ПОЛУЧЕНИЕ ТОКЕНА (ВХОД) ---
        print("\n--- Сценарий 2: Вход и получение JWT токенов ---")
        # Вход для user1
        login_response1 = session.post(f"{BASE_URL}/token", data={"username": user1_username, "password": user1_password})
        if login_response1.status_code == 200:
            user1_token = login_response1.json().get("access_token")
            print_status(f"Токен для '{user1_username}' успешно получен.")
        else:
            print_status(f"Ошибка входа для '{user1_username}'. Статус: {login_response1.status_code}", success=False)
            return

        # Вход для user2
        login_response2 = session.post(f"{BASE_URL}/token", data={"username": user2_username, "password": user2_password})
        if login_response2.status_code == 200:
            user2_token = login_response2.json().get("access_token")
            print_status(f"Токен для '{user2_username}' успешно получен.")
        else:
            print_status(f"Ошибка входа для '{user2_username}'. Статус: {login_response2.status_code}", success=False)
            return
            
        # --- ТЕСТ 3: РАБОТА С ИЗБРАННЫМ (USER 1) ---
        print("\n--- Сценарий 3: Работа с избранным для User 1 ---")
        headers1 = {"Authorization": f"Bearer {user1_token}"}
        
        # Проверяем, что список избранного изначально пуст
        fav_get_resp1 = session.get(f"{BASE_URL}/favorites", headers=headers1)
        if fav_get_resp1.status_code == 200 and fav_get_resp1.json() == []:
            print_status("Список избранного для нового пользователя корректно пуст.")
        else:
            print_status(f"Ошибка: список избранного не пуст для нового пользователя. Ответ: {fav_get_resp1.json()}", success=False)
            return

        # Добавляем поле в избранное для user1
        field_data1 = {"name": "Поле пользователя 1", "bbox": [10.0, 20.0, 11.0, 21.0]}
        fav_post_resp1 = session.post(f"{BASE_URL}/favorites", headers=headers1, json=field_data1)
        if fav_post_resp1.status_code == 200:
            print_status(f"Поле '{field_data1['name']}' успешно добавлено для '{user1_username}'.")
        else:
            print_status(f"Ошибка добавления поля для '{user1_username}'. Ответ: {fav_post_resp1.text}", success=False)
            return

        # Проверяем, что поле появилось в списке
        fav_get_resp1_after = session.get(f"{BASE_URL}/favorites", headers=headers1)
        if fav_get_resp1_after.status_code == 200 and len(fav_get_resp1_after.json()) == 1:
            print_status("Список избранного для user1 корректно содержит 1 поле.")
        else:
            print_status(f"Ошибка: список избранного user1 не обновился. Ответ: {fav_get_resp1_after.json()}", success=False)
            return

        # --- ТЕСТ 4: ПРОВЕРКА ИЗОЛЯЦИИ ДАННЫХ (USER 2) ---
        print("\n--- Сценарий 4: Проверка изоляции данных (самый важный тест) ---")
        headers2 = {"Authorization": f"Bearer {user2_token}"}

        # Проверяем, что user2 НЕ видит избранные поля user1
        fav_get_resp2 = session.get(f"{BASE_URL}/favorites", headers=headers2)
        if fav_get_resp2.status_code == 200 and fav_get_resp2.json() == []:
            print_status(f"Пользователь '{user2_username}' НЕ видит данные '{user1_username}'. Изоляция работает!")
        else:
            print_status(f"КРИТИЧЕСКАЯ ОШИБКА: '{user2_username}' видит данные других пользователей! Ответ: {fav_get_resp2.json()}", success=False)
            return

        # Добавляем поле для user2 и проверяем, что оно добавилось только ему
        field_data2 = {"name": "Секретное поле user2", "bbox": [30.0, 40.0, 31.0, 41.0]}
        session.post(f"{BASE_URL}/favorites", headers=headers2, json=field_data2)
        fav_get_resp2_after = session.get(f"{BASE_URL}/favorites", headers=headers2)
        if fav_get_resp2_after.status_code == 200 and len(fav_get_resp2_after.json()) == 1:
            print_status(f"Поле успешно добавлено и отображается только для '{user2_username}'.")
        else:
            print_status(f"Ошибка добавления или отображения поля для '{user2_username}'.", success=False)
            return
            
    except requests.exceptions.ConnectionError:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к серверу.")
        print(f"Убедитесь, что сервер запущен по адресу {BASE_URL} и доступен.")
        return
    
    print("\n--- Все тесты успешно завершены! ---")


if __name__ == "__main__":
    run_tests()