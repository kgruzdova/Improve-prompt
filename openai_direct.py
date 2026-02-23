#!/usr/bin/env python3
"""
Скрипт для работы с OpenAI API
Создание напоминалки для оплаты услуг и другие заготовленные запросы.
Использует JSON: роль — контекст — задача — формат ответа.
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPTS_FILE = "prompts_direct.json"
STATE_FILE = "payment_state.json"


class OpenAIClient:
    """Класс для работы с OpenAI API"""

    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = "gpt-4.1-2025-04-14"
        self.max_tokens = 1000
        self.temperature = 0.7

        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY не найден в файле .env")

        self.client = OpenAI(api_key=self.api_key)
        # logger.info(f"OpenAI клиент инициализирован (model={self.model})")
        print(f"\n✨💬 model={self.model}")

    def send_message(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> Optional[Dict]:
        """Отправить сообщение и получить ответ от модели. Возвращает dict с текстом и usage."""
        try:
            temp = temperature if temperature is not None else self.temperature
            max_toks = max_tokens if max_tokens is not None else self.max_tokens

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_toks
            )

            answer = response.choices[0].message.content
            usage = response.usage
            logger.info(f"Получен ответ от модели (длина: {len(answer)} символов)")

            return {
                "text": answer,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }

        except Exception as e:
            logger.error(f"Ошибка при обращении к OpenAI API: {e}")
            return None


def load_prompt(filepath: str = PROMPTS_FILE):
    """Загружает промпт из JSON файла"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            prompts = data.get('prompts', [])
            if not prompts:
                print("❌ В файле нет промпта")
                return None
            return prompts[0]
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {filepath} не найден")
        return None
    except json.JSONDecodeError:
        print("❌ Ошибка: Неверный формат JSON файла")
        return None


def get_settings():
    """Запрашивает temperature и max_tokens"""
    temperature = 0.7
    temp_input = input(f"🌡️ Температура (0.0-1.0, по умолчанию {temperature}): ").strip()
    if temp_input:
        try:
            temperature = float(temp_input)
            if not (0.0 <= temperature <= 1.0):
                print("⚠️ Некорректное значение, используется 0.7")
                temperature = 0.7
        except ValueError:
            print("⚠️ Некорректное значение, используется 0.7")
            temperature = 0.7

    max_tokens = 1000
    tokens_input = input(f"🔢 Max tokens (по умолчанию {max_tokens}): ").strip()
    if tokens_input:
        try:
            max_tokens = int(tokens_input)
            if max_tokens <= 0:
                print("⚠️ Некорректное значение, используется 1000")
                max_tokens = 1000
        except ValueError:
            print("⚠️ Некорректное значение, используется 1000")
            max_tokens = 1000

    return temperature, max_tokens


def load_saved_state():
    """Загружает сохранённый список из файла (для новой сессии)"""
    try:
        path = Path(STATE_FILE)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return None


def save_state(data: dict):
    """Сохраняет актуальный список в файл"""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except IOError:
        pass


def try_parse_checklist(text: str) -> Optional[dict]:
    """Пытается извлечь валидный JSON чек-листа из ответа"""
    text = text.strip()
    for start in ('{', '\n{'):
        if start in text:
            idx = text.index(start)
            chunk = text[idx:]
            for end in range(len(chunk), 0, -1):
                try:
                    obj = json.loads(chunk[:end])
                    if isinstance(obj, dict) and 'steps' in obj:
                        return obj
                except json.JSONDecodeError:
                    continue
    return None


def build_system_message(prompt, saved_state: Optional[dict] = None):
    """Формирует системное сообщение: роль — контекст — задача — формат ответа"""
    task = prompt.get('task') or prompt.get('question', '')
    msg = (
        f"{prompt['role']}\n\n"
        f"Контекст: {prompt['context']}\n\n"
        f"Задача: {task}\n\n"
        f"Формат ответа: {prompt['format']}"
    )
    if saved_state and isinstance(saved_state, dict) and 'steps' in saved_state:
        msg += f"\n\nТекущее состояние списка (продолжай с него):\n{json.dumps(saved_state, ensure_ascii=False)}"
    return msg


def print_help():
    print("\n📖 Доступные команды:")
    print("     /помощь       — показать эту справку")
    print("     /очистить     — очистить историю диалога")
    print("     /история      — показать текущую историю сообщений")
    print("     /настройки    — изменить temperature и max_tokens")
    print("     /выход        — завершить работу")
    print()


def main():
    print("\n🚀 OpenAI - Напоминалка по оплате услуг")
    print("=" * 60)

    prompt = load_prompt()
    if not prompt:
        sys.exit(1)
    try:
        client = OpenAIClient()
    except ValueError as e:
        print(e)
        sys.exit(1)

    try:
        temperature, max_tokens = get_settings()
    except KeyboardInterrupt:
        print("\nОтменено пользователем")
        sys.exit(0)

    saved_state = load_saved_state()
    messages = [{"role": "system", "content": build_system_message(prompt, saved_state)}]
    total_session_tokens = 0
    message_count = 0

    if saved_state:
        print("📂 Загружен сохранённый список из предыдущей сессии")

    print_help()

    first_input = input("💬 Введите ваш вопрос/данные (или Enter для пустого запроса): ").strip() or None

    while True:
        try:
            if first_input is not None:
                user_input = first_input
                first_input = None
            else:
                user_input = input(f"\n💬 Вы: ").strip()

            if not user_input:
                continue

            if user_input.startswith('/'):
                cmd = user_input.lower()

                if cmd in ('/выход', '/exit', '/quit'):
                    print(f"\n👋 Сессия завершена.")
                    print(f"   Всего сообщений: {message_count}")
                    print(f"   Всего токенов за сессию: {total_session_tokens}")
                    break

                elif cmd in ('/очистить', '/clear'):
                    if Path(STATE_FILE).exists():
                        Path(STATE_FILE).unlink()
                    messages = [{"role": "system", "content": build_system_message(prompt)}]
                    print("🧹 История диалога и сохранённый список очищены.")
                    continue

                elif cmd in ('/история', '/history'):
                    print(f"\n📜 История диалога ({message_count} сообщений):")
                    print("-" * 60)
                    for msg in messages:
                        role_label = {"system": "⚙️ Система", "user": "👤 Вы", "assistant": "🤖 Ассистент"}.get(msg["role"], msg["role"])
                        text_preview = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
                        print(f"{role_label}: {text_preview}")
                        print("-" * 40)
                    continue

                elif cmd in ('/настройки', '/settings'):
                    temperature, max_tokens = get_settings()
                    print(f"✅ Настройки обновлены: temperature={temperature}, max_tokens={max_tokens}")
                    continue

                elif cmd in ('/помощь', '/help'):
                    print_help()
                    continue

                else:
                    print(f"⚠️ Неизвестная команда: {user_input}")
                    print_help()
                    continue

            messages.append({"role": "user", "content": user_input})

            print(f"\n📤 Отправляем запрос к OpenAI...")
            print("=" * 60)

            result = client.send_message(messages, temperature, max_tokens)

            if result is None:
                print("❌ Не удалось получить ответ от OpenAI")
                messages.pop()
                continue

            messages.append({"role": "assistant", "content": result["text"]})
            message_count += 1
            total_session_tokens += result["total_tokens"]

            checklist = try_parse_checklist(result["text"])
            if checklist:
                save_state(checklist)

            print(f"\n🤖 Ответ:")
            print("=" * 60)
            print(result["text"])
            print("=" * 60)
            print(f"  ℹ️  Токены: промпт={result['prompt_tokens']}, ответ={result['completion_tokens']}, "
                  f"всего={result['total_tokens']} | За сессию: {total_session_tokens}")

        except KeyboardInterrupt:
            print(f"\n\n👋 Сессия завершена (Ctrl+C).")
            print(f"   Всего сообщений: {message_count}")
            print(f"   Всего токенов за сессию: {total_session_tokens}")
            break

        except Exception as e:
            error_str = str(e)
            if "unsupported_country_region_territory" in error_str:
                print("❌ Ошибка: OpenAI API недоступен в вашем регионе")
                print("💡 Используйте VPN или попробуйте genapi_client.py")
            else:
                print(f"❌ Ошибка: {e}")
            continue


if __name__ == "__main__":
    main()
