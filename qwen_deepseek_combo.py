"""
Программа объединяет Qwen и DeepSeek:
- Отправляет запрос обоим моделям параллельно
- Синтезирует финальный ответ с помощью одной из моделей
- Поддерживает режим "голосования" (voting) и "синтеза" (synthesis)

Установка зависимостей:
    pip install openai python-dotenv

API ключи (создать файл .env):
    QWEN_API_KEY=ваш_ключ
    DEEPSEEK_API_KEY=ваш_ключ
"""

import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ──────────────────────────────────────────────
# Клиенты для обеих моделей (OpenAI-совместимый API)
# ──────────────────────────────────────────────

qwen_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY", "YOUR_QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ──────────────────────────────────────────────
# Функции вызова моделей
# ──────────────────────────────────────────────

def call_qwen(prompt: str, system: str = "Ты полезный ассистент.") -> str:
    """Запрос к Qwen (Alibaba)."""
    try:
        response = qwen_client.chat.completions.create(
            model="qwen-plus",          # или qwen-turbo, qwen-max
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Qwen ERROR]: {e}"


def call_deepseek(prompt: str, system: str = "Ты полезный ассистент.") -> str:
    """Запрос к DeepSeek."""
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",      # или deepseek-reasoner
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[DeepSeek ERROR]: {e}"


# ──────────────────────────────────────────────
# Параллельный вызов обеих моделей
# ──────────────────────────────────────────────

def ask_both(prompt: str, system: str = "Ты полезный ассистент.") -> dict:
    """Параллельно спрашивает Qwen и DeepSeek, возвращает оба ответа."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_qwen     = executor.submit(call_qwen,     prompt, system)
        future_deepseek = executor.submit(call_deepseek, prompt, system)

        qwen_answer     = future_qwen.result()
        deepseek_answer = future_deepseek.result()

    return {
        "qwen":     qwen_answer,
        "deepseek": deepseek_answer,
    }


# ──────────────────────────────────────────────
# Режим 1: SYNTHESIS — синтез ответов
# ──────────────────────────────────────────────

def synthesize(prompt: str, answers: dict) -> str:
    """
    DeepSeek читает оба ответа и создаёт единый, улучшенный ответ.
    """
    synthesis_prompt = f"""Пользователь задал вопрос:
\"\"\"{prompt}\"\"\"

Ниже два независимых ответа от разных ИИ-моделей:

--- Ответ Qwen ---
{answers['qwen']}

--- Ответ DeepSeek ---
{answers['deepseek']}

Твоя задача: объедини лучшее из обоих ответов в один чёткий, точный и полный ответ.
Не упоминай, что ты синтезируешь ответы — просто дай финальный ответ."""

    return call_deepseek(synthesis_prompt, system="Ты эксперт-аналитик.")


# ──────────────────────────────────────────────
# Режим 2: VOTING — голосование / выбор лучшего
# ──────────────────────────────────────────────

def vote_best(prompt: str, answers: dict) -> dict:
    """
    Qwen выступает судьёй и выбирает лучший ответ.
    Возвращает победителя и обоснование.
    """
    judge_prompt = f"""Вопрос пользователя:
\"\"\"{prompt}\"\"\"

Ответ A:
{answers['qwen']}

Ответ B:
{answers['deepseek']}

Определи, какой ответ лучше (A или B), и объясни почему в 1-2 предложениях.
Формат ответа:
WINNER: A или B
REASON: <обоснование>"""

    verdict = call_qwen(judge_prompt, system="Ты строгий и справедливый судья.")

    winner = "qwen" if "WINNER: A" in verdict else "deepseek"
    return {
        "winner":  winner,
        "answer":  answers[winner],
        "verdict": verdict,
    }


# ──────────────────────────────────────────────
# Режим 3: CHAIN — цепочка (Qwen → DeepSeek)
# ──────────────────────────────────────────────

def chain(prompt: str) -> str:
    """
    Qwen даёт первичный ответ, DeepSeek его улучшает/дополняет.
    """
    qwen_answer = call_qwen(prompt)

    refine_prompt = f"""Пользователь задал вопрос:
\"\"\"{prompt}\"\"\"

Другая модель дала такой ответ:
{qwen_answer}

Улучши этот ответ: исправь ошибки, добавь детали, сделай его точнее и понятнее."""

    return call_deepseek(refine_prompt)


# ──────────────────────────────────────────────
# Главная функция — интерактивный чат
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🤖 Qwen + DeepSeek — объединённый ИИ-ассистент")
    print("=" * 60)
    print("Режимы работы:")
    print("  1 — synthesis : синтез лучшего из двух ответов")
    print("  2 — vote      : голосование (выбор лучшего)")
    print("  3 — chain     : цепочка (Qwen → DeepSeek)")
    print("  4 — both      : показать оба ответа без объединения")
    print("  q — выход")
    print("-" * 60)

    mode_map = {"1": "synthesis", "2": "vote", "3": "chain", "4": "both"}

    while True:
        mode_input = input("\nВыберите режим (1/2/3/4): ").strip()
        if mode_input.lower() == "q":
            print("До свидания!")
            break

        mode = mode_map.get(mode_input)
        if not mode:
            print("Неверный режим. Введите 1, 2, 3 или 4.")
            continue

        prompt = input("Ваш вопрос: ").strip()
        if not prompt:
            continue

        print("\n⏳ Запрашиваю модели...\n")

        if mode == "synthesis":
            answers = ask_both(prompt)
            print("── Qwen ──────────────────────────────────")
            print(answers["qwen"])
            print("\n── DeepSeek ──────────────────────────────")
            print(answers["deepseek"])
            print("\n── 🔀 Синтез ─────────────────────────────")
            print(synthesize(prompt, answers))

        elif mode == "vote":
            answers = ask_both(prompt)
            result = vote_best(prompt, answers)
            print("── Qwen ──────────────────────────────────")
            print(answers["qwen"])
            print("\n── DeepSeek ──────────────────────────────")
            print(answers["deepseek"])
            print("\n── 🏆 Победитель:", result["winner"].upper(), "──────────")
            print(result["answer"])
            print("\n── Вердикт судьи ─────────────────────────")
            print(result["verdict"])

        elif mode == "chain":
            print("── 🔗 Цепочка: Qwen → DeepSeek ───────────")
            print(chain(prompt))

        elif mode == "both":
            answers = ask_both(prompt)
            print("── Qwen ──────────────────────────────────")
            print(answers["qwen"])
            print("\n── DeepSeek ──────────────────────────────")
            print(answers["deepseek"])

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
