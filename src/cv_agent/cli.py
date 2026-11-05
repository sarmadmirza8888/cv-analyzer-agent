"""Interactive command-line chat: python -m cv_agent"""
import os

from cv_agent.bootstrap import build_assistant


def main():
    assistant = build_assistant()

    print("=" * 60)
    print("       CV ANALYZER AGENT  (summary memory + persistent history)")
    print("Ask about CVs, calculations, or general information.")
    print("Commands: memory | cache | save | reset | exit")
    print(f"History file: {assistant.history_path}")
    print("=" * 60)
    assistant.cache.stats()

    while True:
        try:
            user_query = input("\nYou: ").strip()
        except EOFError:
            assistant.save()
            break

        if not user_query:
            continue

        low = user_query.lower()

        if low in ["exit", "quit"]:
            assistant.save()
            print("Agent stopped.")
            break

        if low == "memory":
            assistant.memory.show()
            continue

        if low == "cache":
            assistant.cache.stats()
            continue

        if low == "save":
            assistant.save()
            continue

        if low == "reset":
            assistant.memory.clear()
            assistant.cache.clear()
            if os.path.exists(assistant.history_path):
                os.remove(assistant.history_path)
            print("Memory, cache and history.json cleared.")
            continue

        try:
            answer = assistant.ask(user_query)
            print("\nAgent:", answer)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
