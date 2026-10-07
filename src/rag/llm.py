import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def generate_response(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    if not response.ok:
        print("OLLAMA STATUS:", response.status_code)
        print("OLLAMA RESPONSE:", response.text)

    response.raise_for_status()

    data = response.json()

    return data["response"]