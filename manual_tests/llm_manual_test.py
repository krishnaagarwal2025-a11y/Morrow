import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.llm import generate_response

prompt = """
Explain what PostgreSQL is in two sentences.
"""


answer = generate_response(prompt)

print("\nGemma response:\n")
print(answer)