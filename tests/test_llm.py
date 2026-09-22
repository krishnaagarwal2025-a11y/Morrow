from .llm import generate_response


prompt = """
Explain what PostgreSQL is in two sentences.
"""


answer = generate_response(prompt)

print("\nGemma response:\n")
print(answer)