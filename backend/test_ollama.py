
from ollama import Client

client = Client()

response = client.chat(
    model="gemma4:31b-cloud",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Ollama is working."
        }
    ],
)

print(response.message.content)