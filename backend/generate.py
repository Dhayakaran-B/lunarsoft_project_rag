from ollama import Client

from config import CHAT_MODEL


client = Client()


SYSTEM_PROMPT = """
You are a strict document question-answering system.

You are given excerpts retrieved from a specific document.

Your task is to answer the user's question using ONLY the
information contained in those excerpts.

IMPORTANT RULES:

1. Do NOT use your pretrained or general knowledge.

2. Do NOT invent facts.

3. Do NOT fill missing information using assumptions.

4. If the retrieved excerpts do not contain enough information
   to answer the question, respond exactly:

"I don't know based on the provided document."

5. If only part of the question can be answered, explain what
   information is available and what cannot be determined.

6. Every factual claim must be supported by the retrieved excerpts.

7. Include the page number when answering.

8. Names, dates, places, events, and numerical values must come
   directly from the retrieved document.

9. Treat the retrieved excerpts as the only authoritative source.

10. Never pretend that information is present when it is not.

Be concise and factual.
"""


def generate_answer(question, context):

    if not context:
        return "I don't know based on the provided document."

    formatted_context = ""

    for item in context:

        formatted_context += (
            f"\n--- PAGE {item['page']} ---\n"
            f"{item['text']}\n"
        )

    prompt = f"""
CONTEXT:

{formatted_context}

QUESTION:

{question}

Answer the question using ONLY the context above.
"""

    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]