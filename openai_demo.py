import os
from openai import OpenAI

OPENAI_API_KEY = 'sk-proj-OnVbC2bDNYQ6hMFkRKwbT3BlbkFJC4vPQ8pH0VOq7xP6uztP'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI()


def call_GPT(system_prompt, prompt, model_name="gpt-3.5-turbo"):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": 'system_prompt',
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        seed=42,
        max_tokens=300,
        temperature=0.0,
    )

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    system_prompt = 'You are a chatbot.'
    prompt = 'Hello! I am Nikhil. Who are you?'
    print(call_GPT(system_prompt=system_prompt, prompt=prompt))
