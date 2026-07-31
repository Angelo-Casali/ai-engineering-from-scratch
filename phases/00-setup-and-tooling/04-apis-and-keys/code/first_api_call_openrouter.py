from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=os.environ['OPENROUTER_API_KEY'],
)

response = client.chat.completions.create(
    model='google/gemma-4-26b-a4b-it:free',
    max_tokens=256,
    messages=[{'role': 'user', 'content': 'What is a neural network in one sentence?'}]
)

print(f"Response: {response.choices[0].message.content}")