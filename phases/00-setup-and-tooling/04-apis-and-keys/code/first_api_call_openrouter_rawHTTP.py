from dotenv import load_dotenv
import os
import urllib.request
import json

load_dotenv()

url = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}",
}
body = json.dumps({
    'model': 'google/gemma-4-26b-a4b-it:free',
    'max_tokens': 256,
    'messages': [{'role': 'user', 'content': 'What is a neural network in one sentence?'}],
}).encode()

req = urllib.request.Request(url, data=body, headers=headers, method='POST')
with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result['choices'][0]['message']['content'])