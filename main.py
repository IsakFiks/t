import ollama
response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Tell me a joke.',
  },
])
print(response['message']['content'])
