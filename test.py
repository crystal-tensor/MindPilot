from openai import OpenAI

client = OpenAI(api_key="sk-5c35391ff9f04c73a3ccafff36fed371", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是谁？"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)