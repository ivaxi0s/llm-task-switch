from openai import OpenAI
client = OpenAI()

def predict(model_name, prompt):
    if model_name == 'gpt3.5':
        msgs = [{"role":"user", "content":prompt}]
        response = client.chat.completions.create(model='gpt-3.5-turbo', messages=msgs, temperature=0)
        return response.choices[0].message.content