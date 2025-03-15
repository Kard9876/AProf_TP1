import csv
import requests

# Defina sua chave da API DeepInfra
API_KEY = ""

# URL da API
url = "https://api.deepinfra.com/v1/openai/chat/completions"

# Nome do arquivo CSV
nome_arquivo = "dadosAI.csv"

# Cabeçalho do CSV
cabecalho = ["ID", "Text"]

# Criando e escrevendo o cabeçalho no CSV
with open(nome_arquivo, mode="w", newline="", encoding="utf-8") as arquivo:
    fd = csv.writer(arquivo, delimiter="\t")
    fd.writerow(cabecalho)

    for i in range(1,10000):
        prompt = "Write a text Escreva um texto simples continuous with approximately 100 to 120 words on a topic related to biology. The text must be direct, without subdivisions, topics or detailed explanations, just a fluid paragraph."

        # Configuração da requisição
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        # Fazendo a requisição
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            print(text)
            fd.writerow([i, text])
        else:
            print(f"Erro na requisição {i}: {response.status_code} - {response.text}")