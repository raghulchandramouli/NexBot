import requests

url = "http://127.0.0.1:8001/query/"
data = {"question": "what is FHO?"}
response = requests.post(url, json=data)

print(response.json())
