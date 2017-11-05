import requests

response=requests.get("http://localhost:8080/lang")
print(response.json())