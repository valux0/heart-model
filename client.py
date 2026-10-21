import requests

body = {
    "carat": 1.10,
    "cut": "Ideal",
    "color": "H",
    "clarity": "SI2",
    "depth": 62.0,
    "table": 55.0,
    "x": 6.61,
    "y": 6.65,
    "z": 4.11
}

response = requests.post(url='http://127.0.0.1:8000/score', json=body)
print(response.json())
