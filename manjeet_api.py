import requests
import json
url = "https://fakestoreapi.com/products/"

# make get request
reaponse = requests.get(url)
# create json object
json_data = json.loads(reaponse.text)
rate = json_data[0]["rating"]["rate"]
count = json_data[0]["rating"]["count"] 