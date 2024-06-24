import requests


def get_replicas():

    url = "https://tavusapi.com/v2/replicas"

    headers = {"x-api-key": "3ae090d0c5b548439faef2a067b3cfc6"}

    response = requests.request("GET", url, headers=headers)

    print(response.text)


# get_replicas()
