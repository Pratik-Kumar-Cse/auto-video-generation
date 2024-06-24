import http.client
import json
import serpapi
from dotenv import load_dotenv
import os
from typing import Annotated


# Load environment variables
load_dotenv("../.env")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

client = serpapi.Client(api_key=SERPER_API_KEY)


def web_search(
    query: Annotated[str, "The search query"], type="/search"
) -> Annotated[str, "The search results"]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "hl": "en", "gl": "us"})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))
    return data.decode("utf-8")


def images_search(
    query: Annotated[str, "The search query"], type="/search"
) -> Annotated[str, "The search results"]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "hl": "en", "gl": "us"})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/images", payload, headers)
    res = conn.getresponse()
    data = res.read()

    response_data = json.loads(data.decode("utf-8"))

    # Get the list of image results
    images = response_data.get("images", [])

    # Initialize an empty list to store image URLs
    image_urls = []

    # Iterate over the image results and extract the 'imageUrl' for each image
    for image in images:
        image_url = image.get("imageUrl")
        if image_url:
            image_urls.append(image_url)

    # Print the list of image URLs
    print(image_urls)
    return image_urls


def video_search(
    query: Annotated[str, "The search query"]
) -> Annotated[str, "The search results"]:
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query, "hl": "en", "gl": "us"})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    conn.request("POST", "/videos", payload, headers)
    res = conn.getresponse()
    data = res.read()

    response_data = json.loads(data.decode("utf-8"))

    # Get the list of image results
    images = response_data.get("videos", [])

    # Initialize an empty list to store image URLs
    video_urls = []

    # Iterate over the image results and extract the 'imageUrl' for each image
    for image in images:
        video_url = image.get("link")
        if video_url:
            video_urls.append(video_url)

    # Print the list of image URLs
    print(video_urls)
    return video_urls


def search_images_videos(
    query,
    api_key=SERPER_API_KEY,
    engine="google",
    location="Austin,Texas",
    google_domain="google.com",
    gl="us",
    hl="en",
):
    """
    Performs a search on Google for images and videos related to the given query.

    Args:
        query (str): The search query string.
        api_key (str): Your SerpAPI API key.
        engine (str, optional): The search engine to use (e.g., 'google', 'bing'). Defaults to 'google'.
        location (str, optional): The location to use for the search. Defaults to 'Austin,Texas'.
        google_domain (str, optional): The Google domain to use for the search. Defaults to 'google.com'.
        gl (str, optional): The Google country code for the search. Defaults to 'us'.
        hl (str, optional): The Google language code for the search. Defaults to 'en'.

    Returns:
        dict: A dictionary containing the search results for images and videos from SerpAPI.
    """
    params = {
        "q": query,
        "api_key": api_key,
        "engine": engine,
        "location": location,
        "google_domain": google_domain,
        "gl": gl,
        "hl": hl,
        # "tbm": "isch,vid",  # Search for images and videos
    }

    results = client.search(params)

    return results


# query = "GPT-4o"
# search_results = images_search(query)
