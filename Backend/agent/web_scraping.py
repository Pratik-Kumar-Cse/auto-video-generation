import os
from typing import Annotated
import json

# from dotenv import load_dotenv
# from langchain.adapters.openai import convert_openai_messages
# from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader

import requests
from bs4 import BeautifulSoup
import re
from termcolor import colored


# def scrape_website(
#     urls: Annotated[str, "The list of urls"]
# ) -> Annotated[str, "The search results"]:
#     """
#     Scrapes the content of a website using WebBaseLoader from LangChain.

#     Args:
#         urls (str): The URL of the website to scrape.

#     Returns:
#         list: A list of dictionaries containing the scraped content.
#     """

#     print("scrape_website", urls)
#     loader = WebBaseLoader(urls)
#     print("loader", loader)
#     data = json.load(loader)
#     print("data", data)
#     return data


def scrape_website(
    urls: Annotated[list, "The list of urls of the website"]
) -> Annotated[str, "The search results"]:
    """
    Scrape a website and fetch video or image URLs

    Args:
        url (str): The URL of the website to scrape

    Returns:
        list: A list of URLs of the scraped media
    """

    # Extract the media URLs from the elements
    media_urls = []

    for url in urls:
        try:
            # Send a request to the website and get the HTML response
            response = requests.get(url)
            html = response.content

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")

            # Find all media elements on the page
            image_media_elements = []
            video_media_elements = []

            video_media_elements = soup.find_all("video")

            image_media_elements = soup.find_all("img")

            media_elements = soup.find_all()

            for element in video_media_elements:
                src = element.get("src")
                if src:
                    media_urls.append(src)

            for element in image_media_elements:
                src = element.get("src")
                if src:
                    media_urls.append(src)

            for element in media_elements:
                # Check if the element is a link to a video or image
                if element.name == "a" and element.get("href"):
                    # Check if the link is to a video or image
                    if element.get("href").endswith(
                        (".mp4", ".webm", ".ogg", ".jpg", ".jpeg", ".png", ".gif")
                    ):
                        media_urls.append(element.get("href"))

            # Find all script tags that contain media URLs
            script_tags = soup.find_all("script")

            for script in script_tags:
                script_text = script.text
                urls = re.findall(r'src=[\'"]?([^\'" >]+)', script_text)
                media_urls.extend(urls)

        except Exception as err:
            print(colored(f"[-] Error: {str(err)}", "red"))

    # Remove duplicates and return the list of media URLs
    media_urls = list(set(media_urls))

    return media_urls
