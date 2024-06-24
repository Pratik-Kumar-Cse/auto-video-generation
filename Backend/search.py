import requests
import time

from typing import List
from termcolor import colored

from utils import generate_hmac

from gpt import generate_similar_search_terms


def search_for_stock_videos(
    query: str, api_key: str, it: int, min_dur: int
) -> List[str]:
    """
    Searches for stock videos based on a query.

    Args:
        query (str): The query to search for.
        api_key (str): The API key to use.

    Returns:
        List[str]: A list of stock videos.
    """

    # Build headers
    headers = {"Authorization": api_key}

    # Build URL
    qurl = f"https://api.pexels.com/videos/search?query={query}&per_page={it}"

    # Send the request
    r = requests.get(qurl, headers=headers)

    # Parse the response
    response = r.json()

    # Parse each video
    raw_urls = []
    video_url = []
    video_res = 0
    try:
        # loop through each video in the result
        for i in range(it):
            # check if video has desired minimum duration
            if response["videos"][i]["duration"] < min_dur:
                continue
            raw_urls = response["videos"][i]["video_files"]
            temp_video_url = ""

            # loop through each url to determine the best quality
            for video in raw_urls:
                # Check if video has a valid download link
                if ".com/video-files" in video["link"]:
                    # Only save the URL with the largest resolution
                    if (video["width"] * video["height"]) > video_res:
                        temp_video_url = video["link"]
                        video_res = video["width"] * video["height"]

            # add the url to the return list if it's not empty
            if temp_video_url != "":
                video_url.append(temp_video_url)

    except Exception as e:
        print(colored("[-] No Videos found.", "red"))
        print(colored(e, "red"))

    # Let user know
    print(colored(f'\t=> "{query}" found {len(video_url)} Videos', "cyan"))

    # Return the video url
    return video_url


def search_for_stock_videos_on_story_block(
    queries: list, api_key: str, secret_key: str, it=4, min_dur=10
) -> List[str]:
    """
    Searches for stock videos based on a query.

    Args:
        query (str): The query to search for.
        api_key (str): The API key to use.

    Returns:
        List[str]: A list of stock videos.
    """
    try:
        # Parse each video
        video_url = []

        for query in queries:
            data = call_story_block_api(query, api_key, secret_key, it, min_dur)
            if len(data["results"]) != 0:
                break

        if len(data["results"]) == 0:

            result_data = generate_similar_search_terms(query)

            print(colored(f"result_data : {result_data}", "yellow"))

            for result in result_data:
                data = call_story_block_api(result, api_key, secret_key, it, min_dur)
                if len(data["results"]) != 0:
                    break

        # loop through each video in the result
        for i in range(it):
            temp_video_url = data["results"][i]["preview_urls"]["_720p"]
            # add the url to the return list if it's not empty
            if temp_video_url != "":
                video_url.append(temp_video_url)

    except Exception as e:
        print(colored("[-] No Videos found.", "red"))
        print(colored(e, "red"))

    # Let user know
    print(colored(f'\t=> "{query}" found {len(video_url)} Videos', "cyan"))

    # Return the video url
    return video_url


def call_story_block_api(query: str, api_key: str, secret_key: str, it, min_dur):
    # Build headers
    headers = {
        "Content-Type": "application/json",
    }

    expires = str(int(time.time()) + (1 * 60 * 60))

    params = {
        "APIKEY": api_key,
        "EXPIRES": expires,  # Replace with expiration timestamp
        "HMAC": generate_hmac(secret_key, expires),
        "project_id": "475b6acd-c6d1-4361-8ce8-bdfce2e8a096",
        "user_id": "475b6acd-c6d1-4361-8ce8-bdfce2e8a094",
        "keywords": query,
        "content_type": "motionbackgrounds,templates",
        # "Content-Type": application/json
        # "quality": quality,
        "min_duration": min_dur,
        # "max_duration": 25,
        # "has_talent_released": has_talent_released,
        # "has_property_released": has_property_released,
        # "has_alpha": has_alpha,
        # "is_editorial": is_editorial,
        # "categories": categories,
        "page": 1,
        "results_per_page": it,
        # "sort_by": sort_by,
        # "sort_order": sort_order,
        # "required_keywords":required_keywords,
        # "filtered_keywords":filtered_keywords,
    }

    # Build URL
    url = "https://api.storyblocks.com/api/v2/videos/search"

    response = requests.get(url, headers=headers, params=params)

    response.raise_for_status()

    return response.json()
