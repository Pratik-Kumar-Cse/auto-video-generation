import requests
import json
import os
import time

from dotenv import load_dotenv

load_dotenv("../.env")

# Set environment variables
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
WEBHOOK_ENDPOINT = os.getenv("WEBHOOK_ENDPOINT")


def get_video(video_id):

    url = "https://api.heygen.com/v1/video_status.get"

    headers = {
        "accept": "application/json",
        "x-api-key": HEYGEN_API_KEY,
    }

    # Define the query parameters
    params = {
        "video_id": video_id,
    }

    response = requests.get(url, params=params, headers=headers)

    data = json.loads(response.text)

    print("data", data["data"])

    return data["data"]["video_url"]


def create_video(title, input_text, template):

    url = "https://api.heygen.com/v2/video/generate"

    if template == "temp3":
        avatar_id = "83e1ce3cdb21477f9fa07df472ba6214"
    else:
        avatar_id = "4cfeb3afb1a1497e9274b15197796537"

    payload = {
        "test": False,
        "caption": True,
        "title": title,
        "dimension": {"width": 960, "height": 1080},
        "video_inputs": [
            {
                "character": {
                    "type": "talking_photo",
                    "talking_photo_id": "fed23affadcb496c94e3e1ddf50fccf4",
                    # "avatar_style": "normal",
                },
                "voice": {
                    "type": "text",
                    "input_text": input_text,
                    "voice_id": "f35bf5153bef4a2cb94f56b29ec67e7f",
                    "pitch": -5,
                    "speed": 1,
                },
            }
        ],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": HEYGEN_API_KEY,
    }

    response = requests.post(url, json=payload, headers=headers)

    video_id = ""

    # Check if the request was successful
    if response.ok:

        print("response data", response.text)

        response = json.loads(response.text)

        video_id = response["data"]["video_id"]

    else:
        # Print the error message if the request was not successful
        print("error :", response.text)

    return video_id


def add_webhook(entity_id):
    """
    Add a webhook endpoint to the HeyGen API.

    Args:
        url (str): The URL where the webhook events will be sent.
        events (list): A list of event types to listen for (e.g., ["avatar_video.success", "avatar_video.fail"]).
        entity_id (str): The ID of the entity (e.g., video_id or project_id) to receive notifications for.
        api_key (str): Your HeyGen API key.

    Returns:
        dict: The response from the HeyGen API.
    """
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": HEYGEN_API_KEY,
    }

    data = {
        "url": WEBHOOK_ENDPOINT,
        "events": ["avatar_video.success", "avatar_video.fail"],
        "entity_id": entity_id,
    }

    api_url = "https://api.heygen.com/v1/webhook/endpoint.add"

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to add webhook. Status code: {response.json()}"}


def create_video_with_voice(title, audio_urls, template, video_type):

    url = "https://api.heygen.com/v2/video/generate"

    if template == "temp3":
        avatar_id = "83e1ce3cdb21477f9fa07df472ba6214"
    else:
        avatar_id = "4cfeb3afb1a1497e9274b15197796537"

    if video_type == "shorts":
        dimension = {"width": 1080, "height": 1920}
        aspect_ratio = "16:9"
    else:
        dimension = {"width": 1920, "height": 1080}
        aspect_ratio = "9:16"

    video_inputs = []

    for audio_url in audio_urls:
        video_inputs.append(
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "avatar_style": "normal",
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url,
                },
                "background": {
                    "type": "image",
                    "url": "https://resource.heygen.ai/image/9f0038e172e748a183338b33895632bc/original",
                    # "image_asset_id": "9f0038e172e748a183338b33895632bc",
                },
            }
        )

    payload = {
        "test": False,
        "caption": True,
        "title": title,
        "dimension": dimension,
        "aspect_ratio": aspect_ratio,
        "video_inputs": video_inputs,
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": HEYGEN_API_KEY,
    }

    response = requests.post(url, json=payload, headers=headers)

    video_id = ""

    # Check if the request was successful
    if response.ok:

        print("response data", response.text)

        response = json.loads(response.text)

        video_id = response["data"]["video_id"]

    else:
        # Print the error message if the request was not successful
        print("error :", response.text)

    return video_id


def get_templates():
    template_list_url = "https://api.heygen.com/v2/templates"

    headers = {"accept": "application/json", "x-api-key": HEYGEN_API_KEY}

    response = requests.get(template_list_url, headers=headers)

    print("response data", response.text)

    response = json.loads(response.text)

    print("response", response)

    template_id = response.json()["data"]["templates"][0][
        "template_id"
    ]  # Get the first template_id

    print("template_id", template_id)


def upload_asset(path, content_type):
    url = "https://upload.heygen.com/v1/asset"
    with open(path, "rb") as f:
        resp = requests.post(
            url,
            data=f,
            headers={"Content-Type": content_type, "x-api-key": HEYGEN_API_KEY},
        )
    response = json.loads(resp.text)

    print("response", response)

    return response["data"]["id"]


# get_templates()


# video_id = create_video(
#     "testing image",
#     "Is there any way to provide our users with the option to clone their avatars directly within our platform? If so, what are the technical requirements and steps involved in integrating this functionality seamlessly?",
#     "temp2",
# )

# print("video_id", video_id)

# url = get_video("824cdf72f3c341a39852b0b5c7a34750")

# print("url", url)

# upload_asset("../videos/bg.png", "image/png")


def create_template_video(template_id):

    headers = {"Accept": "application/json", "X-API-KEY": HEYGEN_API_KEY}

    FRUIT_NAME = "APPLE"
    Subtitle = "The Wonderful World of Apples"
    scene_2_script = "Apples are one of the most widely consumed fruits worldwide. They come in various colors, shapes, and sizes, with over 7,500 known apple cultivars around the world."
    scene_3_script = (
        "Some popular varieties include Granny Smith, Red Delicious, Gala, and Fuji."
    )
    scene_4_script = "They are a rich source of dietary fiber, vitamin C, and antioxidants. Eating apples regularly has been linked to numerous health benefits, including improved digestion, lowered risk of heart disease, and enhanced immune function..."
    scene_5_script = "Additionally, apples are often associated with folklore and cultural traditions. They symbolize knowledge, temptation, and forbidden fruit in various mythologies and stories."

    # Modify Template Elements and Generate Video
    generate_url = f"https://api.heygen.com/v2/template/{template_id}/generate"
    payload = {
        "test": True,
        "caption": False,
        "title": FRUIT_NAME + " | Fruit Presentation",
        "variables": {
            "Subtitle": {
                "name": "Subtitle",
                "type": "text",
                "properties": {"content": Subtitle},
            },
            "FRUIT_NAME": {
                "name": "FRUIT_NAME",
                "type": "text",
                "properties": {"content": FRUIT_NAME},
            },
            "scene_2_script": {
                "name": "scene_2_script",
                "type": "text",
                "properties": {"content": scene_2_script},
            },
            "scene_3_script": {
                "name": "scene_3_script",
                "type": "text",
                "properties": {"content": scene_3_script},
            },
            "scene_4_script": {
                "name": "scene_4_script",
                "type": "text",
                "properties": {"content": scene_4_script},
            },
            "scene_5_script": {
                "name": "scene_5_script",
                "type": "text",
                "properties": {"content": scene_5_script},
            },
            "scene_1_image_1": {
                "name": "scene_1_image_1",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1567974772901-1365e616baa9",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_3_image_1": {
                "name": "scene_3_image_1",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/bD6x1R5.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_3_image_2": {
                "name": "scene_3_image_2",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/GmT8jqJ.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_3_image_3": {
                "name": "scene_3_image_3",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/kFXc1Dg.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_4_image_1": {
                "name": "scene_4_image_1",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1552255349-450c59a5ec8e",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_4_image_2": {
                "name": "scene_4_image_2",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_4_image_3": {
                "name": "scene_4_image_3",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/OeCAhPD.png",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_5_image_1": {
                "name": "scene_5_image_1",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/pFJPqUx.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_5_image_2": {
                "name": "scene_5_image_2",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/8MwSsQG.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_5_image_3": {
                "name": "scene_5_image_3",
                "type": "image",
                "properties": {
                    "url": "https://i.imgur.com/HxAM8yR.jpg",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_2_bg_image": {
                "name": "scene_2_bg_image",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1545160788-dc8fd1971a96",
                    "asset_id": None,
                    "fit": "contain",
                },
            },
            "scene_4_bg_image": {
                "name": "scene_4_bg_image",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1620992467768-3cb360bf2aac",
                    "asset_id": None,
                    "fit": "crop",
                },
            },
            "scene_5_bg_image": {
                "name": "scene_5_bg_image",
                "type": "image",
                "properties": {
                    "url": "https://images.unsplash.com/photo-1605153322277-dd0d7f608b4d",
                    "asset_id": None,
                    "fit": "crop",
                },
            },
        },
    }
    headers["Content-Type"] = "application/json"
    response = requests.post(generate_url, headers=headers, json=payload)
    if not response.json()["data"]:
        print(response)
        print(response.json()["error"])
        exit()

    video_id = response.json()["data"]["video_id"]
    print("video_id:", video_id)

    # Check Video Generation Status
    video_status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    while True:
        response = requests.get(video_status_url, headers=headers)
        status = response.json()["data"]["status"]

        if status == "completed":
            video_url = response.json()["data"]["video_url"]
            thumbnail_url = response.json()["data"]["thumbnail_url"]
            print(
                f"Video generation completed! \nVideo URL: {video_url} \nThumbnail URL: {thumbnail_url}"
            )

            # Save the video to a file
            video_filename = "generated_video.mp4"
            with open(video_filename, "wb") as video_file:
                video_content = requests.get(video_url).content
                video_file.write(video_content)

            break
        elif status == "processing" or status == "pending":
            print("Video is still processing. Checking status...")
            time.sleep(5)  # Sleep for 5 seconds before checking again
        elif status == "failed":
            error = response.json()["data"]["error"]
            print(f"Video generation failed. '{error}'")
            break
