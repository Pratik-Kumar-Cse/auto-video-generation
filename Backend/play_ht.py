import requests
import os
from pyht import Client, TTSOptions, Format
from dotenv import load_dotenv
import json
from termcolor import colored
import time

# Load environment variables
load_dotenv("../.env")

PLAY_API_KEY = os.getenv("PLAY_API_KEY")
PLAY_USER_ID = os.getenv("PLAY_USER_ID")


from utils import split_long_string


def create_voice(text):
    try:
        url = "https://api.play.ht/api/v2/tts"
        payload = {
            "text": text,
            "voice": "s3://voice-cloning-zero-shot/8f197327-3682-4004-92c1-e8549d36f53f/enhanced/manifest.json",
            "output_format": "mp3",
            "voice_engine": "PlayHT2.0",
            "quality": "high",
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "AUTHORIZATION": PLAY_API_KEY,
            "X-USER-ID": PLAY_USER_ID,
        }

        response = requests.post(url, json=payload, headers=headers)

        response.raise_for_status()  # Raise an error for bad responses

        print(response.text)

        id = json.loads(response.text)["id"]

        # id = "GWV4ZelbssSiQqHPYs"

        request_url = f"https://api.play.ht/api/v2/tts/{id}"

        # Maximum number of retries
        max_retries = 10
        retry_count = 0

        # Loop until we get the video_url or reach the maximum number of retries
        while retry_count < max_retries:
            # Call the get_video function
            response = requests.get(url=request_url, headers=headers)

            print(response.text)

            data = json.loads(response.text)

            status = data["status"]

            # Check if video_url is obtained
            if status == "complete":
                response_url = data["output"]["url"]
                # If video_url is obtained, break out of the loop
                break
            else:
                # Code execution will pause for 5 minutes before continuing
                print(colored("waiting for play ht creation...", "red"))
                # If video_url is not obtained, wait for 30 seconds before retrying
                time.sleep(30)
                retry_count += 1
                
        return response_url

    except Exception as err:
        print(colored(f"[-] Error: {str(err)}", "red"))


def create_voices(text):
    scripts = split_long_string(text)
    voice_links = []
    for script in scripts:
        voice_link = create_voice(script)
        voice_links.append(voice_link)
    return voice_links


def get_voices():

    url = "https://api.play.ht/api/v2/voices"

    headers = {
        "AUTHORIZATION": PLAY_API_KEY,
        "X-USER-ID": PLAY_USER_ID,
    }

    response = requests.get(url, headers=headers)

    print(response.text)


def create_audio(text):
    # Initialize PlayHT API with your credentials
    client = Client(
        user_id=PLAY_USER_ID,
        api_key=PLAY_API_KEY,
    )

    # Configure stream options
    options = TTSOptions(
        voice="s3://voice-cloning-zero-shot/8f197327-3682-4004-92c1-e8549d36f53f/enhanced/manifest.json",
        sample_rate=44_100,
        format=Format.FORMAT_MP3,
        speed=1,
    )

    # Open a file to write the audio stream
    output_file = "../data/output_audio.mp3"

    with open(output_file, "wb") as f:
        # Stream the audio and write it to the file
        for chunk in client.tts(
            text=text, voice_engine="PlayHT2.0-turbo", options=options
        ):
            f.write(chunk)

    print(f"Audio saved to: {os.path.abspath(output_file)}")
