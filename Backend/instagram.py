import os
import instagrapi
import json
from dotenv import load_dotenv

from instagrapi import Client
from instagrapi.exceptions import ChallengeRequired
from instagrapi.types import Usertag, Location

load_dotenv("../.env")


INSTAGRAM_ID = os.getenv("INSTAGRAM_ID")
INSTAGRAM_PASS = os.getenv("INSTAGRAM_PASS")


def upload_reel_to_instagram(reel_video_path, reel_caption):
    """
    Upload a reel to Instagram using instagrapi

    Args:
        username (str): Your Instagram username
        password (str): Your Instagram password
        reel_video_path (str): Path to the video file for the reel
        reel_caption (str): Caption for the reel

    Returns:
        bool: True if the reel is uploaded successfully, False otherwise
    """

    try:
        # Initialize the instagrapi client
        client = instagrapi.Client()

        # Login to Instagram
        client.login(INSTAGRAM_ID, INSTAGRAM_PASS)

        # user = client.user_info_by_username("INSTAGRAM_ID")

        # print(user)

        # Upload the reel video
        clip = client.clip_upload(reel_video_path, caption=reel_caption)

        return clip

    except Exception as e:
        print(f"Error uploading reel: {e}")
        return False


def upload_reel(username, password, video_path, caption):
    try:
        cl = Client()
        cl.login(username, password)
        media = cl.video_upload(video_path, caption=caption)
        print(f"Reel uploaded successfully! Media ID: {media.media_id}")
        return True

    except ChallengeRequired as challenge:
        try:
            # Solve the challenge
            cl.challenge_code_handler(challenge)

            # Retry uploading the reel
            media = cl.video_upload(video_path, caption=caption)

            print(f"Reel uploaded successfully! Media ID: {media.media_id}")
            return True

        except Exception as e:
            print(f"Error uploading reel: {e}")
            return False

    except Exception as e:
        print(f"Error uploading reel: {e}")
        return False


# username = "pratik_rapid"
# password = "Rapid@123"
# reel_video_path = "../data/rapid_video.mp4"
# reel_caption = "testing"
# location = Location(name="Russia, Saint-Petersburg", lat=59.96, lng=30.29)
# # tags = [Usertag(user="rapidinnovation.io", x=0.5, y=0.5),]

# upload_reel_to_instagram(username, password, reel_video_path, reel_caption, location)
# # upload_reel(username, password, reel_video_path, reel_caption)
