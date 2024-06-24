import os
import sys
import json
import random
import logging
import zipfile
import requests
import hmac
import hashlib
import re


from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_dir(path: str) -> None:
    """
    Removes every file in a directory.

    Args:
        path (str): Path to directory.

    Returns:
        None
    """
    try:
        if not os.path.exists(path):
            os.mkdir(path)
            logger.info(f"Created directory: {path}")

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            os.remove(file_path)
            logger.info(f"Removed file: {file_path}")

        logger.info(colored(f"Cleaned {path} directory", "green"))
    except Exception as e:
        logger.error(f"Error occurred while cleaning directory {path}: {str(e)}")


def fetch_songs(zip_url: str) -> None:
    """
    Downloads songs into songs/ directory to use with geneated videos.

    Args:
        zip_url (str): The URL to the zip file containing the songs.

    Returns:
        None
    """
    try:
        logger.info(colored(f" => Fetching songs...", "magenta"))

        files_dir = "../Songs"
        if not os.path.exists(files_dir):
            os.mkdir(files_dir)
            logger.info(colored(f"Created directory: {files_dir}", "green"))
        else:
            # Skip if songs are already downloaded
            return

        # Download songs
        response = requests.get(zip_url)

        # Save the zip file
        with open("../Songs/songs.zip", "wb") as file:
            file.write(response.content)

        # Unzip the file
        with zipfile.ZipFile("../Songs/songs.zip", "r") as file:
            file.extractall("../Songs")

        # Remove the zip file
        os.remove("../Songs/songs.zip")

        logger.info(colored(" => Downloaded Songs to ../Songs.", "green"))

    except Exception as e:
        logger.error(colored(f"Error occurred while fetching songs: {str(e)}", "red"))


def choose_random_song() -> str:
    """
    Chooses a random song from the songs/ directory.

    Returns:
        str: The path to the chosen song.
    """
    try:
        songs = os.listdir("../Songs")
        song = random.choice(songs)
        logger.info(colored(f"Chose song: {song}", "green"))
        return f"../Songs/{song}"
    except Exception as e:
        logger.error(
            colored(f"Error occurred while choosing random song: {str(e)}", "red")
        )


def check_env_vars() -> None:
    """
    Checks if the necessary environment variables are set.

    Returns:
        None

    Raises:
        SystemExit: If any required environment variables are missing.
    """
    try:
        required_vars = ["PEXELS_API_KEY"]
        missing_vars = [
            var + os.getenv(var)
            for var in required_vars
            if os.getenv(var) is None or (len(os.getenv(var)) == 0)
        ]

        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            logger.error(
                colored(
                    f"The following environment variables are missing: {missing_vars_str}",
                    "red",
                )
            )
            logger.error(
                colored(
                    "Please consult 'EnvironmentVariables.md' for instructions on how to set them.",
                    "yellow",
                )
            )
            sys.exit(1)  # Aborts the program
    except Exception as e:
        logger.error(f"Error occurred while checking environment variables: {str(e)}")
        sys.exit(1)  # Aborts the program if an unexpected error occurs


def generate_hmac(secret_key, expires):

    # url info
    resource = "/api/v2/videos/search"

    hmacBuilder = hmac.new(
        bytearray(secret_key + expires, "utf-8"),
        resource.encode("utf-8"),
        hashlib.sha256,
    )
    hmacHex = hmacBuilder.hexdigest()

    return hmacHex


def srt_to_text_with_timestamps(file_path):
    
    with open(file_path, "r", encoding="utf-8") as file:
        srt_content = file.read()

    # Define a regular expression pattern to match the SRT format
    pattern = re.compile(
        r"(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s+([\s\S]+?)(?=\n\n|\Z)",
        re.MULTILINE,
    )

    matches = pattern.findall(srt_content)

    text_output = []
    for match in matches:
        start_time = match[1]
        end_time = match[2]
        text = match[3].replace(
            "\n", " "
        )  # Replace newlines within subtitles with spaces for cleaner output

        text_output.append(f"{start_time} --> {end_time}\n{text}")

    return "\n\n".join(text_output)


def download_image(image_url, index):
    try:
        save_path = f"../temp/image_{index}.jpg"
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful

        # Save the image to the specified path
        with open(save_path, "wb") as file:
            file.write(response.content)
        print(f"Image downloaded and saved as {save_path}")

        return save_path

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the image: {e}")
    except OSError as e:
        print(f"Failed to delete the image: {e}")


def remove_file(save_path):
    # Delete the image file after use
    os.remove(save_path)
    print(f"Image {save_path} deleted after use")
    
def split_long_string(input_string):
    if len(input_string) <= 2000:
        return [input_string]
    else:
        substrings = []
        start = 0
        end = 1800
        while start < len(input_string):
            substrings.append(input_string[start:end])
            start = end
            end += 1800
            if end > len(input_string):
                end = len(input_string)
        return substrings
