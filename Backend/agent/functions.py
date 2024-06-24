from langchain_community.document_loaders import YoutubeLoader
from moviepy.editor import VideoFileClip
import youtube_dl
from pytube import YouTube
from typing import Annotated
import subprocess
import requests
import time
import socket
import os
from bs4 import BeautifulSoup

from youtube_transcript_api import YouTubeTranscriptApi

from youtube_transcript_api import YouTubeTranscriptApi


def get_transcription_from_yt_video(video_url):
    # Extract video ID from URL
    video_id = video_url.split("v=")[-1]

    try:
        # Fetch the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Format the transcript with timestamps
        transcript_with_timestamps = [
            f"{entry['start']}s: {entry['text']}" for entry in transcript
        ]

        return transcript_with_timestamps

    except Exception as e:
        return f"An error occurred: {e}"


# def get_transcription_from_yt_video(video_url):
#     # Get the video ID from the URL
#     video_id = video_url.split("v=")[-1]

#     # Fetch the transcript from the YouTube Transcript API
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)

#     # Initialize an empty list to store the transcript with timestamps
#     transcript_with_timestamps = []

#     # Iterate through the transcript and append each text chunk with its start and end time
#     for chunk in transcript:
#         text = chunk["text"]
#         start_time = chunk["start"]
#         end_time = chunk["start"] + chunk["duration"]
#         timestamp = (start_time, end_time)
#         transcript_with_timestamps.append((text, timestamp))

#     return transcript_with_timestamps


def get_transcription_from_video(
    video_url: Annotated[str, "youtube video url"]
) -> Annotated[str, "The search results"]:
    # Transcribe the videos to text
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    docs = loader.load()
    # Combine doc
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    return text


output_path = "./data"


def download_video(url, output_path=output_path):
    yt = YouTube(url)
    video = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
    )
    video.download(output_path=output_path)
    return video.default_filename


# def download_video(url, output_path='.'):
#     try:
#         # Create a YouTube object
#         yt = YouTube(url)

#         # Get the highest resolution stream available
#         stream = yt.streams.get_highest_resolution()

#         # Download the video and get the file path
#         file_path = stream.download(output_path)

#         print(f"Video downloaded successfully and saved to {file_path}")
#         return file_path
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# def download_video(video_url):
#     # Extract the video ID from the URL
#     video_id = video_url.split("=")[1]

#     # Create a YouTube object
#     yt = YouTube(video_url)

#     # Get the first stream with the highest resolution
#     stream = (
#         yt.streams.filter(progressive=True, file_extension="mp4")
#         .order_by("resolution")
#         .desc()
#         .first()
#     )

#     # Download the video
#     stream.download(output_path=output_path, filename=video_id)

#     # Return the video path
#     return os.path.join(os.getcwd(), video_id + ".mp4")


def get_timestamp(video_file):
    clip = VideoFileClip(video_file)
    duration = clip.duration
    clip.close()
    return duration


def download_and_get_timestamp(video_url):
    video_filename = download_video(video_url, output_path)
    print(video_filename)
    video_file = f"{output_path}/{video_filename}"
    timestamp = get_timestamp(video_file)
    return timestamp


def create_and_save_clips(
    clip_timestamps: Annotated[list, "list of sub clips time"],
    video_path: Annotated[str, "string of video path"],
) -> Annotated[str, "successfully"]:
    """
    Create and save multiple clips from a video by passing the clip's timestamp.

    Args:
        video_file (str): The path to the video file.
        clip_timestamps (list): A list of tuples containing the start and end timestamps for each clip.

    Returns:
        None
    """
    video = VideoFileClip(video_path)
    for i, (start_timestamp, end_timestamp) in enumerate(clip_timestamps):
        subclip = video.subclip(start_timestamp, end_timestamp)
        clip_filename = f"./data/clip_{i}.mp4"
        subclip.write_videofile(clip_filename)
        print(f"Clip {i} saved as {clip_filename}")

    return "success"


# res = get_transcription_from_video("https://www.youtube.com/watch?v=YLnmUnVKrXo")

# print(res)


def generate_script_from_article(article_url):
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    response = requests.get(article_url, headers=header)
    print(response.status_code)
    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    soup.text
    blog_title = soup.title.text
    print("blog_title", blog_title)
    blog_content = soup.find("div", class_="md-content")

    print("blog_content", blog_content)
    if blog_content:
        print(len(blog_content.text))


def get_article_content(url):
    """
    Returns the content of an article from the given URL
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        content_section = soup.find("div", {"id": "article-body"})
        if content_section:
            article_content = ""
            for p in content_section.find_all("p"):
                article_content += p.text + "\n"
            article_content = article_content.strip()

        print(article_content)

        return article_content
    else:
        return None


def extract_article(url):
    # Send a request to the URL and get the HTML content
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the article content
    article_content = soup.find("article")
    if article_content:
        content = article_content.get_text()
    else:
        content = soup.get_text()
        
    return content


# generate_script_from_article('https://www.nvidia.com/en-in/geforce/news/g-assist-ai-assistant/')
# extract_article("https://www.nvidia.com/en-in/geforce/news/g-assist-ai-assistant/")
