import os
import uuid
import cv2
import numpy as np
import json
from collections import defaultdict

import requests
import srt_equalizer
import assemblyai as aai

import urllib.request

from typing import List
from moviepy.editor import *
from termcolor import colored
from dotenv import load_dotenv
from datetime import timedelta
from moviepy.video.fx.all import crop
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    clips_array,
    CompositeVideoClip,
    VideoClip,
    CompositeAudioClip,
)
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.tools.drawing import circle
import moviepy.editor as mpy

from transition import (
    fade,
    add_effect_transition,
    add_multi_transition,
    add_zoom_transition,
    add_animation_transition,
    add_burn_transition,
    cross_dissolve,
)

from moviepy.video.fx.margin import margin
from moviepy.video.tools.segmenting import findObjects
from moviepy.video.tools.drawing import color_gradient
from moviepy.video.fx.mask_color import mask_color
from moviepy.video.VideoClip import ColorClip
from moviepy.video.fx.mask_color import mask_color
from moviepy.audio.fx.volumex import volumex


import mediapipe as mp


from PIL import Image, ImageDraw

load_dotenv("../.env")

ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")


def save_video(
    video_url: str, video_id=uuid.uuid4(), directory: str = "../temp"
) -> str:
    """
    Saves a video from a given URL and returns the path to the video.

    Args:
        video_url (str): The URL of the video to save.
        directory (str): The path of the temporary directory to save the video to

    Returns:
        str: The path to the saved video.
    """
    # video_id = uuid.uuid4()
    print("Saving video id ==>", video_id)
    video_path = f"{directory}/{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(requests.get(video_url).content)

    return video_path


def __generate_subtitles_assemblyai(audio_path: str, voice: str) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.

    Returns:
        str: The generated subtitles
    """
    try:
        language_mapping = {
            "br": "pt",
            "id": "en",  # AssemblyAI doesn't have Indonesian
            "jp": "ja",
            "kr": "ko",
        }

        if voice in language_mapping:
            lang_code = language_mapping[voice]
        else:
            lang_code = voice

        aai.settings.api_key = ASSEMBLY_AI_API_KEY
        config = aai.TranscriptionConfig(language_code=lang_code)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        if transcript.status == aai.TranscriptStatus.error:
            print(transcript.error)
        else:
            print(transcript.text)
        subtitles = transcript.export_subtitles_srt()

        return subtitles
    except Exception as err:
        print(colored(f"Error: {str(err)}", "red"))


def __generate_subtitles_locally(
    sentences: List[str], audio_clips: List[AudioFileClip]
) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track
    Returns:
        str: The generated subtitles
    """

    def convert_to_srt_time_format(total_seconds):
        # Convert total seconds to the SRT time format: HH:MM:SS,mmm
        if total_seconds == 0:
            return "0:00:00,0"
        return str(timedelta(seconds=total_seconds)).rstrip("0").replace(".", ",")

    start_time = 0
    subtitles = []

    for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips), start=1):
        duration = audio_clip.duration
        end_time = start_time + duration

        # Format: subtitle index, start time --> end time, sentence
        subtitle_entry = f"{i}\n{convert_to_srt_time_format(start_time)} --> {convert_to_srt_time_format(end_time)}\n{sentence}\n"
        subtitles.append(subtitle_entry)

        start_time += duration  # Update start time for the next subtitle

    return "\n".join(subtitles)


def generate_subtitles(
    audio_path: str, sentences: List[str], audio_clips: List[AudioFileClip], voice: str
) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track

    Returns:
        str: The path to the generated subtitles.
    """

    def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
        # Equalize subtitles
        srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)

    # Save subtitles
    subtitles_path = f"../subtitles/{uuid.uuid4()}.srt"

    if ASSEMBLY_AI_API_KEY is not None and ASSEMBLY_AI_API_KEY != "":
        print(colored("[+] Creating subtitles using AssemblyAI", "blue"))
        subtitles = __generate_subtitles_assemblyai(audio_path, voice)
    else:
        print(colored("[+] Creating subtitles locally", "blue"))
        subtitles = __generate_subtitles_locally(sentences, audio_clips)
        # print(colored("[-] Local subtitle generation has been disabled for the time being.", "red"))
        # print(colored("[-] Exiting.", "red"))
        # sys.exit(1)

    with open(subtitles_path, "w") as file:
        file.write(subtitles)

    # Equalize subtitles
    equalize_subtitles(subtitles_path)

    print(colored("[+] Subtitles generated.", "green"))

    return subtitles_path


def combine_videos(
    video_paths: List[str], max_duration: int, max_clip_duration: int, threads: int
) -> str:
    """
    Combines a list of videos into one video and returns the path to the combined video.

    Args:
        video_paths (List): A list of paths to the videos to combine.
        max_duration (int): The maximum duration of the combined video.
        max_clip_duration (int): The maximum duration of each clip.
        threads (int): The number of threads to use for the video processing.

    Returns:
        str: The path to the combined video.
    """
    # Required duration of each clip
    req_dur = max_duration / len(video_paths)

    print(colored("[+] Combining videos...", "blue"))
    print(colored(f"[+] Each clip will be maximum {req_dur} seconds long.", "blue"))

    clips = []
    tot_dur = 0

    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    while tot_dur < max_duration:
        for video_path in video_paths:
            clip = VideoFileClip(video_path)
            clip = clip.without_audio()
            # Check if clip is longer than the remaining audio
            if (max_duration - tot_dur) < clip.duration:
                clip = clip.subclip(2, 2 + (max_duration - tot_dur))
            # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
            elif req_dur < clip.duration:
                clip = clip.subclip(2, 2 + req_dur)
            clip = clip.set_fps(30)

            # # Not all videos are same size,
            # # so we need to resize them
            # if round((clip.w / clip.h), 4) < 0.4625:
            #     clip = crop(
            #         clip,
            #         width=clip.w,
            #         height=round(clip.w / 0.4625),
            #         x_center=clip.w / 2,
            #         y_center=clip.h / 2,
            #     )
            # else:
            #     clip = crop(
            #         clip,
            #         width=round(0.4625 * clip.h),
            #         height=clip.h,
            #         x_center=clip.w / 2,
            #         y_center=clip.h / 2,
            #     )

            # clip = clip.resize((540, 960))

            if clip.duration > max_clip_duration:
                clip = clip.subclip(0, max_clip_duration)

            clips.append(clip)
            tot_dur += clip.duration

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_fps(30)

    final_clip.write_videofile("../data/combine.mp4", threads=threads or 2)
    return final_clip


def generate_video(
    combined_video_path: str,
    tts_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
) -> str:
    """
    This function creates the final video, with subtitles and audio.

    Args:
        combined_video_path (str): The path to the combined video.
        tts_path (str): The path to the text-to-speech audio.
        subtitles_path (str): The path to the subtitles.
        threads (int): The number of threads to use for the video processing.
        subtitles_position (str): The position of the subtitles.

    Returns:
        str: The path to the final video.
    """
    # Make a generator that returns a TextClip when called with consecutive
    generator = lambda txt: TextClip(
        txt,
        font="../fonts/bold_font.ttf",
        fontsize=100,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Split the subtitles position into horizontal and vertical
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)

    result = CompositeVideoClip(
        [
            VideoFileClip(combined_video_path),
            subtitles.set_pos(
                (horizontal_subtitles_position, vertical_subtitles_position)
            ),
        ]
    )

    # Add the audio
    audio = AudioFileClip(tts_path)
    result = result.set_audio(audio)

    result.write_videofile("../temp/output.mp4", threads=threads or 2)

    return "output.mp4"


def combine_two_videos(video1, video2):

    print("size", video1.size)

    clip1 = video1.crop(x1=100, y1=0, x2=1180, y2=720).resize(newsize=(1080, 960))

    clip2 = video2

    clip1 = clip1.set_duration(clip2.duration)

    # Get the video dimensions
    video_width, video_height = clip2.size

    # Calculate the coordinates for cropping
    x1 = (video_width - 500) // 2
    y1 = (video_height - 530) // 2
    x2 = x1 + 550
    y2 = y1 + 530

    clip2 = clip2.crop(x1=x1, y1=y1, x2=x2, y2=y2).resize(newsize=(1080, 960))

    # Stack the resized video clips on top of each other
    final_clip = clips_array([[clip1], [clip2]])

    final_clip = final_clip.resize(newsize=(1080, 1920))

    return final_clip


def generate_final_video(
    combined_video_clip: str,
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    video_paths,
) -> str:
    """
    This function creates the final video, with subtitles and audio.

    Args:
        combined_video_path (str): The path to the combined video.
        custom_video_path (str): The path to the custom video.
        subtitles_path (str): The path to the subtitles.
        threads (int): The number of threads to use for the video processing.
        subtitles_position (str): The position of the subtitles.
        text_color (str): The color of the subtitles text.

    Returns:
        str: The path to the final video.
    """

    final_clip = None
    output_path = "../data/rapid_video.mp4"

    custom_video = VideoFileClip(custom_video_path)

    if templates == "temp3":
        # Write the stacked clips to a temporary file
        final_clip = add_clips_between_videos(custom_video, video_paths)
        output_path = "../data/reel_temp3.mp4"

    elif templates == "temp2":
        # Write the stacked clips to a temporary file
        final_clip = masking_video(video1=combined_video_clip, video2=custom_video)

        output_path = "../data/reel_temp2.mp4"

    else:
        # Write the stacked clips to a temporary file
        final_clip = combine_two_videos(video1=combined_video_clip, video2=custom_video)

        output_path = "../data/reel_temp1.mp4"

    # Define a generator for creating TextClip objects
    generator = lambda txt: TextClip(
        txt=txt.upper(),
        font="../fonts/poppins-black.ttf",
        fontsize=80,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Extract the subtitles position
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Create a SubtitlesClip object
    subtitles = SubtitlesClip(subtitles_path, generator)

    # Composite the main video with the subtitles
    result = CompositeVideoClip(
        [
            final_clip,
            subtitles.set_position(
                (horizontal_subtitles_position, vertical_subtitles_position),
                relative=True,
            ),
        ]
    )

    video2 = VideoFileClip("../videos/rapid-outro.mp4")

    # Get the video dimensions
    width, height = video2.size

    # Calculate the aspect ratio of the input video
    aspect_ratio = width / height

    # Desired aspect ratio for reels/shorts
    target_aspect_ratio = 9 / 16

    if aspect_ratio > target_aspect_ratio:
        # Video is wider than the target aspect ratio
        new_width = int(height * target_aspect_ratio)
        x1 = (width - new_width) // 2
        x2 = x1 + new_width
        y1 = 0
        y2 = height
    else:
        # Video is taller than the target aspect ratio
        new_height = int(width / target_aspect_ratio)
        x1 = 0
        x2 = width
        y1 = (height - new_height) // 2
        y2 = y1 + new_height

    # Crop the video to the target aspect ratio
    video2 = crop(video2, x1=x1, y1=y1, x2=x2, y2=y2)

    video2 = video2.resize(width=result.w, height=result.h)

    # Concatenate the resized main video with the outro video
    final_video = concatenate_videoclips([result, video2], method="compose")

    # Write the final concatenated video to a file
    final_video.write_videofile(output_path, codec="libx264", fps=24)

    return output_path


def new_generate_final_video(
    combined_video_path: str,
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
) -> str:
    """
    This function creates the final video, with subtitles and audio.

    Args:
        combined_video_path (str): The path to the combined video.
        tts_path (str): The path to the text-to-speech audio.
        subtitles_path (str): The path to the subtitles.
        threads (int): The number of threads to use for the video processing.
        subtitles_position (str): The position of the subtitles.

    Returns:
        str: The path to the final video.
    """

    clip1 = VideoFileClip(combined_video_path)
    clip2 = VideoFileClip(custom_video_path)

    # # Crop the video clip
    # clip2 = clip2.crop(x1=290, y1=240, x2=850, y2=790)

    # Crop the video clip
    clip2 = clip2.crop(x1=290, y1=240, x2=850, y2=790)

    print("clip1 size:  ====> ", clip1.size)
    print("clip2 size:  ====> ", clip2.size)

    # Resize the videos to the same dimensions
    clip1_resized = clip1.resize(height=240, width=240)  # Adjust height as needed
    clip2_resized = clip2.resize(height=240, width=240)  # Adjust height as needed

    # Stack the two videos on top of each other
    final_clip = clips_array([[clip1_resized], [clip2_resized]])

    final_clip.write_videofile("../data/output1.mp4", codec="libx264", fps=24)

    # Make a generator that returns a TextClip when called with consecutive
    generator = lambda txt: TextClip(
        txt=txt.upper(),
        font="../fonts/poppins-black.ttf",
        fontsize=20,
        color=text_color,
        stroke_color="black",
        stroke_width=1,
    )

    # Split the subtitles position into horizontal and vertical
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)

    result = CompositeVideoClip(
        [
            VideoFileClip("../data/output1.mp4"),
            subtitles.set_pos(
                (horizontal_subtitles_position, vertical_subtitles_position)
            ),
        ]
    )

    result.write_videofile("../data/output.mp4", threads=threads or 2)

    video1 = VideoFileClip("../data/output.mp4")

    video2 = VideoFileClip("../videos/rapid-outro.mp4")

    print("video1 size:  ====> ", video1.size)
    print("video2 size:  ====> ", video2.size)

    # Resize the second video to match the dimensions of the first video
    video1_resized = video1.resize(width=video2.w, height=video2.h)

    # Concatenate the two videos vertically
    final_video = concatenate_videoclips([video1_resized, video2], method="compose")

    # Write the final concatenated video to a file
    final_video.write_videofile("../data/rapid_video.mp4", codec="libx264", fps=24)

    return "output.mp4"


def convert_video_to_audio(input_video_file, output_audio_file) -> str:
    # Load the video file
    video = VideoFileClip(input_video_file)

    # Extract the audio from the video
    audio = video.audio

    # Save the audio to an MP3 file
    audio.write_audiofile(output_audio_file)

    # Close the video and audio files
    video.close()
    audio.close()
    return output_audio_file


def crop_video(input_path, output_path, x1, y1, x2, y2):
    # Load the video clip
    clip = VideoFileClip(input_path)

    # Crop the video clip
    cropped_clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Write the cropped video to the output file
    cropped_clip.write_videofile(output_path)

    # Close the video clips
    clip.close()
    cropped_clip.close()


def new_masking_video():
    # create_mask_rectangle()
    # create_mask_circle()
    # Load background image
    # Load background image
    background = ImageClip("../videos/background.jpg").set_duration(10).set_fps(30)

    # Load videos
    video1 = VideoFileClip("../data/combine_videos.mp4").resize(background.size)
    video2 = VideoFileClip("../data/test.mp4").resize(background.size)

    # Create clip masks
    clip_mask1 = ImageClip("../data/mask1.png", ismask=True).resize(
        video1.size
    )  # A PNG image with a green rectangle as the mask
    clip_mask2 = ImageClip("../data/mask2.png", ismask=True).resize(
        video2.size
    )  # A PNG image with a green circle as the mask

    # # Apply clip masks to videos
    # masked_video1 = CompositeVideoClip(
    #     [video1.set_duration(10).set_fps(30), clip_mask1.set_position("center")]
    # )
    # masked_video2 = CompositeVideoClip(
    #     [video2.set_duration(10).set_fps(30), clip_mask2.set_position("center")]
    # )

    # Apply clip masks to videos
    masked_video1 = CompositeVideoClip([video1, clip_mask1])
    masked_video2 = CompositeVideoClip([video2, clip_mask2])

    # Composite the masked videos onto the background
    final_clip = CompositeVideoClip(
        [
            background,
            masked_video1.set_position(("center", "center")),
            masked_video2.set_position(("right", "bottom")),
        ]
    )

    # Set the duration of the final clip
    final_clip = final_clip.set_duration(max(video1.duration, video2.duration))

    # # Add subtitles
    # subtitles = TextClip("Subtitles", fontsize=70, color="white", bg_color="gray")
    # subtitles = subtitles.set_position(("center", "bottom"), relative=True)
    # subtitles = subtitles.set_duration(final_clip.duration)
    # final_clip = CompositeVideoClip([final_clip, subtitles])

    # Write the final video
    final_clip.write_videofile("../data/output2.mp4", threads=2)


def create_mask_rectangle():
    # Create a new transparent image
    width, height = 1920, 1080  # Adjust dimensions as needed
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Create a drawing object
    draw = ImageDraw.Draw(mask)

    # Draw a green rectangle
    rect_coords = (100, 100, 800, 600)  # (x1, y1, x2, y2)
    draw.rectangle(rect_coords, fill=(0, 255, 0))

    # Save the mask image
    mask.save("../data/mask1.png")


def create_mask_circle():
    # Create a new transparent image
    width, height = 1920, 1080  # Adjust dimensions as needed
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Create a drawing object
    draw = ImageDraw.Draw(mask)

    # Draw a green circle
    center_x, center_y = width // 2, height // 2  # Center coordinates
    radius = 200  # Adjust the radius as needed
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=(0, 255, 0),
    )

    # Save the mask image
    mask.save("../data/mask2.png")


# def create_mask_circle(size, center, radius):
#     height, width = size
#     mask = np.zeros((height, width), dtype=np.float32)
#     circle_mask = circle(mask.shape, center, radius, (1.0, 1.0, 1.0))
#     mask[circle_mask] = 1.0
#     return mask


def clip_mask_video(background_path, video_path, mask_center, mask_radius, output_path):
    # Load background image
    background = ImageClip(background_path).set_duration(4)

    # Load video
    video = VideoFileClip(video_path)

    # Get the size of the video
    video_size = video.size

    # Resize video to match background dimensions
    video = video.resize(background.size)

    # # Create circular mask
    mask = ImageClip("../data/mask2.png", ismask=True)

    # masked_video = video.copy()
    # masked_video = masked_video.add_mask()
    # masked_video = masked_video.set_mask(mask)
    # masked_video.mask.duration = video.duration

    # Apply circular mask to video
    masked_video = CompositeVideoClip([video.set_mask(mask)])

    # Create a circular mask
    # mask = create_mask_circle()

    # # Apply the circular mask to the video
    # circular_video = video.set_mask(mask)

    # # Write the circular video to file
    # circular_video.write_videofile(output_path, codec="libx264", fps=video.fps)

    # Write the final video to file
    masked_video.write_videofile("../data/mask_video.mp4", codec="libx264", fps=30)

    # Composite masked video onto the background
    final_clip = CompositeVideoClip([background, circular_video])

    # Write the final video to file
    final_clip.write_videofile(output_path, codec="libx264", fps=30)


def example_function():
    # Load the image specifying the regions.
    im = ImageClip("../videos/background.jpg")

    # Loacate the regions, return a list of ImageClips
    regions = findObjects(im)

    # Load 7 clips from the US National Parks. Public Domain :D
    clips = [
        VideoFileClip(n, audio=False).subclip(3, 22)
        for n in [
            "../temp/7.mp4",
            "../temp/7.mp4",
            "../temp/7.mp4",
            "../temp/7.mp4",
            "../temp/7.mp4",
            "../temp/7.mp4",
            "../temp/7.mp4",
        ]
    ]

    # fit each clip into its region
    comp_clips = [
        c.resize(r.size).set_mask(r.mask).set_pos(r.screenpos)
        for c, r in zip(clips, regions)
    ]

    cc = CompositeVideoClip(comp_clips, im.size)
    cc.resize(0.6).write_videofile("../data/composition.mp4")


def create_circular_mask(size, center, radius):
    height, width = size
    mask = np.zeros((height, width), dtype=np.float32)

    print(size, center, radius)
    # circle_mask = color_gradient(
    #     size, center, r=radius, col1=0.0, col2=1.0, shape="radial"
    # )
    circle_mask = circle(size, center=center, radius=radius)
    mask[:] = circle_mask
    return mask


# Define the circle mask
def circle_masks(frame, radius_percent):
    mask = np.zeros_like(frame, dtype=np.uint8)
    radius = int((min(frame.shape[1], frame.shape[0]) * radius_percent) / 100)
    cv2.circle(
        mask,
        (1020, 540),
        # min(frame.shape[1], frame.shape[0]) // 2,
        radius,
        (255, 255, 255),
        -1,
    )
    return mask


def create_rounded_rectangle_mask(frame, img_shape, corner_radius):
    """
    Creates a mask image with a rounded rectangle shape.

    Args:
        img_shape (tuple): Shape of the output mask image (height, width).
        corner_radius (int): Radius of the rounded corners.

    Returns:
        numpy.ndarray: A binary mask image with the rounded rectangle shape.
    """

    x, y = (0, 0)
    w, h = img_shape

    mask = np.zeros_like(frame, dtype=np.uint8)

    # Create a rounded rectangle mask
    cv2.rectangle(
        mask,
        (x + corner_radius, y),
        (x + w - corner_radius, y + h),
        (255, 255, 255),
        -1,
        corner_radius,
    )
    cv2.rectangle(
        mask,
        (x, y + corner_radius),
        (x + w, y + h - corner_radius),
        (255, 255, 255),
        -1,
        corner_radius,
    )
    cv2.circle(
        mask, (x + corner_radius, y + corner_radius), corner_radius, (255, 255, 255), -1
    )
    cv2.circle(
        mask,
        (x + w - corner_radius, y + corner_radius),
        corner_radius,
        (255, 255, 255),
        -1,
    )
    cv2.circle(
        mask,
        (x + corner_radius, y + h - corner_radius),
        corner_radius,
        (255, 255, 255),
        -1,
    )
    cv2.circle(
        mask,
        (x + w - corner_radius, y + h - corner_radius),
        corner_radius,
        (255, 255, 255),
        -1,
    )

    # Save the image to a file
    cv2.imwrite("../data/chamfered_rectangle_mask.png", mask)

    return mask


def circle_mask_with_margins(video, radius_percent, margin_thickness=15):
    # Create a single image with the circle
    img = np.zeros((video.h, video.w, 4), dtype=np.uint8)  # Add an alpha channel

    radius = int(
        (
            min(video.get_frame(0).shape[1], video.get_frame(0).shape[0])
            // 2
            * radius_percent
        )
        / 100
    )

    cv2.circle(
        img, (540, 540), radius, (*(0, 0, 255), 255), margin_thickness
    )  # Set the alpha channel to 255

    return img


def circle_mask_with_margin(video, radius_percent, margin_thickness=15):
    # Create a single image with the circle
    img = np.zeros((video.h, video.w, 4), dtype=np.uint8)  # Add an alpha channel

    radius = int(
        (min(video.get_frame(0).shape[1], video.get_frame(0).shape[0]) * radius_percent)
        / 100
    )

    # Define the colors
    colors = [
        (25, 28, 255),  # 191cff
        (173, 52, 255),  # ad34ff
        (25, 28, 255),  # 191cff
        (8, 178, 225),  # 08b2e1
    ]

    for i in range(360):  # iterate over all degrees in a circle
        angle = i * 3.14159 / 180  # convert degree to radian
        x = int(1020 + radius * np.cos(angle))  # calculate x coordinate
        y = int(540 + radius * np.sin(angle))  # calculate y coordinate

        # Calculate the color using interpolation
        color_index = int(i / 90) % len(colors)
        next_color_index = (color_index + 1) % len(colors)
        ratio = (i % 90) / 90.0
        r = int(
            colors[color_index][0] * (1 - ratio) + colors[next_color_index][0] * ratio
        )
        g = int(
            colors[color_index][1] * (1 - ratio) + colors[next_color_index][1] * ratio
        )
        b = int(
            colors[color_index][2] * (1 - ratio) + colors[next_color_index][2] * ratio
        )
        color = (r, g, b, 255)

        cv2.circle(
            img, (x, y), 1, color, margin_thickness
        )  # draw a small circle at the calculated position

    return img


def masking_video(video1, video2):
    # Load the background image as a clip with infinite duration
    background_image_path = "../videos/background.jpg"

    video1_clip = video1.resize(width=800, height=800)

    final_video_duration = video1_clip.duration

    background_clip = ImageClip(
        background_image_path, duration=final_video_duration
    ).set_position("center")

    # Get the size of the video
    width, height = video1_clip.size

    # Calculate the crop coordinates
    x1 = (width - 800) // 2
    y1 = (height - 800) // 2
    x2 = x1 + 800
    y2 = y1 + 800

    # Crop the video to a square
    video1_clip = video1_clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Set the position of the cropped video
    video1_clip = video1_clip.set_position((140, 100))

    video2_clip = video2.subclip(
        0, final_video_duration
    )  # Use only the first 10 seconds of the video

    # Generate the mask image
    mask_image = circle_masks(video2_clip.get_frame(0), 57)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    masked_video = video2_clip.set_mask(mask_clip)

    # Set the position of the cropped video
    video2_clip_mask = masked_video.set_position((30, 1000))

    print("video2_clip_mask.size", video2_clip_mask.size)

    # Apply a mask (circular for video2, square for video1 is by default)
    video1_clip_mask = (
        video1_clip.add_mask()
        .set_duration(final_video_duration)
        .resize(width=800, height=800)
        .margin(10)
    )

    # Create a single image with the circle
    img = circle_mask_with_margin(video2_clip, 58)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video2_clip.duration).set_position(
        (30, 1000)
    )

    # Combine everything
    final_clip = CompositeVideoClip(
        [background_clip, video1_clip_mask, circle_clip, video2_clip_mask]
    )

    # # Write the result to a file
    # output_path = "../data/rapid_video2.mp4"
    # final_clip.write_videofile(output_path, codec="libx264", fps=24)
    return final_clip


def crop_to_circle():
    # Load the video and background image
    video = VideoFileClip("../temp/7.mp4")
    background = ImageClip("../videos/background.jpg")

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 50)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    masked_video = video.set_mask(mask_clip)

    # Composite the masked video over the background image
    final_video = CompositeVideoClip([background, masked_video])

    final_video.duration = video.duration

    # Write the output video
    final_video.write_videofile("../data/background.mp4")


def crop_video_to_rounded_square(video_path, output_path, square_size, radius):
    """
    Crops a video into a rounded square shape using MoviePy.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the cropped video.
        square_size (int): Width and height of the square (in pixels).
        radius (int): Radius of the rounded corners (in pixels).

    Returns:
        None
    """
    # Load the video file
    video = VideoFileClip(video_path)

    # Get the video dimensions
    video_width, video_height = video.size

    # Calculate the coordinates for cropping
    x1 = (video_width - square_size) // 2
    y1 = (video_height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Crop the video to a square shape
    cropped_video = video.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Create a mask with rounded corners
    rounded_video = mask_color(cropped_video, color=[0, 0, 0], thr=1)

    # # Apply the mask to the cropped video
    # rounded_video = cropped_video.set_mask(mask)

    # Write the rounded video to a file
    rounded_video.write_videofile(output_path)

    # Close the video clips
    video.close()
    cropped_video.close()
    rounded_video.close()


def create_crossfade_transition(
    video1_path, video2_path, output_path, transition_duration=1
):
    """
    Creates a crossfade transition between two video clips and saves the resulting video.

    Args:
        video1_path (str): Path to the first video file.
        video2_path (str): Path to the second video file.
        output_path (str): Path to save the output video file.
        transition_duration (float): Duration of the crossfade transition in seconds (default: 1 second).

    Returns:
        None
    """
    # Load the video clips
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    # Ensure the videos have the same dimensions
    video2 = video2.resize(video1.size)

    # Create the crossfade transition clip
    transition_clip = CompositeVideoClip([video1, video2])

    transition_clip = transition_clip.crossfadein(transition_duration)

    # Combine the video clips with the transition
    final_clip = CompositeVideoClip([video1, transition_clip])

    # Write the final video to a file
    final_clip.write_videofile(output_path)

    # Close the video clips
    video1.close()
    video2.close()
    final_clip.close()


def add_clips_between_videos(main_video, clip_paths):
    """
    Adds two video clips between two segments of a main video.

    Args:
        main_video_path (str): Path to the main video file.
        clip1_path (str): Path to the first video clip to be inserted.
        clip2_path (str): Path to the second video clip to be inserted.
        insert_time (float): Time (in seconds) to insert the clips in the main video.
        output_path (str): Path to save the output video file.

    Returns:
        None
    """

    # Get the video dimensions
    width, height = main_video.size

    # Desired aspect ratio for reels/shorts
    target_aspect_ratio = 9 / 16

    # Calculate the aspect ratio of the input video
    aspect_ratio = width / height

    if aspect_ratio > target_aspect_ratio:
        # Video is wider than the target aspect ratio
        new_width = int(height * target_aspect_ratio)
        x1 = (width - new_width) // 2
        x2 = x1 + new_width
        y1 = 0
        y2 = height
    else:
        # Video is taller than the target aspect ratio
        new_height = int(width / target_aspect_ratio)
        x1 = 0
        x2 = width
        y1 = (height - new_height) // 2
        y2 = y1 + new_height

    # Crop the video to the target aspect ratio
    main_video = crop(main_video, x1=x1, y1=y1, x2=x2, y2=y2)

    video_clips = []
    start_time = 0
    end_time = 5

    for index, clip_path in enumerate(clip_paths):

        clip = VideoFileClip(clip_path)
        clip = clip.without_audio()
        clip = clip.subclip(2, 4)
        clip = clip.set_fps(30)

        # Get the video dimensions
        width, height = clip.size

        # Calculate the aspect ratio of the input video
        aspect_ratio = width / height

        if aspect_ratio > target_aspect_ratio:
            # Video is wider than the target aspect ratio
            new_width = int(height * target_aspect_ratio)
            x1 = (width - new_width) // 2
            x2 = x1 + new_width
            y1 = 0
            y2 = height
        else:
            # Video is taller than the target aspect ratio
            new_height = int(width / target_aspect_ratio)
            x1 = 0
            x2 = width
            y1 = (height - new_height) // 2
            y2 = y1 + new_height

        # Crop the video to the target aspect ratio
        clip = crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)

        clip = clip.resize(main_video.size)

        # clip.write_videofile(f"../data/{start_time}.mp4", threads=2)

        if main_video.duration < end_time:
            segment = main_video.subclip(start_time, main_video.duration)
        else:
            # Extract the segments from the main video
            segment = main_video.subclip(start_time, end_time)

        # Calculate the start and end times based on the clip durations
        start_time = end_time + clip.duration
        end_time = start_time + 5

        video_clips.append(segment)

        video_clips.append(clip)

        clip.close()

        if main_video.duration < start_time:
            break

    if main_video.duration > end_time:
        segment = main_video.subclip(start_time, main_video.duration)
        video_clips.append(segment)

    # final_clip = video_clips[0]
    # for index, video_clip in enumerate(video_clips):
    #     if index == 0:
    #         pass
    #     else:
    #         if index % 2 == 0:
    #             final_clip = fade(final_clip, video_clip, 0.3)
    #         else:
    #             final_clip = add_burn_transition(final_clip, video_clip)

    # Concatenate the video segments and clips
    final_clip = concatenate_videoclips(video_clips)

    # Add the audio
    audio = AudioFileClip("../temp/audio.mp3")

    final_clip = final_clip.set_audio(audio).resize(newsize=(1080, 1920))

    final_clip = final_clip.set_fps(30)

    return final_clip


def auto_detect_and_extract_object(video_path, output_path, confidence_threshold=0.5):
    """
    Automatically detects an object from a video and extracts it as a separate video clip.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output video file with the extracted object.
        confidence_threshold (float, optional): Confidence threshold for object detection (between 0 and 1). Default is 0.5.

    Returns:
        None
    """

    # Get the current working directory
    cwd = os.getcwd()

    # Load the pre-trained object detection model
    net = cv2.dnn.readNetFromCaffe(
        os.path.join(cwd, "MobileNetSSD_deploy.prototxt.txt"),
        os.path.join(cwd, "MobileNetSSD_deploy.caffemodel"),
    )
    # # Load the pre-trained object detection model
    # net = cv2.dnn.readNetFromCaffe(
    #     "path/to/MobileNetSSD_deploy.prototxt.txt",
    #     "path/to/MobileNetSSD_deploy.caffemodel",
    # )

    # Load the input video
    clip = VideoFileClip(video_path)

    # Get the video dimensions
    width, height = clip.size

    # Create a list to store the frames with the detected object
    object_frames = []

    # Process each frame of the video
    for frame in clip.iter_frames():
        # Convert the frame to a numpy array
        frame = np.array(frame)

        # Get the frame dimensions
        (h, w) = frame.shape[:2]

        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )

        # Pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > confidence_threshold:
                # Extract the index of the class label from the detections
                idx = int(detections[0, 0, i, 1])

                # Compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Create a new frame with only the detected object
                object_frame = np.zeros((h, w, 3), dtype=np.uint8)
                object_frame[startY:endY, startX:endX] = frame[startY:endY, startX:endX]

                # Append the object frame to the list
                object_frames.append(object_frame)

    # Create a new video clip from the extracted object frames
    object_clip = VideoFileClip(object_frames, fps=clip.fps)

    # Write the object clip to a new video file
    object_clip.write_videofile(output_path)

    # Close the clips
    clip.close()
    object_clip.close()


def trim_object_and_add_background(video_path, image_path, output_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load the background image
    background_image = cv2.imread(image_path)

    # Create a mask to detect the object
    lower_bound = np.array([0, 0, 0])  # adjust these values to detect the object
    upper_bound = np.array([255, 255, 255])  # adjust these values to detect the object
    kernel = np.ones((5, 5), np.uint8)

    # Create a list to store the trimmed frames
    trimmed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV and apply thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find the contour of the object
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]  # assume only one object
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # adjust these values to filter out small objects
            trimmed_frame = frame[y : y + h, x : x + w]
            trimmed_frames.append(trimmed_frame)

    # Release the video capture
    cap.release()

    # Create a new video with the trimmed frames and background image
    trimmed_video = []
    for frame in trimmed_frames:
        frame = cv2.resize(frame, (width, height))
        background_image_resized = cv2.resize(background_image, (width, height))
        frame_with_background = cv2.addWeighted(
            frame, 1, background_image_resized, 1, 0
        )
        trimmed_video.append(frame_with_background)

    # Convert the trimmed frames to a video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in trimmed_video:
        out.write(frame)

    out.release()

    # Create a moviepy clip from the output video
    video_clip = VideoFileClip(output_path)

    # Create a moviepy clip from the background image
    background_clip = ImageClip(background_image)

    # Composite the video clip over the background clip
    final_clip = CompositeVideoClip(
        [background_clip, video_clip.set_position("center")]
    )

    # Write the final clip to a new video file
    final_clip.write_videofile(output_path, fps=fps)


def remove_background_and_add_new_background(
    video_path, background_video_path, output_path
):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load the background video
    background_cap = cv2.VideoCapture(background_video_path)
    background_fps = background_cap.get(cv2.CAP_PROP_FPS)
    background_width = int(background_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    background_height = int(background_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a list to store the frames with removed background
    frames_with_removed_background = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSV and apply thresholding
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, 255, 255]))

        # Remove the background from the frame
        frame_with_removed_background = cv2.bitwise_and(frame, frame, mask=mask)

        # Add the new background to the frame
        background_ret, background_frame = background_cap.read()
        if not background_ret:
            break
        background_frame = cv2.resize(background_frame, (width, height))
        frame_with_new_background = cv2.addWeighted(
            frame_with_removed_background, 1, background_frame, 1, 0
        )

        frames_with_removed_background.append(frame_with_new_background)

    # Release the video captures
    cap.release()
    background_cap.release()

    # Create a new video with the frames with removed background
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames_with_removed_background:
        out.write(frame)

    out.release()

    # Create a moviepy clip from the output video
    video_clip = VideoFileClip(output_path)

    # Return the moviepy clip
    return video_clip


# # Example usage
# video_path = "../data/test.mp4"
# output_path = "../data/test1.mp4"
# background_video_path = "../data/test2.mp4"
# clip = remove_background_and_add_new_background(
#     video_path, background_video_path, output_path
# )
# clip.write_videofile(output_path)


def clip_main_object(video_path, output_path, yolo_weights, yolo_config, yolo_classes):
    """
    Clip the main object from a video using YOLO object detection.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output video file.
        yolo_weights (str): Path to the YOLO weights file.
        yolo_config (str): Path to the YOLO configuration file.
        yolo_classes (str): Path to the YOLO classes file.

    Returns:
        None
    """
    # Load YOLO
    net = cv2.dnn.readNet(yolo_weights, yolo_config)
    classes = []
    with open(yolo_classes, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Object detection
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)

        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Create a mask for the main object
        mask = np.zeros((height, width), dtype=np.uint8)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # Clip the main object from the original frame
        clipped_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Write the clipped frame to the output video
        out.write(clipped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def get_new_final_video(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    clip_paths,
):
    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    video_duration = main_video.duration

    # Required duration of each clip
    req_dur = video_duration / len(clip_paths)

    # Get the size of the video
    width, height = main_video.size

    # Calculate the crop coordinates
    x1 = (width - 500) // 2
    y1 = (height - 500) // 2
    x2 = x1 + 500
    y2 = y1 + 500

    # # Crop the video to a square
    # main_video = main_video.crop(x1=x1, y1=y1, x2=x2, y2=y2).resize((1080, 1080))

    # main_video.write_videofile("../data/template_4.mp4", threads=threads)

    # return

    video_clips = []
    tot_dur = 0

    # Desired aspect ratio for reels/shorts
    target_aspect_ratio = 9 / 16

    for i, clip_path in enumerate(clip_paths):
        clip = VideoFileClip(clip_path)
        clip = clip.without_audio()

        # Check if clip is longer than the remaining audio
        if (video_duration - tot_dur) < clip.duration:
            clip = clip.subclip(2, 2 + (video_duration - tot_dur))
        # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
        elif req_dur < clip.duration:
            clip = clip.subclip(2, 2 + req_dur)

        clip = clip.set_fps(30)

        # Get the video dimensions
        width, height = clip.size

        # Calculate the aspect ratio of the input video
        aspect_ratio = width / height

        if aspect_ratio > target_aspect_ratio:
            # Video is wider than the target aspect ratio
            new_width = int(height * target_aspect_ratio)
            x1 = (width - new_width) // 2
            x2 = x1 + new_width
            y1 = 0
            y2 = height
        else:
            # Video is taller than the target aspect ratio
            new_height = int(width / target_aspect_ratio)
            x1 = 0
            x2 = width
            y1 = (height - new_height) // 2
            y2 = y1 + new_height

        # Crop the video to the target aspect ratio
        clip = crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)

        clip = clip.resize((1080, 1920))

        tot_dur += clip.duration

        # If not the last video, add a transition
        if i < len(clip_paths) - 1 and i >= 1:
            # Get the next video
            current_clip = video_clips[len(video_clips) - 1]
            # Perform the transition
            if i % 2 == 0:
                transition = cross_dissolve(current_clip, clip, 1)
            else:
                transition = slide(current_clip, clip, 1)
            # Add the transition to the list
            video_clips.append(transition)

        video_clips.append(clip)

        clip.close()

    # Concatenate the video segments and clips
    final_clip = concatenate_videoclips(video_clips)

    final_clip = final_clip.set_duration(main_video.duration)

    # 3. Define the Scaling Function for Text Resizing
    def resize(t):
        if t == 5:
            return 20
        # Define starting and ending scale factors
        # Compute the scaling factor linearly over the clip's duration
        scale_factor = 80 - ((30 + t * 5) / 30)
        if scale_factor < 20:
            return 20
        return scale_factor

    # 4. Define the Positioning Function to Center the Text
    def translate(t):
        max_time = 5
        if t > 5:
            return (30, 1100)
        y = 600 + (500 * (t / 5))
        return (30, y)

    # Generate the mask image
    mask_image = circle_mask_with_margin(main_video.get_frame(0), 50)

    # # Create a new video with the white outer line
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # fps = main_video.fps
    # size = main_video.size
    # out = cv2.VideoWriter("output.avi", fourcc, fps, size)

    # for t in range(int(main_video.fps * main_video.duration)):
    #     frame = main_video.get_frame(t / main_video.fps)
    #     mask = circle_mask_with_margin(frame, 50)
    #     frame[mask == 255] = (
    #         255,
    #         255,
    #         255,
    #     )  # set the pixels where the mask is white to white
    #     out.write(frame)

    # out.release()

    # Apply the circle mask to the new video
    mask_clip = ImageClip(mask_image, ismask=True)
    main_video = main_video.set_mask(mask_clip)

    # # Generate the mask image
    # mask_image = circle_mask_with_margin(main_video.get_frame(0), 50)

    # # Apply the circle mask to the video
    # mask_clip = ImageClip(mask_image, ismask=True)

    # main_video = main_video.set_mask(mask_clip)

    # Set the position of the cropped video
    main_video = main_video.set_position(translate)

    # Add the audio
    audio = AudioFileClip("../temp/audio.mp3")

    final_clip = final_clip.set_audio(audio).resize(newsize=(1080, 1920))

    # Combine everything
    final_clip = CompositeVideoClip([final_clip, main_video])

    # Define a generator for creating TextClip objects
    generator = lambda txt: TextClip(
        txt=txt.upper(),
        font="../fonts/poppins-black.ttf",
        fontsize=100,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Extract the subtitles position
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Create a SubtitlesClip object
    subtitles = SubtitlesClip(subtitles_path, generator)

    # Composite the main video with the subtitles
    result = CompositeVideoClip(
        [
            final_clip,
            subtitles.set_position(
                (horizontal_subtitles_position, vertical_subtitles_position),
                relative=True,
            ),
        ]
    )

    result = result.set_fps(30)

    result.write_videofile("../data/template6.mp4", threads=threads)

    return "../data/template4.mp4"


def get_new_final_videos(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    clip_paths,
):
    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    video_duration = main_video.duration

    # Required duration of each clip
    req_dur = video_duration / len(clip_paths)

    # Get the size of the video
    width, height = main_video.size

    # Calculate the crop coordinates
    x1 = (width - 500) // 2
    y1 = (height - 500) // 2
    x2 = x1 + 500
    y2 = y1 + 500

    # Crop the video to a square
    main_video = main_video.crop(x1=x1, y1=y1, x2=x2, y2=y2).resize((1080, 1080))

    # main_video.write_videofile("../data/template_4.mp4", threads=threads)

    video_clips = []
    tot_dur = 0

    # Desired aspect ratio for reels/shorts
    target_aspect_ratio = 9 / 16

    for clip_path in clip_paths:
        clip = VideoFileClip(clip_path)
        clip = clip.without_audio()

        # Check if clip is longer than the remaining audio
        if (video_duration - tot_dur) < clip.duration:
            clip = clip.subclip(2, 2 + (video_duration - tot_dur))
        # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
        elif req_dur < clip.duration:
            clip = clip.subclip(2, 2 + req_dur)

        clip = clip.set_fps(30)

        # Get the video dimensions
        width, height = clip.size

        # Calculate the aspect ratio of the input video
        aspect_ratio = width / height

        if aspect_ratio > target_aspect_ratio:
            # Video is wider than the target aspect ratio
            new_width = int(height * target_aspect_ratio)
            x1 = (width - new_width) // 2
            x2 = x1 + new_width
            y1 = 0
            y2 = height
        else:
            # Video is taller than the target aspect ratio
            new_height = int(width / target_aspect_ratio)
            x1 = 0
            x2 = width
            y1 = (height - new_height) // 2
            y2 = y1 + new_height

        # Crop the video to the target aspect ratio
        clip = crop(clip, x1=x1, y1=y1, x2=x2, y2=y2)

        clip = clip.resize((1080, 1920))

        video_clips.append(clip)

        tot_dur += clip.duration

        clip.close()

    # Concatenate the video segments and clips
    final_clip = concatenate_videoclips(video_clips)

    # Combine everything
    final_clip = create_mask_and_clip_person_from_video(main_video, final_clip)

    # Add the audio
    audio = AudioFileClip("../temp/audio.mp3")

    final_clip = final_clip.set_audio(audio).resize(newsize=(1080, 1920))

    # Define a generator for creating TextClip objects
    generator = lambda txt: TextClip(
        txt=txt.upper(),
        font="../fonts/poppins-black.ttf",
        fontsize=100,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Extract the subtitles position
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Create a SubtitlesClip object
    subtitles = SubtitlesClip(subtitles_path, generator)

    # Composite the main video with the subtitles
    result = CompositeVideoClip(
        [
            final_clip,
            subtitles.set_position(
                (horizontal_subtitles_position, vertical_subtitles_position),
                relative=True,
            ),
        ]
    )

    result = result.set_fps(30)

    result.write_videofile("../data/template5.mp4", threads=threads)

    return "../data/template5.mp4"


def create_mask_and_clip_person_from_video(video_clip, background_video_clip):
    """
    Create a mask of a person object in a video using MediaPipe's Solutions API,
    and then clip the person object from the video using MoviePy.

    Args:
        video_clip (moviepy.editor.VideoFileClip): The input video clip.
        background_video_clip (moviepy.editor.VideoFileClip): The background video clip.
        output_path (str): Path to the output video file.

    Returns:
        None
    """
    # Create a MediaPipe Solutions object
    mp_solutions = mp.solutions

    # Create a person detector object
    person_detector = mp_solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    )

    # Get the video properties
    fps = background_video_clip.fps
    width, height = background_video_clip.size

    # Create a list to store the masked frames
    masked_frames = []

    # Iterate over the frames of the input video clip and background video clip
    for frame, background_frame in zip(
        video_clip.iter_frames(), background_video_clip.iter_frames()
    ):
        # Resize the background frame to match the size of the original frame
        # background_frame = cv2.resize(background_frame, (width, height))

        frame = cv2.resize(frame, (width, height))

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect the person in the frame
        results = person_detector.process(rgb_frame)

        # Get the segmentation mask
        mask = results.segmentation_mask

        # Check if the mask is valid
        if mask is None:
            print("Error: Mask is None")
            continue

        # Convert the mask to a binary mask
        _, thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # Check if the thresholded image is valid
        if thresh is None:
            print("Error: Thresholded image is None")
            continue

        # Apply the mask to the original frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=thresh.astype(np.uint8))

        # Remove the background video from the detected object area
        background_frame = cv2.bitwise_and(
            background_frame,
            background_frame,
            mask=cv2.bitwise_not(thresh.astype(np.uint8)),
        )

        # Add the background frame to the masked frame
        masked_frame = cv2.addWeighted(masked_frame, 1, background_frame, 1, 0)

        # Add the masked frame to the list
        masked_frames.append(masked_frame)

    # Create a MoviePy clip from the masked frames
    clip = ImageSequenceClip(masked_frames, fps=fps)

    return clip


def create_video_1(main_video: str, time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(0, time)

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 47)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((290, 180))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 48)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((290, 180))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1280, 720)),
            new_video,
        ]
    )

    intro_video_path = "../videos/intro.mp4"

    intro_clip = VideoFileClip(intro_video_path)

    intro_audio_path = "../videos/intro_audio.mp3"

    audio_clip = AudioFileClip(intro_audio_path)

    # Set the start time of the audio to 2 seconds before the end of the first video
    audio_start_time = time - 2

    # Composite the original audio of the first video with the new audio
    composite_audio = CompositeAudioClip(
        [final_clip.audio, audio_clip.subclip(0, 6).set_start(audio_start_time)]
    )

    final_clip = fade(final_clip, intro_clip, 0.2)

    final_clip.audio = composite_audio

    return final_clip


def video_1():

    # logo_path = "../videos/logo1.png"

    # # Load the logo
    # logo = ImageClip(logo_path, duration=time)

    # logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # # Set the logo position to top right corner
    # logo_pos = (1920 - logo.w - 60, 60)

    # # Combine everything
    # final_clip = CompositeVideoClip(
    #     [
    #         background_clip,
    #         logo.set_position(logo_pos),
    #         circle_clip.resize((1280, 720)),
    #         new_video,
    #     ]
    # )

    intro_video_path = "../videos/intro.mp4"

    intro_clip = VideoFileClip(intro_video_path)

    intro_audio_path = "../videos/intro_audio.mp3"

    audio_clip = AudioFileClip(intro_audio_path)

    # Set the start time of the audio to 2 seconds before the end of the first video
    audio_start_time = 2

    # Composite the original audio of the first video with the new audio
    composite_audio = CompositeAudioClip([audio_clip.subclip(1, audio_clip.duration)])

    # final_clip = fade(final_clip, intro_clip, 0.2)

    intro_clip.audio = composite_audio

    return intro_clip


def create_video_2(main_video, title, stock_video, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 41)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((290, 280))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    # Create a text clip with the title text
    title_clip = TextClip(
        title,
        fontsize=200,
        color="white",
        font="../fonts/poppins-black.ttf",
    )

    # 4. Define the Positioning Function to Center the Text
    def translate(t):
        if t > 3:
            return (450, 160)
        y = 0 + (160 * (t / 3))
        return (450, y)

    # Animate the title clip with a fade-in effect
    title_clip = (
        title_clip.set_position(translate).set_duration(time).fadein(2).fadeout(2)
    )

    # Resize the title clip to 50% of the video width
    title_clip = title_clip.resize(width=background_clip.w // 2)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 42)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((290, 280))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1280, 720)),
            new_video,
            title_clip,
        ]
    )

    # # Add the audio
    # audio = AudioFileClip("../temp/audio.mp3").subclip(end_time, end_time + 5)

    # stock_subclip = (
    #     stock_video.subclip(end_time, end_time + 5)
    #     .resize((1920, 1080))
    #     .set_audio(audio)
    # )

    # final_clip = add_burn_transition(final_clip, stock_subclip)

    return final_clip


def combine_long_stock_video(clip_paths, video_duration):

    video_clips = []
    tot_dur = 0

    req_dur = video_duration / len(clip_paths)

    # Desired aspect ratio for reels/shorts
    target_aspect_ratio = 9 / 16

    for i, clip_path in enumerate(clip_paths):
        clip = VideoFileClip(clip_path)
        clip = clip.without_audio()

        # Check if clip is longer than the remaining audio
        if (video_duration - tot_dur) < clip.duration:
            clip = clip.subclip(2, 2 + (video_duration - tot_dur))
        # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
        elif req_dur < clip.duration:
            clip = clip.subclip(2, 2 + req_dur)

        clip = clip.set_fps(30)

        width = 1280
        height = 720

        # Get the video dimensions
        video_width, video_height = clip.size

        # Calculate the crop dimensions
        if width > video_width or height > video_height:
            print("Error: Desired dimensions are larger than the input video.")
            return

        x = (video_width - width) // 2
        y = (video_height - height) // 2

        # Crop the video
        clip = clip.crop(x1=x, y1=y, x2=x + width, y2=y + height)

        clip = clip.resize((1280, 720))

        tot_dur += clip.duration

        # If not the last video, add a transition
        if i < len(clip_paths) - 1 and i >= 1:
            # Get the next video
            current_clip = video_clips[len(video_clips) - 1]
            # Perform the transition
            # if i % 2 == 0:
            #     transition = cross_dissolve(current_clip, clip, 1.5)
            # else:
            # transition = fade(current_clip, clip, 1)

            # # Add the transition to the list
            # video_clips.append(transition)

        video_clips.append(clip)

        clip.close()

    # Concatenate the video segments and clips
    final_clip = concatenate_videoclips(video_clips)

    final_clip = final_clip.set_duration(video_duration)

    return final_clip


def create_video_3(main_video, stock_video, subtitles_path, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 56)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    video = video.set_position((800, 0))

    logo_path = "../videos/logo.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 4, height=logo.h // 4)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w, 0)

    # Define a generator for creating TextClip objects
    generator = lambda txt: TextClip(
        txt=txt.upper(),
        font="../fonts/Satoshi-Regular.otf",
        fontsize=100,
        color="#FFFFFF",
        stroke_color="White",
        stroke_width=2,
    )

    # Create a SubtitlesClip object
    subtitles = (
        SubtitlesClip(subtitles_path, generator)
        .subclip(start_time, end_time)
        .set_position((100, 400))
    )

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 57)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((800, 0))

    # Combine everything
    final_clip = CompositeVideoClip(
        [background_clip, logo.set_position(logo_pos), circle_clip, video, subtitles]
    )

    # stock_subclip = stock_video.subclip(end_time, end_time + 5).resize((1920, 1080))
    # final_clip = fade(final_clip, stock_subclip, 1)

    return final_clip


def curve_edge(clip, curve_amount):
    """
    Curve the edge of a video.

    Args:
        video_path (str): Path to the input video file.
        curve_amount (float): Amount of curvature (0.0 to 1.0).

    Returns:
        moviepy.editor.VideoClip: The curved video clip.
    """

    # Define the curve function
    def curve(x, y, amount, w, h):
        return x + amount * (x - w / 2) ** 2 / (w**2), y

    # Apply the curve transformation to each frame
    def curve_frame(t):
        frame = clip.get_frame(t)
        curved_frame = np.zeros_like(frame)
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                x_curved, y_curved = curve(x, y, curve_amount, clip.w, clip.h)
                x_curved = int(x_curved)
                y_curved = int(y_curved)
                if 0 <= x_curved < frame.shape[1] and 0 <= y_curved < frame.shape[0]:
                    curved_frame[y_curved, x_curved] = frame[y, x]
        return curved_frame

    # Create a new clip with the curved frames
    curved_clip = VideoClip(lambda t: curve_frame(t), duration=clip.duration)

    return curved_clip


def create_video_4(main_video, stock_video, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 45)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # # Add a margin around the circular region
    # video = video.fx(margin, 8, (165, 42, 42))

    # Set the position of the cropped video
    video = video.set_position((1050, 280))

    logo_path = "../videos/logo.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 4, height=logo.h // 4)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w, 0)

    stock_subclip = stock_video.subclip(start_time, end_time).resize((1450, 850))

    # stock_subclip = curve_edge(stock_subclip, 0.5)

    # Set the position of the cropped video
    stock_subclip = stock_subclip.set_position((80, 100)).fx(
        vfx.margin, 12, (165, 42, 42)
    )

    # Apply a mask (circular for video2, square for video1 is by default)
    stock_subclip_mask = stock_subclip.add_mask()

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 46)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((1050, 280))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            stock_subclip_mask,
            circle_clip,
            video,
        ]
    )

    return final_clip


def create_video_5(main_video, stock_video, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 40)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    video = video.set_position((1070, 300))

    stock_subclip = stock_video.subclip(start_time, end_time).resize((1920, 1080))

    logo_path = "../videos/logo.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 4, height=logo.h // 4)

    # Set the logo position to top right corner
    logo_pos = (stock_subclip.w - logo.w, 0)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 41)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((1070, 300))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            stock_subclip,
            logo.set_position(logo_pos),
            circle_clip,
            video,
        ]
    )

    return final_clip


def create_video_6(main_video: str, start_time):

    print("main_video size", main_video.size)

    time = main_video.duration - start_time

    video = main_video.subclip(start_time, main_video.duration)

    # video = video.crop(x1=0, y1=280, x2=960, y2=800).resize((1920, 1080))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (video.w - logo.w - 60, 60)

    # Combine everything
    final_clip = CompositeVideoClip([video, logo.set_position(logo_pos)])

    outro_audio_path = "../videos/intro_audio.mp3"

    audio_clip = AudioFileClip(outro_audio_path)

    outro_video_path = "../videos/outro.mp4"

    outro_clip = VideoFileClip(outro_video_path)

    outro_clip.audio = audio_clip.subclip(0, audio_clip.duration)

    final_clip = fade(final_clip, outro_clip, 0.3)

    return final_clip


def create_long_video(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    clip_paths,
):
    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    video1 = create_video_1(main_video, 5.2)

    stock_video = combine_long_stock_video(clip_paths, main_video.duration)

    video2 = create_video_2(main_video, stock_video, 5.1, 22)

    final_clip = add_burn_transition(video1, video2)

    video3 = create_video_3(main_video, stock_video, subtitles_path, 27, 42)

    final_clip = add_burn_transition(final_clip, video3)

    video4 = create_video_4(main_video, stock_video, 42, 62)

    final_clip = add_burn_transition(final_clip, video4)

    video3 = create_video_3(main_video, stock_video, subtitles_path, 62, 75)

    final_clip = add_burn_transition(final_clip, video3)

    video4 = create_video_4(main_video, stock_video, 75, 90)

    final_clip = add_zoom_transition(final_clip, video4)

    video5 = create_video_5(main_video, stock_video, 90, 110)

    final_clip = add_effect_transition(final_clip, video5)

    video6 = create_video_6(main_video, 110)

    final_clip = add_burn_transition(final_clip, video6)

    final_clip.write_videofile("../data/output1.mp4", threads=5)

    return "../data/output1.mp4"


def create_video_3_copy(main_video, clip, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 40)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    new_video = video.resize((1180, 664)).set_position((900, 180))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 41)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((900, 180))

    # Generate the mask image
    mask_image = create_rounded_rectangle_mask(clip.get_frame(0), clip.size, 60)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    # Apply a mask (circular for video2, square for video1 is by default)
    clip = clip.set_mask(mask_clip)

    clip = clip.set_position((100, 180))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1180, 664)),
            new_video,
            clip,
        ]
    )

    return final_clip


def create_video_4_copy(main_video, stock_video, start_time, end_time, end_clip):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 36)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((900, 506)).set_position((1150, 560))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    stock_subclip = stock_video.subclip(0, end_clip).resize((1450, 850))

    # Generate the mask image
    mask_image = create_rounded_rectangle_mask(
        stock_subclip.get_frame(0), stock_subclip.size, 65
    )

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    # Apply a mask (circular for video2, square for video1 is by default)
    stock_subclip_mask = stock_subclip.set_mask(mask_clip)

    # Set the position of the cropped video
    stock_subclip = stock_subclip_mask.set_position((80, 100))

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 36)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((1150, 560))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            stock_subclip,
            circle_clip.resize((900, 506)),
            new_video,
        ]
    )

    return final_clip


def video_4_copy(stock_video):

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(
        background_image_path, duration=stock_video.duration
    ).set_position("center")

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=stock_video.duration)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    stock_subclip = stock_video.resize((1450, 850))

    # Generate the mask image
    mask_image = create_rounded_rectangle_mask(
        stock_subclip.get_frame(0), stock_subclip.size, 65
    )

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    # Apply a mask (circular for video2, square for video1 is by default)
    stock_subclip_mask = stock_subclip.set_mask(mask_clip)

    # Set the position of the cropped video
    stock_subclip = stock_subclip_mask.set_position((235, 115))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            stock_subclip,
        ]
    )

    return final_clip


def create_video_5_copy(main_video, stock_video, start_time, end_time, end_clip):

    video_duration = main_video.duration

    width, height = main_video.size

    video = main_video.subclip(start_time, end_time)

    time = end_time - start_time

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 35)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # # Add a margin around the circular region
    # video = video.fx(margin, 8, (165, 42, 42))

    # Set the position of the cropped video
    new_video = video.resize((900, 506)).set_position((1200, 580))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    stock_subclip = stock_video.subclip(0, end_clip).resize((1920, 1080))

    # Set the logo position to top right corner
    logo_pos = (stock_subclip.w - logo.w - 60, 60)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 35)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((1200, 580))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            stock_subclip,
            logo.set_position(logo_pos),
            circle_clip.resize((900, 506)),
            new_video,
        ]
    )

    return final_clip


def video_5_copy(stock_video):

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=stock_video.duration)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    stock_subclip = stock_video.resize((1920, 1080))

    # Set the logo position to top right corner
    logo_pos = (stock_subclip.w - logo.w - 60, 60)

    final_clip = CompositeVideoClip(
        [
            stock_subclip,
            logo.set_position(logo_pos),
        ]
    )

    return final_clip


def create_video_6_copy(main_video: str, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    time = end_time - start_time

    video = main_video.subclip(start_time, end_time)

    background_image_path = "../videos/bg.png"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 47)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((290, 180))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 75, height=logo.h // 75)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 60, 60)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 48)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((290, 180))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1280, 720)),
            new_video,
        ]
    )

    return final_clip


def test_long_video(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    title: str,
):

    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    final_clip = create_video_1(main_video, 7.5)

    end_clip_time = 0

    with open("./content/clips.json", "r") as file:
        video_data = json.load(file)

    for key, value in video_data.items():
        clips = json.loads(key)

    clips = sorted(clips, key=lambda x: x["atTime"])

    print("Clips", clips)

    with open("./content/images.json", "r") as file:
        image_data = json.load(file)

    for key, value in image_data.items():
        images = json.loads(key)

    images = sorted(images, key=lambda x: x["atTime"])

    print("images", images)

    for index, clip in enumerate(clips):

        clip_num = int(clip["clip"])
        at_time = int(clip["atTime"])
        # background_placement = bool(clip["background_placement"])

        if index < len(clips) - 1:
            next_at_time = int(clips[index + 1]["atTime"])
        else:
            next_at_time = main_video.duration - 6

        if index < len(images):
            image = int(images[index]["image"])
            image_at_time = int(images[index]["atTime"])

        if index == 0:
            clip = create_video_2(main_video, title, final_clip, 7.5, 12)
            final_clip = add_burn_transition(final_clip, clip)
            end_clip_time = 12

        if end_clip_time < at_time:

            print("end_clip_time", end_clip_time)

            if image_at_time < at_time and index % 2 == 0:
                image_path = f"../temp/image_{image}.jpg"
                clip = ImageClip(image_path, duration=(at_time - end_clip_time)).resize(
                    (1000, 650)
                )
                video3 = create_video_3_copy(main_video, clip, end_clip_time, at_time)
            else:
                video3 = create_video_6_copy(main_video, end_clip_time, at_time)

            final_clip = add_animation_transition(final_clip, video3)

            end_clip_time = at_time

        print("end_clip_time", end_clip_time)

        clip = VideoFileClip(f"./data/clip_{clip_num - 1}.mp4")

        end_time = 0

        # if background_placement:

        # Create a new audio clip with the lowered volume
        audio = clip.audio.volumex(0.02)

        clip = clip.set_audio(audio)

        start_time = at_time if at_time >= end_clip_time else end_clip_time

        if (start_time + clip.duration) > next_at_time:
            end_time = next_at_time
        else:
            end_time = start_time + clip.duration

        if index % 2 == 0:

            video4 = create_video_4_copy(
                main_video,
                clip,
                start_time,
                end_time,
                end_time - start_time,
            )
            final_clip = add_burn_transition(final_clip, video4)

        else:

            video5 = create_video_5_copy(
                main_video,
                clip,
                start_time,
                end_time,
                end_time - start_time,
            )
            final_clip = add_animation_transition(final_clip, video5)

        end_clip_time = end_time
        # else:
        #     # Concatenate the video segments and clips
        #     final_clip = add_burn_transition(final_clip, clip.resize((1920, 1080)))

        print(f"Clip {clip_num} at time {at_time} end_clip_time {end_clip_time}")
        # if index == 1:
        #     break

    if end_clip_time < main_video.duration:
        video6 = create_video_6(main_video, end_clip_time)
        final_clip = add_burn_transition(final_clip, video6)

    print("size", final_clip.size)

    print("fps", final_clip.fps)

    final_clip = final_clip.resize((1280, 720))

    final_clip = final_clip.set_fps(24)

    final_clip.write_videofile(
        "../data/output_test12.mp4",
        threads=5,
        codec="libx264",
        preset="fast",
    )

    return "../data/output_test12.mp4"


def test_long_video1(
    audio_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    title: str,
):

    # Load the main video and video clips
    main_audio = AudioFileClip(audio_path)

    with open("./content/images.json", "r") as file:
        image_data = json.load(file)

    for key, value in image_data.items():
        images = json.loads(key)

    with open("./content/clips.json", "r") as file:
        video_data = json.load(file)

    for key, value in video_data.items():
        clips = json.loads(key)

    with open("./content/search_terms.json", "r") as file:
        video_data = json.load(file)

    # Create a defaultdict to store the merged data, sorted by 'at_time'
    merged_data = defaultdict(list)

    # Add data1 items to the merged_data dictionary
    for item in images:
        merged_data[item["atTime"]].append({"image": item["image"]})

    # Add data1 items to the merged_data dictionary
    for item in clips:
        merged_data[item["atTime"]].append({"clip": item["clip"]})

    # Add data2 items to the merged_data dictionary
    for item in video_data["data"]:
        merged_data[item["at_time"]].append({"search_terms": item["search_terms"]})

    # Sort the keys (at_time values) in increasing order
    sorted_keys = sorted(merged_data.keys())

    merged_json = [
        {"at_time": k, **{key: value for d in v for key, value in d.items()}}
        for k, v in zip(sorted_keys, [merged_data[key] for key in sorted_keys])
    ]

    print("merged_json", merged_json)

    clip_num_stock = 0

    last_image_path = 0

    final_clip = video_1()

    endtime = final_clip.duration

    for index, data in enumerate(merged_json):

        at_time = int(data["at_time"])

        if index < len(merged_json) - 1:
            next_at_time = int(merged_json[index + 1]["at_time"])
            is_next_video_clip = "clip" in merged_json[index + 1]
        else:
            next_at_time = main_audio.duration

        if endtime < at_time and endtime != 0:

            print(f"hello 1,{at_time} ")

            clip = ImageClip(last_image_path, duration=(at_time - endtime))
            audio = main_audio.subclip(endtime, at_time)

            clip = clip.set_audio(audio)

            video4 = video_4_copy(
                clip,
            )

            final_clip = add_burn_transition(final_clip, video4)

            endtime = at_time

        if endtime <= at_time or endtime < next_at_time:

            if "image" in data:
                print(f"hello 2,{at_time} ")
                image = data["image"]
                image_path = f"../temp/image_{image}.jpg"
                duration = next_at_time - endtime
                audio = main_audio.subclip(endtime, next_at_time)
                clip = ImageClip(image_path, duration=duration)
                last_image_path = image_path
                endtime = next_at_time

            elif "clip" in data:
                print(f"hello 3,{at_time} ")
                clip_num = data["clip"]
                clip = VideoFileClip(f"./data/clip_{clip_num - 1}.mp4")

                end = (
                    next_at_time
                    if next_at_time < endtime + clip.duration and is_next_video_clip
                    else endtime + clip.duration
                )
                audio = main_audio.subclip(endtime, endtime + clip.duration)
                endtime = end

            else:
                print(f"hello 4,{at_time} ")
                clip = VideoFileClip(f"../temp/{clip_num_stock}.mp4")
                clip = clip.subclip(1, clip.duration)
                end = (
                    next_at_time
                    if next_at_time < endtime + clip.duration
                    else endtime + clip.duration
                )
                clip_num_stock = clip_num_stock + 1
                audio = main_audio.subclip(endtime, end)
                clip = clip.subclip(0, end - endtime)
                endtime = end

            clip = clip.set_audio(audio)

            if index % 2 == 0:

                video = video_4_copy(
                    clip,
                )

                final_clip = add_burn_transition(final_clip, video)

            # elif index % 3 == 1:
            #     video = video_4_copy(
            #         clip,
            #     )

            #     final_clip = cross_dissolve(final_clip, video)

            else:

                video = video_5_copy(
                    clip,
                )

                final_clip = add_animation_transition(final_clip, video)

        # if index == 15:
        #     break

    # if end_clip_time < main_video.duration:
    #     video6 = create_video_6(main_video, end_clip_time)
    #     final_clip = add_burn_transition(final_clip, video6)

    print("size", final_clip.size)

    print("fps", final_clip.fps)

    final_clip = final_clip.resize((1280, 720))

    final_clip = final_clip.set_fps(24)

    final_clip.write_videofile(
        "../data/output_test11.mp4",
        threads=5,
        codec="libx264",
        preset="fast",
    )

    return "../data/output_test11.mp4"


def create_reel_video_1(main_video, clip, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    time = end_time - start_time

    video = main_video.subclip(start_time, end_time)

    background_image_path = "../videos/background.jpg"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 45)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((-120, 1150))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 50, height=logo.h // 50)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 30, 40)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 46)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((-120, 1150))

    if clip:
        # # Apply a mask (circular for video2, square for video1 is by default)
        video_clip_mask = clip.add_mask()

        # Combine everything
        final_clip = CompositeVideoClip(
            [
                background_clip,
                logo.set_position(logo_pos),
                circle_clip.resize((1280, 720)),
                new_video,
                clip,
            ]
        )
    else:
        # Combine everything
        final_clip = CompositeVideoClip(
            [background_clip, logo.set_position(logo_pos), circle_clip, video]
        )

    return final_clip


def create_reel_video_2(main_video, clip, start_time, end_time):

    # Get the original video dimensions
    original_width, original_height = clip.size

    # Calculate the aspect ratio of the original video
    original_aspect_ratio = original_width / original_height

    # Define the desired aspect ratio for Shorts (9:16)
    shorts_aspect_ratio = 9 / 16

    # Determine if we need to crop horizontally or vertically
    if original_aspect_ratio > shorts_aspect_ratio:
        # Crop horizontally
        new_width = int(original_height * shorts_aspect_ratio)
        new_height = original_height
        x1 = (original_width - new_width) // 2
        x2 = x1 + new_width
        y1 = 0
        y2 = original_height
    else:
        # Crop vertically
        new_height = int(original_width / shorts_aspect_ratio)
        new_width = original_width
        x1 = 0
        x2 = original_width
        y1 = (original_height - new_height) // 2
        y2 = y1 + new_height

    # Crop the video
    clip = clip.crop(x1=x1, x2=x2, y1=y1, y2=y2)

    clip = clip.resize((1080, 1920))

    video_duration = main_video.duration

    width, height = main_video.size

    time = end_time - start_time

    video = main_video.subclip(start_time, end_time)

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 40)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((-120, 1200))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 50, height=logo.h // 50)

    # Set the logo position to top right corner
    logo_pos = (clip.w - logo.w - 30, 40)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 40)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((-120, 1200))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1280, 720)),
            new_video,
        ]
    )

    return final_clip


def create_reel_video_3(main_video, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    time = end_time - start_time

    video = main_video.subclip(start_time, end_time)

    background_image_path = "../videos/background.jpg"

    background_clip = ImageClip(background_image_path, duration=time).set_position(
        "center"
    )

    # Generate the mask image
    mask_image = circle_masks(video.get_frame(0), 46)

    # Apply the circle mask to the video
    mask_clip = ImageClip(mask_image, ismask=True)

    video = video.set_mask(mask_clip)

    # Set the position of the cropped video
    new_video = video.resize((1280, 720)).set_position((-120, 700))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 50, height=logo.h // 50)

    # Set the logo position to top right corner
    logo_pos = (background_clip.w - logo.w - 30, 40)

    # Create a single image with the circle
    img = circle_mask_with_margin(video, 46)

    # Create a clip from the image
    circle_clip = ImageClip(img)

    # Repeat the image over time
    circle_clip = circle_clip.set_duration(video.duration).set_position((-120, 700))

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            background_clip,
            logo.set_position(logo_pos),
            circle_clip.resize((1280, 720)),
            new_video,
        ]
    )

    return final_clip


def create_reel_video_4(main_video, stock_video, start_time, end_time):

    video_duration = main_video.duration

    width, height = main_video.size

    time = end_time - start_time

    video = main_video.subclip(start_time, end_time)

    video = video.crop(x1=480, y1=0, x2=1550, y2=960).resize(newsize=(1080, 960))

    stock_video = stock_video.resize((1606, 960))

    # Get the video dimensions
    video_width, video_height = stock_video.size

    # Calculate the coordinates for cropping
    x1 = (video_width - 1080) // 2
    y1 = (video_height - 960) // 2
    x2 = video_width - x1
    y2 = video_height - y1

    stock_video = stock_video.crop(x1=x1, y1=y1, x2=x2, y2=y2).resize(
        newsize=(1080, 960)
    )

    # Stack the resized video clips on top of each other
    final_clip = clips_array([[stock_video], [video]])

    final_clip = final_clip.resize(newsize=(1080, 1920))

    logo_path = "../videos/logo1.png"

    # Load the logo
    logo = ImageClip(logo_path, duration=time)

    logo = logo.resize(width=logo.w // 50, height=logo.h // 50)

    # Set the logo position to top right corner
    logo_pos = (final_clip.w - logo.w - 30, 40)

    # Combine everything
    final_clip = CompositeVideoClip(
        [
            final_clip,
            logo.set_position(logo_pos),
        ]
    )

    return final_clip


def test_reel_video(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    clip_paths,
):
    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    with open("./content/clips.json", "r") as file:
        video_data = json.load(file)

    for key, value in video_data.items():
        clips = json.loads(key)

    with open("./content/images.json", "r") as file:
        image_data = json.load(file)

    for key, value in image_data.items():
        images = json.loads(key)

    print(clips, images)

    video_clips = []

    end_clip_time = 0

    final_clip = ""

    for index, clip in enumerate(clips):
        clip_num = int(clip["clip"])
        at_time = int(clip["atTime"])

        if index < len(clips) - 1:
            next_at_time = int(clips[index + 1]["atTime"])
        else:
            next_at_time = main_video.duration

        if index < len(images):
            image = int(images[index]["image"])
            image_at_time = int(images[index]["atTime"])

        if end_clip_time == 0:
            end_time = at_time if at_time < image_at_time else 6
            # Define a generator for creating TextClip objects
            generator = lambda txt: TextClip(
                txt=txt.upper(),
                font="../fonts/Satoshi-Regular.otf",
                fontsize=100,
                color="#FFFFFF",
                stroke_color="White",
                stroke_width=2,
            )

            # Create a SubtitlesClip object
            subtitles = SubtitlesClip(subtitles_path, generator).subclip(0, end_time)

            final_clip = create_reel_video_1(
                main_video, subtitles.set_position((250, 350)), 0, end_time
            )
            end_clip_time = end_time

        if image_at_time < at_time and end_clip_time < at_time:

            image_path = f"../temp/image_{image}.jpg"

            clip = ImageClip(image_path, duration=(at_time - end_clip_time))

            clip = clip.resize(newsize=(900, 750))

            video = create_reel_video_1(
                main_video, clip.set_position((90, 200)), end_clip_time, at_time
            )

            final_clip = fade(final_clip, video)

            end_clip_time = at_time

        clip = VideoFileClip(f"./data/clip_{clip_num - 1}.mp4")

        # Create a new audio clip with the lowered volume
        audio = clip.audio.volumex(0.04)

        clip = clip.set_audio(audio)

        start_time = at_time if at_time >= end_clip_time else end_clip_time

        if (start_time + clip.duration) > next_at_time:
            end_time = next_at_time
        else:
            end_time = start_time + clip.duration

        time = end_time - start_time

        if index % 3 == 0:

            video = create_reel_video_4(
                main_video,
                clip.subclip(0, time),
                start_time,
                end_time,
            )
            final_clip = fade(final_clip, video)

        elif index % 3 == 1:

            clip = clip.crop(x1=150, y1=0, x2=1130, y2=720).resize(newsize=(900, 750))

            video = create_reel_video_1(
                main_video,
                clip.subclip(0, time).set_position((90, 200)),
                start_time,
                end_time,
            )
            final_clip = fade(final_clip, video)

        else:
            video = create_reel_video_2(
                main_video,
                clip.subclip(0, time).resize((1280, 720)),
                start_time,
                end_time,
            )

            final_clip = fade(final_clip, video)

        end_clip_time = end_time

    if end_clip_time < main_video.duration:

        video = create_reel_video_3(
            main_video,
            end_clip_time,
            main_video.duration,
        )

        final_clip = fade(final_clip, video)

    final_clip.write_videofile("../data/output_test9.mp4", threads=2)

    return "../data/output_test9.mp4"


def test_reel_video1(
    custom_video_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    templates: str,
    clip_paths,
):
    # Load the main video and video clips
    main_video = VideoFileClip(custom_video_path)

    with open("./content/search_terms.json", "r") as file:
        video_data = json.load(file)

    with open("./content/images.json", "r") as file:
        image_data = json.load(file)

    for key, value in image_data.items():
        images = json.loads(key)

    # Create a defaultdict to store the merged data, sorted by 'at_time'
    merged_data = defaultdict(list)

    # Add data1 items to the merged_data dictionary
    for item in images:
        merged_data[item["atTime"]].append({"image": item["image"]})

    # Add data2 items to the merged_data dictionary
    for item in video_data["data"]:
        merged_data[item["at_time"]].append({"search_terms": item["search_terms"]})

    # Sort the keys (at_time values) in increasing order
    sorted_keys = sorted(merged_data.keys())

    merged_json = [
        {"at_time": k, **{key: value for d in v for key, value in d.items()}}
        for k, v in zip(sorted_keys, [merged_data[key] for key in sorted_keys])
    ]

    print(merged_json)

    video_clips = []

    end_clip_time = 0

    final_clip = None

    clip_num = 0

    for index, data in enumerate(merged_json):

        at_time = int(data["at_time"])

        if index < len(merged_json) - 1:
            next_at_time = int(merged_json[index + 1]["at_time"])
        else:
            next_at_time = main_video.duration

        if end_clip_time == 0 or end_clip_time < at_time:
            end_time = at_time
            # Define a generator for creating TextClip objects
            generator = lambda txt: TextClip(
                txt=txt.upper(),
                font="../fonts/Satoshi-Regular.otf",
                fontsize=80,
                color="#FFFFFF",
                stroke_color="White",
                stroke_width=2,
            )

            # Create a SubtitlesClip object
            subtitles = SubtitlesClip(subtitles_path, generator).subclip(
                end_clip_time, end_time
            )

            video = create_reel_video_1(
                main_video, subtitles.set_position((230, 350)), end_clip_time, end_time
            )

            if final_clip is not None:
                final_clip = fade(final_clip, video)
            else:
                final_clip = video

            end_clip_time = end_time

        if "image" in data:
            image = data["image"]
            image_path = f"../temp/image_{image}.jpg"
            clip = ImageClip(image_path, duration=4)

        else:
            clip = VideoFileClip(f"../temp/{clip_num}.mp4")
            clip_num += 1

        start_time = at_time if at_time >= end_clip_time else end_clip_time

        if (start_time + clip.duration) > next_at_time:
            end_time = next_at_time
        else:
            end_time = start_time + clip.duration

        time = end_time - start_time

        if "image" in data:

            clip = clip.resize(newsize=(900, 750))

            video = create_reel_video_1(
                main_video,
                clip.set_position((90, 200)).subclip(0, time),
                start_time,
                end_time,
            )

            final_clip = fade(final_clip, video)

        else:

            if clip_num % 3 == 0:

                video = create_reel_video_4(
                    main_video,
                    clip.subclip(1, time + 1),
                    start_time,
                    end_time,
                )
                final_clip = fade(final_clip, video)

            elif clip_num % 3 == 1:

                clip = clip.crop(x1=150, y1=0, x2=1130, y2=720).resize(
                    newsize=(900, 750)
                )

                video = create_reel_video_1(
                    main_video,
                    clip.subclip(1, time + 1).set_position((90, 200)),
                    start_time,
                    end_time,
                )
                final_clip = fade(final_clip, video)

            else:
                video = create_reel_video_2(
                    main_video,
                    clip.subclip(1, time + 1).resize((1280, 720)),
                    start_time,
                    end_time,
                )

                final_clip = fade(final_clip, video)

        end_clip_time = end_time

    if end_clip_time < main_video.duration:

        video = create_reel_video_3(
            main_video,
            end_clip_time,
            main_video.duration,
        )

        final_clip = fade(final_clip, video)

    final_clip.write_videofile("../data/output_test10.mp4", threads=2)

    return "../data/output_test10.mp4"
