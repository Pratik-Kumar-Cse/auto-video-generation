import os
import requests
from utils import *
from dotenv import load_dotenv

import urllib.request

# Load environment variables
load_dotenv("../.env")
# Check if all required environment variables are set
# This must happen before importing video which uses API keys without checking
check_env_vars()

from gpt import *
from video import *
from custom_video import *
from play_ht import *
from search import *
from uuid import uuid4
from tiktokvoice import *
from flask_cors import CORS
from termcolor import colored
from youtube import *
from apiclient.errors import HttpError
from flask import Flask, request, jsonify, render_template
from moviepy.config import change_settings

from agent.workflow import ScriptCreator
from instagram import upload_reel_to_instagram
from agent.fetch_image import generate_images
from agent.event_clip import generate_sub_clips

openai_api_key = os.getenv("OPENAI_API_KEY")
change_settings({"IMAGEMAGICK_BINARY": os.getenv("IMAGEMAGICK_BINARY")})

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Constants
HOST = "0.0.0.0"
PORT = 8080
GENERATING = False


@app.route("/api/generate-script", methods=["POST"])
def generate_script_route():
    try:
        # Parse JSON
        data = request.get_json()

        video_type = data.get("videoType")

        videoInputType = data.get("videoInputType")

        linkInputField = data.get("linkInputField")

        topic = data.get("videoSubject")

        # Print little information about the video which is to be generated
        print(colored("[Script to be generated]", "blue"))

        print(colored("Subject: " + data["videoSubject"], "blue"))

        script_creator = ScriptCreator()

        # script_creator.generate_script_in_file(
        #     topic, video_type, videoInputType, linkInputField
        # )

        # Specify the path to your JSON file
        script_file_path = "./content/script.json"

        # Open the JSON file in read mode
        with open(script_file_path, "r") as json_file:
            # Load the JSON data from the file
            script_data = json.load(json_file)

        script = script_data["script"]

        print(colored("script: " + script, "yellow"))

        return jsonify({"success": True, "script": script})

    except Exception as err:
        print(colored(f"[-] Error: {str(err)}", "red"))
        return jsonify({"success": False, "script": ""})


# Generation Endpoint
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        # Set global variable
        global GENERATING
        GENERATING = True

        # # Clean
        # clean_dir("../temp/")
        # clean_dir("../subtitles/")

        # Parse JSON
        data = request.get_json()

        print("data", data)

        video_type = data.get("videoType")

        videoInputType = data.get("videoInputType")

        linkInputField = data.get("linkInputField")

        templates = data.get("templates")

        ai_model = data.get("aiModel")  # Get the AI model selected by the user

        n_threads = data.get("3")  # Amount of threads to use for video generation

        subtitles_position = data.get(
            "subtitlesPosition"
        )  # Position of the subtitles in the video

        text_color = data.get("color")  # Color of subtitle text

        # Get 'useMusic' from the request data and default to False if not provided
        use_music = data.get("useMusic", False)

        # Get 'useMusic' from the request data and default to False if not provided
        automate_instagram_upload = data.get("automateInstagramUpload", False)

        video_provider = data.get("videoProvide", "storyblocks")

        # Get 'automateYoutubeUpload' from the request data and default to False if not provided
        automate_youtube_upload = data.get("automateYoutubeUpload", False)

        # Get the ZIP Url of the songs
        songs_zip_url = data.get("zipUrl")

        topic = data.get("videoSubject")

        # Download songs
        if use_music:
            # Downloads a ZIP file containing popular TikTok Songs
            if songs_zip_url:
                fetch_songs(songs_zip_url)
            else:
                # Default to a ZIP file containing popular TikTok Songs
                fetch_songs(
                    "https://filebin.net/2avx134kdibc4c3q/drive-download-20240209T180019Z-001.zip"
                )

        # Print little information about the video which is to be generated
        print(colored("[Video to be generated]", "blue"))

        print(colored("Subject: " + data["videoSubject"], "blue"))

        print(colored("AI Model: " + ai_model, "blue"))  # Print the AI model being used

        if not GENERATING:
            return jsonify(
                {
                    "status": "error",
                    "message": "Video generation was cancelled.",
                    "data": [],
                }
            )

        voice = None

        if not voice:
            voice = "en_us_001"
            voice_prefix = voice[:2]

        script = """WWDC 2024: The Jaw-Dropping Apple Upgrades You Need to See!**  
        Hey, good morning everyone! Welcome to Apple Park! Today, we're diving into the most exciting updates from WWDC 2024 that are gonna blow your mind. So, let's jump right in and check out the revolutionary stuff Apple just unleashed! First up, VisionOS 2. Apple is really flexing its machine-learning muscles here. Imagine turning your stunning 2D photos into spatial photos with some serious depth. Just a tap, and boom—your photos come to life! But wait, there’s more. SharePlay just got supercharged. Now, you can share these epic spatial photos and panoramas with your family and friends. Think about those family reunions—everyone's there, feeling the moment, no matter the distance. Navigating Vision Pro is easier than ever. New gestures like just tapping in the air to open Home view or checking time and battery? Super intuitive. Mac virtual display is getting a sweet upgrade this year. Think higher resolution, wider sizes—even an ultrawide that’s like having two 4K monitors side by side. Travel mode now supports trains, and there’s some fresh Apple immersive video content coming to the TV app. Plus, Apple Vision Pro is expanding to China, Japan, the UK, and five other countries. Big moves, right? Now, let's get into iOS 18. This update is all about customization and control. You can now arrange your app icons and widgets to make your custom wallpaper really pop. Here’s a game-changer: you can lock sensitive apps with Face ID, Touch ID, or a passcode, and even hide them in a locked folder. Privacy at its finest. Managing app access to contacts and pairing accessories is now in your hands. And messaging? Whole new level. You can use any Emoji or sticker as a tapback, schedule messages, and format your texts with bold, italics, underline—you name it. Plus, for those off-the-grid moments, iPhone 14 and later now support SMS via satellite. Super clutch when you need to stay connected out in the wild. Managing your emails just got way easier with on-device categorization. Maps now show detailed trail networks and hiking routes for all 63 US National Parks—perfect for the adventurers out there. And check this out, Tap to Cash lets you exchange Apple Cash just by holding your phones together. Fast, private, and simple. The Photos app? Biggest redesign ever. It’s now a unified view with super smooth filtering and themed collections to organize your memories just the way you like. Talking AirPods—interacting with Siri just got more natural. Nod your head for 'yes' or shake it for 'no’. Plus, voice isolation will make sure every call sounds crystal clear. And on Apple TV, the new 'Insight' feature pops up real-time info about actors, characters, and even the music tracks. WatchOS 11 introduces 'Training Load' for a deeper look at your workouts and the new 'Vitals App' for tracking your health metrics like heart rate and respiratory rate. It's all about keeping you in check. iPadOS 18 is making multitasking smoother with a floating tab bar that changes into a sidebar. Plus, SharePlay now allows remote control of devices, and yes, we finally have a calculator on iPad—with a twist! It’s got a math notes experience for you to scribble down equations. MacOS Sequoia brings 'iPhone Mirroring.' View and control your iPhone right from your Mac. Enhanced window management and the new 'Passwords App' make your workflow seamless and secure. Seriously, everything's where you need it to be. Apple Intelligence? It's everywhere. Creating text, images, and even handling personalized tasks all while keeping your data private. And get this—Siri now integrates with ChatGPT for some serious brainpower. The future is here, folks. That’s a wrap on WWDC 2024's mind-blowing announcements. Each one aiming to take your Apple experience to the next level. But now, I want to hear from you! What feature or update are you most excited about? How do you think these changes will elevate your Apple experience? Drop all your thoughts in the comments below. And if you enjoyed this video, don't forget to hit the like button, subscribe for more tech updates, and share this video with your fellow Apple fans. Thanks for watching, and stay excited for what’s next from Apple! """

        script_creator = ScriptCreator()

        # script_creator.generate_script_in_file(
        #     topic, video_type, videoInputType, linkInputField
        # )

        # # Specify the path to your JSON file
        # script_file_path = "./content/script.json"

        # # Open the JSON file in read mode
        # with open(script_file_path, "r") as json_file:
        #     # Load the JSON data from the file
        #     script_data = json.load(json_file)

        # script = script_data["script"]

        print(colored("script: " + script, "yellow"))

        # Generate a script
        title = get_title(
            script, ai_model
        )  # Pass the AI model to the script generation

        print(colored("title of script: " + title, "yellow"))

        # voice_urls = create_voices(script)

        voice_urls = [
            "https://peregrine-results.s3.amazonaws.com/hNS03gUAaAvsGk56P3.mp3",
            "https://peregrine-results.s3.amazonaws.com/Ifz53t71uyXwroEz26.mp3",
            "https://peregrine-results.s3.amazonaws.com/4400EbWGbq6H28teHt.mp3",
        ]

        if len(voice_urls) == 0:
            return jsonify(
                {
                    "status": "error",
                    "message": "voice url not found.",
                    "data": [],
                }
            )

        print(colored(f"voice_urls:{voice_urls}", "yellow"))

        # video_id = create_video_with_voice(
        #     title=topic,
        #     audio_urls=voice_urls,
        #     template=templates,
        #     video_type=video_type,
        # )

        # video_id = create_video(
        #     title=data["videoSubject"], input_text=script, template=templates
        # )

        # Long video
        # video_id = "082e9c48e0b2404cb52ffc074f67da49"
        # video_id = "0073804ddf3448488490598b294ab650"

        # Short video
        # video_id = "2e73c736cdb346d0a45f8d8b8b72d6e4"

        # if not video_id:
        #     return jsonify(
        #         {
        #             "status": "error",
        #             "message": "Video id not found.",
        #             "data": [],
        #         }
        #     )

        # webhook_res = add_webhook(video_id)
        # print(colored(webhook_res, "yellow"))

        # print(colored("video_id of custom video " + video_id, "yellow"))

        # Split script into sentences
        sentences = script.split(". ")

        # Remove empty strings
        sentences = list(filter(lambda x: x != "", sentences))
        paths = []

        tts_path = f"../temp/audio.mp3"

        # # Maximum number of retries
        # max_retries = 10
        # retry_count = 0

        # # Loop until we get the video_url or reach the maximum number of retries
        # while retry_count < max_retries:
        #     # Call the get_video function
        #     video_url = get_video(video_id=video_id)

        #     # Check if video_url is obtained
        #     if video_url:
        #         # If video_url is obtained, break out of the loop
        #         break
        #     else:
        #         # Code execution will pause for 5 minutes before continuing
        #         print(colored("waiting for heygen video creation...", "red"))
        #         # If video_url is not obtained, wait for 30 seconds before retrying
        #         time.sleep(30)
        #         retry_count += 1

        # video_path = save_video(video_url)

        # video_path = "../temp/f278859c-0c79-4a9d-8cf8-6af4b03f3eae.mp4"

        # print("video_path =================>" + video_path)

        # convert_video_to_audio(input_video_file=video_url, output_audio_file=tts_path)

        # audio_clips = []
        # for index, voice_url in enumerate(voice_urls):
        #     # Download the audio files
        #     file_path = os.path.join(os.getcwd(), f"../temp/temp_audio{index}.mp3")
        #     urllib.request.urlretrieve(voice_url, file_path)
        #     print(file_path)
        #     audio_clips.append(AudioFileClip(file_path))
        #     print(voice_url)

        # audio_clip = concatenate_audioclips(audio_clips)

        # # Write the combined audio to the output file
        # audio_clip.write_audiofile(tts_path)

        audio_clip = AudioFileClip(tts_path)

        paths.append(audio_clip)

        try:
            # subtitles_path = generate_subtitles(
            #     audio_path=tts_path,
            #     sentences=sentences,
            #     audio_clips=paths,
            #     voice=voice_prefix,
            # )

            subtitles_path = "../subtitles/9636bc67-cf1f-47b0-be95-4a66e92a4349.srt"

            print("subtitles_path", subtitles_path)

            video_script_with_timestamps = srt_to_text_with_timestamps(subtitles_path)

        except Exception as e:
            print(colored(f"[-] Error generating subtitles: {e}", "red"))
            subtitles_path = None

        # script_creator.generate_search_terms(script, video_script_with_timestamps)

        # Specify the path to your JSON file
        search_terms_file_path = "./content/search_terms.json"

        # Open the JSON file in read mode
        with open(search_terms_file_path, "r") as json_file:
            # Load the JSON data from the file
            search_terms_data = json.load(json_file)

        # Extract search_terms string from the dictionary
        search_terms_list = search_terms_data.get("data")

        search_terms = []
        for data in search_terms_list:
            search_terms.append(data["search_terms"])

        print(colored(search_terms, "yellow"))

        # Search for a video of the given search term
        video_urls = []

        # Defines how many results it should query and search through
        it = 2

        # Defines the minimum duration of each clip
        min_dur = 5

        # # Loop through all search terms,
        # # and search for a video of the given search term
        # for search_term in search_terms:
        #     if not GENERATING:
        #         return jsonify(
        #             {
        #                 "status": "error",
        #                 "message": "Video generation was cancelled.",
        #                 "data": [],
        #             }
        #         )

        #     found_urls = []

        #     if video_provider == "storyblocks":
        #         found_urls = search_for_stock_videos_on_story_block(
        #             search_term,
        #             os.getenv("STORYBLOCKS_API_KEY"),
        #             os.getenv("STORYBLOCKS_SCRECT_KEY"),
        #             it,
        #             min_dur,
        #         )
        #     else:
        #         found_urls = search_for_stock_videos(
        #             search_term, os.getenv("PEXELS_API_KEY"), it, min_dur
        #         )

        #     # Check for duplicates
        #     for url in found_urls:
        #         if url not in video_urls:
        #             video_urls.append(url)
        #             break

        # # Check if video_urls is empty
        # if not video_urls:
        #     print(colored("[-] No videos found to download.", "red"))
        #     return jsonify(
        #         {
        #             "status": "error",
        #             "message": "No videos found to download.",
        #             "data": [],
        #         }
        #     )

        # # Define video_paths
        # video_paths = []

        # # Let user know
        # print(colored(f"[+] Downloading {len(video_urls)} videos...", "blue"))

        # # Save the videos
        # for index, video_url in enumerate(video_urls):
        #     if not GENERATING:
        #         return jsonify(
        #             {
        #                 "status": "error",
        #                 "message": "Video generation was cancelled.",
        #                 "data": [],
        #             }
        #         )
        #     try:
        #         saved_video_path = save_video(video_url, video_id=index)
        #         video_paths.append(saved_video_path)
        #     except Exception:
        #         print(colored(f"[-] Could not download video: {video_url}", "red"))

        # if not GENERATING:
        #     return jsonify(
        #         {
        #             "status": "error",
        #             "message": "Video generation was cancelled.",
        #             "data": [],
        #         }
        #     )

        # print(video_paths)

        # try:
        #     generate_sub_clips(
        #         "https://www.youtube.com/watch?v=sBXdyUA6A88&t=37s",
        #         script,
        #         video_script_with_timestamps,
        #     )
        # except Exception as e:
        #     print(colored(f"[-] Error generating video clips: {e}", "red"))

        # try:
        #     generate_images(topic, script, video_script_with_timestamps)
        # except Exception as e:
        #     print(colored(f"[-] Error generating images: {e}", "red"))

        # Put everything together
        try:

            final_video_path = test_long_video1(
                tts_path,
                subtitles_path,
                n_threads or 2,
                subtitles_position,
                text_color or "#FFFF00",
                title,
            )

            # final_video_path = test_long_video(
            #     video_path,
            #     subtitles_path,
            #     n_threads or 2,
            #     subtitles_position,
            #     text_color or "#FFFF00",
            #     title,
            # )

            # final_video_path = test_reel_video1(
            #     video_path,
            #     subtitles_path,
            #     n_threads or 2,
            #     subtitles_position,
            #     text_color or "#FFFF00",
            #     templates,
            #     video_paths,
            # )

            return

            if video_type == "long":
                final_video_path = create_long_video(
                    video_path,
                    subtitles_path,
                    n_threads or 2,
                    subtitles_position,
                    text_color or "#FFFF00",
                    templates,
                    video_paths,
                )
            else:
                temp_video = VideoFileClip(video_path)

                combined_video_clip = combine_videos(
                    video_paths, temp_video.duration, 10, n_threads or 2
                )

                final_video_path = generate_final_video(
                    combined_video_clip,
                    video_path,
                    subtitles_path,
                    n_threads or 2,
                    subtitles_position,
                    text_color or "#FFFF00",
                    templates,
                    video_paths,
                )

        except Exception as e:
            print(colored(f"[-] Error generating final video: {e}", "red"))
            final_video_path = None

        if automate_youtube_upload:
            # Define metadata for the video, we will display this to the user, and use it for the YouTube upload
            title, description, keywords = generate_metadata(
                data["videoSubject"], script, ai_model
            )

            print(colored("[-] Metadata for YouTube upload:", "blue"))
            print(colored("   Title: ", "blue"))
            print(colored(f"   {title}", "blue"))
            print(colored("   Description: ", "blue"))
            print(colored(f"   {description}", "blue"))
            print(colored("   Keywords: ", "blue"))
            print(colored(f"  {', '.join(keywords)}", "blue"))
            # Start Youtube Uploader
            # Check if the CLIENT_SECRETS_FILE exists
            client_secrets_file = os.path.abspath("./client_secret.json")
            SKIP_YT_UPLOAD = False
            if not os.path.exists(client_secrets_file):
                SKIP_YT_UPLOAD = True
                print(
                    colored(
                        "[-] Client secrets file missing. YouTube upload will be skipped.",
                        "yellow",
                    )
                )
                print(
                    colored(
                        "[-] Please download the client_secret.json from Google Cloud Platform and store this inside the /Backend directory.",
                        "red",
                    )
                )

            # Only proceed with YouTube upload if the toggle is True  and client_secret.json exists.
            if not SKIP_YT_UPLOAD:
                # Choose the appropriate category ID for your videos
                video_category_id = "28"  # Science & Technology
                privacyStatus = "public"  # "public", "private", "unlisted"
                video_metadata = {
                    "video_path": os.path.abspath(f"../data/{final_video_path}"),
                    "title": title,
                    "description": description,
                    "category": video_category_id,
                    "keywords": ",".join(keywords),
                    "privacyStatus": privacyStatus,
                }

                # Upload the video to YouTube
                try:
                    # Unpack the video_metadata dictionary into individual arguments
                    video_response = upload_video(
                        video_path=video_metadata["video_path"],
                        title=video_metadata["title"],
                        description=video_metadata["description"],
                        category=video_metadata["category"],
                        keywords=video_metadata["keywords"],
                        privacy_status=video_metadata["privacyStatus"],
                    )
                    print(f"Uploaded video ID: {video_response.get('id')}")
                except HttpError as e:
                    print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")

        if automate_instagram_upload:

            caption = generate_reel_captions(data["videoSubject"], script, ai_model)

            print(colored(f"caption: {caption} ", "blue"))

            # Upload the video to YouTube
            try:
                # Unpack the video_metadata dictionary into individual arguments
                video_response = upload_reel_to_instagram(
                    reel_video_path=os.path.abspath(f"../data/{final_video_path}"),
                    reel_caption=caption,
                )

                print(f"Uploaded reel  successfully")
            except HttpError as e:
                print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")

        if use_music:
            video_clip = VideoFileClip(f"../data/{final_video_path}")
            # Select a random song
            song_path = choose_random_song()

            # Add song to video at 30% volume using moviepy
            original_duration = video_clip.duration
            original_audio = video_clip.audio
            song_clip = AudioFileClip(song_path).set_fps(44100)

            # Set the volume of the song to 10% of the original volume
            song_clip = song_clip.volumex(0.1).set_fps(44100)

            # Add the song to the video
            comp_audio = CompositeAudioClip([original_audio, song_clip])
            video_clip = video_clip.set_audio(comp_audio)
            video_clip = video_clip.set_fps(30)
            video_clip = video_clip.set_duration(original_duration)
            video_clip.write_videofile(f"../{final_video_path}", threads=n_threads or 1)

        # Let user know
        print(colored(f"[+] Video generated: {final_video_path}!", "green"))

        # Stop FFMPEG processes
        if os.name == "nt":
            # Windows
            os.system("taskkill /f /im ffmpeg.exe")
        else:
            # Other OS
            os.system("pkill -f ffmpeg")

        GENERATING = False

        # Return JSON
        return jsonify(
            {
                "status": "success",
                "message": "Video generated! See ..data/rapid_video.mp4 for result.",
                "data": final_video_path,
            }
        )
    except Exception as err:
        print(colored(f"[-] Error: {str(err)}", "red"))
        return jsonify(
            {
                "status": "error",
                "message": f"Could not retrieve stock videos: {str(err)}",
                "data": [],
            }
        )


@app.route("/api/cancel", methods=["POST"])
def cancel():
    print(colored("[!] Received cancellation request...", "yellow"))

    global GENERATING
    GENERATING = False

    return jsonify({"status": "success", "message": "Cancelled video generation."})


@app.route("/api/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json()
    print(f"Received webhook event: {data}")
    # Process the webhook event data as needed
    return "OK"


if __name__ == "__main__":
    # Run Flask App
    app.run(debug=True, host=HOST, port=PORT)
