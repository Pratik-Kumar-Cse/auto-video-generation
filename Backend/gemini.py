import google.generativeai as genai
import os
import time
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)


from utils import srt_to_text_with_timestamps, remove_file, download_image

# from agent.functions import create_and_save_clips


def text_call(prompt):
    model = genai.GenerativeModel("gemini-pro")

    response = model.generate_content(prompt)
    print(response.text)

    return response.text


def image_call(prompt, image_path, name="test"):

    sample_file = genai.upload_file(path=image_path, display_name=name)

    print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    response = model.generate_content([prompt, sample_file])

    genai.delete_file(sample_file.name)

    print(f"Deleted {sample_file.display_name}.")

    print(response.text)

    return response.text


def video_call(prompt, video_file_path):

    print(f"Uploading file...")

    video_file = genai.upload_file(path=video_file_path)

    print(f"Completed upload: {video_file.uri}")

    print(f"Completed upload: {video_file.name}")

    time.sleep(100)

    # Set the model to Gemini 1.5 Pro.
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content(
        [prompt, video_file], request_options={"timeout": 600}
    )

    genai.delete_file(video_file.name)
    print(f"Deleted file {video_file.uri}")

    print(response.text)

    return response.text


def get_video_clips_details(script, video_path):
    print("video_path", video_path)
    return video_call(
        f"""
        Use this video file from which I need specific clip segments extracted.

        Your goal is:
        1. Read this new script: {script}, and analyze this script and find the relevant section with this video related visual and transcript.
        2. To compile a collection of short video clips (around 5-20 seconds each) that are relevant to visualize the topics covered in my scripted narration.
        3. Fetch the accurate timestamp of the video clips the sub clips are total related to script.
        4. Remove the initial intro part more focus on features and good visual representation.
        5. Ensure that the total duration of the clips does not exceed the actual video duration, and that each clip's duration is accurate.

        For each main point in my script, please identify an applicable clip segment from the provided video file that illustrates or relates to that topic.

        Please provide the following details for each recommended clip:
        2. A brief description of the relevant script section
        2. The start and end time of the clip segment from the original video
        3. Any other context about why this clip is a good fit for that point in the script
        

        I will use these clips alongside the voiceover narration to create an edited video production.
        The clips should match and visualize the concepts I'm describing in the script.

        Please provide your recommendations in the following format:
            Clip 1
            Duration: (start_time in seconds, end_time in seconds)
            Description: [explain the about the section]

            Clip 2
            Duration: (start_time in seconds, end_time in seconds)
            Description: [explain the about the section]

        """,
        video_path,
    )


# get_video_clips_details(script, "../data/data.mp4")


def get_reels_clips_details(script, video_path):

    return video_call(
        f"""
        Use this video file data from which I need specific clip segments extracted. 
        
        Your goal is: 
        1. Read this new script: {script}, and analyze this script to find relevant sections that can be visually represented by clips from the provided video.
        2. Compile a collection of short video clips (around 10-15 seconds each) that are relevant to visualize the topics covered in my scripted narration.
        3. Fetch the accurate timestamps of the video clips, ensuring that the sub-clips are directly related to the script.
        4. Remove the initial intro part and focus more on features and good visual representations.
        5. Ensure that the total duration of the clips does not exceed the actual video duration, and that each clip's duration is accurate.

        For each main point in my script, please identify an applicable clip segment from the provided video file that illustrates or relates to that topic. 
        
        Please provide the following details for each recommended clip:
        1. A brief description of the relevant script section
        2. The start and end timestamps of the clip segment from the original video (in seconds) (mandatory)
        3. Any other context about why this clip is a good fit for that point in the script
        
        I will use these clips alongside the as background stock footage. 
        The clips should match and visualize the concepts I'm describing in the script.

        Please provide your recommendations in the following format:
        Clip 1
        Duration: (start_time in seconds, end_time in seconds)
        Description: [explain the relevant section of the script]

        Clip 2
        Duration: (start_time in seconds, end_time in seconds)
        Description: [explain the relevant section of the script]
    
        ....

        """,
        video_path,
    )


# video_call(
#     f"""
#     Use this video file and Analyze the content and divide it into multiple sections based on the main topics covered.

#     For each section, provide the following:
#     1. Start and end times (in minutes:seconds format)
#     2. A concise description summarizing the main topic of that section.
#     3. These subsection I want to use in my new video so thing like a video editor.

#     Please ensure that the sections are coherent and logically divided based on the changes in topic throughout the video.
#     If possible, include timestamps that correspond to natural breaks or transitions in the content.
#     The descriptions should be clear and accurately capture the essence of what is being discussed in each section."

#     """,
#     "../data/data.mp4",
# )


def at_clip_in_video(video_script_with_timestamps):

    res = video_call(
        f"""
        Use this video file and Analyze the content and divide it into multiple sections based on the main topics covered.

        For each section, provide the following:
        1. Start and end times (in minutes:seconds format)
        2. A concise description summarizing the main topic of that section.
        3. These subsection I want to use in my new video so thing like a video editor.
        

        Please ensure that the sections are coherent and logically divided based on the changes in topic throughout the video.
        If possible, include timestamps that correspond to natural breaks or transitions in the content.
        The descriptions should be clear and accurately capture the essence of what is being discussed in each section."
        
        Please provide your recommendations in the following format:

        Clip 1
        Duration: (start_time in seconds, end_time in seconds)
        Description: [explain the about the section]

        Clip 2 
        Duration: (start_time in seconds, end_time in seconds)
        Description: [explain the about the section]
        
        ...

        """,
        "../data/data.mp4",
    )

    print(res)

    prompt = f"""
    
        You are a video editor working on creating a new video by incorporating various video clips into a given video script. Your task is to analyze the script and the provided video clips, and then determine the most appropriate placement for clip within the script.

        You will be provided with the following information:

        1. The full video script text, which will be inserted in place of {video_script_with_timestamps} also have timestamp for reference for clip insert timing.
        2. A list of video clips, including their descriptions and timings, which will be inserted in place of {res}. 
        3. It is not necessary to add all clips in the script only add those clips that the related to script
        4. thing like these clips are the stock clips for a new video it and not mandate to insert all clips
        
        The format for clip will be:
        [Clip Number]
        [Clip Duration]:(start_time in seconds, end_time in seconds)
        [Clip Description]: 
        
        Your goal is to analyze the script content and the provided video clips, and then recommend where clip should be inserted into the script to create a cohesive and well-structured final video. 
        Consider the context and flow of the script, as well as the relevance of clip's content to the script.

        Please provide your recommendations in the following format:

        Clip 1 Description: [description]
        Recommended Insertion Point: [timestamp or specific location in the script]
        Justification: [explain why this clip should be inserted at this point]

        Clip 2 Description: [description]
        Recommended Insertion Point: [timestamp or specific location in the script]
        Justification: [explain why this clip should be inserted at this point]

        ... (repeat for all provided clips)

        Please be as detailed and specific as possible in your recommendations and justifications, ensuring that the final video will flow smoothly and effectively convey the intended message.
        
        """

    res = text_call(prompt)

    print(res)

    return res


# at_clip_in_video(video_script_with_timestamps)


def extract_clips_call():

    res = """
        Here are the recommended clips from the provided video with timestamps and descriptions that best match your script:

        **Clip 1**
        Duration: (28, 59) 
        Description: Mira Murati announces that OpenAI is releasing GPT-4o for free to all users. 

        **Clip 2**
        Duration: (1:58, 2:18) 
        Description: OpenAI emphasizes the ease of use and accessibility of GPT-4o, highlighting a refreshed UI for a natural user experience. 

        **Clip 3**
        Duration: (9:17, 10:01)
        Description:  Two research leads demonstrate real-time conversational speech using GPT-4o on a phone, showcasing its intuitive and responsive nature.

        **Clip 4**
        Duration: (10:14, 10:51)
        Description: The research lead asks GPT-4o for feedback on his breathing exercise and jokes with GPT-4o about breathing like a vacuum cleaner, highlighting the model's emotion recognition and humorous capabilities.

        **Clip 5**
        Duration: (11:51, 13:17) 
        Description:  The research lead asks GPT-4o to tell a bedtime story about robots and love, demonstrating the model's ability to generate creative content in different styles, including a singing voice.

        **Clip 6**
        Duration: (13:56, 16:29) 
        Description:  The research lead interacts with GPT-4o by uploading an image of a handwritten linear equation, showcasing GPT-4o's vision capabilities and its ability to help solve math problems interactively.

        **Clip 7**
        Duration: (17:17, 17:52) 
        Description:   The research lead asks GPT-4o to analyze a selfie to determine his emotions, demonstrating the model's ability to recognize facial expressions and understand emotional cues.

        **Clip 8**
        Duration: (18:20, 20:21)
        Description:  The research leads demonstrate GPT-4o's ability to analyze and understand code, as well as its capability to generate and interpret visual data like plots. 

        **Clip 9**
        Duration: (21:56, 23:48)
        Description:  The research leads showcase GPT-4o's real-time translation capabilities by having Mira and Mark communicate in Italian and English respectively, with GPT-4o seamlessly translating their conversation. 

        **Clip 10**
        Duration: (25:24, 25:59) 
        Description:  The segment concludes with the research leads reiterating that GPT-4o is designed to be intuitive, accessible, and powerful, emphasizing its potential to revolutionize AI interaction.
        """

    prompt = f"""
    clips data : {res}
    Using the clips_data, 
    return an list of tuples representing the start and end times of each clip in the following format:
    don't include the startTime or start , endTime or end text in response 
    Result: [(start_time_1, end_time_1), (start_time_2, end_time_2), ...]
    
    note: always return time in seconds and result in list
    """

    # res = text_call(prompt)

    res = [
        (28, 59),
        (118, 138),
        (557, 601),
        (614, 651),
        (711, 797),
        (836, 989),
        (1032, 1072),
        (1092, 1221),
        (1316, 1428),
        (1524, 1559),
    ]

    # create_and_save_clips((res))

    print(res)


def get_images_details(script, images):

    details = []

    image_number = 1

    for index, image in enumerate(images):

        image_path = download_image(image, image_number)

        if image_path:
            try:

                prompt = """Analyze the provided image and perform image segmentation to identify the various elements and objects present. Based on the results of the image segmentation, provide the following information:

                A brief description of the overall scene or contents of the image.
                A list of the main objects, elements, or regions identified through image segmentation, along with their approximate locations or positions within the image (e.g., top-left, center, bottom-right).
                An assessment of whether the image is relevant and suitable for visually representing or supporting the given script.
                For each main point or topic covered in the script, indicate whether the image contains relevant visual elements or objects that could effectively illustrate or complement that part of the script.

                In your response, please include the following sections with clear headings for better readability:
                [Image Number]
                {image_number}
                [Image Description]
                ...
                [Segmentation Results]
                ...
                [Relevance Assessment]
                ...
                
                If you need any clarification or have additional requirements, feel free to ask.
                Script:
                {script}
                """

                detail = image_call(
                    prompt,
                    image_path,
                )

                details.append(detail)

                image_number = image_number + 1
            except Exception as err:
                print(f"[-] Error: {str(err)}")
                remove_file(image_path)

    print("details", details)

    return details
