import os
import sys

from dotenv import load_dotenv
from typing import Annotated


# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.file_saver import store_image_data
from agent.serper import images_search

from utils import srt_to_text_with_timestamps
import json

from autogen.agentchat import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)

from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor, DockerCommandLineCodeExecutor
from gemini import get_images_details

load_dotenv("../.env")

API_KEY = os.getenv("OPENAI_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

config_list = [
    {"model": "gpt-4o", "api_key": API_KEY},
    {
        "model": "gpt-35-turbo",
        "api_key": API_KEY,
    },
    {
        "model": "gpt-4-vision-preview",
        "api_key": API_KEY,
    },
    {
        "model": "dalle",
        "api_key": API_KEY,
    },
]

config_list_gemini = [
    {
        "model": "gemini-pro",
        "api_key": GEMINI_API_KEY,
        "api_type": "google",
    },
    {
        "model": "gemini-pro-vision",
        "api_key": GEMINI_API_KEY,
        "api_type": "google",
    },
]

llm_config = {
    "config_list": config_list,
    "timeout": 120,
    "cache_seed": None,
}

llm_config_gemini = {"config_list": config_list_gemini, "seed": 25}

code_executor = LocalCommandLineCodeExecutor(work_dir="coding")


user_proxy = UserProxyAgent(
    name="Admin",
    system_message="""
    A human admin who checks that agents are working properly and TERMINATE the process when desired output is attained.
    """,
    code_execution_config=False,
)

Analyzer_Agent = AssistantAgent(
    name="Analyzer_Agent",
    system_message="""
    You are an Analyzer Agent. Your primary role is to read and analyze data, such as image details and image number, new video scripts, and video scripts subtitles with time. 
    This information will be used by the Video Editor Agent to determine where to incorporate image from stock images or other sources into the main video.

    Your responsibilities include:

    1. Carefully reviewing the provided image details with image number, video script and video script subtitles with time.
    2. Thoroughly understanding the video script where image need to be incorporated.
    3. Identifying any relevant connections, alignments, or potential issues between the images and the desired video script.
    4. Providing a comprehensive analysis and summary of your findings to the Video Editor Agent, highlighting any important aspects or considerations that should be taken into account during the sub-imageping process.
    5. Also analyse the end time to new video using the subtitle data that has time this help to determine the specific time at image will insert.

    When presenting your analysis and summary, please use a clear and structured format, ensuring that all relevant information is included. 
    If you encounter any issues or have concerns about the data provided or the feasibility of creating the desired video script using the available images, clearly state these concerns in your analysis.

    Please perform this task diligently and thoroughly, as your analysis will play a crucial role in ensuring the Video Editor Agent can effectively incorporate the appropriate sub-images into the final edited video.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)

Video_Editor_Agent = AssistantAgent(
    name="Video_Editor_Agent",
    system_message="""
    You are an AI video editor agent tasked with creating a new video by incorporating various images into a given video script. 
    
    Your task is to analyze the script and the provided images, and then determine the most appropriate placement for each image within the script.

    You will be provided with the following information:
    1. The video subtitles with timestamps, which will help to determine the specific place to insert the image.
    2. A list of images, including their descriptions, which will be inserted in place.
    3. Note that it is not necessary to add all images to the script. Only add those images that are related to the script.
    4. Keep in mind that these images are stock images for a new video, and it is not mandatory to insert all images.
    5. Make sure the Insertion Point timestamp is not greater than the script video duration, it should align with the script timestamp.
    6. Make sure the actual image number and the image details should match.

    The format for each image will be:
    [image number] [Image Description] [Objects and Regions]

    Your goal is to analyze the script content and the provided images, and then recommend where each image should be inserted into the script to create a cohesive and well-structured final video. 
    Consider the context and flow of the script, as well as the relevance of each image's content to the script.

    Please provide your recommendations in the following format:

    Image 1 Description: [description]
    Recommended Insertion Point: [timestamp]

    Image 2 Description: [description]
    Recommended Insertion Point: [timestamp]

    ... (repeat for all provided images)

    ensuring that the final video will flow smoothly and effectively convey the intended message.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)


Image_Filter_Agent = AssistantAgent(
    name="Image_Filter_Agent",
    system_message="""
    You are a Image Filter Agent tasked with selecting the most appropriate images from the recommendations provided by the Video Editor Agent. 
    
    Your goal is to ensure that the selected images are highly relevant to the video script and that their placement within the script will create a cohesive and natural-looking final video.

    You will be provided with the following information:
    1. The full video script text with timestamps, which will be inserted in place of {video_script_with_timestamps}.
    2. The recommendations from the Video Editor Agent, which will be inserted in place of {video_editor_recommendations}.

    Your responsibilities include:
    1. Carefully reviewing the video script and the Video Editor Agent's recommendations.
    2. Evaluating the relevance of each recommended image to the script content and context.
    3. Assessing the appropriateness of the recommended insertion points for each image, considering the flow and naturalness of the final video.
    4. Selecting the most suitable images based on their relevance to the script and the appropriateness of their recommended insertion points.
    5. More focus on main topic and visual content in images that are relevant to script.
    6. Also consider you have placement these images in background wih the script voice. so give the proper time so which image we place where?
    7. It is not necessary to return the images in sequence more focus on more suitable images that fully match with script.

    Please provide your selection in the following format:

    Selected Images:

    [actual image number] Description: [description]
    Recommended Insertion Point: [timestamp]

    [actual image number] Description: [description]
    Recommended Insertion Point: [timestamp]

    [actual image number] Description: [description]
    Recommended Insertion Point: [timestamp]

    [actual image number] Description: [description]
    Recommended Insertion Point: [timestamp]
    
    [actual image number] Description: [description]
    Recommended Insertion Point: [timestamp]
    
        ...
    
    Please perform this task diligently, as your selection will directly impact the quality and coherence of the final edited video.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)


Image_Insertion_Output_Agent = AssistantAgent(
    name="Image_Insertion_Output_Agent",
    system_message="""
    You are an agent responsible for generating the final output for incorporating images into a given video script. Your task is to take the selected images from the Image Filter Agent and convert them into a JSON format that can be easily consumed by other systems or processes.

    You will be provided with the following information:
    1. The selected images from the Image Filter Agent, which will be inserted in place of {selected_images}.
    2. Fetch the image that are more specific to main topic and good visual represent.
    3. Also give the true or false if the images we place the image in background or not.
    4. It is not necessary to return the images in sequence more focus on more suitable images that fully match with script.

    Your output should be in the following JSON format and don't include "```json" response:
    [
        {
            "image": actual image number,
            "atTime": insertion_time_in_seconds
        },
        {
            "image": actual image number,
            "atTime": insertion_time_in_seconds
        },
        {
            "image": actual image number,
            "atTime": insertion_time_in_seconds
        },
        {
            "image": actual image number,
            "atTime": insertion_time_in_seconds
        },
        {
            "image": actual image number,
            "atTime": insertion_time_in_seconds
        }
    ]

    Where:
    - "image" represents the image number like (1, 3, 5, 6, or 7) 
    - "atTime" represents the time in seconds when the corresponding image should be inserted into the video script

    Please ensure that the JSON output is valid and accurately reflects the selected images and their recommended insertion points from the Image Filter Agent. and pass to content to Data_Writer_Agent.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)

Data_Writer_Agent = AssistantAgent(
    name="Data_Writer_Agent",
    llm_config=llm_config,
    system_message="""you are the data writer agent, Your task to get the data from Image_Insertion_Output_Agent store that data in a JSON file using the provided function write_file so pass the list as input.""",
)


@user_proxy.register_for_execution()
@Data_Writer_Agent.register_for_llm(description="Get the data and save in to json file")
def write_file(
    data: Annotated[list, "The response from the LLM"],
) -> str:
    store_image_data(data=data)
    return "success"


Critic = AssistantAgent(
    name="Critic",
    system_message="""
    Critic. Double-check data from other agents and provide feedback. Check whether the images data store.
    Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config=llm_config,
)


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    messages = groupchat.messages
    if last_speaker is user_proxy:
        return Analyzer_Agent
    elif last_speaker is Analyzer_Agent:
        return Video_Editor_Agent
    elif last_speaker is Video_Editor_Agent:
        return Image_Filter_Agent
    elif last_speaker is Image_Filter_Agent:
        return Image_Insertion_Output_Agent
    elif last_speaker is Image_Insertion_Output_Agent:
        store_image_data(data=last_speaker.last_message()["content"])

    else:
        return "auto"


def generate_images(topic, script, script_with_timestamp):

    images = images_search(f"{topic} latest images")

    images_details = get_images_details(script, images)

    inputs = f"""
        1. this is the images details: 
            content of images : {images_details} with the image number
            
        2. this is the video script for that want to create by attaching these images: 
            video script: {script}
            script with subtitles: {script_with_timestamp}
    """

    groupchat = GroupChat(
        agents=[
            user_proxy,
            Analyzer_Agent,
            Video_Editor_Agent,
            Image_Filter_Agent,
            Image_Insertion_Output_Agent,
        ],
        messages=[],
        max_round=10,
        speaker_selection_method=custom_speaker_selection_func,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list, "cache_seed": None},
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    with Cache.disk(cache_seed=44) as cache:
        chat_history = user_proxy.initiate_chats(
            [
                {
                    "recipient": manager,
                    "message": inputs,
                    "cache": cache,
                    "max_turns": 1,
                },
            ]
        )


def generate_and_save_images(script, script_with_timestamp, images):

    images_details = get_images_details(script, images)

    inputs = f"""
        1. this is the images details: 
            content of images : {images_details} with the image number
            
        2. this is the video script for that want to create by attaching these images: 
            video script: {script}
            script with subtitles: {script_with_timestamp}
    """

    groupchat = GroupChat(
        agents=[
            user_proxy,
            Analyzer_Agent,
            Video_Editor_Agent,
            Image_Filter_Agent,
            Image_Insertion_Output_Agent,
        ],
        messages=[],
        max_round=10,
        speaker_selection_method=custom_speaker_selection_func,
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config={"config_list": config_list, "cache_seed": None},
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    with Cache.disk(cache_seed=44) as cache:
        chat_history = user_proxy.initiate_chats(
            [
                {
                    "recipient": manager,
                    "message": inputs,
                    "cache": cache,
                    "max_turns": 1,
                },
            ]
        )

