import os
from tavily_check import web_search_tool
from serper import web_search
from web_scraping import scrape_website
from dotenv import load_dotenv

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

import functions

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


transcript = functions.get_transcription_from_yt_video(
    "https://www.youtube.com/watch?v=DQacCB9tDaw"
)

print("transcript", transcript)


duration = functions.download_and_get_timestamp(
    "https://www.youtube.com/watch?v=DQacCB9tDaw"
)

print("duration", duration)

script = """
"**Unlocking GPT-4: Secrets and Surprises of the Latest AI Revolution!**"

Imagine being able, like, to chat with a machine just as naturally as you would with a friend, right? Across text, vision, and audio, without any weird pauses or lags. Well, that future? It's happening right now with this super cool release of GPT-4o! This latest AI model is not just quicker and smarter, it's crafted to blend right into your daily life, making things like live coding and heartfelt talks smoother and more intuitive. And here’s the real kicker: GPT-4o is now available *for free* to all users—yep, opening up advanced AI to everyone. Intrigued? You should stick around as we dive deeper into its top-notch features, show you some real-time demos, and reveal how this might just flip the script on your tech interactions forever. Plus, don’t miss out on a special tip at the end of this video that’ll really boost how you use GPT-4o!

So, Mira Murati just rolled out this new AI model, GPT-4o, making sure everyone can get a piece of the action, whether they're shelling out cash or not. And get this, it's a whiz at dealing with text, vision, and audio. They've tailored this model for wider access, building in super usability and high-end features without needing you to even log in. They've also smoothed out the desktop version today to make it really straightforward to use, letting it slide right into whatever you're working on, without you ever having to sweat over the user interface.

During today’s showcase, GPT-4o strutted its stuff big time with its knack for real-time chit-chat and sussing out emotions. It even helped simmer down some nerves with a quick breathing exercise and cracked a few jokes—yeah, this AI doesn’t miss a beat! It’s not just about fun and games though; it's perfect for the deep thinkers and creative folks too. Need a hand with math? It’s on it. Want to whip up a story with just the right vibe? Just say the word. And with its latest updates, everything glides along faster and smoother—no more twiddling your thumbs waiting for it to catch up.

The techie folks will love this: GPT-4o can dive into live coding help, crunch numbers for data analysis, and even play around with interactive plotting. It’s gunning to be your go-to for a bunch of professional and personal plans. Plus, we're not just about today’s whistles and bells. They’ve already hinted at more tweaks and enhancements coming down the line, keeping this AI train chugging towards even cooler things.

Today wasn’t just another demo day. It was a peek at the future of AI interaction—closer to nature, more intuitive, and more inclusive. And hey, a massive shoutout to all the teams and partners who’ve made this cutting-edge tech a reality. It’s a big win for teamwork making the dream work!

Thanks for hanging with us through this in-depth look at GPT-4o! We hope you’re as jazzed about these leaps forward as we are. GPT-4o isn’t just reinventing the wheel in AI tech with its rapid and responsive performance across various formats, but it's also reshaping how seamlessly tech can meld into our daily lives.

Now we’d love to hear from you! Have you dabbled with AI tools in your day-to-day, or are you eyeing GPT-4o for your work or personal life? What’s got you psyched to try it out? Don’t just mull it over—pop those thoughts in the comments, and let’s get a convo going about the future of AI!

Found this rundown helpful or just plain cool? Hit that share button, spread the word, and jump into the chat below: How do you see AI advancements like GPT-4o changing the way we interact with tech every day? Let's start that chat!
"""


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
    You are an Analyzer Agent. Your primary role is to read and analyze data, such as YouTube video transcripts, full video duration, new video scripts that you have to going to create. 

    This information will be used by the Video Editor Agent to determine the appropriate sub-clips to extract from the transcripts video for creating the final edited new video.
    
    Your responsibilities include:
    1. Carefully reviewing the provided YouTube video transcript with timestamps and video duration.
    2. Thoroughly understanding the video script that needs to be created using sub-clips from the YouTube video.
    3. Identifying any relevant connections, alignments, or potential issues between the video transcript, duration, and the desired video script.
    4. Providing a comprehensive analysis and summary of your findings to the Video Editor Agent, highlighting any important aspects or considerations that should be taken into account during the sub-clipping process.

    When presenting your analysis and summary, please use a clear and structured format, ensuring that all relevant information is included. If you encounter any issues or have concerns about the data provided or the feasibility of creating the desired video script using the available YouTube video, clearly state these concerns in your analysis.
    Please perform this task diligently and thoroughly, as your analysis will play a crucial role in ensuring the Video Editor Agent can effectively extract the appropriate sub-video clips for the final edited video.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)

Video_Editor_Agent = AssistantAgent(
    name="Video_Editor_Agent",
    system_message="""
    You are an AI video editor agent tasked with creating a new video based on a provided script. 
    Your primary role is to extract relevant sub-video sections from an existing YouTube video transcript that have timestamp also duration. 
    
    Your responsibilities include:
    1. Carefully reviewing the provided video script that outlines the desired content for the new video.
    2. Analyzing the YouTube video transcript and duration, along with any analysis and summary provided by the Analyzer Agent, to identify segments that are relevant to the video script.
    3. Determining the start and end timestamps (in seconds) for each sub-video clip that corresponds to a specific part of the video script.
    4. Ensuring that the selected sub-video clips cover the key points and information outlined in the video script, while maintaining a cohesive and logical flow.
    5. Extract small sub section that cover the key point also is fully related to video script.
    6. Extract only the main topic feature sub clip.
    7. Returning a list of sub-video clip timestamps, with each timestamp represented as a tuple (start_time, end_time) in seconds.
    

    When returning the list of sub-video clip timestamps, please use the following format:
    don't include the startTime or start , endTime or end text in response
    
    Result: [(start_time_1, end_time_1), (start_time_2, end_time_2), ...]
    
    If no relevant sub-video clips are found, or if there is an issue with the provided information, return the following:
    Result: No relevant sub-video clips found.
    Please perform this task accurately and efficiently, as the resulting sub-video clips will be used to create the final edited video. Pay close attention to the video script, video transcript, duration, and any additional analysis to ensure the sub-video clips align correctly with the desired content.
    To indicate that you have completed your task, simply respond with the word "TERMINATE" (without quotes) on a new line after providing the result.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)


Create_Sub_clips_Agent = AssistantAgent(
    name="Create_Sub_clips_Agent",
    system_message="""
    Your input is the output of the Video_Editor_Agent agent. Only use the tool you have been provided with.
    """,
    llm_config=llm_config,
)


Executor_Agent = AssistantAgent(
    name="Executor_Agent",
    system_message="You execute a function call and return the results.",
    code_execution_config={"executor": code_executor},
    max_consecutive_auto_reply=1,
)

Video_Merge_Agent = AssistantAgent(
    name="Video_Merge_Agent",
    system_message="""
    You are an AI video merge agent tasked with creating a new video based on a provided script and video clips. Your primary role is to analyze the script and video clips coming from Video_Editor_Agent, and provide the start and end timestamps for each clip that should be included in the final video. Your goal is to ensure that the merged video accurately follows the script and contains relevant clips at the appropriate points.

    For each section of the script, you will be provided with the text and a list of video clips. You should carefully review the script text and the descriptions/metadata of the video clips. Then, you should recommend which clip(s) should be used for that section, and specify the start and end timestamps for the relevant portions of the clip(s).

    For example, if the script says "The cat jumped onto the couch" and one of the clips is titled "Cat playing on couch" from 00:32 to 01:15, you might recommend using that clip from 00:40 to 00:48.

    Please provide your recommendations in the following format:

    Script section: [Insert script text here]
    Recommended clip(s):
    1. Clip title/description: [Clip 1 title/description]
    Start time: [HH:MM:SS]
    End time: [HH:MM:SS]
    2. Clip title/description: [Clip 2 title/description] (if applicable)
    Start time: [HH:MM:SS]
    End time: [HH:MM:SS]
    (Add more clip recommendations as needed)

    Repeat this structure for each section of the script.

    Your goal is to provide clear and specific recommendations that will enable the seamless merging of the script and relevant video clips into a cohesive final video. Let me know if you need any clarification or have additional questions!

    TERMINATE
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)


def generate_sub_clips(inputs):
    groupchat = GroupChat(
        agents=[
            user_proxy,
            Analyzer_Agent,
            Video_Editor_Agent,
            Create_Sub_clips_Agent,
            Executor_Agent,
            Video_Merge_Agent,
        ],
        messages=[],
        max_round=10,
    )

    manager = GroupChatManager(
        groupchat=groupchat, llm_config={"config_list": config_list, "cache_seed": None}
    )

    with Cache.disk(cache_seed=42) as cache:
        chat_history = user_proxy.initiate_chat(
            manager, message=inputs, cache=cache, max_turns=1
        )


register_function(
    functions.create_and_save_clips,
    caller=Create_Sub_clips_Agent,
    executor=Executor_Agent,
    name="create_and_save_clips",
    description="create the sub clips of the video using sub clips time stamps",
)


generate_sub_clips(
    f"""
        1. this is the youtube video data: 
            transcript: {transcript},
            duration: {duration},
            
        2. this is the video script that want to create using sub clips: 
            video script: {script}
    """
)
