from typing import Annotated
import autogen
import random
import os
from dotenv import load_dotenv
import json
import sys

from autogen.cache import Cache

from autogen.agentchat import (
    UserProxyAgent,
    AssistantAgent,
    GroupChat,
    GroupChatManager,
)

from agent.file_saver import store_script_title


# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import srt_to_text_with_timestamps

# Load environment variables
load_dotenv("../.env")

# Set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


config_list = [
    {
        "model": "gpt-4o",
        "api_key": OPENAI_API_KEY,
    },  # another Azure OpenAI API endpoint for gpt-4
    {
        "model": "gpt-4",
        "api_key": OPENAI_API_KEY,
    },  # another Azure OpenAI API endpoint for gpt-4
    {
        "model": "gpt-3.5-turbo",
        "api_key": OPENAI_API_KEY,
    },  # OpenAI API endpoint for gpt-3.5-turbo
    {
        "model": "gpt-3.5-turbo-16k",
        "api_key": OPENAI_API_KEY,
    },  # Azure OpenAI API endpoint for gpt-3.5-turbo
]


llm_config = {
    "timeout": 120,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0.5,  # changing temperature to 0.5 from 0.3
}


user_proxy = UserProxyAgent(
    name="User_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    system_message="A human admin who checks that agents are working properly and TERMINATE the process when desired output is attained.",
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "groupchat",
        "use_docker": False,
    },
)

Analyzer_Agent = AssistantAgent(
    name="Analyzer_Agent",
    system_message="""
    You are an expert Analyzer Agent responsible for dissecting a video script and its subtitles with timestamps. 
    
    Your analysis will be crucial for video editing and clip insertion.
    
    Tasks:
    1. Thoroughly examine the provided video script and subtitles with timestamps.
    2. Segment the script into more/multiple logical scenes or sections based on content shifts, theme changes, or narrative progression.
    3. For each scene/section:
       a. Analyze the content in-depth.
       b. Identify key topics, themes, actions, or visual elements that may require supporting video clips.
       
    4. Utilize subtitle timestamps to pinpoint optimal time frames for video clip insertions within each scene/section.
    5. Provide a comprehensive analysis to the Video Editor Agent, including:
       a. A detailed breakdown of scenes/sections in the script, with clear demarcations.
       b. Key topics, themes, actions, and visual elements that may require video clips, along with their significance to the narrative.
       c. Suggested time frames (based on subtitles) for clip insertion in each scene, with rationale.
    
    Present your analysis in a structured, easy-to-follow format. Use bullet points, numbered lists, or tables where appropriate to enhance clarity. Ensure all relevant information is conveyed concisely yet comprehensively.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)


Video_Editor_Agent = AssistantAgent(
    name="Video_Editor_Agent",
    system_message="""
    You are an expert AI video editor tasked with crafting a compelling video by strategically incorporating relevant clips into a given script. 
    
    Your role is crucial in bringing the narrative to life visually.
    Input:
    1. Original video script and video transcript
    2. Detailed analysis from Analyzer_Agent, including:
       - Scene/section breakdown
       - Key topics, themes, actions, and visual elements
       - Suggested time frames for clip insertion

    Your Mission:
    1. Thoroughly review the script, analysis, and suggested clip insertions.
    2. For each scene/section, recommend precise clip insertions that enhance the narrative.
    3. For each recommended clip, provide:
       a. Clip Description: Detailed visual description of the ideal footage.
       b. Recommended Insertion Point: Exact timestamp (in seconds).
    
    Consider:
    - Relevance and reinforcement of the scene's content and emotional tone.
    - Smooth flow and continuity of the overall video.
    - Alignment with suggested time frames from the analysis.
    - Pacing and rhythm of the narrative.
    - Visual variety and engagement for the viewer.

    Aim to create a visually rich, well-structured final video that effectively conveys the intended message while maintaining viewer engagement.

    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)

search_terms_generator = AssistantAgent(
    name="search_terms_generator",
    llm_config=llm_config,
    system_message="""
    You are the search terms generator for finding relevant video stock footage based on a given script.

    Follow these instructions:
    1. Read the provided video content from Analyzer_Agent and Video_Editor_Agent agent.
    2. For each scene/section, get relevant search terms with 5-7 word search team that directly relate to the scene, actions, or visuals described in that part of the script. also describe the scene
    3. Ensure the search terms accurately capture the essence of what video clips would be needed for that scene/section.
    4. Order the search terms in the array to follow the sequence of scenes/sections in the script.
    5. For each set of all search terms, also provide the approximate timestamp (in seconds) where the corresponding clip(s) should be inserted, based on the scene/section timing in the script.
    
    Return your output as a JSON object in the following format and don't include ```json , ```markdown :
    [
        {
            "search_terms": ["city street scene", "car chase", "explosion", ...],
            "at_time": 120.5
        }
        ...
    ]

    Repeat this JSON object for each scene/section, with the corresponding search terms and timestamp.

    ONLY return the JSON object(s). Do not include any additional text or explanations.
    """,
)

search_terms_writer = AssistantAgent(
    name="search_terms_writer",
    llm_config=llm_config,
    system_message="""you are the search terms writer, Your task to create a JSON file using the provided function write_file and store the array of search terms in this file.""",
)


@user_proxy.register_for_execution()
@search_terms_writer.register_for_llm(
    description="Get the data and save in to json file"
)
def write_file(
    data: Annotated[str, "The response from the LLM"],
) -> str:
    store_script_title(search_terms=data)
    return "success"


critic = AssistantAgent(
    name="Critic",
    system_message="""
    Critic. Double-check search terms from other agents and provide feedback. Check whether the search terms in the sequence of video script.
    Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config=llm_config,
)

groupchat = GroupChat(
    agents=[
        user_proxy,
        Analyzer_Agent,
        Video_Editor_Agent,
        search_terms_generator,
        search_terms_writer,
        # critic,
    ],
    messages=[],
    max_round=10,
)


search_terms_manager = GroupChatManager(
    groupchat=groupchat,
    name="search_terms_manager",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)
