import os

from dotenv import load_dotenv
import sys
from typing import Annotated


# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from gemini import get_video_clips_details, text_call

from agent.functions import download_video, create_and_save_clips
from agent.file_saver import store_clip_data


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
    You are an Analyzer Agent. Your primary role is to read and analyze data, such as video clip details, new video scripts, and video scripts subtitles with time. This information will be used by the Video Editor Agent to determine where to incorporate sub-clips from stock videos or other sources into the main video.

    Your responsibilities include:

    1. Carefully reviewing the provided video clip details , video script and video script subtitles with time.
    2. Thoroughly understanding the video script where sub-clips need to be incorporated.
    3. Identifying any relevant connections, alignments, or potential issues between the video clips and the desired video script.
    4. Providing a comprehensive analysis and summary of your findings to the Video Editor Agent, highlighting any important aspects or considerations that should be taken into account during the sub-clipping process.
    5. Also analyse the end time to new video using the subtitle data that has time this help to determine the specific time at clip will insert.

    When presenting your analysis and summary, please use a clear and structured format, ensuring that all relevant information is included. If you encounter any issues or have concerns about the data provided or the feasibility of creating the desired video script using the available video clips, clearly state these concerns in your analysis.

    Please perform this task diligently and thoroughly, as your analysis will play a crucial role in ensuring the Video Editor Agent can effectively incorporate the appropriate sub-video clips into the final edited video.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)

Video_Editor_Agent = AssistantAgent(
    name="Video_Editor_Agent",
    system_message="""
    You are an AI video editor agent tasked with creating a new video by incorporating various video clips into a given video script. Your task is to analyze the script and the provided video clips, and then determine the most appropriate placement for each clip within the script.

    You will be provided with the following information:
    1. The video subtitles with timestamps, which will help to determine the specific place to insert the clip.
    2. A list of video clips, including their descriptions and timings, which will be inserted in place.
    3. Note that it is not necessary to add all clips to the script. Only add those clips that are related to the script.
    4. Keep in mind that these clips are stock clips for a new video, and it is not mandatory to insert all clips.
    5. Make sure the Insertion Point timestamp is not greater than the script video duration, it should align with the script timestamp

    The format for each clip will be:
    [Clip Number] [Clip Duration]:(start_time in seconds, end_time in seconds) [Clip Description]

    Your goal is to analyze the script content and the provided video clips, and then recommend where each clip should be inserted into the script to create a cohesive and well-structured final video. Consider the context and flow of the script, as well as the relevance of each clip's content to the script.

    Please provide your recommendations in the following format:

    Clip 1 Description: [description]
    Recommended Insertion Point: [timestamp]
    Justification: [explain why this clip should be inserted at this point]

    Clip 2 Description: [description]
    Recommended Insertion Point: [timestamp]
    Justification: [explain why this clip should be inserted at this point]

    ... (repeat for all provided clips)

    Please be as detailed and specific as possible in your recommendations and justifications, ensuring that the final video will flow smoothly and effectively convey the intended message.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)


Clip_Filter_Agent = AssistantAgent(
    name="Clip_Filter_Agent",
    system_message="""
    You are a Clip Filter Agent tasked with selecting the most appropriate video clips from the recommendations provided by the Video Editor Agent. 
    
    Your goal is to ensure that the selected clips are highly relevant to the video script and that their placement within the script will create a cohesive and natural-looking final video.

    You will be provided with the following information:
    1. The full video script text with timestamps, which will be inserted in place of {video_script_with_timestamps}.
    2. The recommendations from the Video Editor Agent, which will be inserted in place of {video_editor_recommendations}.

    Your responsibilities include:
    1. Carefully reviewing the video script and the Video Editor Agent's recommendations.
    2. Evaluating the relevance of each recommended clip to the script content and context.
    3. Assessing the appropriateness of the recommended insertion points for each clip, considering the flow and naturalness of the final video.
    4. Selecting the most suitable clips based on their relevance to the script and the appropriateness of their recommended insertion points.
    5. More focus on main topic and visual content in clips that are relevant to script.
    6. Also consider you have placement these clips in two way one play in background second play clip after the script video clip. so give the proper info so which clip we place where?
    7. It is not necessary to return the clips in sequence more focus on more suitable clips that fully match with script.

    Please provide your selection in the following format:

    Selected Clips:

    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement

    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement

    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement

    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement
    
    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement
    
    [Clip number] Description: [description]
    Recommended Insertion Point: [timestamp]
    clip placement
    
    ...

    Justification:
    [Provide a detailed explanation for your selection, highlighting why these clips are the most appropriate choices based on their relevance to the script content and the natural flow of the final video. Address any potential concerns or considerations regarding the selected clips or their recommended insertion points.]

    Please perform this task diligently, as your selection will directly impact the quality and coherence of the final edited video.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
)


Create_Sub_clips_Agent = AssistantAgent(
    name="Create_Sub_clips_Agent",
    system_message="""
    Your task is to take the input video clips data and return a list of tuples representing the start and end times (in seconds) of each clip and pass this data and video_path in Executor_Agent. 
    
    The output should be in the following format:
    Results: [(start_time_1, end_time_1), (start_time_2, end_time_2), ...]

    Do not include the words "startTime", "start", "endTime", or "end" in the response. Always return the times in seconds as a list of tuples. 
    Pass this Results in Executor_Agent as input so it will execute the function.
    """,
    llm_config=llm_config,
)


Executor_Agent = AssistantAgent(
    name="Executor_Agent",
    system_message="Get the results data from Create_Sub_clips_Agent and You execute a function call and return the results.",
    code_execution_config={"executor": code_executor},
    max_consecutive_auto_reply=1,
)

Clip_Insertion_Output_Agent = AssistantAgent(
    name="Clip_Insertion_Output_Agent",
    system_message="""
    You are an agent responsible for generating the final output for incorporating video clips into a given video script. Your task is to take the selected clips from the Clip Filter Agent and convert them into a JSON format that can be easily consumed by other systems or processes.

    You will be provided with the following information:
    1. The selected clips from the Clip Filter Agent, which will be inserted in place of {selected_clips}.
    2. Fetch the clip that are more specific to main topic and good visual represent.
    3. Also give the true or false if the clips we place the clip in background or not.
    4. It is not necessary to return the clips in sequence more focus on more suitable clips that fully match with script.

    Your output should be in the following JSON format and don't mention json:
    [
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        },
        {
            "clip": actual clip number,
            "atTime": insertion_time_in_seconds
            "background_placement": boolean value
        }
        
        ...
    ]

    Where:
    - "clip" represents the clip number like (1, 3, 5, 6, or 7) 
    - "atTime" represents the time in seconds when the corresponding clip should be inserted into the video script

    Please ensure that the JSON output is valid and accurately reflects the selected clips and their recommended insertion points from the Clip Filter Agent. and pass to content to Data_Writer_Agent.
    """,
    llm_config={"config_list": config_list, "cache_seed": None},
    is_termination_msg=lambda x: x.get("content", "")
    and x.get("content", "").rstrip().endswith("TERMINATE"),
)

Data_Writer_Agent = AssistantAgent(
    name="Data_Writer_Agent",
    llm_config=llm_config,
    system_message="""you are the data writer agent, Your task to get the data from Clip_Insertion_Output_Agent store that data in a JSON file using the provided function write_file so pass the list as input.""",
)


@user_proxy.register_for_execution()
@Data_Writer_Agent.register_for_llm(description="Get the data and save in to json file")
def write_file(
    data: Annotated[list, "The response from the LLM"],
) -> str:
    store_clip_data(data=data)
    return "success"


Critic = AssistantAgent(
    name="Critic",
    system_message="""
    Critic. Double-check data from other agents and provide feedback. Check whether the clips data store.
    Reply "TERMINATE" in the end when everything is done.
    """,
    llm_config=llm_config,
)


register_function(
    create_and_save_clips,
    caller=Create_Sub_clips_Agent,
    executor=Executor_Agent,
    name="create_and_save_clips",
    description="create the sub clips of the video using sub clips time stamps",
)


def custom_speaker_selection_func(last_speaker: Agent, groupchat: GroupChat):
    messages = groupchat.messages
    if last_speaker is user_proxy:
        return Analyzer_Agent
    elif last_speaker is Analyzer_Agent:
        return Video_Editor_Agent
    elif last_speaker is Video_Editor_Agent:
        return Clip_Filter_Agent
    elif last_speaker is Clip_Filter_Agent:
        return Clip_Insertion_Output_Agent
    elif last_speaker is Clip_Insertion_Output_Agent:
        store_clip_data(data=last_speaker.last_message()["content"])
    else:
        return "auto"


def custom_speaker_selection_func1(last_speaker: Agent, groupchat: GroupChat):
    messages = groupchat.messages
    if last_speaker is user_proxy:
        return Create_Sub_clips_Agent
    elif last_speaker is Create_Sub_clips_Agent:
        return Executor_Agent
    elif last_speaker is Executor_Agent:
        pass
    else:
        return "auto"


def generate_sub_clips(video_link, script, script_with_timestamp):

    # print(video_link, script)

    video_filename = download_video(video_link)

    # video_filename = "data1.mp4"

    print("video_filename", video_filename)

    video_file_path = f"./data/{video_filename}"

    print("video_file_path", video_file_path)

    video_clips_details = get_video_clips_details(script, video_file_path)

    # video_clips_details = """

    #     OK, here is a breakdown of the clips with their start and end times to align with your script, skipping the intro as requested:

    #     **Clip 1: Gemini 1.5 Pro**
    #     Duration: (1:29, 1:48)
    #     Description:  This clip showcases the announcement of Gemini 1.5 Pro, highlighting its availability to developers globally. The visual of the "Context Window" expanding dramatically illustrates the increased token capacity and its power to handle vast amounts of information.

    #     **Clip 2: License Plate Number**
    #     Duration: (0:26, 0:38)
    #     Description: A man on stage is showcasing a new AI feature that helps you find the number of your car's license plate using photos you have taken. This segment visually demonstrates the powerful capabilities of Gemini to efficiently solve real-life problems using multi-modal understanding.

    #     **Clip 3: Summarizing a Meeting**
    #     Duration: (1:55, 2:12)
    #     Description: This clip demonstrates how Gemini can summarize an hour-long Google Meet meeting recording, highlighting key points and even drafting an email reply based on the content of the meeting. It showcases Gemini's advanced comprehension and ability to process lengthy audio and generate relevant summaries and actions.

    #     **Clip 4: NotebookLM in Action**
    #     Duration: (2:16, 2:59)
    #     Description: A man on stage describes and shows the NotebookLM in action. It helps a father and son learn together with personalized, age-appropriate information.

    #     **Clip 5:  Project Astra: Identify a Speaker**
    #     Duration: (3:51, 4:07)
    #     Description: This clip showcases the practical applications of Project Astra by using a smartphone to identify objects and provide information. A woman asks the phone to identify an object that makes sound, and Astra correctly recognizes it as a speaker and provides additional context about the speaker's function.

    #     **Clip 6: Project Astra: Optimize System Performance**
    #     Duration: (4:27, 4:38)
    #     Description:  A whiteboard with a system design on it is shown, and a person using Project Astra asks the AI how to make the system faster.  Astra intelligently suggests adding a cache between the server and the database to improve speed, showcasing its problem-solving and optimization capabilities.

    #     **Clip 7: Imagen 3**
    #     Duration: (4:47, 5:07)
    #     Description: This clip introduces Imagen 3, emphasizing its enhanced photorealism and detailed image generation. The comparison between the generated wolf image and its description highlights the advancements in AI-powered image creation.

    #     **Clip 8: Music AI Sandbox**
    #     Duration: (5:07, 5:20)
    #     Description: A man talks about the Music AI Sandbox on stage, and this segment focuses on its tools that create instrumental sections from scratch, transfer styles between tracks, and more. The visual focuses on the tools in the app.

    #     **Clip 9: Veo Video Generation**
    #     Duration: (5:25, 6:14)
    #     Description: This segment introduces Veo, Google's generative video model. The visual shows a sunflower blooming as an example of how Veo can generate complex videos based on a user's text prompt.

    #     **Clip 10:  Trillium TPUs**
    #     Duration: (6:14, 6:32)
    #     Description: This clip highlights the sixth generation of Google's TPUs, Trillium, showcasing its superior performance capabilities. The visual shows a diagram of the Trillium chip along with a comparison of its performance boost over the previous generation.

    #     **Clip 11: Multi-step Reasoning in Google Search**
    #     Duration: (6:32, 7:13)
    #     Description: A woman talks about a new feature called multi-step reasoning in Google Search on stage. The visual showcases how the technology can break down a complex question into smaller parts, understand the context, and then connect information from different sources to provide a comprehensive answer.

    #     **Clip 12: Planning a Trip with Gemini Advanced**
    #     Duration: (7:13, 7:29)
    #     Description: This segment shows how Gemini can be used for planning. It shows how to create a three-day vacation itinerary, with dynamic adjustments based on a user's preferences. The visual shows the interface as the person makes the adjustments.

    #     **Clip 13: Gemini Help for Side Hustles**
    #     Duration: (13:00, 13:19)
    #     Description: A woman is talking about how Gemini can help with side hustles by analyzing and visualizing data from spreadsheets.  The visual shows a line graph generated by Gemini, representing profit over time for different products.

    #     **Clip 14:  Gemini Nano With Multimodality**
    #     Duration: (13:49, 15:24)
    #     Description: A man is showcasing a new model called Gemini Nano with Multimodality. The segment shows how Gemini Nano, using multimodal understanding, can generate images from text, answer questions about videos, and even work offline.

    #     **Clip 15: PaliGemma and Gemma 2 Announcement**
    #     Duration: (15:24, 16:03)
    #     Description:  This clip features the exciting announcement of two new models: PaliGemma, Google's first vision-language open model, and Gemma 2. The visual emphasizes the names of these models, signifying the rapid advancements in AI technology.

    #     **Clip 16: Educational Tools with LearnLM**
    #     Duration: (16:17, 16:34)
    #     Description:  This segment introduces LearnLM, a family of AI models tailored for education. The focus on personalized learning experiences emphasizes Google's commitment to making AI accessible and beneficial in the education sector.

    #     **Clip 17:  Making AI Affordable**
    #     Duration: (15:24, 15:53)
    #     Description: This segment highlights the affordability of Gemini models, showing a significant price reduction.  The clear presentation of pricing details emphasizes Google's effort to make cutting-edge AI technology accessible to a wider audience.

    #     """

    inputs = f"""
        1. this is the youtube video clips details: 
            content of video clips : {video_clips_details},
            
        2. this is the video script for that want to create using sub clips: 
            video script: {script}
            script with subtitles: {script_with_timestamp}
    """

    groupchat = GroupChat(
        agents=[
            user_proxy,
            Analyzer_Agent,
            Video_Editor_Agent,
            Clip_Filter_Agent,
            Clip_Insertion_Output_Agent,
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

    groupchat1 = GroupChat(
        agents=[
            user_proxy,
            Create_Sub_clips_Agent,
            Executor_Agent,
        ],
        messages=[],
        max_round=5,
        speaker_selection_method=custom_speaker_selection_func1,
    )

    manager1 = GroupChatManager(
        groupchat=groupchat1,
        llm_config={"config_list": config_list, "cache_seed": None},
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    input1 = f"This is the clips data: {video_clips_details} and the video path : {video_file_path}"

    with Cache.disk(cache_seed=44) as cache:
        chat_history = user_proxy.initiate_chats(
            [
                {
                    "recipient": manager,
                    "message": inputs,
                    "cache": cache,
                    "max_turns": 1,
                },
                {
                    "recipient": manager1,
                    "message": input1,
                    "cache": cache,
                    "max_turns": 1,
                },
            ]
        )


# video_script_with_timestamps = create_and_save_clips(
#     [
#         (10, 23),
#         (24, 31),
#         (35, 46),
#         (46, 61),
#         (61, 70),
#         (98, 119),
#         (137, 167),
#         (191, 216),
#         (217, 227),
#         (228, 269),
#         (270, 292),
#         (293, 304),
#         (312, 327),
#         (327, 340),
#         (360, 407),
#         (412, 429),
#         (436, 461),
#         (462, 477),
#         (522, 562),
#         (597, 607),
#         (607, 622),
#         (623, 660),
#         (690, 718),
#         (736, 744),
#         (754, 775),
#         (831, 1017),
#         (1018, 1044),
#     ],
#     "/Users/pratikkumar/Desktop/Projects/arkadia/python/video-generator/Backend/data/Apple WWDC 2024 keynote in 18 minutes.mp4",
# )

# generate_sub_clips(
#     "https://www.youtube.com/watch?v=MzHCWZB5ZpE", script, video_script_with_timestamps
# )
