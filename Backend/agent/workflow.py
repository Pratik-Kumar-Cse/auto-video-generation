import json
import sys
import os

from autogen.cache import Cache

from agent.agents import (
    Master_Agent,
    user_proxy,
    generate_script_using_video,
    generate_script_using_article,
    generate_script_using_data,
    generate_script_using_topic,
)

from agent.functions import extract_article

from agent.agent import user_proxy, search_terms_manager
from agent.serper import video_search


class ScriptCreator:
    def __init__(self):
        pass

    def generate_script_in_file(
        self, topic, video_type, videoInputType, linkInputField
    ):
        task = topic

        with Cache.disk(cache_seed=44) as cache:
            chat_history = user_proxy.initiate_chat(
                Master_Agent, message=task, cache=cache, max_turns=1
            )

        if videoInputType == "youtubeURL":
            task = f"topic is: {topic} and video url: {linkInputField}"
            generate_script_using_video(task, video_type)

        elif videoInputType == "blogLink":

            content = extract_article(linkInputField)

            task = f"topic is: {topic} and article content : {content}"
            generate_script_using_article(task, video_type)

        elif videoInputType == "script":
            task = f"this is the topic or script data with instructions: {topic}"
            generate_script_using_data(task, video_type)

        else:
            task = f"topic is: {topic}"
            generate_script_using_topic(task, video_type)

    def generate_search_terms(self, script, video_script_with_timestamps):

        with Cache.disk(cache_seed=44) as cache:
            chat_history = user_proxy.initiate_chat(
                search_terms_manager,
                message=f"""
                this is the content -
                    script: {script} 
                    video_transcript: {video_script_with_timestamps}
                """,
                cache=cache,
                max_turns=1,
            )
