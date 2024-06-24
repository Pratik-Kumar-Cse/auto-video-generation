import re
import os
import g4f
import json
import openai
import google.generativeai as genai

from g4f.client import Client
from termcolor import colored
from dotenv import load_dotenv
from typing import Tuple, List

# Load environment variables
load_dotenv("../.env")

# Set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def generate_response(prompt: str, ai_model: str) -> str:
    """
    Generate a script for a video, depending on the subject of the video.

    Args:
        video_subject (str): The subject of the video.
        ai_model (str): The AI model to use for generation.


    Returns:

        str: The response from the AI model.

    """

    if ai_model == "g4f":
        # Newest G4F Architecture
        client = Client()
        response = (
            client.chat.completions.create(
                model="gpt-3.5-turbo",
                provider=g4f.Provider.You,
                messages=[{"role": "user", "content": prompt}],
            )
            .choices[0]
            .message.content
        )

    elif ai_model in ["gpt3.5-turbo", "gpt4"]:

        model_name = (
            "gpt-3.5-turbo" if ai_model == "gpt3.5-turbo" else "gpt-4-1106-preview"
        )

        response = (
            openai.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            .choices[0]
            .message.content
        )
    elif ai_model == "gemmini":
        model = genai.GenerativeModel("gemini-pro")
        response_model = model.generate_content(prompt)
        response = response_model.text

    else:

        raise ValueError("Invalid AI model selected.")

    return response


def generate_script(
    video_subject: str,
    paragraph_number: int,
    ai_model: str,
    voice: str,
    customPrompt: str,
) -> str:
    """
    Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.



    Args:

        video_subject (str): The subject of the video.

        paragraph_number (int): The number of paragraphs to generate.

        ai_model (str): The AI model to use for generation.



    Returns:

        str: The script for the video.

    """

    # Build prompt

    if customPrompt:
        prompt = customPrompt
    else:
        prompt = """
            Generate a script for a video, depending on the subject of the video.

            The script is to be returned as a string with the specified number of paragraphs.

            Here is an example of a string:
            "This is an example string."

            Do not under any circumstance reference this prompt in your response.

            Get straight to the point, don't start with unnecessary things like, "welcome to this video".

            Obviously, the script should be related to the subject of the video.

            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
            YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
            ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.

        """

    prompt += f"""
    
    Subject: {video_subject}
    Number of paragraphs: {paragraph_number}
    Language: {voice}

    """

    # Generate script
    response = generate_response(prompt, ai_model)

    print(colored(response, "cyan"))

    # Return the generated script
    if response:
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        final_script = "\n\n".join(selected_paragraphs)

        # Print to console the number of paragraphs used
        print(
            colored(f"Number of paragraphs used: {len(selected_paragraphs)}", "green")
        )

        return final_script
    else:
        print(colored("[-] GPT returned an empty response.", "red"))
        return None


def get_search_terms(
    video_subject: str, amount: int, script: str, ai_model: str
) -> List[str]:
    """
    Generate a JSON-Array of search terms for stock videos,
    depending on the subject of a video.

    Args:
        video_subject (str): The subject of the video.
        amount (int): The amount of search terms to generate.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        List[str]: The search terms for the video subject.
    """

    # # Build prompt
    # prompt = f"""
    # Generate {amount} search terms for stock videos,
    # depending on the subject of a video.
    # Subject: {video_subject}

    # The search terms are to be returned as
    # a JSON-Array of strings.

    # Each search term should consist of 1-3 words,
    # always add the main subject of the video.

    # YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    # YOU MUST NOT RETURN ANYTHING ELSE.
    # YOU MUST NOT RETURN THE SCRIPT.

    # The search terms must be related to the subject of the video.
    # Here is an example of a JSON-Array of strings:
    # ["search term 1", "search term 2", "search term 3"]

    # For context, here is the full text:
    # {script}
    # """

    # Build prompt
    prompt = f"""
    Generating relevant {amount} search terms for video stock footage based on a given script. 
    
    Follow these instructions carefully:

    1. Divide the script into distinct sections or scenes.
    2. For each section, generate 3-5 search terms that directly relate to the content of that section or scenes and would be suitable for finding relevant stock footage.
    3. Ensure the search terms are in the correct sequence, following the order of the script sections.
    4. Return the search terms as a JSON array of strings, with each string representing one search terms.
    5. Do not include any additional text or explanations. Only return the JSON array.
    
    YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    YOU MUST NOT RETURN ANYTHING ELSE.
    YOU MUST NOT RETURN THE SCRIPT.
    REMOVE this remove ```json ``` this we don't want in response
    
    Example JSON array: 
    ["search term 1", "search term 2", "title for section 3", ...]

    Script: {script}
    """

    # Generate search terms
    response = generate_response(prompt, ai_model)
    print(response)

    # Parse response into a list of search terms
    search_terms = []

    try:
        search_terms = json.loads(response)
        if not isinstance(search_terms, list) or not all(
            isinstance(term, str) for term in search_terms
        ):
            raise ValueError("Response is not a list of strings.")

    except (json.JSONDecodeError, ValueError):
        # Get everything between the first and last square brackets
        response = response[response.find("[") + 1 : response.rfind("]")]

        print(
            colored(
                "[*] GPT returned an unformatted response. Attempting to clean...",
                "yellow",
            )
        )

        # Attempt to extract list-like string and convert to list
        match = re.search(r'\["(?:[^"\\]|\\.)*"(?:,\s*"[^"\\]*")*\]', response)
        print(match.group())
        if match:
            try:
                search_terms = json.loads(match.group())
            except json.JSONDecodeError:
                print(colored("[-] Could not parse response.", "red"))
                return []

    # Let user know
    print(
        colored(
            f"\nGenerated {len(search_terms)} search terms: {', '.join(search_terms)}",
            "cyan",
        )
    )

    # Return search terms
    return search_terms


def generate_metadata(
    video_subject: str, script: str, ai_model: str
) -> Tuple[str, str, List[str]]:
    """
    Generate metadata for a YouTube video, including the title, description, and keywords.

    Args:
        video_subject (str): The subject of the video.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        Tuple[str, str, List[str]]: The title, description, and keywords for the video.
    """

    # Build prompt for title
    title_prompt = f"""  
    Generate a catchy and SEO-friendly title for a YouTube shorts video about {video_subject}.  
    """

    # Generate title
    title = generate_response(title_prompt, ai_model).strip()

    # Build prompt for description
    description_prompt = f"""  
    Write a brief and engaging description for a YouTube shorts video about {video_subject}.  
    The video is based on the following script:  
    {script}  
    
    also add hashTag #RapidInnovation
    
    also add these lines and links at the end 

    Discover More From Rapid Innovation:

    Rapid Innovation: ```https://www.rapidinnovation.io/```
    Portfolio: ```https://www.rapidinnovation.io/portfolio```
    Facebook: ```https://www.facebook.com/rapidinnovation.io/```
    Instagram: ```https://www.instagram.com/rapidinnovation.io/```
    Twitter: ```https://twitter.com/InnovationRapid```
    LinkedIn: ```https://www.linkedin.com/company/rapid-innovation/mycompany/```
    Medium: ```https://bit.ly/RapidInnovationMedium```

    Are you ready to automate with AI?

    Contact Rapid Innovation for modern solutions!
    Contact Us: hello@rapidinnovation.io
    
    """

    # Generate description
    description = generate_response(description_prompt, ai_model).strip()

    # Generate keywords
    keywords = get_search_terms(video_subject, 6, script, ai_model)

    return title, description, keywords


def generate_similar_search_terms(input_text, num_terms=5):
    """
    Generate similar search terms for finding stock videos based on the given input text.

    Args:
        input_text (str): The input text describing the desired stock video.
        num_terms (int, optional): The number of similar search terms to generate. Default is 5.

    Returns:
        list: A list of similar search terms.
    """
    prompt = f"Generate {num_terms} similar search terms that help to finding stock videos related to the following text: \n\n{input_text} , don't need stock videos keyword remove it. YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS. "

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.7,
    )

    return json.loads(response.choices[0].message.content)


def generate_reel_captions(subject, script, ai_model):
    Prompt = f"""
    You are an expert social media copywriter tasked with creating engaging captions for Instagram reels. Your goal is to craft captions that are attention-grabbing, relevant, and encourage viewers to watch the reel.

    You will be provided with the following information:
    1. The subject or theme of the reel (e.g., Tech, Science, Marketing, etc.)
    2. A brief script or summary of the reel's content

    Using this information, you should generate a compelling caption that:

    - Hooks the viewer's interest from the beginning
    - Provides context or a teaser about the reel's content
    - Incorporates relevant hashtags and emojis
    - Keeps the caption concise (around 1-3 sentences)
    - Maintains a tone and voice appropriate for the subject matter

    The caption should be written in a way that entices viewers to watch the reel and engage with the content. Remember to focus on creating a catchy, attention-grabbing caption that accurately represents the reel's theme and content.

    Subject/Theme: {subject}
    Reel Script/Summary: {script}

    Please generate an Instagram reel caption based on the provided information.
    """

    return generate_response(Prompt, ai_model).strip()


def get_title(script, ai_model):

    Prompt = f"""
    Your task to write the title (5-10) words of the video using the video script.
    video Script is: {script}
    """

    return generate_response(Prompt, ai_model).strip()
