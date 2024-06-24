import os
from tavily import TavilyClient
from typing import Annotated
from dotenv import load_dotenv
from langchain.adapters.openai import convert_openai_messages
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv("../.env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

client = TavilyClient(api_key=TAVILY_API_KEY)


def web_search_tool(
    query: Annotated[str, "The search query"]
) -> Annotated[str, "The search results"]:
    return client.get_search_context(query=query, search_depth="advanced")


def search_tool(
    query: Annotated[str, "The search query"]
) -> Annotated[str, "The search results"]:
    content = client.search(query=query, search_depth="advanced")["results"]
    prompt = [
        {
            "role": "system",
            "content": f"You are an AI critical thinker research assistant. "
            f"Your sole purpose is to write well written, critically acclaimed,"
            f"It must include the thoughts of the industry leaders.It is a mandatory in the report"
            f"objective and structured reports on given text."
            f"The report should contain the economic and market growth impact."
            f"The report should be detailed and it should mention the source of the information."
            f"The report should be extremely informative."
            f"Always mention the examples related to the facts.",
        },
        {
            "role": "user",
            "content": f'Information: """{content}"""\n\n'
            f"Using the above information, answer the following"
            f'query: "{query}" in a detailed report do not include conclusion and references and summary --',
        },
    ]
    lc_messages = convert_openai_messages(prompt)
    return (
        ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"])
        .invoke(lc_messages)
        .content
    )
