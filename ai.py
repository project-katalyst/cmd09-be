import os
from dotenv import load_dotenv
import re
import pandas as pd
from typing import Tuple
from pydantic import BaseModel
from openai import OpenAI
from batching import URL_PROMPT, SUMMARY_PROMPT

# Configure your OpenAI API key via environment variable or directly
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pydantic models for parsing GPT outputs
class URLs(BaseModel):
    """Pydantic model for a list of URL strings from GPT output."""
    data: list[str]


class Tags(BaseModel):
    """Pydantic model for hierarchical tags from GPT."""
    data: list[str]


class Text(BaseModel):
    """Pydantic model for a textual summary from GPT."""
    data: str


class BestTags(BaseModel):
    """Pydantic model to pick the best tagset (0 or 1)."""
    data: int


def ask_gpt(context: str, prompt_text: str, output_class: BaseModel, model: str = 'gpt-4', temp: float = 0.1) -> Tuple:
    """
    Sends a chat completion request to OpenAI and parses the response into the given Pydantic model.
    Returns the parsed data along with input and output token counts.
    """
    
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": context}
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_class,
        temperature=temp
    )

    # Parse the JSON response using the provided Pydantic model
    data = response.output_parsed.data

    # input_tokens = response.usage.prompt_tokens
    # output_tokens = response.usage.completion_tokens

    return data


def get_relevant_links(visited_links: list[str], model: str = 'gpt-4o-mini') -> Tuple[list[str], int, int]:
    """
    Filters a list of URLs, returning only those relevant based on GPT's judgment.
    """
    if len(visited_links) > 200:
        visited_links = visited_links[:200]
        print(f'{len(visited_links)} exceeds the maximum number of allowed links (200). Only considering the first 200.')

    context = ';'.join(visited_links)
    response_links = ask_gpt(
        context,
        URL_PROMPT,
        URLs,
        model
    )
    print(f'Links validated: {len(visited_links)} -> {len(response_links)}')
    return response_links

def summarize_company(pages: dict, model: str = 'gpt-4') -> Tuple[str, int, int]:
    """
    Generates a plain-text summary of a company given page contents.
    """
    print('\nAsking GPT to summarize the company...')
    context = ';'.join(f'{link} -> {content}' for link, content in pages.items())
    summary_text = ask_gpt(
        context,
        SUMMARY_PROMPT,
        Text,
        model
    )
    return markdown_to_plaintext(summary_text)


def get_tags(possible_tags: list[str], summary: str, tag_level: int, previous_tags, company_name: str, model: str) -> Tuple[list[str], int, int]:
    """
    Determines hierarchical tags for a company based on its summary.
    """
    print(f'Sending description to GPT to determine tag{tag_level}...')

    prompt = (
        f"I have a summary for a company. Based on these texts, you are going to determine the company's tag{tag_level}. "
        f"Choose from: {possible_tags}. Provide at least one option."
    )

    tags = ask_gpt(
        summary,
        prompt,
        Tags,
        model,
        temp=0.1
    )
    return tags


def check_outputs(iq_tags: 'pd.DataFrame', un_tags: 'pd.DataFrame', summary: str) -> int:
    """
    Chooses the better tag set between two DataFrames based on the company description.
    """
    prompt = (
        "I'll provide two pandas DataFrames and a company's business description. "
        "Pick the better set of tags: 0 for the first, 1 for the second." \
        f"Description: {summary}"
    )
    context = f'[{iq_tags.to_string()}], [{un_tags.to_string()}]'
    best, _, _ = ask_gpt(
        context,
        prompt,
        BestTags,
        model='gpt-4'
    )
    return int(best)


def markdown_to_plaintext(md_text: str) -> str:
    """
    Strips markdown formatting to plain text.
    """
    try:
        text = re.sub(r'\n#+\s*', '\n', md_text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\n[-*+]\s*', '\n', text)
        text = re.sub(r'\n\d+\.\s*', '\n', text)
        return text.strip()
    except TypeError:
        return md_text

if __name__ == '__main__':
    print( ask_gpt(
        'Hello, how are you?',
        'You are a helpful assistant.',
        Text,
        model='gpt-4o-mini'
    ))