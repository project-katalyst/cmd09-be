from concurrent.futures import ThreadPoolExecutor, as_completed
from scrapping import clean_pages, collect_home_links, get_relevant_contents
from datetime import datetime, timezone, timedelta
from typing import Tuple
from nslookup import Nslookup
import pandas as pd
import os
import dotenv
import json
import time
import copy

dotenv.load_dotenv()
from tqdm import tqdm
from openai import OpenAI
import openai

API_KEY = os.environ.get("OPENAI_API_KEY")

CLIENT = OpenAI(api_key=API_KEY)


RESPONSE_FORMATS = {
    "URL": {
        "type": "json_schema",
        "json_schema": {
            "name": "URLs",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "links": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["links"],
                "additionalProperties": False
            }
        }
    },
    "SUMMARY": {
        "type": "json_schema",
        "json_schema": {
            "name": "SummaryResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "has_content": {"type": "boolean"}
                },
                "required": ["summary", "has_content"],
                "additionalProperties": False
            }
        }
    },
    "TAGS": {
        "type": "json_schema",
        "json_schema": {
            "name": "Tags",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["tags"],
                "additionalProperties": False
            }
        }
    }
}

URL_PROMPT = "I have a list of URLs. Based only on the URL text (and not any actual page content), please only return the links you believe contain information about the company, i.e. FAQ, about, quem somos, investors, mission and so on. Remove urls that are product pages from e-comerces or any other that does not help understand the company. Only respond with the final JSON array, no other text. Maximum 20 links"

SUMMARY_PROMPT = "I'll provide you with a list of urls and the corresponding text from its webpage. Based solely on the information provided, please provide a summary of the company.Give me a short business description and their main products/services offered include all services or product categories offered by the company. Make it in english and plain text."


def get_reference_object(reference_file='generic.jsonl'):

    with open(reference_file, "r") as file:
        json_data = json.load(file)

    return json_data


def create_batch_item(id: str, system_message: str, user_message: str, output_schema: dict, reference_object: dict, model='gpt-4o-mini-2024-07-18'):

    json_data = copy.deepcopy(reference_object)

    json_data["custom_id"] = str(id)
    json_data["body"]["model"] = model
    json_data["body"]["messages"][0]["content"] = system_message
    json_data["body"]["messages"][1]["content"] = user_message
    json_data["body"]["response_format"] = output_schema

    return json_data


def compose_batch_file(json_obj: dict, batch_file: str):
    with open(batch_file, "a") as file:
        file.write(json.dumps(json_obj) + "\n")


def upload_file(file_name: str, time_interval=1):

    file = CLIENT.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )

    status = 'pending'
    while status not in ("error", "processed", "canceled"):
        file_response = CLIENT.files.retrieve(file.id)
        status = file_response.status.lower()
        time.sleep(time_interval)
    return file.id


def queue_batch(file_id: str) -> Tuple[str, float]:

    batch_response = CLIENT.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    start_time = time.time()

    batch_id = batch_response.id

    return batch_id, start_time


def check_batch_status(batch_id: str, start_time=0.0, time_interval=7) -> openai.types.Batch:

    status = "validating"
    while status not in ("completed", "failed", "canceled"):
        batch_response = CLIENT.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"\r[{datetime.now(timezone(timedelta(hours=-3))).strftime('%d/%m/%y %H:%M:%S')}] Batch Id: {batch_id},  Status: {status}, Elapsed time {format_time(int(time.time()-start_time))}\t", end='')
        time.sleep(time_interval)
    print('')
    if batch_response.status == "failed":
        print(f"Error code {batch_response.errors}")
        return batch_response
    return batch_response


def get_batch_responses(batch_response: openai.types.Batch) -> list[dict]:
    output_file_id = batch_response.output_file_id
    responses = []
    if not output_file_id:
        output_file_id = batch_response.error_file_id

    if output_file_id:
        file_response = CLIENT.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')

        for raw_response in raw_responses:
            json_response = json.loads(raw_response)
            responses.append(json_response)
    return responses


def batch_cycle(batch_file: str, delete=True):

    file_id = upload_file(batch_file, 1)
    batch_id, start_time = queue_batch(file_id)
    batch_response = check_batch_status(batch_id, start_time)
    if batch_response.status != 'failed':
        responses = get_batch_responses(batch_response)
        if delete:
            os.remove(batch_file)

        return responses
    else:
        return [{}]


def format_time(elapsed: int) -> str:
    hour = elapsed//3600
    minutes = (elapsed - hour*3600)//60
    seconds = (elapsed - hour*3600 - minutes*60)
    return f'{hour:02d}:{minutes:02d}:{seconds:02d}'


def validate_dns(url: str) -> bool:
    dns_query = Nslookup(verbose=False, tcp=False)

    url = url.strip().lower().replace('http://', '').replace('https://', '').replace('/', '')
    
    if not url:
        return False

    try:
        result = dns_query.dns_lookup(url).answer
        valid = len(result) >= 1
        return valid
    except Exception as e:
        print(f'{url} - {e}')
        return False


def process_single_row(idx: int, url: str, ref_obj: dict, validate: bool) -> dict:
    dns_valid = validate_dns(url) if validate else True

    if dns_valid:
        links, destination_url = collect_home_links(url)
    else:
        links = []
        destination_url = 'NOT REACHED'

    row_result = {
        'idx': idx,
        'valid': len(links) > 0,
        'destination_url': destination_url,
        'item': None,
        'links': links
    }

    if links:
        truncated = ';'.join(links[:150])
        item = create_batch_item(
            url, URL_PROMPT, truncated, RESPONSE_FORMATS['URL'], ref_obj)
        row_result['item'] = item

    return row_result


def get_selected_links(links: list[str], validate=True, max_workers=10) -> Tuple[str, pd.DataFrame]:
    ref_obj = get_reference_object()
    batch_file = f'batch_file_{str(datetime.now().timestamp()).replace(".", "_")}.jsonl'
    batch_lines = []
    result_df = pd.DataFrame(
        columns=['site_valid', 'destination_url', 'links'])

    tasks = [
        (idx, link)
        for idx, link in enumerate(links)
    ]

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, idx, url, ref_obj, validate): idx
            for idx, url in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Parallel Processing', ncols=100):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                print(
                    f"Error on row {idx} / Iteration number {idx}: {e}")

    for res in results:
        idx = res['idx']
        result_df.at[idx, 'site_valid'] = res['valid']
        result_df.at[idx, 'destination_url'] = res['destination_url']
        result_df.at[idx, 'links'] = res['links']

        if res['item']:
            batch_lines.append(json.dumps(res['item']))

    with open(batch_file, "w", encoding='utf-8') as bf:
        for line in batch_lines:
            bf.write(line + "\n")

    return batch_file, result_df


def get_level_tags(df: pd.DataFrame, tags_df: pd.DataFrame, level: int, all_tags: dict) -> list[dict]:
    batch_file = f'batch_file_{str(datetime.now().timestamp()).replace(".","_")}.jsonl'
    json_reference = get_reference_object('./generic.jsonl')

    for id, possible_tags in tqdm(all_tags.items(), ncols=180):

        all_tags_for_level = []
        try:
            possible_prev_tags = possible_tags[level-1]
        except KeyError as e:
            print(e)
            break
        summary = df[df['ticker'] == id]['resumo_final'].iloc[0]

        for possible_prev_tag in possible_prev_tags:
            all_tags_for_level += list(tags_df[(tags_df[f'tag{level-1}'] == possible_prev_tag) & ~(
                tags_df[f'tag{level}'].isna())][f'tag{level}'].unique())
        TAGS_PROMPT = f"I have a summary for a company. Based on these texts, you are going to determine the company's tag{level}. To determine the tag{level}, you should choose between these options: {all_tags_for_level}. Choose at least 1 option"

        item = create_batch_item(
            id, TAGS_PROMPT, summary, RESPONSE_FORMATS['TAGS'], json_reference, model='gpt-4o-mini-2024-07-18')
        compose_batch_file(item, batch_file)

    return batch_cycle(batch_file)


def get_level_one_tags(df: pd.DataFrame, tags_df: pd.DataFrame, id_column=3, summary_column=-1) -> list[dict]:
    batch_file = f'batch_file_{str(datetime.now().timestamp()).replace(".","_")}.jsonl'
    json_reference = get_reference_object('./generic.jsonl')

    all_tags = list(tags_df['tag1'].unique())
    TAGS_PROMPT = f"I have a summary for a company. Based on these texts, you are going to determine the company's tag1. To determine the tag1, you should choose between these options: {all_tags}. Choose at least 1 option"

    for row in tqdm(df.itertuples(), ncols=180, total=df.shape[0]):
        id = row[id_column]
        summary = row[summary_column]
        item = create_batch_item(
            id, TAGS_PROMPT, summary, RESPONSE_FORMATS['TAGS'], json_reference, model='gpt-4o-mini-2024-07-18')
        compose_batch_file(item, batch_file)

    return batch_cycle(batch_file)

def batch_get_relevant_content(list_of_dict_of_url_lists: list[dict]) -> dict[str, list[str]]:
    final_results = []
    results = {}
    for urls in tqdm(list_of_dict_of_url_lists, desc="Batch get_relevant_content", ncols=100):
        try:
            content = get_relevant_contents(urls['good_links'], verbose=False)

        except Exception as e:
            content = {}

        results[urls['valor']] = content
    if not results:
        return {' ': ' '}
    return results

def batch_sumirize_company(pages: list[dict], model: str = 'gpt-4o-mini-2024-07-18') -> list[dict]:
    batch_file = f'batch_file_{str(datetime.now().timestamp()).replace(".","_")}.jsonl'
    print(batch_file)
    json_reference = get_reference_object()

    for page in tqdm(pages, ncols=180):
        print(page)
        if not page:
            continue
        page = clean_pages(page)
        context = ';'.join(f'{link} -> {content}' for link, content in page.items())
        item = create_batch_item(
            page.get('id', 'unknown'), SUMMARY_PROMPT, context, RESPONSE_FORMATS['SUMMARY'], json_reference, model=model)
        compose_batch_file(item, batch_file)

    print(f"Batch file created: {batch_file}")
    print("Starting batch processing...")
    return batch_cycle(batch_file, delete=False)