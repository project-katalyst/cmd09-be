import httpx
from bs4 import BeautifulSoup
from bs4.element import Comment
from typing import Tuple
from urllib.parse import urlparse, urljoin, urlunparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


# PROXY_API_KEY = 'sWIVIO6Qbw4k3xLqvi5m2aJhkXbhEQaP9sVo24GR'
TIMEOUT = 90
# PROXY_API_KEY = os.getenv('PROXY_API_KEY', 'fallback_or_error')


def fetch_html_source(url: str, timeout=10):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        )
    }

    with httpx.Client(timeout=timeout, headers=headers, verify=False) as client:
        try:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()

            # Check if the content is a PDF
            if 'application/pdf' in response.headers.get('Content-Type', ''):
                return BeautifulSoup('', "html.parser"), str(response.url)

            # Parse the content with BeautifulSoup
            return BeautifulSoup(response.content, "html.parser", from_encoding="iso-8859-1"), str(response.url)

        except httpx.HTTPError as e:
            if '403' not in e.args[0]:
                print(f"Error fetching {url}: {e}")
            return BeautifulSoup('', "html.parser"), str(url)

    # try:
    #     response = httpx.get(url, headers=headers, timeout=timeout)
    #     response.raise_for_status()

    #     return BeautifulSoup(response.content, "html.parser", from_encoding="iso-8859-1"), response.url

    # except httpx.HTTPError as e:
    #     print(f"Error fetching {url}: {e}")
    #     return BeautifulSoup('', "html.parser"), url


# def fetch_with_proxy(url: str, timeout=TIMEOUT, residencial=False) -> Tuple[BeautifulSoup, str]:
#     # print('\rFetching with proxy...', end='')
#     endpoint = 'https://xfzo864hcl.execute-api.sa-east-1.amazonaws.com/Prod/brazil-ip'
#     headers = {
#         "x-api-key": PROXY_API_KEY,
#         "x-url": url,
#         "x-use-residential-proxy": 'true' if residencial else 'false'
#     }
#     try:
#         response = requests.get(endpoint, headers=headers, timeout=timeout)
#         response.raise_for_status()

#         destination_url = response.json()['finalUrl']
#         content = response.json()['htmlContent']

#         if 'application/pdf' in response.headers.get('Content-Type', ''):
#             return BeautifulSoup('', "html.parser"), destination_url

#         return BeautifulSoup(content, "html.parser"), destination_url

#     except requests.exceptions.RequestException:
#         if not residencial:
#             return fetch_with_proxy(url, residencial=True)
#         else:
#             return BeautifulSoup('', "html.parser"), url


# def fetch_with_js(url: str, timeout=TIMEOUT, residencial=False) -> Tuple[BeautifulSoup, str]:
#     endpoint = 'https://xfzo864hcl.execute-api.sa-east-1.amazonaws.com/Prod/process-js'
#     headers = {
#         "x-api-key": PROXY_API_KEY,
#         "x-url": url,
#         "x-use-residential-proxy": "true" if residencial else 'false'
#     }

#     try:
#         response = requests.get(endpoint, headers=headers, timeout=timeout)
#         response.raise_for_status()

#         destination_url = response.json()['finalUrl']
#         content = response.json()['htmlContent']

#         if 'application/pdf' in response.headers.get('Content-Type', ''):
#             return BeautifulSoup('', "html.parser"), destination_url

#         return BeautifulSoup(content, "html.parser", from_encoding="iso-8859-1"), destination_url

#     except requests.exceptions.RequestException:
#         if not residencial:
#             return fetch_with_js(url, residencial=True)
#         else:
#             return BeautifulSoup('', "html.parser"), url


# def fetch(url: str) -> Tuple[bool, bool, BeautifulSoup, str]:

#     soup, need_proxy, destination_url = fetch_html_source(url)

#     if need_proxy:
#         soup, destination_url = fetch_with_proxy(url)

#     if len(soup) == 0:
#         return False, False, soup, url

#     text = extract_visible_text(soup)
#     minimal_content = len(text) < 300

#     placeholder_flag = any(
#         placeholder in str(soup)
#         for placeholder in ['<div id="root">', '<div id="app">', '<div id="main">']
#     )

#     is_js_site = minimal_content or placeholder_flag

#     if is_js_site:
#         soup, destination_url = fetch_with_js(url)

#     return is_js_site, need_proxy, soup, destination_url


def fetch_content(link: str, timeout=TIMEOUT) -> str:

    # if not requires_js and used_proxy:
    #     soup, _ = fetch_with_proxy(link, timeout=timeout)

    # elif not requires_js and not used_proxy:
    soup, _ = fetch_html_source(link, timeout=timeout)

    # else:
    #     soup, _ = fetch_with_js(link, timeout=timeout)

    text_content = extract_visible_text(soup) if (len(soup) != 0) else ''

    return text_content


def remove_duplicate_slashes_from_path(url: str) -> str:
    p = urlparse(url)
    new_path = re.sub(r'/+', '/', p.path)
    return urlunparse((p.scheme, p.netloc, new_path, p.params, p.query, p.fragment))


def remove_suffix(url: str) -> str:
    no_parameter = re.sub(r'\?.*$', '', url)
    no_hash = re.sub(r'#.*$', '', no_parameter)
    return re.sub(r'#.*$', '', no_hash)


def normalize_url(base_url: str, link: str) -> str:
    link = "https:" + link if link.startswith("//") else link
    joined = urljoin(base_url, link)
    no_hash = remove_suffix(joined)
    normalized = remove_duplicate_slashes_from_path(no_hash)
    return normalized


def remove_unwanted_links(domain: str, links: list[str]) -> list[str]:
    return [
        link for link in links
        if not link.lower().endswith((".pdf", ".jpg", ".png", ".jpeg", '.mp4', '.mp3', '.md', ".xml"))
        and domain in urlparse(link).netloc.lower()
    ]


def tag_visible(element) -> bool:
    if element.parent.name in ('style', 'script', 'head', 'title', 'meta', '[document]'):
        return False
    if isinstance(element, Comment):
        return False
    return True


def extract_visible_text(soup: BeautifulSoup) -> str:
    texts = soup.find_all(string=True)
    visible_texts = filter(tag_visible, texts)
    combined = " ".join(t.strip() for t in visible_texts)
    return re.sub(r"\[/?li\]|\n", " ", combined).strip()


def collect_home_links(url: str) -> Tuple[list[str], str]:
    visited = set()
    visited.add(url)

    soup, destination_url = fetch_html_source(url)
    visited.add(destination_url)

    base_domain = urlparse(destination_url).netloc

    if len(soup) == 0:
        return [], url

    new_links = [a["href"] for a in soup.select("a[href]") if a["href"]]
    new_links = [normalize_url(url, link)
                 for link in new_links if type(link) == str]
    new_links = remove_unwanted_links(base_domain, new_links)
    new_links = list(set(new_links))
    for link in new_links:
        visited.add(link)

    return list(visited), destination_url


def get_relevant_contents(relevant_links: list[str], verbose=True, timeout=TIMEOUT) -> dict[str, str]:
    relevant_contents = {}
    if len(relevant_links) == 0:
        return relevant_contents

    def fetch_content_wrapper(link):
        try:
            return link, fetch_content(link, timeout)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            print(f'Encoding error on link: {link} - {e}')
            return link, ""

    # Limit to 20 threads for efficiency
    max_threads = min(10, len(relevant_links))

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_url = {executor.submit(
            fetch_content_wrapper, link): link for link in relevant_links[:80]}  # Limit to 80 links
        for future in as_completed(future_to_url):
            link, content = future.result()
            if verbose:
                print(f"\rFetched content from: {link} {' '*50}", end="")
            relevant_contents[link] = content

    return relevant_contents


def common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""

    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def common_suffix(strings: list[str]) -> str:
    if not strings:
        return ""

    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""
    return suffix


def clean_pages(pages_content: dict) -> dict[str, str]:
    if len(pages_content) == 0:
        return {' ': ' '}
    prefix = common_prefix(list(pages_content.values()))
    suffix = common_suffix(list(pages_content.values()))
    for link, content in pages_content.items():
        pages_content[link] = content[len(prefix): len(content) - len(suffix)]
    pages_content[list(pages_content.keys())[0]] = prefix + \
        pages_content[list(pages_content.keys())[0]] + suffix

    return pages_content
