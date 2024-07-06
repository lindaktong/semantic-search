import csv
# import umap.umap_ as umap
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from requests.exceptions import HTTPError, ConnectionError, Timeout
from openai import OpenAI
from openai import AsyncOpenAI
import tiktoken
from itertools import islice
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import time

import requests
from bs4 import BeautifulSoup
import math
import concurrent.futures

import asyncio
import logging
import re
import sys
from typing import IO
import urllib.error
import urllib.parse

import aiofiles
import aiohttp
from aiohttp import ClientSession

import certifi
import ssl
import chardet

ssl_context = ssl.create_default_context(cafile=certifi.where())

# ---------- OpenAI stuff ----------

client = AsyncOpenAI()

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# ---------- Embedding links ----------

# Groups tokens into chunks
def chunk_tokens(text, encoding_name, chunk_length):
    chunked_tokens = []
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    num_chunks = math.ceil(len(tokens) / chunk_length)
    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, len(text))
        chunk = tokens[start:end]
        chunked_tokens.append(chunk)
    return chunked_tokens

# Get safe embedding
async def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunked_tokens = chunk_tokens(text, encoding_name=EMBEDDING_ENCODING, chunk_length=EMBEDDING_CTX_LENGTH)
    # this takes a long time to come back, just like requests.get -- can run other things while waiting for this!
    chunk_embeddings = (await client.embeddings.create(input=chunked_tokens, model=EMBEDDING_MODEL)).data
    chunk_embeddings = [chunk.embedding for chunk in chunk_embeddings]
    chunk_lens = [len(chunk) for chunk in chunked_tokens]
    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

# @retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type((ConnectionError, Timeout)))
async def robust_get(url, session: ClientSession):
    return await session.request(method="GET", url=url, ssl=ssl_context)

# async def process_link(url, session: ClientSession):
#     print(f"Processing URL: {url}")
#     try:
#         link_response = await robust_get(url, session)
#         link_response.raise_for_status()
#         # if link_response.status_code == 200:
#         html = await link_response.text()
#         soup = BeautifulSoup(html, 'html.parser')
#         text = ' '.join(soup.stripped_strings)
#         if text:
#             return await len_safe_get_embedding(text, model="text-embedding-3-small")
#     except Exception as e:
#         print(f"Failed to process URL {url}: {e}")
#     return None

async def process_link(url, session: ClientSession):
    print(f"Processing URL: {url}")
    try:
        link_response = await robust_get(url, session)
        link_response.raise_for_status()
        # Read the raw content
        raw_content = await link_response.read()
        # Detect encoding
        result = chardet.detect(raw_content)
        encoding = result['encoding']
        # Decode the content using the detected encoding
        html = raw_content.decode(encoding, errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
        text = ' '.join(soup.stripped_strings)
        if text:
            return await len_safe_get_embedding(text, model="text-embedding-3-small")
    except Exception as e:
        print(f"Failed to process URL {url}: {e}")
    return None

# for openai
async def fetch_and_process_pages(links):
    async with ClientSession() as session:
        tasks = []
        for url in links:
            tasks.append(
                process_link(url, session)
            )
        embeddings_list = await asyncio.gather(*tasks)
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     embeddings_list = executor.map(await process_link, links)
    embeddings_list = [embedding for embedding in embeddings_list if embedding]
    embeddings_array = np.array(embeddings_list)
    # average_embedding = np.mean(embeddings_array, axis=0)
    # normalized_average_embedding = average_embedding / np.linalg.norm(average_embedding)
    return embeddings_array

def main():

    # URL of the page
    url = "https://www.paulgraham.com/articles.html"

    # Fetch the content of the page
    response = requests.get(url)
    html_content = response.content

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all links to blog posts
    # Assuming the blog post links are in <a> tags within <td> tags
    links = soup.find_all('a', href=True)

    # Extract and print the URLs
    blog_post_urls = []

    for link in links:
        href = link['href']
        # Check if the link is a relative URL, then prepend the base URL
        if not href.startswith('http'):
            href = url.rsplit('/', 1)[0] + '/' + href
        blog_post_urls.append(href)

    links = []

    # Save the blog post URLs
    for post_url in blog_post_urls:
        links.append(post_url)

    t0 = time.time()
    pg_embeddings = asyncio.run(fetch_and_process_pages(links))
    t1 = time.time()
    print(t1-t0)

if __name__ == "__main__":
    main()