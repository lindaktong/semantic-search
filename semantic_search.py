import csv
import umap.umap_ as umap
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from requests.exceptions import HTTPError, ConnectionError, Timeout
from openai import OpenAI
import tiktoken
from itertools import islice
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

import requests
from bs4 import BeautifulSoup
import math
import concurrent.futures

# ---------- OpenAI stuff ----------

client = OpenAI()

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# ---------- Populate links ----------

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
def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunked_tokens = chunk_tokens(text, encoding_name=EMBEDDING_ENCODING, chunk_length=EMBEDDING_CTX_LENGTH)
    chunk_embeddings = client.embeddings.create(input=chunked_tokens, model=EMBEDDING_MODEL).data
    chunk_embeddings = [chunk.embedding for chunk in chunk_embeddings]
    chunk_lens = [len(chunk) for chunk in chunked_tokens]
    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type((ConnectionError, Timeout)))
def robust_get(url):
    return requests.get(url)

def process_link(url):
    print(f"Processing URL: {url}")
    try:
        link_response = robust_get(url)
        if link_response.status_code == 200:
            html = link_response.text
            soup = BeautifulSoup(html, 'html.parser')
            text = ' '.join(soup.stripped_strings)
            if text:
                return len_safe_get_embedding(text, model="text-embedding-3-small")
    except Exception as e:
        print(f"Failed to process URL {url}: {e}")
    return None

# for openai
def fetch_and_process_pages(links):
    embeddings_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embeddings_list = executor.map(process_link, links)
        embeddings_list = [embedding for embedding in embeddings_list if embedding]
    embeddings_array = np.array(embeddings_list)
    # average_embedding = np.mean(embeddings_array, axis=0)
    # normalized_average_embedding = average_embedding / np.linalg.norm(average_embedding)
    return embeddings_array

pg_embeddings = fetch_and_process_pages(links)