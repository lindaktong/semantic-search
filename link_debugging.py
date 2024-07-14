import json
import asyncio
from aiohttp import ClientSession, ClientConnectorError
import certifi
import ssl
from openai import AsyncOpenAI
import tiktoken
import math
import numpy as np
import chardet
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

ssl_context = ssl.create_default_context(cafile=certifi.where())

# ---------- OpenAI stuff ----------
client = AsyncOpenAI()

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

#---------------------------------------------------------------

async def robust_get(url, session: ClientSession, retries=3, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
        "DNT": "1",  # Do Not Track Request Header
        "Upgrade-Insecure-Requests": "1"
    }
    for attempt in range(retries):
        try:
            response = await session.request(method="GET", url=url, ssl=ssl_context, headers=headers)
            response.raise_for_status()
            return response
        except (ClientConnectorError, asyncio.TimeoutError) as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"All {retries} attempts failed. Giving up.")
                raise

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

async def process_link(url, session: ClientSession):
    print(f"Embedding URL: {url}")
    try:
        link_response = await robust_get(url, session)
        link_response.raise_for_status()
        # Read the raw content
        raw_content = await link_response.read()
        # Detect content type
        content_type = link_response.headers.get('Content-Type', '').lower()
        
        if 'pdf' in content_type:
            # Process PDF
            pdf_document = fitz.open(stream=raw_content, filetype="pdf")
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
        else:
            # Process HTML
            result = chardet.detect(raw_content)
            encoding = result['encoding']
            if not encoding:
                raise ValueError("Failed to detect encoding")
            html = raw_content.decode(encoding, errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            text = ' '.join(soup.stripped_strings)
        
        if text:
            print(f"URL okay: {url}")
            return 1
            # return await len_safe_get_embedding(text, model="text-embedding-3-small")
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
    #     embeddings_list = await asyncio.gather(*tasks)
        full_list = await asyncio.gather(*tasks)
        list_of_nones = [val for val in full_list if val == None]
        print(len(list_of_nones)/len(full_list))
    # embeddings_list = [embedding for embedding in embeddings_list if embedding]
    # return embeddings_list


#---------------------------------------------------------------

def main():

    # Specify the path to your JSON file
    file_path = 'links.json'

    # Open the JSON file and load its contents into a dictionary
    with open(file_path, 'r') as file:
        data_dict = json.load(file)

    # Now data_dict contains the JSON data as a dictionary
    urls_list = [item["link"] for item in data_dict.values()]
    print(len(urls_list))

    embeddings_list = asyncio.run(fetch_and_process_pages(urls_list[100000:100200]))
    # Map the elements of the embeddings list to the nested dictionary
    # for key, embedding in zip(big_dict.keys(), embeddings_list):
    #     big_dict[key]["embedding"] = embedding

if __name__ == "__main__":
    main()