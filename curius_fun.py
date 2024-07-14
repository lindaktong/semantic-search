import json
import asyncio
from aiohttp import ClientSession
import certifi
import ssl
from openai import AsyncOpenAI
import tiktoken
import math
import numpy as np
import chardet
from bs4 import BeautifulSoup

ssl_context = ssl.create_default_context(cafile=certifi.where())

# ---------- OpenAI stuff ----------

client = AsyncOpenAI()

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

#---------------------------------------------------------------

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
    embeddings_list = [embedding for embedding in embeddings_list if embedding]
    return embeddings_list

#---------------------------------------------------------------

async def robust_get(url, session: ClientSession):
    return await session.request(method="GET", url=url, ssl=ssl_context)

async def process_all_users(ids, big_dict):
    # grab all the article ids and urls - parallelize this over pages and parallelize this over users
    async with ClientSession() as session:
        tasks = []
        for id in ids:
            base_url = f"https://curius.app/api/users/{id}/searchLinks"
            tasks.append(process_user(id, base_url, big_dict, session))
        await asyncio.gather(*tasks)

async def process_user(id, base_url, big_dict, session):
    print(f"Processing user: {id}")
    link_response = await robust_get(base_url, session)
    data = await link_response.json()
    if data['links']:  
        for item in data['links']:
            article_id = item['id']
            link_url = item['link']
            big_dict[article_id] = {}
            big_dict[article_id]["link"] = link_url

#---------------------------------------------------------------

def main():

    # Read the data from a text file
    with open('sorted_users.txt', 'r') as file:
        data = file.read()

    # Convert the string to a dictionary
    data_dict = json.loads(data)
    users_list = data_dict.get('users', [])

    # Extract all the ids
    ids = [user['id'] for user in users_list]

    # Get all the links
    big_dict = {}
    asyncio.run(process_all_users(ids, big_dict))
    big_dict = dict(sorted(big_dict.items(), key=lambda item: int(item[0])))
    json.dump(big_dict, open('links.json', 'w'))

    print("------------------Done retrieving links--------------------")

    # # Embed everything
    # urls_list = [item["link"] for item in big_dict.values()]
    # embeddings_list = asyncio.run(fetch_and_process_pages(urls_list))
    # # Map the elements of the embeddings list to the nested dictionary
    # for key, embedding in zip(big_dict.keys(), embeddings_list):
    #     big_dict[key]["embedding"] = embedding

    print("----------------------Done embedding----------------------")

if __name__ == "__main__":
    main()



    


