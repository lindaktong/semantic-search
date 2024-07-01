def fetch_and_process_pages(links):
    embeddings_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for url in links:
            print(f"Processing URL: {url}")
            embedding = process_link(url)
            if embedding is not None:
                embeddings_list.append(embedding)
    if not embeddings_list:
        return None
    embeddings_array = np.array(embeddings_list)
    # average_embedding = np.mean(embeddings_array, axis=0)
    # normalized_average_embedding = average_embedding / np.linalg.norm(average_embedding)
    return embeddings_array