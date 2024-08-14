import requests

# Define the order of importance
importance_order = ["Miscellaneous", "Person", "Organization", "Location"]
NUM_ENTITIES = 2

# Sample test data
test_data = [
    {},  # No options
    {"Person": "Gordon Ramsay"},  # One item
    {"Person": "Elon Musk", "Miscellaneous": "Covid-19", "Organization": "Tesla", "Location": "Mars"}  # Four items
]

def get_top_entities(entities):
    sorted_entities = sorted(entities.items(), key=lambda item: importance_order.index(item[0]))
    return sorted_entities[:NUM_ENTITIES]

def search_wikipedia(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "namespace": 0,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if len(data) > 3 and len(data[2]) > 0:
        return data[2][0], data[3][0]  # Return the first snippet and link
    else:
        return "No information found", None

def get_article_content(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|description",
        "explaintext": True,
        "redirects": 1,
        "format": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    if pages:
        for page_id, page_data in pages.items():
            extract = page_data.get("extract", "")
            description = page_data.get("description", "")
            return description, extract
    return "No description found", "No article content found"

def get_wikipedia_info(entities):
    top_entities = get_top_entities(entities)
    info = {}
    for entity_type, entity_value in top_entities:
        snippet, link = search_wikipedia(entity_value)
        if link:
            article_title = link.split("/")[-1].replace('_', ' ')
            description, article_content = get_article_content(article_title)
            info[entity_value] = {
                "snippet": snippet,
                "link": link,
                "description": description,
                "full_article": article_content
            }
        else:
            info[entity_value] = {
                "snippet": snippet,
                "link": None,
                "description": None,
                "full_article": None
            }
    return info

# Process each test data set
for index, data in enumerate(test_data):
    print(f"Test Set {index + 1}:")
    if not data:
        print("No entities to search.")
    else:
        info = get_wikipedia_info(data)
        for entity, details in info.items():
            print(f"{entity}: {details['snippet']}")
            if details['link']:
                print(f"Link: {details['link']}")
                print(f"Description: {details['description']}")
                print(f"Full article (first 200 chars): {details['full_article'][:200]}")
            print(f"Full article length: {len(details['full_article'])} characters")
    print("\n" + "-"*40 + "\n")
