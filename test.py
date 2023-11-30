# from googleapiclient.discovery import build
# # def google_search(query):
# #     api_key = "AIzaSyB_uFeOOa8GWykvF6SLPbnox4D1LMfxyIk"
# #     service = build("customsearch", "v1", developerKey=api_key)
# #     cse_id = "c21943deeeb464fb6"
# #     query = query

# #     result = service.cse().list(q=query, cx=cse_id).execute()
# #     Snippets = ""
# #     for item in result.get("items", []):
# #         Snippets = Snippets + (item['snippet'])

# #     return "###" + Snippets

# # from googleapiclient.discovery import build
# # google_search("hello")

# def google_search(query):
#     # api_key = "AIzaSyDfUcyZawS98_T3GN7dGC8H0tgG80l-7Kg"
#     # service = build("customsearch", "v1", developerKey=api_key)
#     # cse_id = "b29dee810a6a84bbf"

#     api_key = "AIzaSyB_uFeOOa8GWykvF6SLPbnox4D1LMfxyIk"
#     service = build("customsearch", "v1", developerKey=api_key)
#     cse_id = "c21943deeeb464fb6"    
#     print(query)
#     query = query

#     result = service.cse().list(q=query, cx=cse_id).execute()
#     Snippets = ""
#     for item in result.get("items", []):
#         Snippets = Snippets + (item['snippet'])

#     return "###" + Snippets

# google_search('apple has iron')

# import requests

# response = requests.get('https://www.youtube.com/')

# if response.status_code == 200:
#     print("成功连接到 youtube.com")
# else:
#     print("连接到 youtube.com，但响应码不是200，响应码为:", response.status_code)

import requests

def extract_search_results(data):
    """
    Extract search results from the API response data.

    :param data: The JSON data from the API response.
    :return: A list of search results.
    """
    # example
    # search_results = []
    # if 'items' in data:
    #     for item in data['items']:
    #         title = item.get('title')
    #         link = item.get('link')
    #         snippet = item.get('snippet')
    #         search_results.append({'title': title, 'link': link, 'snippet': snippet})
    Snippets = ""    
    if 'items' in data:
        for item in data['items']:
            snippet = item.get('snippet')    
            Snippets = Snippets + snippet
    return "###" + Snippets 

    # return search_results

def google_search(query, api_key, cse_id):
    """
    Perform a search using the Google Custom Search JSON API.

    :param query: Search query string.
    :param api_key: Your Google API key.
    :param cse_id: Your custom search engine ID.
    :return: The search results.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id
    }
    # params.update(kwargs)
    response = requests.get(search_url, params=params)
    results = response.content

    data = response.json()
    search_results = extract_search_results(data)
    return search_results

api_key = "AIzaSyDfUcyZawS98_T3GN7dGC8H0tgG80l-7Kg"
cse_id = "b29dee810a6a84bbf"
print(google_search('apple has iron', api_key, cse_id))

# Replace 'YOUR_API_KEY' and 'YOUR_CSE_ID' with

