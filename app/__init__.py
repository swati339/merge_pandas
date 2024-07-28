
import requests
from bs4 import BeautifulSoup
from app.parser import parse_states_data

def main():
    URL = 'https://pastebin.com/raw/G0VH1LpS'
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

    response = requests.get(URL, headers={'User-Agent': USER_AGENT})

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title_tag = soup.find('title')
        
        if title_tag:
            title = title_tag.get_text()
            print(f'Title: {title}')
        else:
            print('Title tag not found in the HTML content.')
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')

    data = parse_states_data('states_data.txt')
    for entry in data:
        print(entry)

if __name__ == '__main__':
    main()
