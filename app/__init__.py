import requests
from bs4 import BeautifulSoup
import json
import re

def fetch_website_data(url, user_agent):
    response = requests.get(url, headers={'User-Agent': user_agent})
    if response.status_code == 200:
        return response.content.decode('utf-8')
    else:
        print(f'Failed to retrieve the webpage. Status code: {response.status_code}')
        return None

def parse_states_data(content):
    lines = content.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Skip the first two lines which are headers
    lines = lines[2:]

    state_data = []

    for line in lines:
        # Split by multiple spaces or tabs
        parts = re.split(r'\s{2,}|\t', line)
        
        if len(parts) == 6:  # Two states in one line
            state_data.append({
                "State": parts[0].strip(),
                "Postal": parts[1].strip(),
                "FIPS": parts[2].strip()
            })
            state_data.append({
                "State": parts[3].strip(),
                "Postal": parts[4].strip(),
                "FIPS": parts[5].strip()
            })
        elif len(parts) == 3:  # One state in one line
            state_data.append({
                "State": parts[0].strip(),
                "Postal": parts[1].strip(),
                "FIPS": parts[2].strip()
            })
        else:
            print(f"Unexpected format in line: {line}")

    return state_data

def main():
    URL = 'https://pastebin.com/raw/G0VH1LpS'
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    
    content = fetch_website_data(URL, USER_AGENT)

    if content:
        state_data = parse_states_data(content)

        # Prepare the data to be stored in JSON
        website_data = {
           
            'states info': state_data
        }

        with open('states_data.json', 'w', encoding='utf-8') as json_file:
            json.dump(website_data, json_file, ensure_ascii=False, indent=4)
        print('Website data has been saved to states_data.json')

if __name__ == '__main__':
    main()
