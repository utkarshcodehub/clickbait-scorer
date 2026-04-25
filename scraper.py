import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_toi():
    """Scrape Times of India headlines - typically clickbaity"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    headlines = []
    
    urls = [
        'https://timesofindia.indiatimes.com/india',
        'https://timesofindia.indiatimes.com/entertainment',
        'https://timesofindia.indiatimes.com/sports',
    ]
    
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            # TOI headline selectors
            for tag in soup.find_all(['h2', 'h3'], class_=lambda c: c and ('heading' in c.lower() or 'title' in c.lower())):
                text = tag.get_text(strip=True)
                if 20 < len(text) < 200:
                    headlines.append({'headline': text, 'source': 'TOI', 'label': 1})
            time.sleep(1)
        except Exception as e:
            print(f"TOI scrape failed: {e}")
    
    return headlines

def scrape_the_wire():
    """Scrape The Wire - typically factual"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    headlines = []
    
    try:
        r = requests.get('https://thewire.in', headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup.find_all(['h2', 'h3', 'h4']):
            text = tag.get_text(strip=True)
            if 20 < len(text) < 200:
                headlines.append({'headline': text, 'source': 'TheWire', 'label': 0})
    except Exception as e:
        print(f"Wire scrape failed: {e}")
    
    return headlines

def scrape_ndtv():
    """NDTV - mix of both"""
    headers = {'User-Agent': 'Mozilla/5.0'}
    headlines = []
    try:
        r = requests.get('https://www.ndtv.com', headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        for tag in soup.find_all(['h2', 'h3']):
            text = tag.get_text(strip=True)
            if 20 < len(text) < 200:
                headlines.append({'headline': text, 'source': 'NDTV', 'label': 1})
    except Exception as e:
        print(f"NDTV scrape failed: {e}")
    return headlines

all_headlines = scrape_toi() + scrape_the_wire() + scrape_ndtv()
df = pd.DataFrame(all_headlines).drop_duplicates('headline') if all_headlines else pd.DataFrame()

if df.empty:
    print("No headlines scraped (sites blocked bots). Skipping — seed data will be used for training.")
else:
    df.to_csv('headlines_raw.csv', index=False)
    print(f"Scraped {len(df)} headlines")
    print(df['source'].value_counts())