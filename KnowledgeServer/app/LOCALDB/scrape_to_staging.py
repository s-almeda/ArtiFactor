#!/usr/bin/env python3
"""
Automated WikiArt Scraper to Staging JSON

This script automates the process of scraping artists and artworks from WikiArt,
preparing the data for staging review before database updates.

Usage:
    python scrape_to_staging.py --download true --limit 50
    python scrape_to_staging.py --download false --limit 100
"""

import argparse
import json
import os
import re
#using sqlean to make using extensions easy,,
import sqlite_vec
import sqlean as sqlite3

import sys
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

from typing import Dict, List, Optional, Tuple


# Get the absolute path to the current directory (which should be in LOCALDB, same place as the images and .db file))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database and image paths
DB_PATH = os.path.join(BASE_DIR, "knowledgebase.db")
IMAGES_PATH = os.path.join(BASE_DIR, "images")
STAGING_PATH = os.path.join(BASE_DIR, "staging")


# --- UTILITY FUNCTIONS. --- #

# Get a database connection with sqlite-vec loaded
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn




def normalize_text(text, for_slug=False):
    """Normalize text for slugs or fuzzy matching."""
    if not text:
        return ""
    text = text.lower().strip()
    accents = {
        'à': 'a', 'á': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a', 'å': 'a', 'ā': 'a',
        'è': 'e', 'é': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
        'ì': 'i', 'í': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
        'ò': 'o', 'ó': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ø': 'o', 'ō': 'o',
        'ù': 'u', 'ú': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
        'ñ': 'n', 'ç': 'c', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'ą': 'a', 'ę': 'e', 'ł': 'l', 'ń': 'n', 'š': 's', 'č': 'c', 'ř': 'r',
        'ð': 'd', 'þ': 'th', 'ß': 'ss'
    }
    for accent, replacement in accents.items():
        text = text.replace(accent, replacement)
    for old, new in {'“': '"', '”': '"', '‘': "'", '’': "'", '–': '-', '—': '-'}.items():
        text = text.replace(old, new)
    if for_slug:
        text = text.replace(' ', '-')
        text = re.sub(r'[^a-z0-9-]', '', text)
        text = re.sub(r'-+', '-', text)
        return text.strip('-')
    return text

slugify = lambda name: normalize_text(name, for_slug=True)



class LimitReachedException(Exception):
    """Custom exception to signal scraping limit reached"""
    pass

class WikiArtScraper:
    def __init__(self, download: bool = False, limit: int = 5, api_base_url: str = "http://localhost:5000"):
        self.download = download
        self.limit = limit
        self.api_base_url = api_base_url.rstrip('/')
        
        # Session for WikiArt scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Session for API calls
        self.api_session = requests.Session()
        self.api_session.headers.update({
            'Content-Type': 'application/json'
        })
        
        # Initialize paths
        self.db_path = DB_PATH
        self.images_dir = IMAGES_PATH
        self.staging_dir = STAGING_PATH
        
        # Create directories if they don't exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.staging_dir, exist_ok=True)
        
        # Initialize database connection
        self.db = get_db_connection()
        
        # Load existing database data
        self.existing_artists = self.load_existing_artists()
        self.existing_keywords = self.load_existing_keywords()
        self.existing_artworks = self.load_existing_artworks()
        
        # Initialize staging data structure
        self.staging_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "download_enabled": self.download,
                "limit": self.limit,
                "total_processed": 0,
                "total_artists": 0,
                "total_artworks": 0
            },
            "artists": [],
            "artworks": []
        }
        

        
        self.new_artists_count = 0
        self.new_artworks_count = 0
    
    def get_database_operations_count(self) -> int:
        """Get the total number of database operations that would be performed"""
        return self.new_artists_count + self.new_artworks_count
    
    def would_exceed_limit(self, additional_artists: int = 0, additional_artworks: int = 0) -> bool:
        """Check if adding items would exceed the limit"""
        total_operations = self.get_database_operations_count() + additional_artists + additional_artworks
        return total_operations >= self.limit
    
    def load_existing_artists(self) -> Dict:
        """Load existing artists from database"""
        existing = {}
        try:
            cursor = self.db.execute('''
                SELECT entry_id, value, relatedKeywordIds, relatedKeywordStrings 
                FROM text_entries 
                WHERE isArtist = 1
            ''')
            for row in cursor.fetchall():
                existing[row[1]] = {  # Use artist name as key
                    'entry_id': row[0],
                    'name': row[1],
                    'relatedKeywordIds': json.loads(row[2]) if row[2] else [],
                    'relatedKeywordStrings': json.loads(row[3]) if row[3] else []
                }
        except Exception as e:
            print(f"Error loading existing artists: {e}")
        return existing

    def load_existing_keywords(self) -> Dict:
        """Load existing keywords from database (non-artist entries)"""
        keywords = {}
        try:
            cursor = self.db.execute('SELECT entry_id, value FROM text_entries WHERE isArtist = 0')
            for row in cursor.fetchall():
                keywords[row[1]] = row[0]  # keyword_text: entry_id
        except Exception as e:
            print(f"Error loading existing keywords: {e}")
        return keywords

    def normalize_artwork_title(self, title: str) -> str:
        return normalize_text(title)

    def load_existing_artworks(self) -> Dict:
        """Load existing artworks from database"""
        artworks = {}
        try:
            cursor = self.db.execute('''
                SELECT image_id, value, artist_names, filename
                FROM image_entries
            ''')
            for row in cursor.fetchall():
                title = row[1]  # artwork title from 'value' column
                normalized_title = self.normalize_artwork_title(title)
                
                # Store both normalized and original title for better matching
                artwork_data = {
                    'image_id': row[0],
                    'title': title,
                    'normalized_title': normalized_title,
                    'artist_names': json.loads(row[2]) if row[2] else [],
                    'filename': row[3]
                }
                
                # Use normalized title as key, but also store original
                artworks[normalized_title] = artwork_data
                
                # Also store under original title if different from normalized
                if title != normalized_title:
                    artworks[title] = artwork_data
                    
        except Exception as e:
            print(f"Error loading existing artworks: {e}")
        return artworks

    def find_existing_artwork(self, artwork_title: str, artist_name: str) -> Optional[Dict]:
        """Find existing artwork with fuzzy matching, requiring both title and artist match"""
        if not artwork_title or not artist_name:
            return None

        normalized_title = self.normalize_artwork_title(artwork_title)
        normalized_artist = normalize_text(artist_name)

        # Try exact match first
        for artwork in self.existing_artworks.values():
            if (
                normalized_title == self.normalize_artwork_title(artwork['title']) and
                any(normalized_artist == normalize_text(a) for a in artwork.get('artist_names', []))
            ):
                return artwork

        # Try case-insensitive match
        for artwork in self.existing_artworks.values():
            if (
                normalized_title.lower() == self.normalize_artwork_title(artwork['title']).lower() and
                any(normalized_artist.lower() == normalize_text(a).lower() for a in artwork.get('artist_names', []))
            ):
                return artwork

        # Try fuzzy match (removing punctuation and extra spaces)
        import re
        cleaned_title = re.sub(r'[^\w\s]', '', normalized_title.lower()).strip()
        cleaned_title = re.sub(r'\s+', ' ', cleaned_title)
        cleaned_artist = re.sub(r'[^\w\s]', '', normalized_artist.lower()).strip()
        cleaned_artist = re.sub(r'\s+', ' ', cleaned_artist)

        for artwork in self.existing_artworks.values():
            existing_title = self.normalize_artwork_title(artwork['title'])
            cleaned_existing_title = re.sub(r'[^\w\s]', '', existing_title.lower()).strip()
            cleaned_existing_title = re.sub(r'\s+', ' ', cleaned_existing_title)
            for a in artwork.get('artist_names', []):
                existing_artist = normalize_text(a)
                cleaned_existing_artist = re.sub(r'[^\w\s]', '', existing_artist.lower()).strip()
                cleaned_existing_artist = re.sub(r'\s+', ' ', cleaned_existing_artist)
                if cleaned_title == cleaned_existing_title and cleaned_artist == cleaned_existing_artist:
                    return artwork

        return None

    def lookup_keywords_via_api(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Use the existing lookup_text API to find similar keywords"""
        try:
            response = self.api_session.post(
                f"{self.api_base_url}/lookup_text",
                json={
                    "query": query_text,
                    "top_k": top_k
                },
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            print(f"[DEBUG] lookup_keywords_via_api: got {len(results)} results for query: {query_text[:80]}")
            # Filter to only keyword entries (isArtist = 0, robust to string/int)
            keyword_results = []
            for result in results:
                # print(f"[DEBUG] API result: {result}")
                if str(result.get('isArtist')) == '0':
                    keyword_results.append(result)
            print(f"[DEBUG] lookup_keywords_via_api: returning {len(keyword_results)} keyword results")
            return keyword_results
        except Exception as e:
            print(f"Error calling lookup_text API: {e}")
            return []

    def get_artist_keywords(self, artist_name: str, artist_info: Dict) -> Tuple[List[str], List[str]]:
        """Get keywords for an artist using the existing lookup_text API"""
        if artist_name in self.existing_artists:
            # For existing artists, start with their current keywords
            existing_keywords_ids = self.existing_artists[artist_name]['relatedKeywordIds']
            existing_keywords_strings = self.existing_artists[artist_name]['relatedKeywordStrings']
        else:
            existing_keywords_ids = []
            existing_keywords_strings = []
        # Extract text for semantic matching
        search_text_parts = [artist_name]
        # Add structured data for context
        if 'structured_data' in artist_info:
            structured = artist_info['structured_data']
            for field in ['nationality', 'style', 'genre', 'period', 'medium']:
                if field in structured:
                    search_text_parts.append(structured[field])
        # Add Wikipedia excerpt
        if 'wikipedia' in artist_info and 'description' in artist_info['wikipedia']:
            search_text_parts.append(artist_info['wikipedia']['description'])
        search_text = ' '.join(search_text_parts)
        print(f"  Searching for keywords with text: {search_text[:100]}...")
        # Find similar keywords using API
        similar_keywords = self.lookup_keywords_via_api(search_text, top_k=10)
        #print(f"[DEBUG] get_artist_keywords: similar_keywords={similar_keywords}")
        # Combine existing and new keywords
        final_keyword_ids = existing_keywords_ids.copy()
        final_keyword_strings = existing_keywords_strings.copy()
        # Track which IDs we already have to avoid duplicates
        existing_ids_set = set(existing_keywords_ids)
        for result in similar_keywords:
            keyword_id = result.get('entry_id')
            keyword_text = result.get('value')
            distance = result.get('distance')
            #print(f"[DEBUG] Considering keyword: id={keyword_id}, text={keyword_text}, distance={distance}")
            # Only add if not already present and distance indicates relevance
            if keyword_id not in existing_ids_set and distance is not None and distance < 0.9:
                final_keyword_ids.append(keyword_id)
                final_keyword_strings.append(keyword_text)
                existing_ids_set.add(keyword_id)
                print(f"    Added keyword: {keyword_text} (distance: {distance:.3f})")
        print(f"[DEBUG] get_artist_keywords: final_keyword_ids={final_keyword_ids}")
        print(f"[DEBUG] get_artist_keywords: final_keyword_strings={final_keyword_strings}")
        # Limit to 3-7 keywords total
        max_keywords = min(7, max(3, len(final_keyword_ids)))
        return final_keyword_ids[:max_keywords], final_keyword_strings[:max_keywords]

    def get_artwork_keywords(self, artwork_title: str, artist_keywords_ids: List[str], 
                           artist_keywords_strings: List[str], artwork_data: Dict) -> Tuple[List[str], List[str]]:
        """Get keywords for an artwork (inherits from artist + API matches)"""
        # Start with artist keywords
        keyword_ids = artist_keywords_ids.copy()
        keyword_strings = artist_keywords_strings.copy()
        # Extract text for semantic matching
        search_text_parts = [artwork_title]
        # Add artwork-specific data
        if 'descriptions' in artwork_data and 'wikiart' in artwork_data['descriptions']:
            wikiart_desc = artwork_data['descriptions']['wikiart']
            for field in ['medium', 'style', 'genre', 'period', 'series']:
                if field in wikiart_desc:
                    search_text_parts.append(wikiart_desc[field])
        search_text = ' '.join(search_text_parts)
        print(f"    Searching for artwork keywords with text: {search_text[:100]}...")
        # Find additional similar keywords using API
        similar_keywords = self.lookup_keywords_via_api(search_text, top_k=5)
        #print(f"[DEBUG] get_artwork_keywords: similar_keywords={similar_keywords}")
        # Track which IDs we already have to avoid duplicates
        existing_ids_set = set(keyword_ids)
        # Add top 5 new keywords
        for result in similar_keywords:
            keyword_id = result.get('entry_id')
            keyword_text = result.get('value')
            distance = result.get('distance')
            print(f"[DEBUG] Considering artwork keyword: id={keyword_id}, text={keyword_text}, distance={distance}")
            # Only add if not already present and distance indicates relevance
            if keyword_id not in existing_ids_set and distance is not None and distance < 0.9:
                keyword_ids.append(keyword_id)
                keyword_strings.append(keyword_text)
                existing_ids_set.add(keyword_id)
                print(f"      Added artwork keyword: {keyword_text} (distance: {distance:.3f})")
        print(f"[DEBUG] get_artwork_keywords: keyword_ids={keyword_ids}")
        print(f"[DEBUG] get_artwork_keywords: keyword_strings={keyword_strings}")
        return keyword_ids, keyword_strings

    def download_image(self, image_url: str, image_id: str) -> Optional[str]:
        """Download image to local storage"""
        if not self.download:
            return None
        
        try:
            # Generate filename
            file_ext = os.path.splitext(urlparse(image_url).path)[1] or '.jpg'
            filename = f"{image_id}{file_ext}"
            filepath = os.path.join(self.images_dir, filename)
            
            # Check if file already exists
            if os.path.exists(filepath):
                return filename
            
            # Download image
            response = self.session.get(image_url, timeout=15)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Downloaded image: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None

    def parse_wikiart_html(self, html_content: str) -> Dict:
        """Parse WikiArt HTML to extract artist information"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        artist_info = {
            'wikipedia': {},
            'structured_data': {}
        }
        
        try:
            # Get Wikipedia article content (stop at first <br />)
            wiki_tab = soup.find('div', id='info-tab-wikipediaArticle')
            if wiki_tab:
                first_p = wiki_tab.find('p')
                if first_p:
                    content_parts = []
                    for element in first_p.contents:
                        if hasattr(element, 'name') and element.name == 'br':
                            break
                        if hasattr(element, 'get_text'):
                            content_parts.append(element.get_text())
                        else:
                            content_parts.append(str(element))
                    
                    artist_info['wikipedia']['description'] = ''.join(content_parts).strip() #article_excerpt
            
            # Get Wikipedia link
            wiki_link = soup.find('a', class_='wiki-link')
            if wiki_link and wiki_link.get('href'):
                artist_info['wikipedia']['wikipediaLink'] = wiki_link['href']
            
            # Get structured microdata
            microdata_fields = [
                ('birthDate', 'birth'),
                ('birthPlace', 'birthPlace'),
                ('deathDate', 'death'),
                ('deathPlace', 'deathPlace'),
                ('nationality', 'nationality'),
                ('artMovement', 'art_movement'),
                ('paintingSchool', 'painting_school'),
                ('field', 'field'),
                ('influencedBy', 'influenced_by'),
                ('influencedOn', 'influenced_on'),
                ('artInstitution', 'art_institution'),
                ('friendsAndCoWorkers', 'friends_and_co_workers'),
                ('activeYears', 'active_years')
            ]
            
            for itemprop, key in microdata_fields:
                elem = soup.find('span', itemprop=itemprop)
                if elem:
                    artist_info['structured_data'][key] = elem.get_text().strip()
            
            # Parse dictionary values
            dict_items = soup.find_all('li', class_='dictionary-values')
            for item in dict_items:
                label_elem = item.find('s')
                if not label_elem:
                    continue
                
                label = label_elem.get_text().strip().rstrip(':').lower()
                
                # Extract values
                value_parts = []
                links = item.find_all('a')
                for link in links:
                    link_text = link.get_text().strip()
                    if link_text:
                        value_parts.append(link_text)
                
                if value_parts:
                    clean_label = label.replace(' ', '_').replace(':', '')
                    artist_info['structured_data'][clean_label] = ', '.join(value_parts)
            
            # Get artist name
            meta_name = soup.find('meta', itemprop='name')
            if meta_name and meta_name.get('content'):
                artist_info['structured_data']['name'] = meta_name['content'].strip()
            
        except Exception as e:
            print(f"Error parsing WikiArt HTML: {e}")
        
        return artist_info

    def parse_wikiart_artworks(self, html_content: str) -> List[Dict]:
        """Parse artwork data from WikiArt artist page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        artworks = []
        
        try:
            masonry_container = soup.find('ul', class_='wiki-masonry-container')
            if masonry_container:
                artwork_items = masonry_container.find_all('li')
                
                for item in artwork_items:
                    try:
                        img = item.find('img')
                        if not img:
                            continue
                        
                        img_url = img.get('src', '')
                        if 'lazy-load-placeholder' in img_url:
                            lazy_source = img.get('lazy-load') or img.get('img-source', '')
                            if lazy_source:
                                img_url = lazy_source.strip("'\"")
                        
                        title_block = item.find('div', class_='title-block')
                        if not title_block:
                            continue
                        
                        artwork_link = title_block.find('a', class_='artwork-name')
                        if not artwork_link:
                            continue
                        
                        title = artwork_link.get_text().strip()
                        artwork_path = artwork_link.get('href', '')
                        wikiart_url = f"https://www.wikiart.org{artwork_path}" if artwork_path.startswith('/') else artwork_path
                        
                        year_span = title_block.find('span', class_='artwork-year')
                        year = year_span.get_text().strip() if year_span else None
                        
                        artworks.append({
                            'title': title,
                            'date': year,
                            'thumbnail_url': img_url,
                            'wikiart_url': wikiart_url,
                            'wikiart_path': artwork_path
                        })
                        
                    except Exception as e:
                        print(f"Error parsing individual artwork: {e}")
                        continue
        
        except Exception as e:
            print(f"Error parsing artworks: {e}")
        
        return artworks

    def parse_artwork_details(self, html_content: str, artwork_title: str) -> Dict:
        """Parse detailed artwork information from WikiArt artwork page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        image_id = 'w_' + str(uuid.uuid4()).replace('-', '')[:23]
        
        artwork_data = {
            'image_id': image_id,
            'value': artwork_title,
            'artist_names': [],
            'image_urls': {'large': None, 'medium': None, 'small': None},
            'filename': None,
            'rights': None,
            'descriptions': {}
        };
        
        try:
            # Get artist name
            artist_span = soup.find('span', itemprop='name')
            if artist_span:
                artist_link = artist_span.find('a')
                if artist_link:
                    artist_name = artist_link.get_text().strip()
                    artwork_data['artist_names'] = [artist_name]
            
            # Get rights information
            copyright_link = soup.find('a', class_='copyright')
            if copyright_link:
                artwork_data['rights'] = copyright_link.get_text().strip()
            
            # Get artwork descriptions
            wikiart_data = {}
            
            # Date
            date_span = soup.find('span', itemprop='dateCreated')
            if date_span:
                wikiart_data['date'] = date_span.get_text().strip()
            
            # Dictionary values
            dict_items = soup.find_all('li', class_=re.compile(r'dictionary-values'))
            for item in dict_items:
                try:
                    label_elem = item.find('s')
                    if not label_elem:
                        continue
                    
                    label = label_elem.get_text().strip().rstrip(':').lower()
                    
                    value_parts = []
                    main_span = item.find('span')
                    if main_span:
                        links = main_span.find_all('a')
                        if links:
                            for link in links:
                                link_text = link.get_text().strip()
                                if link_text and link_text != ',':
                                    value_parts.append(link_text)
                    
                    if value_parts:
                        clean_label = label.replace(' ', '_').replace(':', '').replace('-', '_')
                        # Dynamically map all found labels to themselves, except for known overrides
                        label_mappings = {
                            'media': 'medium',
                            'style': 'style',
                            'genre': 'genre',
                            'location': 'collecting_institution',
                            'period': 'period',
                            'series': 'series',
                            'created': 'date'
                        }
                        # Add all other labels as-is if not already mapped
                        if clean_label not in label_mappings:
                            label_mappings[clean_label] = clean_label
                        final_label = label_mappings.get(clean_label, clean_label)
                        wikiart_data[final_label] = ', '.join(value_parts)
                
                except Exception as e:
                    continue
            
            if wikiart_data:
                artwork_data['descriptions']['wikiart'] = wikiart_data
            
            # Parse image URLs from ng-init JSON
            main_element = soup.find('main', attrs={'ng-controller': 'ArtworkViewCtrl'})
            if main_element and main_element.get('ng-init'):
                ng_init_content = main_element['ng-init']
                json_match = re.search(r'thumbnailSizesModel\s*=\s*({.*})', ng_init_content)
                if json_match:
                    json_str = json_match.group(1)
                    import html
                    decoded_json = html.unescape(json_str)
                    
                    try:
                        thumbnail_data = json.loads(decoded_json)
                        if 'ImageThumbnailsModel' in thumbnail_data and thumbnail_data['ImageThumbnailsModel']:
                            first_image = thumbnail_data['ImageThumbnailsModel'][0]
                            if 'Thumbnails' in first_image:
                                thumbnails = first_image['Thumbnails']
                                
                                # Sort by pixels
                                for thumb in thumbnails:
                                    thumb['pixels'] = thumb['Width'] * thumb['Height']
                                thumbnails.sort(key=lambda x: x['pixels'])
                                
                                if thumbnails:
                                    artwork_data['image_urls']['small'] = thumbnails[0]['Url']
                                    
                                    if len(thumbnails) >= 3:
                                        artwork_data['image_urls']['large'] = thumbnails[-1]['Url']
                                        mid_index = len(thumbnails) // 2
                                        artwork_data['image_urls']['medium'] = thumbnails[mid_index]['Url']
                                    elif len(thumbnails) == 2:
                                        artwork_data['image_urls']['large'] = thumbnails[1]['Url']
                    
                    except json.JSONDecodeError:
                        pass
            
            # Generate filename
            if artwork_data['image_urls']['small']:
                small_url = artwork_data['image_urls']['small']
                file_ext = os.path.splitext(urlparse(small_url).path)[1] or '.jpg'
                filename = f"{image_id}{file_ext}"
                artwork_data['filename'] = filename
        
        except Exception as e:
            print(f"Error parsing artwork details: {e}")
        
        return artwork_data
    
    def get_artist_names_to_scrape(self) -> List[str]:
        """Get list of artist names to scrape"""
        # For now, return a sample list - in production you might want to:
        # 1. Get from a predefined list
        # 2. Scrape from WikiArt's artist index
        # 3. Use an API or other source
        
        sample_artists = [
            "Pablo Picasso",
            "Vincent van Gogh",
            "Claude Monet",
            "Leonardo da Vinci",
            "Michelangelo",
            "Rembrandt van Rijn",
            "Johannes Vermeer",
            "Henri Matisse",
            "Paul Cézanne",
            "Edgar Degas",
            "Pierre-Auguste Renoir",
            "Gustav Klimt",
            "Salvador Dalí",
            "Frida Kahlo",
            "Georgia O'Keeffe",
            "Jackson Pollock",
            "Andy Warhol",
            "Edvard Munch",
            "Wassily Kandinsky",
            "Piet Mondrian"
        ]
        
        return sample_artists


    def scrape_artist(self, artist_name: str, stop_at_limit: bool = True) -> Optional[Dict]:
        """Scrape a single artist and their artworks. Returns artist_data with whatever artworks were processed before hitting the limit."""
        if getattr(self, 'limit_reached', False):
            return None
        slug = slugify(artist_name)
        url = f'https://www.wikiart.org/en/{slug}'
        print(f"Scraping artist: {artist_name} ({url})")
        if getattr(self, 'limit_reached', False):
            return None
        response = self.session.get(url, timeout=15)
        if getattr(self, 'limit_reached', False):
            return None
        if response.status_code != 200:
            print(f"  Failed to fetch artist page: {response.status_code}")
            return None
        # Parse artist info
        artist_info = self.parse_wikiart_html(response.text)
        if not artist_info.get('structured_data', {}).get('name'):
            print(f"  No artist name found, skipping")
            return None
        actual_name = artist_info['structured_data']['name']
        # Build descriptions field for artist
        descriptions = {}
        # Place all wikiart/structured data under descriptions['wikiart']
        wikiart_desc = {}
        if 'structured_data' in artist_info:
            wikiart_desc.update(artist_info['structured_data'])
        if 'wikipedia' in artist_info:
            wikiart_desc.update(artist_info['wikipedia'])
        descriptions['wikiart'] = wikiart_desc
        if 'artsy' in artist_info:
            descriptions['artsy'] = artist_info['artsy']

        is_existing = actual_name in self.existing_artists
        # If artist exists, merge in all DB info
        artist_aliases = []
        db_descriptions = {}
        db_keyword_ids = []
        db_keyword_strings = []
        if is_existing:
            db_artist = self.existing_artists[actual_name]
            # Fetch full row from DB
            try:
                row = self.db.execute('SELECT * FROM text_entries WHERE entry_id = ?', (db_artist['entry_id'],)).fetchone()
                if row:
                    row_dict = dict(row)
                    # Merge aliases
                    if row_dict.get('artist_aliases'):
                        try:
                            artist_aliases = json.loads(row_dict['artist_aliases'])
                        except Exception:
                            pass
                    # Merge descriptions
                    if row_dict.get('descriptions'):
                        try:
                            db_descriptions = json.loads(row_dict['descriptions'])
                        except Exception:
                            pass
                    # Merge keywords
                    if row_dict.get('relatedKeywordIds'):
                        try:
                            db_keyword_ids = json.loads(row_dict['relatedKeywordIds'])
                        except Exception:
                            pass
                    if row_dict.get('relatedKeywordStrings'):
                        try:
                            db_keyword_strings = json.loads(row_dict['relatedKeywordStrings'])
                        except Exception:
                            pass
            except Exception as e:
                print(f"  Error loading DB artist info for {actual_name}: {e}")
            # Merge DB descriptions with scraped wikiart
            if db_descriptions:
                # Merge wikiart
                if 'wikiart' in db_descriptions:
                    db_descriptions['wikiart'].update(wikiart_desc)
                else:
                    db_descriptions['wikiart'] = wikiart_desc
                # Merge artsy if present
                if 'artsy' in descriptions:
                    db_descriptions['artsy'] = descriptions['artsy']
                descriptions = db_descriptions
        # Check if we would exceed limit with this artist (if new)
        new_artist_count = 0 if is_existing else 1
        if self.would_exceed_limit(additional_artists=new_artist_count):
            self.limit_reached = True
            return None
        # Get keywords for artist (merge DB and scraped, dedup)
        keyword_ids, keyword_strings = self.get_artist_keywords(actual_name, artist_info)
        all_keyword_ids = list(dict.fromkeys(db_keyword_ids + keyword_ids))
        all_keyword_strings = list(dict.fromkeys(db_keyword_strings + keyword_strings))
        # Parse artworks
        artworks = self.parse_wikiart_artworks(response.text)
        artist_data = {
            'name': actual_name,
            'slug': slug,
            'is_existing': is_existing,
            'existing_id': self.existing_artists[actual_name]['entry_id'] if is_existing else None,
            'artist_aliases': artist_aliases,
            'descriptions': descriptions,
            # Store as JSON string for DB compatibility
            'RelatedKeywordIds': json.dumps(all_keyword_ids),
            'RelatedKeywordStrings': json.dumps(all_keyword_strings),
            'artworks': []
        }
        # Track that we're processing this artist
        if not is_existing:
            self.new_artists_count += 1
        # Process artworks from WikiArt
        for artwork in artworks[:10]:  # Limit artworks per artist
            if getattr(self, 'limit_reached', False):
                break
            # Check if we would exceed limit with this artwork (if new)
            if stop_at_limit and self.would_exceed_limit():
                self.limit_reached = True
                break
            try:
                print(f"  Processing artwork: {artwork['title']}")
                if getattr(self, 'limit_reached', False):
                    break
                # Get artwork details
                artwork_response = self.session.get(artwork['wikiart_url'], timeout=15)
                if getattr(self, 'limit_reached', False):
                    break
                if artwork_response.status_code != 200:
                    continue
                artwork_data = self.parse_artwork_details(artwork_response.text, artwork['title'])
                # Merge any extra scraped fields into descriptions['wikiart']
                extra_fields = {}
                # Collect extra fields from the parsed artwork dict that aren't standard DB columns
                for k, v in artwork.items():
                    if k not in ['title', 'date', 'thumbnail_url', 'wikiart_url', 'wikiart_path']:
                        extra_fields[k] = v
                # Also add any extra fields from artwork_data that aren't standard DB columns
                for k, v in artwork_data.items():
                    if k not in ['image_id', 'value', 'artist_names', 'image_urls', 'filename', 'rights', 'descriptions']:
                        extra_fields[k] = v
                # Ensure descriptions['wikiart'] exists and merge
                descriptions = artwork_data.get('descriptions', {})
                if 'wikiart' not in descriptions:
                    descriptions['wikiart'] = {}
                descriptions['wikiart'].update(extra_fields)
                # Add date to wikiart descriptions if available
                if artwork.get('date'):
                    descriptions['wikiart']['date'] = artwork['date']
                # Check if artwork exists with improved matching (title + artist)
                existing_artwork = self.find_existing_artwork(artwork['title'], actual_name)
                is_existing_artwork = existing_artwork is not None
                if is_existing_artwork:
                    print(f"    Found existing artwork: {artwork['title']} by {actual_name} -> {existing_artwork['image_id']}")
                else:
                    print(f"    New artwork: {artwork['title']} by {actual_name}")
                    # Check if adding this artwork would exceed the limit
                    if stop_at_limit and self.would_exceed_limit(additional_artworks=1):
                        print(f"    [DEBUG] limit is {self.limit}, this addition would be #{self.get_database_operations_count() + 1}")
                        self.limit_reached = True
                        break
                    # Track that we're adding this artwork
                    self.new_artworks_count += 1
                    print(f"    [DEBUG] limit is {self.limit}, this is addition #{self.get_database_operations_count()}")
                # Get keywords for artwork
                artwork_keyword_ids, artwork_keyword_strings = self.get_artwork_keywords(
                    artwork['title'], keyword_ids, keyword_strings, artwork_data
                )
                # Download image if enabled
                downloaded_filename = None
                if artwork_data['image_urls']['small']:
                    downloaded_filename = self.download_image(
                        artwork_data['image_urls']['small'],
                        artwork_data['image_id']
                    )
                artwork_staging = {
                    'value': artwork['title'],  # Use 'value' to match database structure
                    'image_id': artwork_data['image_id'],
                    'is_existing': is_existing_artwork,
                    'existing_id': existing_artwork['image_id'] if is_existing_artwork else None,
                    'artist_names': [actual_name],
                    'image_urls': artwork_data['image_urls'],
                    'filename': downloaded_filename or artwork_data['filename'],
                    'rights': artwork_data['rights'],
                    'descriptions': descriptions,
                    # Store as JSON string for DB compatibility - match database field names
                    'relatedKeywordIds': json.dumps(artwork_keyword_ids),
                    'relatedKeywordStrings': json.dumps(artwork_keyword_strings)
                }
                artist_data['artworks'].append(artwork_staging)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"  Error processing artwork {artwork['title']}: {e}")
                continue

        # --- Add all existing DB artworks for this artist (not already included) ---
        if is_existing:
            db_artist = self.existing_artists[actual_name]
            db_artworks = []
            try:
                cursor = self.db.execute('SELECT * FROM image_entries WHERE json_extract(artist_names, "$[0]") = ?', (actual_name,))
                db_artworks = cursor.fetchall()
            except Exception as e:
                print(f"  Error loading DB artworks for {actual_name}: {e}")
            # Build set of image_ids already included
            included_image_ids = set(a['image_id'] for a in artist_data['artworks'] if a.get('image_id'))
            for row in db_artworks:
                row_dict = dict(row)
                if row_dict.get('image_id') in included_image_ids:
                    continue
                # Parse all JSON/text fields as needed
                def parse_json_field(val):
                    if not val:
                        return None
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
                # Compose artwork entry with all available fields from DB
                db_artwork = {
                    'value': row_dict.get('value', ''),  # Use 'value' to match database structure
                    'image_id': row_dict.get('image_id', ''),
                    'is_existing': True,
                    'existing_id': row_dict.get('image_id', ''),
                    'artist_names': parse_json_field(row_dict.get('artist_names')) or [actual_name],
                    'image_urls': parse_json_field(row_dict.get('image_urls')) or {},
                    'filename': row_dict.get('filename', ''),
                    'rights': row_dict.get('rights', ''),
                    'descriptions': parse_json_field(row_dict.get('descriptions')) or {},
                    'relatedKeywordIds': parse_json_field(row_dict.get('relatedKeywordIds')) or [],
                    'relatedKeywordStrings': parse_json_field(row_dict.get('relatedKeywordStrings')) or []
                }
                artist_data['artworks'].append(db_artwork)

        return artist_data


    def run_scraping(self):
        """Main scraping loop"""
        print(f"Starting scraping process...")
        print(f"Download enabled: {self.download}")
        print(f"Limit: {self.limit}")
        print(f"Loaded {len(self.existing_artists)} existing artists")
        print(f"Loaded {len(self.existing_keywords)} existing keywords")
        print(f"Loaded {len(self.existing_artworks)} existing artworks")

        self.limit_reached = False

        if len(self.existing_artists) == 0:
            print("No existing data found... quitting")
            return

        artist_names = self.get_artist_names_to_scrape()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_artist_files = []
        for artist_name in artist_names:
            if self.limit_reached or self.get_database_operations_count() >= self.limit:
                print(f"[INFO] Scraping stopped: limit of {self.limit} reached.")
                break

            artist_data = self.scrape_artist(artist_name, stop_at_limit=True)
            if artist_data:
                # Save per-artist JSON file
                artist_slug = artist_data.get('slug') or slugify(artist_data.get('name',''))
                artist_filename = f"staging_artist_{artist_slug}_{timestamp}.json"
                artist_filepath = os.path.join(self.staging_dir, artist_filename)
                with open(artist_filepath, 'w', encoding='utf-8') as f:
                    json.dump({
                        "metadata": {
                            "timestamp": timestamp,
                            "artist": artist_data.get('name',''),
                            "slug": artist_slug,
                            "artwork_count": len(artist_data.get('artworks',[])),
                        },
                        "artist": artist_data,
                        "artworks": artist_data.get('artworks',[])
                    }, f, indent=2, ensure_ascii=False)
                print(f"Saved per-artist staging file: {artist_filepath}")
                per_artist_files.append(artist_filepath)
                # Also add to mega file in memory
                self.staging_data['artists'].append(artist_data)
                self.staging_data['artworks'].extend(artist_data['artworks'])

            if self.limit_reached or self.get_database_operations_count() >= self.limit:
                print(f"[INFO] Scraping stopped: limit of {self.limit} reached.")
                break

            time.sleep(1)  # Rate limiting between artists

        # Update metadata and always print summary
        self.staging_data['metadata']['total_artists'] = len(self.staging_data['artists'])
        self.staging_data['metadata']['total_artworks'] = len(self.staging_data['artworks'])
        self.staging_data['metadata']['new_artists_count'] = self.new_artists_count
        self.staging_data['metadata']['new_artworks_count'] = self.new_artworks_count
        self.staging_data['metadata']['total_database_operations'] = self.get_database_operations_count()

        print(f"\nScraping completed!")
        print(f"Total artists processed: {len(self.staging_data['artists'])}")
        print(f"Total artworks processed: {len(self.staging_data['artworks'])}")
        print(f"New artists to be added: {self.new_artists_count}")
        print(f"New artworks to be added: {self.new_artworks_count}")
        print(f"Total database operations: {self.get_database_operations_count()}")
        print(f"Limit: {self.limit}")
        print(f"Per-artist files written: {len(per_artist_files)}")

    def cleanup(self):
        """Clean up resources, especially database connection"""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}")

    def save_staging_data(self):
        """Save staging data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"staging_data_{timestamp}.json"
        filepath = os.path.join(self.staging_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.staging_data, f, indent=2, ensure_ascii=False)
        
        print(f"Staging data saved to: {filepath}")
        return filepath
    



def main():
    print(f"✅ Using database: {DB_PATH}")
    print(f"✅ Using images: {IMAGES_PATH}")
        

    parser = argparse.ArgumentParser(description='Scrape WikiArt data to staging JSON')
    parser.add_argument('--download', type=str, choices=['true', 'false'], 
                       default='false', help='Download images to local storage')
    parser.add_argument('--limit', type=int, default=50, 
                       help='Maximum number of items to scrape')
    parser.add_argument('--api-url', type=str, default='http://localhost:8080',
                       help='Base URL for the Flask API')
    parser.add_argument('--clear', type=str, choices=['true', 'false'],
                       default='false', help='Delete old staging JSON files before scraping')
    args = parser.parse_args()
    
    download = args.download.lower() == 'true'
    limit = args.limit
    api_url = args.api_url
    
    try:
        if args.clear.lower() == 'true':
            for fname in os.listdir(STAGING_PATH):
                if fname.startswith("staging_data_") and fname.endswith(".json"):
                    try:
                        os.remove(os.path.join(STAGING_PATH, fname))
                        print(f"Deleted old staging file: {fname}")
                    except Exception as e:
                        print(f"Error deleting {fname}: {e}")
        scraper = WikiArtScraper(download=download, limit=limit, api_base_url=api_url)
        try:
            scraper.run_scraping()
            staging_file = scraper.save_staging_data()
            
            print(f"\n✅ Scraping completed successfully!")
            print(f"📁 Staging file: {staging_file}")
        finally:
            scraper.cleanup()
        
    except KeyboardInterrupt:
        print("\n❌ Scraping interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()