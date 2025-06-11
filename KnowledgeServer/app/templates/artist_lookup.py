# artist_lookup.py
from flask import Blueprint, render_template, request, jsonify, current_app
import json
import requests
import re
import uuid
import html
import os
from bs4 import BeautifulSoup
from index import get_db
import urllib.request
from urllib.parse import urlparse


artist_lookup_bp = Blueprint('artist_lookup', __name__, url_prefix='/artist_lookup')

def slugify(name):
    """Convert name to slug format (firstname-lastname)"""
    # Convert to lowercase
    name = name.lower().strip()
    
    # Replace accented characters
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
        name = name.replace(accent, replacement)
    
    # Replace spaces with hyphens
    name = name.replace(' ', '-')
    
    # Remove any remaining non-alphanumeric characters except hyphens
    name = re.sub(r'[^a-z0-9-]', '', name)
    
    # Remove multiple consecutive hyphens
    name = re.sub(r'-+', '-', name)
    
    # Remove leading/trailing hyphens
    name = name.strip('-')
    
    return name

def parse_wikiart_html(html_content):
    """Parse WikiArt HTML to extract artist information using WikiArt-specific structure"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    artist_info = {
        'wikipedia': {},
        'structured_data': {}
    }
    
    try:
        # 1. Get Wikipedia article content (stop at first <br />)
        wiki_tab = soup.find('div', id='info-tab-wikipediaArticle')
        if wiki_tab:
            # Get the first paragraph and stop at first <br />
            first_p = wiki_tab.find('p')
            if first_p:
                # Get text content up to first <br /> tag
                content_parts = []
                for element in first_p.contents:
                    if hasattr(element, 'name') and element.name == 'br':
                        break
                    if hasattr(element, 'get_text'):
                        content_parts.append(element.get_text())
                    else:
                        content_parts.append(str(element))
                
                artist_info['wikipedia']['article_excerpt'] = ''.join(content_parts).strip()
        
        # 2. Get Wikipedia link
        wiki_link = soup.find('a', class_='wiki-link')
        if wiki_link and wiki_link.get('href'):
            artist_info['wikipedia']['wikipediaLink'] = wiki_link['href']
        
        # 3. Get license info
        license_p = soup.find('p', class_='wikipedia-licence')
        if license_p:
            license_text = license_p.get_text().strip()
            # Remove the "The full text of the article is here →" part
            cleaned_license_text = license_text.split('The full text of the article is here')[0].strip()
            artist_info['wikipedia']['license'] = cleaned_license_text
        
        # 4. Get structured microdata
        # Birth date
        birth_date = soup.find('span', itemprop='birthDate')
        if birth_date:
            artist_info['structured_data']['birth'] = birth_date.get_text().strip()
        
        # Birth place
        birth_place = soup.find('span', itemprop='birthPlace')
        if birth_place:
            artist_info['structured_data']['birthPlace'] = birth_place.get_text().strip()
        
        # Death date
        death_date = soup.find('span', itemprop='deathDate')
        if death_date:
            artist_info['structured_data']['death'] = death_date.get_text().strip()
        
        # Death place
        death_place = soup.find('span', itemprop='deathPlace')
        if death_place:
            artist_info['structured_data']['deathPlace'] = death_place.get_text().strip()
        
        # Nationality
        nationality = soup.find('span', itemprop='nationality')
        if nationality:
            artist_info['structured_data']['nationality'] = nationality.get_text().strip()
        
        # 5. Parse dictionary values (li class="dictionary-values")
        dict_items = soup.find_all('li', class_='dictionary-values')
        for item in dict_items:
            # Get the label (in <s> tag)
            label_elem = item.find('s')
            if not label_elem:
                continue
            
            label = label_elem.get_text().strip().rstrip(':').lower()
            
            # Get the values - look for links and spans
            value_parts = []
            
            # Look for links with text
            links = item.find_all('a')
            for link in links:
                link_text = link.get_text().strip()
                if link_text:
                    value_parts.append(link_text)
            
            # If no links found, look for spans with itemprop
            if not value_parts:
                spans = item.find_all('span', itemprop=True)
                for span in spans:
                    span_text = span.get_text().strip()
                    if span_text:
                        value_parts.append(span_text)
            
            # If still no values, get all text content except the label
            if not value_parts:
                all_text = item.get_text().strip()
                label_text = label_elem.get_text().strip()
                value_text = all_text.replace(label_text, '').strip()
                if value_text:
                    value_parts.append(value_text)
            
            # Join multiple values with commas
            if value_parts:
                # Clean up the label for use as key
                clean_label = label.replace(' ', '_').replace(':', '')
                artist_info['structured_data'][clean_label] = ', '.join(value_parts)
        
        # 6. Extract artist name from meta tag first, then fallback to page title or h1
        meta_name = soup.find('meta', itemprop='name')
        if meta_name and meta_name.get('content'):
            artist_info['structured_data']['name'] = meta_name['content'].strip()
        else:
            name_elem = soup.find('h1') or soup.find('title')
            if name_elem:
                artist_info['structured_data']['name'] = name_elem.get_text().strip()
        
        # 7. Look for artist image
        img_elem = soup.find('img', src=re.compile(r'artist|portrait'))
        if not img_elem:
            # Fallback to any img in artist-related containers
            img_elem = soup.find('img')
        
        if img_elem and img_elem.get('src'):
            src = img_elem['src']
            if src.startswith('//'):
                src = 'https:' + src
            elif src.startswith('/'):
                src = 'https://www.wikiart.org' + src
            artist_info['structured_data']['image_url'] = src
        
        # Clean up empty dictionaries
        if not artist_info['wikipedia']:
            del artist_info['wikipedia']
        if not artist_info['structured_data']:
            del artist_info['structured_data']
        
    except Exception as e:
        print(f"Error parsing WikiArt HTML: {e}")
    
    return artist_info

@artist_lookup_bp.route('/db_stats')
def get_db_stats():
    """Get database statistics for debugging"""
    try:
        db = get_db()
        
        # Count artists
        artist_count = db.execute('SELECT COUNT(*) FROM text_entries WHERE isArtist = 1').fetchone()[0]
        
        # Count images
        image_count = db.execute('SELECT COUNT(*) FROM image_entries').fetchone()[0]
        
        # Total text entries
        total_count = db.execute('SELECT COUNT(*) FROM text_entries').fetchone()[0]
        
        return jsonify({
            'success': True,
            'artist_count': artist_count,
            'image_count': image_count,
            'total_count': total_count
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@artist_lookup_bp.route('/')
def artist_lookup_page():
    """Artist lookup and form page"""
    return render_template('artist_lookup.html')

@artist_lookup_bp.route('/process', methods=['POST'])
def process_artist_lookup():
    """Process the artist lookup request"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})
            
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        if not first_name or not last_name:
            return jsonify({'success': False, 'error': 'First name and last name are required'})
        
        # Create slug
        full_name = f"{first_name} {last_name}"
        slug = slugify(full_name)
        
        # Check if artist exists in database using slug in artist_aliases
        existing_artist = None
        try:
            db = get_db()
            
            # Search for the slug in the artist_aliases JSON column
            # SQLite doesn't have native JSON search, so we need to use LIKE
            search_pattern = f'%"slug": "{slug}"%'
            cursor = db.execute('''
                SELECT * FROM text_entries 
                WHERE isArtist = 1 
                AND artist_aliases LIKE ?
            ''', (search_pattern,))
            existing_artist = cursor.fetchone()
            
            # If not found by slug, try searching by name in artist_aliases
            if not existing_artist:
                name_pattern = f'%"name": "{full_name}"%'
                cursor = db.execute('''
                    SELECT * FROM text_entries 
                    WHERE isArtist = 1 
                    AND artist_aliases LIKE ?
                ''', (name_pattern,))
                existing_artist = cursor.fetchone()
            
            # If still not found, try case-insensitive search on the value field
            if not existing_artist:
                cursor = db.execute('''
                    SELECT * FROM text_entries 
                    WHERE isArtist = 1 
                    AND LOWER(value) = LOWER(?)
                ''', (full_name,))
                existing_artist = cursor.fetchone()
                
            if existing_artist:
                print(f"Found existing artist: {existing_artist['value']} with slug search: {slug}")
            else:
                print(f"No existing artist found for slug: {slug}")
                
        except Exception as db_error:
            print(f"Database error (continuing without DB check): {db_error}")
            # Continue without database check
    
        # Try to fetch WikiArt page
        url = f'https://www.wikiart.org/en/{slug}'
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                # Parse the HTML to extract artist information
                artist_info = parse_wikiart_html(response.text)
                
                return jsonify({
                    'success': True,
                    'slug': slug,
                    'html_content': response.text[:10000],  # First 10k chars for preview
                    'artist_info': artist_info,
                    'existing_artist': dict(existing_artist) if existing_artist else None
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'WikiArt page not found (status code: {response.status_code})'
                })
        except requests.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch WikiArt page: {str(e)}'
            })
            
    except Exception as e:
        # Catch any other errors and return JSON response
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })
    """Process the artist lookup request"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'})
            
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        
        if not first_name or not last_name:
            return jsonify({'success': False, 'error': 'First name and last name are required'})
        
        # Create slug
        full_name = f"{first_name} {last_name}"
        slug = slugify(full_name)
        
        # Check if artist exists in database (with error handling)
        existing_artist = None
        try:
            # Use your get_db() function
            db = get_db()
            cursor = db.execute('SELECT * FROM text_entries WHERE value = ? AND isArtist = 1', (full_name,))
            existing_artist = cursor.fetchone()
        except Exception as db_error:
            print(f"Database error (continuing without DB check): {db_error}")
            # Continue without database check
    
        # Try to fetch WikiArt page
        url = f'https://www.wikiart.org/en/{slug}'
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                # Parse the HTML to extract artist information
                artist_info = parse_wikiart_html(response.text)
                
                return jsonify({
                    'success': True,
                    'slug': slug,
                    'html_content': response.text[:10000],  # First 10k chars for preview
                    'artist_info': artist_info,
                    'existing_artist': dict(existing_artist) if existing_artist else None
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'WikiArt page not found (status code: {response.status_code})'
                })
        except requests.RequestException as e:
            return jsonify({
                'success': False,
                'error': f'Failed to fetch WikiArt page: {str(e)}'
            })
            
    except Exception as e:
        # Catch any other errors and return JSON response
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })


@artist_lookup_bp.route('/get_artworks', methods=['POST'])
def get_artworks():
    """Fetch artworks for an artist from their WikiArt page"""
    try:
        data = request.json
        slug = data.get('slug')
        
        if not slug:
            return jsonify({'success': False, 'error': 'No slug provided'})
        
        # Fetch the artist's main page to get artworks
        url = f'https://www.wikiart.org/en/{slug}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to fetch artist page (status: {response.status_code})'})
        
        # Parse artworks from the HTML
        artworks = parse_wikiart_artworks(response.text)
        
        return jsonify({
            'success': True,
            'artworks': artworks
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def parse_wikiart_artworks(html_content):
    """Parse artwork data from WikiArt artist page"""
    soup = BeautifulSoup(html_content, 'html.parser')
    artworks = []
    
    try:
        # Look for the masonry container with artworks
        masonry_container = soup.find('ul', class_='wiki-masonry-container')
        
        if masonry_container:
            artwork_items = masonry_container.find_all('li')
            
            for item in artwork_items:
                try:
                    # Get image
                    img = item.find('img')
                    if not img:
                        continue
                    
                    # Get image URL (prefer the actual src over lazy-load placeholder)
                    img_url = img.get('src', '')
                    if 'lazy-load-placeholder' in img_url:
                        # Try to get the actual image from img-source attribute
                        lazy_source = img.get('lazy-load') or img.get('img-source', '')
                        if lazy_source:
                            # Clean up the img-source format (remove quotes)
                            img_url = lazy_source.strip("'\"")
                    
                    # Get artwork title and URL
                    title_block = item.find('div', class_='title-block')
                    if not title_block:
                        continue
                    
                    artwork_link = title_block.find('a', class_='artwork-name')
                    if not artwork_link:
                        continue
                    
                    title = artwork_link.get_text().strip()
                    artwork_path = artwork_link.get('href', '')
                    wikiart_url = f"https://www.wikiart.org{artwork_path}" if artwork_path.startswith('/') else artwork_path
                    
                    # Get year
                    year_span = title_block.find('span', class_='artwork-year')
                    year = year_span.get_text().strip() if year_span else None
                    
                    artworks.append({
                        'title': title,
                        'year': year,
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

@artist_lookup_bp.route('/get_artwork_details', methods=['POST'])
def get_artwork_details():
    """Fetch detailed information for a specific artwork"""
    try:
        data = request.json
        artwork_url = data.get('artwork_url')
        artwork_title = data.get('artwork_title')
        
        if not artwork_url:
            return jsonify({'success': False, 'error': 'No artwork URL provided'})
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(artwork_url, timeout=15, headers=headers)
        
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to fetch artwork page (status: {response.status_code})'})
        
        # Parse artwork details
        artwork_data = parse_artwork_details(response.text, artwork_title)
        
        return jsonify({
            'success': True,
            'artwork_data': artwork_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def parse_artwork_details(html_content, artwork_title):
    """Parse detailed artwork information from WikiArt artwork page"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Generate unique image ID
    image_id = 'w_' + str(uuid.uuid4()).replace('-', '')[:23]
    
    artwork_data = {
        'image_id': image_id,
        'value': artwork_title,
        'artist_names': [],
        'image_urls': {'large': None, 'medium': None, 'small': None},
        'filename': None,
        'rights': None,
        'descriptions': {},
        'relatedKeywordIds': [],
        'relatedKeywordStrings': []
    }
    
    try:
        # 1. Get artist name from itemprop="name"
        artist_span = soup.find('span', itemprop='name')
        if artist_span:
            artist_link = artist_span.find('a')
            if artist_link:
                artist_name = artist_link.get_text().strip()
                artwork_data['artist_names'] = [artist_name]
                print(f"Found artist name: {artist_name}")
            else:
                print("Found span with itemprop='name' but no link inside")
        else:
            print("Could not find span with itemprop='name'")
        
        # Get copyright/rights information
        copyright_link = soup.find('a', class_='copyright')
        if copyright_link:
            rights_text = copyright_link.get_text().strip()
            artwork_data['rights'] = rights_text
            print(f"Found rights: {rights_text}")
        else:
            print("Could not find copyright link")



        # GET ARTWORK DESCRIPTIONS
        artwork_descriptions = {}
        wikiart_data = {}
        
        print("=== EXTRACTING ARTWORK DESCRIPTIONS ===")
        
        # Look for date with itemprop="dateCreated"
        date_span = soup.find('span', itemprop='dateCreated')
        if date_span:
            date_value = date_span.get_text().strip()
            wikiart_data['date'] = date_value
            print(f"Found date: {date_value}")
        
        import re
        # Look for dictionary values (li with class containing "dictionary-values")
        dict_items = soup.find_all('li', class_=re.compile(r'dictionary-values'))
        print(f"Found {len(dict_items)} dictionary items")
        
        for item in dict_items:
            try:
                # Get the label (in <s> tag)
                label_elem = item.find('s')
                if not label_elem:
                    continue
                
                label = label_elem.get_text().strip().rstrip(':').lower()
                print(f"Processing label: {label}")
                
                # Get the values - look for spans and links
                value_parts = []
                
                # Look for the main span containing the values
                main_span = item.find('span')
                if main_span:
                    # First, try to get links with text
                    links = main_span.find_all('a')
                    if links:
                        for link in links:
                            link_text = link.get_text().strip()
                            if link_text and link_text != ',':  # Skip comma separators
                                value_parts.append(link_text)
                        print(f"  Found links: {value_parts}")
                    
                    # If no links, look for spans with itemprop
                    if not value_parts:
                        itemprop_spans = main_span.find_all('span', itemprop=True)
                        for span in itemprop_spans:
                            span_text = span.get_text().strip()
                            if span_text:
                                value_parts.append(span_text)
                        print(f"  Found itemprop spans: {value_parts}")
                    
                    # If still no values, get all text content except commas and the label
                    if not value_parts:
                        # Get all text, clean it up
                        all_text = main_span.get_text()
                        # Remove the label text if it appears at the beginning
                        label_text = label_elem.get_text().strip()
                        if all_text.startswith(label_text):
                            all_text = all_text[len(label_text):].strip()
                        
                        # Split by common separators and clean
                        parts = re.split(r'[,;]\s*', all_text)
                        for part in parts:
                            part = part.strip()
                            if part and part != ',':
                                value_parts.append(part)
                        print(f"  Found text parts: {value_parts}")
                
                # If we found values, add to wikiart_data
                if value_parts:
                    # Clean up the label for use as key
                    clean_label = label.replace(' ', '_').replace(':', '').replace('-', '_')
                    
                    # Handle special label mappings
                    label_mappings = {
                        'media': 'medium',
                        'style': 'style',
                        'genre': 'genre',
                        'location': 'collecting_institution',
                        'period': 'period',
                        'series': 'series',
                        'created': 'date'
                    }
                    
                    final_label = label_mappings.get(clean_label, clean_label)
                    
                    # Join multiple values with commas
                    final_value = ', '.join(value_parts)
                    wikiart_data[final_label] = final_value
                    print(f"  Added: {final_label} = {final_value}")
                
            except Exception as e:
                print(f"Error processing dictionary item: {e}")
                continue
        
        # Add wikiart data to descriptions
        if wikiart_data:
            artwork_descriptions['wikiart'] = wikiart_data
            artwork_data['descriptions'] = artwork_descriptions
            print(f"Final descriptions: {artwork_descriptions}")
        else:
            print("No wikiart data found")
        
        print("=== END ARTWORK DESCRIPTIONS EXTRACTION ===")
        
        # Parse image data from ng-init JSON
        print("=== PARSING NG-INIT JSON DATA ===")
        
        main_element = soup.find('main', attrs={'ng-controller': 'ArtworkViewCtrl'})
        if main_element and main_element.get('ng-init'):
            ng_init_content = main_element['ng-init']
            print(f"Found ng-init content (first 200 chars): {ng_init_content[:200]}...")
            
            # Extract the JSON part from ng-init
            # Look for thumbnailSizesModel = { ... }
            json_match = re.search(r'thumbnailSizesModel\s*=\s*({.*})', ng_init_content)
            if json_match:
                json_str = json_match.group(1)
                print(f"Extracted JSON string (first 200 chars): {json_str[:200]}...")
                
                # The JSON has HTML entities, so we need to decode them
                import html
                decoded_json = html.unescape(json_str)
                print(f"Decoded JSON (first 200 chars): {decoded_json[:200]}...")
                
                try:
                    thumbnail_data = json.loads(decoded_json)
                    print("✅ Successfully parsed JSON data")
                    
                    # Navigate to the thumbnails
                    if 'ImageThumbnailsModel' in thumbnail_data and thumbnail_data['ImageThumbnailsModel']:
                        first_image = thumbnail_data['ImageThumbnailsModel'][0]
                        if 'Thumbnails' in first_image:
                            thumbnails = first_image['Thumbnails']
                            print(f"Found {len(thumbnails)} thumbnails")
                            
                            # Sort thumbnails by total pixels (Width * Height)
                            for thumb in thumbnails:
                                thumb['pixels'] = thumb['Width'] * thumb['Height']
                                print(f"  {thumb['Name']}: {thumb['Width']}x{thumb['Height']} = {thumb['pixels']} pixels - {thumb['Url']}")
                            
                            # Sort by pixels (smallest first)
                            thumbnails.sort(key=lambda x: x['pixels'])
                            print(f"\n=== AFTER SORTING (smallest first) ===")
                            for thumb in thumbnails:
                                print(f"  {thumb['Name']}: {thumb['Width']}x{thumb['Height']} = {thumb['pixels']} pixels")
                            
                            if thumbnails:
                                # Always assign small (smallest available)
                                artwork_data['image_urls']['small'] = thumbnails[0]['Url']
                                print(f"Assigned SMALL: {thumbnails[0]['Name']} - {thumbnails[0]['Url']}")
                                
                                if len(thumbnails) >= 3:
                                    # Three or more thumbnails available - assign all three
                                    artwork_data['image_urls']['large'] = thumbnails[-1]['Url']
                                    mid_index = len(thumbnails) // 2
                                    artwork_data['image_urls']['medium'] = thumbnails[mid_index]['Url']
                                    print(f"3+ thumbnails - MEDIUM: {thumbnails[mid_index]['Name']}, LARGE: {thumbnails[-1]['Name']}")
                                elif len(thumbnails) == 2:
                                    # Two thumbnails available - assign small and large only
                                    artwork_data['image_urls']['large'] = thumbnails[1]['Url']
                                    print(f"2 thumbnails - LARGE: {thumbnails[1]['Name']} (no medium assigned)")
                                else:
                                    # Only 1 thumbnail - just small
                                    print("Only 1 thumbnail - assigned to small only (no medium/large)")
                            else:
                                print("❌ No thumbnails found in data")
                        else:
                            print("❌ No 'Thumbnails' key in first image")
                    else:
                        print("❌ No 'ImageThumbnailsModel' in data or it's empty")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON: {e}")
                    print(f"Problematic JSON: {decoded_json[:500]}...")
            else:
                print("❌ Could not find thumbnailSizesModel in ng-init")
        else:
            print("❌ Could not find main element with ng-controller='ArtworkViewCtrl'")
        
        print("=== END NG-INIT PARSING ===")
        
        # Generate filename based on smallest image URL extension
        if artwork_data['image_urls']['small']:
            small_url = artwork_data['image_urls']['small']
            print(f"Determining file extension from: {small_url}")
            
            # Extract file extension from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(small_url)
            file_ext = os.path.splitext(parsed_url.path)[1]
            
            # Default to .jpg if no extension found
            if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                file_ext = '.jpg'
            
            filename = f"{image_id}_small{file_ext}"
            artwork_data['filename'] = filename
            print(f"Generated filename: {filename}")
        else:
            print("No small image URL available - cannot generate filename")
        
    except Exception as e:
        print(f"Error parsing artwork details: {e}")
        import traceback
        traceback.print_exc()
    
    return artwork_data


@artist_lookup_bp.route('/download_image', methods=['POST'])
def download_image_route():
    """Download an image file"""
    try:
        data = request.json
        image_url = data.get('image_url')
        image_id = data.get('image_id')
        filename = data.get('filename')
        
        if not image_url or not image_id:
            return jsonify({'success': False, 'error': 'Missing image URL or ID'})

        # Check if the file already exists
        # Find the base directory by going up one level from the current (templates) directory
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        images_dir = os.path.join(BASE_DIR, "LOCALDB", "images")
        file_ext = os.path.splitext(urlparse(image_url).path)[1] or '.jpg'
        filename = f"{image_id}{file_ext}"
        filepath = os.path.join(images_dir, filename)

        if os.path.exists(filepath):
            return jsonify({
                'success': True,
                'filename': filename
            })
        
        downloaded_filename = download_image(image_url, image_id)
        
        if downloaded_filename:
            return jsonify({
                'success': True,
                'filename': downloaded_filename
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to download image'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@artist_lookup_bp.route('/validate_data', methods=['POST'])
def validate_data():
    """Validate all data before database insertion"""
    try:
        print("=== STARTING VALIDATION ===")
        
        data = request.json
        if not data:
            print("ERROR: No JSON data received")
            return jsonify({'success': False, 'error': 'No JSON data received'})
        
        artist_data = data.get('artist_data', {})
        selected_artworks = data.get('selected_artworks', [])
        
        print(f"Artist data keys: {list(artist_data.keys())}")
        print(f"Number of selected artworks: {len(selected_artworks)}")
        
        validation = {
            'overall_valid': True,
            'artist_preview': {},
            'artist_sql_preview': {},
            'artist_json_valid': True,
            'artist_json_errors': [],
            'artworks_json_valid': [],
            'files_exist': [],
            'image_urls_valid': []
        }
        
        # Create clean artist preview for display
        validation['artist_preview'] = {
            'entry_id': artist_data.get('entry_id'),
            'value': artist_data.get('value'),
            'type': artist_data.get('type'),
            'isArtist': int(artist_data.get('isArtist', 1)),
            'artist_aliases': None,
            'images': None,
            'descriptions': None,
            'relatedKeywordIds': None,
            'relatedKeywordStrings': None
        }
        
        print(f"Artist preview created: {validation['artist_preview']}")
        
        # Validate artist data JSON fields
        json_fields = ['artist_aliases', 'images', 'descriptions', 'relatedKeywordIds', 'relatedKeywordStrings']
        
        for field in json_fields:
            try:
                field_value = artist_data.get(field, '[]')
                print(f"Validating {field}: {field_value}")
                parsed_json = json.loads(field_value)
                validation['artist_preview'][field] = parsed_json
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {field}: {e}")
                validation['artist_json_valid'] = False
                validation['artist_json_errors'].append(field)
                validation['overall_valid'] = False
                validation['artist_preview'][field] = f"INVALID JSON: {artist_data.get(field, '')}"
        
        print(f"Artist JSON validation complete. Valid: {validation['artist_json_valid']}")
        
        # Validate each artwork
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(BASE_DIR, "LOCALDB", "images")
        print(f"Images directory: {images_dir}")
        
        for i, artwork in enumerate(selected_artworks):
            print(f"Validating artwork {i}: {artwork.get('value', 'Unknown')}")
            
            # JSON validation
            json_valid = True
            try:
                json.loads(json.dumps(artwork.get('artist_names', [])))
                json.loads(json.dumps(artwork.get('image_urls', {})))
                json.loads(json.dumps(artwork.get('descriptions', {})))
                json.loads(json.dumps(artwork.get('relatedKeywordIds', [])))
                json.loads(json.dumps(artwork.get('relatedKeywordStrings', [])))
                print(f"Artwork {i} JSON validation: PASSED")
            except Exception as e:
                print(f"Artwork {i} JSON validation: FAILED - {e}")
                json_valid = False
                validation['overall_valid'] = False
            
            validation['artworks_json_valid'].append(json_valid)
            
            # File existence check
            file_exists = False
            filename = artwork.get('filename')
            if filename:
                filepath = os.path.join(images_dir, filename)
                file_exists = os.path.exists(filepath)
                print(f"Artwork {i} file check: {filepath} exists: {file_exists}")
                if not file_exists:
                    validation['overall_valid'] = False
            else:
                print(f"Artwork {i} has no filename")
            
            validation['files_exist'].append(file_exists)
            
            # Image URL validation
            urls_valid = True
            image_urls = artwork.get('image_urls', {})
            for size, url in image_urls.items():
                if url and not (url.startswith('http://') or url.startswith('https://')):
                    print(f"Artwork {i} invalid URL for {size}: {url}")
                    urls_valid = False
                    validation['overall_valid'] = False
                    break
            
            print(f"Artwork {i} URL validation: {urls_valid}")
            validation['image_urls_valid'].append(urls_valid)
        
        print(f"Overall validation result: {validation['overall_valid']}")
        print("=== VALIDATION COMPLETE ===")
        
        return jsonify({
            'success': True,
            'validation': validation
        })
        
    except Exception as e:
        print(f"ERROR in validate_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@artist_lookup_bp.route('/submit_all_data', methods=['POST'])
def submit_all_data():
    """Submit both artist and artwork data to database"""
    try:
        data = request.json
        artist_data = data.get('artist_data', {})
        selected_artworks = data.get('selected_artworks', [])
        
        db = get_db()
        
        # Clean up JSON fields before insertion
        # Parse and re-stringify to ensure clean JSON
        def clean_json_field(field_value, default='[]'):
            if isinstance(field_value, str):
                try:
                    # Parse to validate and clean
                    parsed = json.loads(field_value)
                    # Re-stringify with no escaping
                    return json.dumps(parsed)
                except:
                    return default
            else:
                # Already an object, just stringify
                return json.dumps(field_value)
        
        # Clean artist data JSON fields
        artist_aliases_clean = clean_json_field(artist_data.get('artist_aliases'), '[]')
        images_clean = clean_json_field(artist_data.get('images'), '[]')
        descriptions_clean = clean_json_field(artist_data.get('descriptions'), '{}')
        relatedKeywordIds_clean = clean_json_field(artist_data.get('relatedKeywordIds'), '[]')
        relatedKeywordStrings_clean = clean_json_field(artist_data.get('relatedKeywordStrings'), '[]')
        
        # Insert or update artist data
        if artist_data.get('entry_id'):
            # Check if updating existing artist
            existing = db.execute('SELECT entry_id FROM text_entries WHERE entry_id = ?', 
                                (artist_data['entry_id'],)).fetchone()
            
            if existing:
                # Update existing artist
                db.execute('''
                    UPDATE text_entries SET 
                    value = ?, artist_aliases = ?, images = ?, descriptions = ?, 
                    relatedKeywordIds = ?, relatedKeywordStrings = ?
                    WHERE entry_id = ?
                ''', (
                    artist_data['value'],
                    artist_aliases_clean,
                    images_clean,
                    descriptions_clean,
                    relatedKeywordIds_clean,
                    relatedKeywordStrings_clean,
                    artist_data['entry_id']
                ))
            else:
                # Insert new artist
                db.execute('''
                    INSERT INTO text_entries (
                        entry_id, value, type, isArtist, artist_aliases, images, 
                        descriptions, relatedKeywordIds, relatedKeywordStrings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    artist_data['entry_id'],
                    artist_data['value'],
                    artist_data['type'],
                    1,  # isArtist
                    artist_aliases_clean,
                    images_clean,
                    descriptions_clean,
                    relatedKeywordIds_clean,
                    relatedKeywordStrings_clean
                ))
        
        # Insert artworks
        artworks_added = 0
        for artwork in selected_artworks:
            try:
                # Ensure artwork JSON fields are clean strings, not double-encoded
                artist_names_json = json.dumps(artwork.get('artist_names', []))
                image_urls_json = json.dumps(artwork.get('image_urls', {}))
                descriptions_json = json.dumps(artwork.get('descriptions', {}))
                relatedKeywordIds_json = json.dumps(artwork.get('relatedKeywordIds', []))
                relatedKeywordStrings_json = json.dumps(artwork.get('relatedKeywordStrings', []))
                
                db.execute('''
                    INSERT INTO image_entries (
                        image_id, value, artist_names, image_urls, filename, 
                        rights, descriptions, relatedKeywordIds, relatedKeywordStrings
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    artwork['image_id'],
                    artwork['value'],
                    artist_names_json,
                    image_urls_json,
                    artwork['filename'],
                    artwork['rights'],
                    descriptions_json,
                    relatedKeywordIds_json,
                    relatedKeywordStrings_json
                ))
                artworks_added += 1
            except Exception as e:
                print(f"Failed to insert artwork {artwork.get('value', 'unknown')}: {e}")
        
        db.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully added artist and {artworks_added} artworks',
            'artworks_added': artworks_added
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
        
def download_image(image_url, image_id):
    """Download image to ../LOCALDB/images/ directory"""
    try:
        import urllib.request
        from urllib.parse import urlparse
        
        # Get the base directory and create images path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(BASE_DIR, "LOCALDB", "images")
        
        # Ensure directory exists
        os.makedirs(images_dir, exist_ok=True)
        
        # Get file extension from URL
        parsed_url = urlparse(image_url)
        file_ext = os.path.splitext(parsed_url.path)[1] or '.jpg'
        
        # Create filename
        filename = f"{image_id}_small{file_ext}"
        filepath = os.path.join(images_dir, filename)
        
        # Download the image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        req = urllib.request.Request(image_url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        
        return filename
        
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None