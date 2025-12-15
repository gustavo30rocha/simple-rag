import os
import wikipedia
import re

wikipedia.set_lang("en")

# Topics to scrape (customizable)
TOPICS = [
    # Technical topics
    "Python (programming language)",
    "Machine learning",
    "Database",
    "REST API",
    "SQL",
    "JavaScript",
    "Linux",
    "Blockchain",
    "Neural network",
    "Git (software)",
    
    # Science & Nature
    "Quantum computing",
    "Photosynthesis",
    "Evolution",
    "Climate change",
    "DNA",
    "Vaccine",
    "Immune system",
    "Black hole",
    "Solar system",
    
    # History & Geography
    "World War II",
    "Ancient Egypt",
    "Mount Everest",
    "Amazon River",
    "Renaissance",
    "Roman Empire",
    "Great Wall of China",
    
    # Technology & Engineering
    "Artificial intelligence",
    "Computer science",
    "Internet",
    "Cryptocurrency",
    "Cybersecurity",
    
    # Arts & Culture
    "Jazz",
    "Literature",
    "Theatre",
    
    # Business & Economics
    "Stock market",
    "Supply chain",
    "Marketing",
    "Entrepreneurship",
    
    # Anime topics
    "Naruto",
    "Attack on Titan (TV series)",
    "One Piece (1999 TV series)",
    "Dragon Ball",
    "Anime",
    "Hunter X Hunter",
    "Studio Ghibli",
    "Manga",
]

DATA_PATH = "data"

def sanitize_filename(title):
    """Convert Wikipedia title to a safe filename"""
    # Remove special characters and replace spaces with underscores
    filename = re.sub(r'[<>:"/\\|?*]', '', title)
    filename = filename.replace(' ', '_')
    filename = filename.replace('(', '').replace(')', '')
    return filename

def download_wikipedia_articles(topics, data_path=DATA_PATH):
    """
    Download Wikipedia articles and save them as markdown files.
    
    Args:
        topics: List of Wikipedia article titles
        data_path: Directory to save the articles
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    downloaded = []
    failed = []
    

    print("Downloading Wikipedia Articles")

    
    for topic in topics:
        try:
            print(f"\nDownloading: {topic}")
            
            # Get the Wikipedia page
            page = wikipedia.page(topic, auto_suggest=False)
            
            # Create filename
            filename = sanitize_filename(topic) + ".md"
            filepath = os.path.join(data_path, filename)
            
            # Save as markdown
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {page.title}\n\n")
                f.write(f"**Source:** {page.url}\n\n")
                f.write("---\n\n")
                f.write(page.content)
            
            print(f"Saved: {filepath}")
            print(f"Length: {len(page.content)} characters")
            downloaded.append(topic)
            
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation page found for '{topic}'")
            print(f"Skipping...")
            failed.append((topic, "Disambiguation"))
            
        except wikipedia.exceptions.PageError:
            print(f"Page not found: {topic}")
            failed.append((topic, "Page not found"))
            
        except Exception as e:
            print(f"Error downloading {topic}: {str(e)}")
            failed.append((topic, str(e)))
        print(f"Successfully downloaded: {len(downloaded)}/{len(topics)}")
    print(f"Failed: {len(failed)}")
    
    if downloaded:
        print("\nDownloaded articles:")
        for topic in downloaded:
            print(f"  - {topic}")
    
    if failed:
        print("\nFailed articles:")
        for topic, reason in failed:
            print(f"  - {topic}: {reason}")
    
    return downloaded, failed

if __name__ == "__main__":
    topics = TOPICS
    downloaded, failed = download_wikipedia_articles(topics)
    print(f"\nDone. Check '{DATA_PATH}/' folder for the downloaded articles.")