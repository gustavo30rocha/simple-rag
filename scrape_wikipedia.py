# download_wikipedia.py
import os
import wikipedia
import re

wikipedia.set_lang("en")

# Topics we want to download for our database
TOPICS = [
    # Technical topics - good for keyword matching (sparse search)
    "Python (programming language)",
    "Machine learning",
    "Database",
    "REST API",
    "SQL",
    
    # Anime topics - good for semantic understanding (dense search)
    "Naruto",
    "Attack on Titan (TV series)",
    "One Piece (1999 TV series)",
    "Dragon Ball",
    "Anime",
    "Hunter X Hunter",
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
    
    print("=" * 60)
    print("Downloading Wikipedia Articles")
    print("=" * 60)
    
    for topic in topics:
        try:
            print(f"\n📥 Downloading: {topic}")
            
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
            print(f"Options: {', '.join(e.options[:5])}")
            print(f"Skipping...")
            failed.append((topic, "Disambiguation"))
            
        except wikipedia.exceptions.PageError:
            print(f"Page not found: {topic}")
            failed.append((topic, "Page not found"))
            
        except Exception as e:
            print(f"Error downloading {topic}: {str(e)}")
            failed.append((topic, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
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
    # You can customize the topics list here
    topics = TOPICS
    
    print(f"Will download {len(topics)} Wikipedia articles to '{DATA_PATH}/'")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        downloaded, failed = download_wikipedia_articles(topics)
        print(f"\nDone! Check the '{DATA_PATH}/' folder for the downloaded articles.")
        print("\nNext steps:")
        print("1. Review the downloaded articles")
        print("2. Run: python create_database.py")
        print("3. Test with: python query_data.py 'your query' --hybrid")
    else:
        print("Cancelled.")