import os
import time

import requests
from bs4 import BeautifulSoup

# URLs for the Latin texts we need
LATIN_TEXTS = {
    "Cicero/De_Natura_Deorum": {
        "base_url": "https://www.thelatinlibrary.com/cicero/deorum",
        "books": [1, 2, 3],  # Books 1-3
    },
    "Seneca/Letters": {
        "base_url": "https://www.thelatinlibrary.com/sen/epistulae",
        "letters": list(range(1, 125)),  # Letters 1-124
    },
    "Virgil/Aeneid": {
        "base_url": "https://www.thelatinlibrary.com/vergil/aen",
        "books": list(range(1, 13)),  # Books 1-12
    },
    "Lucretius/De_Rerum_Natura": {
        "base_url": "https://www.thelatinlibrary.com/lucretius/lucr",
        "books": list(range(1, 7)),  # Books 1-6
    },
    "Tacitus/Annals": {
        "base_url": "https://www.thelatinlibrary.com/tacitus/tac.ann",
        "books": list(range(1, 17)),  # Books 1-16
    },
}


def clean_text(text: str) -> str:
    """Clean the extracted text."""
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")

    # Remove header and navigation elements
    for element in soup.find_all(["header", "nav", "script", "style"]):
        element.decompose()

    # Get the main text content
    text = soup.get_text()

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    return text


def download_text(url: str, output_path: str) -> bool:
    """Download text from URL and save to file."""
    try:
        # Add delay to be respectful to the server
        time.sleep(2)  # Increased delay to be more conservative

        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML and extract text
        text = clean_text(response.text)

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Successfully downloaded: {output_path}")
        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def main():
    """Download all Latin texts."""
    for work_path, config in LATIN_TEXTS.items():
        base_url = config["base_url"]

        if "books" in config:
            # Handle works divided into books
            for book in config["books"]:
                url = f"{base_url}{book}.html"
                output_path = f"data/raw/{work_path}_Book_{book}.txt"
                download_text(url, output_path)

        elif "letters" in config:
            # Handle Seneca's letters
            for letter in config["letters"]:
                # Letters are grouped in sets of 10
                group = ((letter - 1) // 10) + 1
                url = f"{base_url}{group}.html"
                output_path = f"data/raw/{work_path}_Letter_{letter}.txt"
                download_text(url, output_path)


if __name__ == "__main__":
    main()
