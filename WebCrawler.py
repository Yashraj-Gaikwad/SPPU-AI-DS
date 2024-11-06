'''
Build the web crawler to pull product information and links from an e-commerce website.

viva questions
1)
'''

# Import Libraries
# HTTP requests
import requests
# For web scraping
from bs4 import BeautifulSoup
# for time functions
import time
# to create random delays
import random

# Function to get product information from a search results page
def get_amazon_products(search_query):
    # Construct the URL for the search query
    url = f"https://www.amazon.in/s?k={search_query}"
    
    # Set headers to mimic a browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Send GET request
    response = requests.get(url, headers=headers)
    
    # Check if request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find product listings
        products = soup.find_all('div', {'data-component-type': 's-search-result'})
        
        # Open a text file to save data
        with open('amazon_products.txt', 'w', encoding='utf-8') as file:
            for product in products:
                # Extract title
                title = product.h2.text.strip()
                # construct link
                link = "https://www.amazon.in" + product.h2.a['href']
                # exception handling
                # to extract price
                try:
                    price = product.find('span', 'a-price').find('span', 'a-offscreen').text.strip()
                except AttributeError:
                    price = "Price not available"
                
                # Write product information to the file
                file.write(f"Product Name: {title}\n")
                file.write(f"Product Link: {link}\n")
                file.write(f"Product Price: {price}\n")
                file.write('-' * 40 + '\n')
        
        print("Product information saved to amazon_products.txt.")
        
        # Delay between requests to avoid being blocked
        time.sleep(random.uniform(1, 3))
    else:
        print("Failed to retrieve products")

# Example usage
search_term = "smartphone"
get_amazon_products(search_term)
