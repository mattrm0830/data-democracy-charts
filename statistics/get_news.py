import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import os
from dotenv import load_dotenv
import geopandas as gpd
from datetime import datetime, timedelta
import openai
import ssl
import time
from pathlib import Path

# Create directories if they don't exist
Path("statistics").mkdir(exist_ok=True)
Path("visualizations").mkdir(exist_ok=True)

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def get_news_articles(query, days_back=30):
    """
    Fetch news articles from NewsAPI.ai
    """
    base_url = "https://newsapi.ai/api/v4/article/getArticles"
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates as required by NewsAPI.ai
    date_start = start_date.strftime('%Y-%m-%d')
    date_end = end_date.strftime('%Y-%m-%d')
    
    # Build the body for the POST request
    request_body = {
        "action": "getArticles",
        "keyword": query,
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "articlesPage": 1,
        "articlesCount": 100,
        "articleBodyLen": -1,  # Get full article body
        "resultType": "articles",
        "dataType": ["news", "blog"],
        "apiKey": NEWS_API_KEY,
        "forceMaxDataTimeWindow": 31,
        "lang": "eng",
        "dateStart": date_start,
        "dateEnd": date_end
    }
    
    all_articles = []
    page = 1
    max_pages = 10  # Limit to prevent excessive API usage
    
    while page <= max_pages:
        request_body["articlesPage"] = page
        
        try:
            print(f"\nMaking request to NewsAPI.ai with query: {query}, page: {page}")
            
            # Using POST instead of GET for NewsAPI.ai
            response = requests.post(base_url, json=request_body)
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Error response: {response.text[:500]}...")
                break
                
            data = response.json()
            
            # Extract articles from the response
            articles = data.get('articles', {}).get('results', [])
            
            if not articles:
                print("No more articles found")
                break
                
            total_articles = data.get('articles', {}).get('totalResults', 0)
            print(f"Found {len(articles)} articles on page {page} (Total available: {total_articles})")
            
            # Transform the articles to match our expected format
            transformed_articles = []
            for article in articles:
                source_name = article.get('source', {}).get('title', 'Unknown')
                
                # Extract the text content
                body = article.get('body', '')
                title = article.get('title', '')
                description = article.get('description', '') or article.get('extract', '')
                
                transformed_articles.append({
                    'title': title,
                    'description': description,
                    'body': body,
                    'url': article.get('url', ''),
                    'source': {'name': source_name},
                    'publishedAt': article.get('dateTime', datetime.now().isoformat()),
                    'date': article.get('date', datetime.now().strftime('%Y-%m-%d'))
                })
            
            all_articles.extend(transformed_articles)
            
            # If we've reached the end of available articles
            if len(articles) < request_body["articlesCount"]:
                print(f"Reached the end of available articles at page {page}")
                break
                
            page += 1
            
            # Respect API rate limits with a short delay
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response text: {response.text[:500]}...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    print(f"Total articles collected: {len(all_articles)}")
    return all_articles

# Function to analyze political leaning using OpenAI
def analyze_political_leaning(text):
    """
    Analyze the political leaning of a text using OpenAI's GPT-4
    """
    if not text or len(text.strip()) < 10:
        print("Text too short for analysis, returning neutral score")
        return 0
        
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a political analyst. Analyze the following text and determine its political leaning on a scale from -10 (very liberal/left) to 10 (very conservative/right), where 0 is neutral/centrist. Respond with only the numerical score."},
                {"role": "user", "content": text[:800]}  # Using more context for better analysis
            ],
            max_tokens=10
        )
        score_text = response.choices[0].message.content.strip()
        
        # Handle potential non-numeric responses
        try:
            score = float(score_text)
            normalized_score = max(min(score, 10), -10)  # Ensure within -10 to 10 range
            return normalized_score
        except ValueError:
            print(f"Could not convert response to float: '{score_text}'")
            return 0
            
    except Exception as e:
        print(f"Error analyzing political leaning: {e}")
        return 0  # Default to neutral if analysis fails

# Extract state mentions from article
def extract_states(text):
    """
    Extract mentions of US states from article text
    """
    if not text:
        return []
        
    # List of US states and their abbreviations
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
        "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
        "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", 
        "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", 
        "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", 
        "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", 
        "VA", "WA", "WV", "WI", "WY"
    ]
    
    # Map of abbreviations to full names
    abbr_to_name = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
        "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
        "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
        "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
        "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
    }
    
    mentioned_states = []
    
    # Check each potential state name in the text
    # Using case-insensitive word boundary search to avoid false positives
    for state in states:
        # Check if state appears as a whole word
        if f" {state} " in f" {text} " or f" {state}," in text or f" {state}." in text:
            if len(state) == 2:  # It's an abbreviation
                state_name = abbr_to_name.get(state)
                if state_name and state_name not in mentioned_states:
                    mentioned_states.append(state_name)
            elif state not in mentioned_states:
                mentioned_states.append(state)
    
    return mentioned_states

# Main processing function
def compile_news_political_data():
    """
    Compile news data with political leaning analysis by state
    """
    # Get articles for multiple topics to ensure broad coverage
    topics = ["politics", "election", "governor", "senator", "congress", "state legislation"]
    all_articles = []
    
    for topic in topics:
        print(f"\n=== Fetching articles for topic: {topic} ===")
        articles = get_news_articles(topic)
        if articles:
            print(f"Found {len(articles)} articles for topic: {topic}")
            all_articles.extend(articles)
        else:
            print(f"No articles found for topic: {topic}")
    
    if not all_articles:
        print("No articles were fetched. Please check your API key and try again.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Remove duplicates based on URL
    unique_urls = set()
    unique_articles = []
    for article in all_articles:
        if article['url'] not in unique_urls:
            unique_urls.add(article['url'])
            unique_articles.append(article)
    
    print(f"\nProcessing {len(unique_articles)} unique articles...")
    
    # Process each article
    processed_data = []
    for i, article in enumerate(unique_articles):
        if i % 5 == 0:
            print(f"Processing article {i+1}/{len(unique_articles)}")
            
        # Get the most comprehensive text available for analysis
        title = article.get('title', '')
        description = article.get('description', '')
        body = article.get('body', '')
        
        # Combine available text for the most complete content
        full_text = title
        if description:
            full_text += " " + description
        if body:
            full_text += " " + body
        
        # Extract states mentioned
        states = extract_states(full_text)
        
        # Only process if states are mentioned
        if states:
            # Analyze political leaning
            leaning = analyze_political_leaning(full_text)
            
            # Add to processed data
            for state in states:
                processed_data.append({
                    'state': state,
                    'title': title,
                    'url': article['url'],
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'political_leaning': leaning,
                    'date': article.get('date', datetime.now().strftime('%Y-%m-%d'))
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    if not df.empty:
        # Save the processed data
        df.to_csv('news_political_data.csv', index=False)
        print(f"\nSaved {len(df)} processed articles with state mentions to statistics/news_political_data.csv")
    else:
        print("\nNo articles with state mentions were found.")
    
    return df

# Generate visualizations
def generate_visualizations(df):
    """
    Generate visualizations from the processed news data
    """
    print("\nGenerating visualizations...")
    
    # Create a directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Prepare data: Aggregate political leanings by state
    state_leanings = df.groupby('state')['political_leaning'].mean().reset_index()
    state_leanings['normalized_leaning'] = state_leanings['political_leaning'] / 10  # Normalize to -1 to 1 scale
    
    # Sort by political leaning
    state_leanings_sorted = state_leanings.sort_values('political_leaning')
    
    # Add counts for each state
    state_counts = df['state'].value_counts().reset_index()
    state_counts.columns = ['state', 'article_count']
    state_leanings = state_leanings.merge(state_counts, on='state', how='left')
    
    # 2. Bar chart of political leanings by state
    plt.figure(figsize=(14, 12))
    ax = sns.barplot(x='political_leaning', y='state', data=state_leanings_sorted, 
                    palette="coolwarm", hue='political_leaning', dodge=False)
    
    # Add count annotations
    for i, row in state_leanings_sorted.iterrows():
        ax.text(0, i, f" n={row['article_count']}", va='center')
    
    plt.axvline(x=0, color='purple', linestyle='--')
    plt.xlabel('Political Leaning (Left [-10] to Right [+10])')
    plt.ylabel('State')
    plt.title('Average Political Leaning of News Articles by State')
    plt.tight_layout()
    plt.savefig('visualizations/state_political_leanings.png', dpi=300)
    print("Saved state political leanings bar chart")
    
    # 3. Create a US map visualization
    try:
        # Set SSL context to unverified for the geojson download
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Load US states map
        usa = gpd.read_file('https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json')
        
        # Merge the political leaning data with the map data
        merged = usa.merge(state_leanings, left_on='name', right_on='state', how='left')
        
        # Create a choropleth map
        fig, ax = plt.subplots(1, figsize=(16, 10))
        
        # Add data to map
        merged.plot(column='political_leaning', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8',
                   legend=True, legend_kwds={'label': "Political Leaning (Left [-10] to Right [+10])", 
                                            'orientation': "horizontal"}, 
                   missing_kwds={'color': 'lightgrey'})
        
        # Add state names and article counts as annotations
        for idx, row in merged.iterrows():
            if pd.notna(row['political_leaning']):
                plt.annotate(text=f"{row['name']}\n(n={row['article_count']})", 
                            xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                            horizontalalignment='center', fontsize=8)
            else:
                plt.annotate(text=f"{row['name']}\n(no data)", 
                            xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
                            horizontalalignment='center', fontsize=8, color='grey')
        
        plt.title('Political Leaning of News Coverage by State')
        plt.tight_layout()
        plt.savefig('visualizations/us_map_political_leanings.png', dpi=300)
        print("Saved US map visualization")
        
    except Exception as e:
        print(f"Error creating map visualization: {e}")
    
    # 4. Article count by state
    plt.figure(figsize=(14, 10))
    sns.barplot(x='article_count', y='state', 
               data=state_counts.sort_values('article_count', ascending=False).head(20))
    plt.xlabel('Number of Mentions in Articles')
    plt.ylabel('State')
    plt.title('Top 20 States by News Coverage')
    plt.tight_layout()
    plt.savefig('visualizations/state_mention_counts.png', dpi=300)
    print("Saved article count visualization")
    
    # 5. Scatter plot of political leaning vs coverage
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=state_leanings, x='political_leaning', y='article_count', 
                   size='article_count', sizes=(20, 500), hue='political_leaning', 
                   palette='coolwarm')
    
    # Add state labels to points
    for i, row in state_leanings.iterrows():
        plt.annotate(row['state'], (row['political_leaning'], row['article_count']), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.axvline(x=0, color='purple', linestyle='--')
    plt.xlabel('Political Leaning (Left [-10] to Right [+10])')
    plt.ylabel('Number of Articles')
    plt.title('Relationship Between Political Leaning and News Coverage by State')
    plt.tight_layout()
    plt.savefig('visualizations/leaning_vs_coverage.png', dpi=300)
    print("Saved political leaning vs coverage scatter plot")
    
    print("\nAll visualizations saved to the 'visualizations' directory!")

# Main execution
if __name__ == "__main__":
    print("Starting news political leaning analysis...")
    df = compile_news_political_data()
    if not df.empty:
        generate_visualizations(df)
        print("\nAnalysis completed successfully!")
    else:
        print("\nNo data to visualize. Please check your API key and try again.")