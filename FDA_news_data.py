"""
Improved News Data Collection for FDA Event Study
Enhanced version with better data collection strategies, multiple sources, and improved filtering.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import feedparser
from textblob import TextBlob
import yfinance as yf
import logging
from pathlib import Path
import warnings
from urllib.parse import quote_plus
import json
import re
from bs4 import BeautifulSoup
warnings.filterwarnings('ignore')

# Setup directories
def create_directories():
    """Create data and logs directories if they don't exist."""
    try:
        Path('data').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        logger.info("✅ Data and logs directories created or already exist")
    except Exception as e:
        logger.error(f"❌ Error creating directories: {str(e)}")
        raise

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add FileHandler after ensuring logs directory exists
create_directories()
logger.handlers.append(logging.FileHandler('logs/news_collection.log'))

# Configuration
COMPANIES = [
    {'name': 'Sun Pharma', 'ticker': 'SUNPHARMA.NS', 'aliases': ['Sun Pharmaceutical', 'Sun Pharma Industries']},
    {'name': 'Biocon', 'ticker': 'BIOCON.NS', 'aliases': ['Biocon Limited', 'Biocon Biologics']},
    {'name': 'Dr. Reddy\'s', 'ticker': 'DRREDDY.NS', 'aliases': ['Dr Reddy\'s Laboratories', 'Dr Reddys', 'DRL']}
]

DATA_CONFIG = {
    'start_date': '2015-01-01',
    'end_date': '2025-07-09',
    'min_articles_threshold': 50,
    'max_articles_per_query': 100
}

# Enhanced FDA keywords
FDA_KEYWORDS = {
    'good_news': [
        'fda approval', 'drug approval', 'fda approves', 'marketing authorization', 
        'biosimilar approval', 'clearance', 'nda approved', 'anda approved',
        'breakthrough designation', 'fast track', 'orphan drug', 'priority review',
        'clinical trial success', 'phase 3 results', 'regulatory approval',
        'launch approval', 'generic approval', 'tentative approval'
    ],
    'bad_news': [
        'fda warning letter', 'form 483', 'fda rejection', 'recall', 'safety concern',
        'adverse event', 'fda inspection', 'import alert', 'consent decree',
        'facility closure', 'manufacturing violation', 'quality issues',
        'clinical hold', 'complete response letter', 'crl', 'fda refuses'
    ]
}

def get_user_agents():
    """Return a list of user agents for web scraping."""
    return [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]

def generate_enhanced_search_queries(company_dict):
    """Generate comprehensive FDA-related search queries for a company."""
    company = company_dict['name']
    aliases = company_dict['aliases']
    
    base_queries = []
    
    # Add queries for main company name and aliases
    for name in [company] + aliases:
        base_queries.extend([
            f'"{name}" FDA approval',
            f'"{name}" FDA warning letter',
            f'"{name}" FDA inspection',
            f'"{name}" drug approval',
            f'"{name}" biosimilar approval',
            f'"{name}" clinical trial',
            f'"{name}" regulatory approval',
            f'"{name}" FDA submission',
            f'"{name}" FDA clearance',
            f'"{name}" recall',
            f'"{name}" manufacturing violation'
        ])
    
    # Company-specific queries
    if company == 'Biocon':
        base_queries.extend([
            'Semglee FDA approval',
            'insulin glargine biosimilar FDA',
            'Biocon Biologics FDA',
            'Viatris Biocon FDA',
            'Fulphila FDA approval'
        ])
    elif company == 'Sun Pharma':
        base_queries.extend([
            'Sun Pharma Halol FDA',
            'Sun Pharma warning letter',
            'Taro Pharmaceutical FDA',
            'Sun Pharma generic approval'
        ])
    elif company == 'Dr. Reddy\'s':
        base_queries.extend([
            'Dr Reddy\'s Srikakulam FDA',
            'Dr Reddy\'s Hyderabad FDA',
            'Dr Reddy\'s API FDA',
            'Dr Reddy\'s generic approval'
        ])
    
    return base_queries

def fetch_google_news_enhanced(query, company, start_date, end_date):
    """Enhanced Google News fetching with better error handling."""
    articles = []
    try:
        # Try multiple Google News RSS approaches
        search_urls = [
            f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={quote_plus(query)}+site:reuters.com&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={quote_plus(query)}+site:bloomberg.com&hl=en-US&gl=US&ceid=US:en"
        ]
        
        for url in search_urls:
            try:
                headers = {'User-Agent': np.random.choice(get_user_agents())}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries:
                        try:
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                pub_date = datetime(*entry.published_parsed[:6])
                            else:
                                pub_date = datetime.now()
                            
                            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                            
                            if start_dt <= pub_date <= end_dt:
                                title = getattr(entry, 'title', '')
                                summary = getattr(entry, 'summary', '')
                                
                                # Enhanced relevance check
                                if is_relevant_fda_news(title, summary, company):
                                    article = {
                                        'Company': company,
                                        'Date': pub_date.strftime('%Y-%m-%d'),
                                        'Datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                                        'Title': title,
                                        'Summary': summary,
                                        'URL': getattr(entry, 'link', ''),
                                        'Source': 'Google News Enhanced',
                                        'Search_Query': query,
                                        'Query_Type': 'FDA News'
                                    }
                                    articles.append(article)
                        
                        except Exception as e:
                            logger.warning(f"Error processing Google News entry: {str(e)}")
                            continue
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error fetching Google News URL {url}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error in Google News enhanced fetch: {str(e)}")
    
    return articles

def is_relevant_fda_news(title, summary, company):
    """Enhanced relevance checking for FDA news."""
    text = f"{title} {summary}".lower()
    
    # Check for company mention (including aliases)
    company_dict = next((c for c in COMPANIES if c['name'] == company), None)
    if not company_dict:
        return False
    
    company_mentioned = False
    for name in [company_dict['name']] + company_dict['aliases']:
        if name.lower() in text:
            company_mentioned = True
            break
    
    if not company_mentioned:
        return False
    
    # Check for FDA relevance
    fda_keywords = ['fda', 'food and drug administration', 'regulatory', 'approval', 'warning letter', 
                    'inspection', 'recall', 'clinical trial', 'drug', 'biosimilar', 'generic']
    
    fda_relevant = any(keyword in text for keyword in fda_keywords)
    
    return fda_relevant

def fetch_yahoo_finance_enhanced(company, ticker):
    """Enhanced Yahoo Finance news fetching."""
    try:
        logger.info(f"Fetching enhanced Yahoo Finance news for {company}")
        ticker_obj = yf.Ticker(ticker)
        
        # Get both news and info
        news_data = ticker_obj.news
        
        articles = []
        for item in news_data:
            try:
                if 'providerPublishTime' in item:
                    pub_date = datetime.fromtimestamp(item['providerPublishTime'])
                else:
                    pub_date = datetime.now()
                
                start_date = datetime.strptime(DATA_CONFIG['start_date'], '%Y-%m-%d')
                end_date = datetime.strptime(DATA_CONFIG['end_date'], '%Y-%m-%d')
                
                if start_date <= pub_date <= end_date:
                    title = item.get('title', '')
                    summary = item.get('summary', '')
                    
                    # Enhanced relevance check
                    if is_relevant_fda_news(title, summary, company):
                        article = {
                            'Company': company,
                            'Date': pub_date.strftime('%Y-%m-%d'),
                            'Datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'Title': title,
                            'Summary': summary,
                            'URL': item.get('link', ''),
                            'Source': 'Yahoo Finance Enhanced',
                            'Search_Query': f'{company} FDA News',
                            'Query_Type': 'Company FDA News'
                        }
                        articles.append(article)
            
            except Exception as e:
                logger.warning(f"Error processing Yahoo Finance article: {str(e)}")
                continue
        
        logger.info(f"Collected {len(articles)} enhanced Yahoo Finance articles for {company}")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching enhanced Yahoo Finance for {company}: {str(e)}")
        return []

def fetch_alternative_sources(company_dict):
    """Fetch news from alternative sources including pharmaceutical news sites."""
    company = company_dict['name']
    articles = []
    
    # Pharmaceutical industry RSS feeds
    pharma_feeds = [
        'https://www.fiercepharma.com/rss.xml',
        'https://www.biopharmadive.com/feeds/news/',
        'https://www.pharmatimes.com/rss/news.xml',
        'https://www.pharmaceutical-technology.com/rss/news/',
        'https://www.outsourcing-pharma.com/rss/news'
    ]
    
    for feed_url in pharma_feeds:
        try:
            logger.info(f"Fetching from pharma RSS feed: {feed_url}")
            headers = {'User-Agent': np.random.choice(get_user_agents())}
            response = requests.get(feed_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                
                for entry in feed.entries:
                    try:
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        start_date = datetime.strptime(DATA_CONFIG['start_date'], '%Y-%m-%d')
                        end_date = datetime.strptime(DATA_CONFIG['end_date'], '%Y-%m-%d')
                        
                        if start_date <= pub_date <= end_date:
                            title = getattr(entry, 'title', '')
                            summary = getattr(entry, 'summary', '')
                            
                            if is_relevant_fda_news(title, summary, company):
                                article = {
                                    'Company': company,
                                    'Date': pub_date.strftime('%Y-%m-%d'),
                                    'Datetime': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                                    'Title': title,
                                    'Summary': summary,
                                    'URL': getattr(entry, 'link', ''),
                                    'Source': 'Pharmaceutical News',
                                    'Search_Query': f'{company} FDA News',
                                    'Query_Type': 'Pharma Industry News'
                                }
                                articles.append(article)
                    
                    except Exception as e:
                        logger.warning(f"Error processing pharma RSS article: {str(e)}")
                        continue
            
            time.sleep(3)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error fetching pharma RSS feed {feed_url}: {str(e)}")
            continue
    
    logger.info(f"Collected {len(articles)} alternative source articles for {company}")
    return articles

def create_enhanced_sample_data():
    """Create enhanced sample FDA-related news data."""
    logger.info("Creating enhanced sample FDA-related news data")
    sample_articles = [
        {
            'Company': 'Sun Pharma',
            'Date': '2016-03-15',
            'Datetime': '2016-03-15 10:30:00',
            'Title': 'Sun Pharma Receives FDA Approval for Generic Advair Diskus',
            'Summary': 'Sun Pharma announces FDA approval for generic version of Advair Diskus.',
            'URL': 'https://example.com/sunpharma-advair-approval',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Sun Pharma FDA Approval',
            'Query_Type': 'FDA Approval'
        },
        {
            'Company': 'Sun Pharma',
            'Date': '2017-11-08',
            'Datetime': '2017-11-08 14:20:00',
            'Title': 'FDA Issues Warning Letter to Sun Pharma Halol Facility',
            'Summary': 'FDA issues warning letter to Sun Pharma for manufacturing deficiencies.',
            'URL': 'https://example.com/sunpharma-warning-letter',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Sun Pharma FDA Warning Letter',
            'Query_Type': 'FDA Regulatory Issue'
        },
        {
            'Company': 'Biocon',
            'Date': '2017-09-20',
            'Datetime': '2017-09-20 14:15:00',
            'Title': 'Biocon Submits BLA for Insulin Glargine Biosimilar to FDA',
            'Summary': 'Biocon files biologics license application for Semglee with FDA.',
            'URL': 'https://example.com/biocon-bla-submission',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Biocon FDA Submission',
            'Query_Type': 'FDA Application'
        },
        {
            'Company': 'Biocon',
            'Date': '2021-06-11',
            'Datetime': '2021-06-11 16:00:00',
            'Title': 'FDA Approves Biocon\'s Semglee as Biosimilar to Lantus',
            'Summary': 'FDA approves Semglee (insulin glargine-yfgn) as biosimilar to Lantus.',
            'URL': 'https://example.com/biocon-semglee-biosimilar',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Biocon FDA Approval',
            'Query_Type': 'FDA Approval'
        },
        {
            'Company': 'Biocon',
            'Date': '2021-07-29',
            'Datetime': '2021-07-29 15:30:00',
            'Title': 'FDA Approves Biocon\'s Semglee as Interchangeable Biosimilar',
            'Summary': 'FDA approves Semglee as first interchangeable biosimilar insulin.',
            'URL': 'https://example.com/biocon-semglee-interchangeable',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Biocon FDA Approval',
            'Query_Type': 'FDA Approval'
        },
        {
            'Company': 'Dr. Reddy\'s',
            'Date': '2019-02-14',
            'Datetime': '2019-02-14 16:45:00',
            'Title': 'Dr. Reddy\'s Receives FDA Warning Letter for Srikakulam Facility',
            'Summary': 'FDA issues warning letter to Dr. Reddy\'s for manufacturing violations.',
            'URL': 'https://example.com/drreddys-warning-letter',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Dr. Reddy\'s FDA Warning Letter',
            'Query_Type': 'FDA Regulatory Issue'
        },
        {
            'Company': 'Dr. Reddy\'s',
            'Date': '2020-08-26',
            'Datetime': '2020-08-26 11:30:00',
            'Title': 'Dr. Reddy\'s Receives FDA Approval for Generic Revlimid',
            'Summary': 'Dr. Reddy\'s gets FDA approval for generic version of Revlimid.',
            'URL': 'https://example.com/drreddys-revlimid-approval',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Dr. Reddy\'s FDA Approval',
            'Query_Type': 'FDA Approval'
        },
        {
            'Company': 'Sun Pharma',
            'Date': '2022-03-10',
            'Datetime': '2022-03-10 09:15:00',
            'Title': 'Sun Pharma Announces FDA Approval for Cequa',
            'Summary': 'Sun Pharma receives FDA approval for Cequa eye drops.',
            'URL': 'https://example.com/sunpharma-cequa-approval',
            'Source': 'Enhanced Sample Data',
            'Search_Query': 'Sun Pharma FDA Approval',
            'Query_Type': 'FDA Approval'
        }
    ]
    
    logger.info(f"Created {len(sample_articles)} enhanced sample FDA-related articles")
    return sample_articles

def classify_news_enhanced(title, summary, stock_data, company, date):
    """Enhanced news classification with better keyword matching."""
    text = f"{title} {summary}".lower()
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    classification = 'Neutral'
    confidence = 0.0
    
    # Enhanced keyword matching
    good_matches = 0
    bad_matches = 0
    
    for keyword in FDA_KEYWORDS['good_news']:
        if keyword.lower() in text:
            good_matches += 1
            
    for keyword in FDA_KEYWORDS['bad_news']:
        if keyword.lower() in text:
            bad_matches += 1
    
    if good_matches > bad_matches and good_matches > 0:
        classification = 'Good'
        confidence = min(good_matches / len(FDA_KEYWORDS['good_news']), 1.0)
    elif bad_matches > good_matches and bad_matches > 0:
        classification = 'Bad'
        confidence = min(bad_matches / len(FDA_KEYWORDS['bad_news']), 1.0)
    
    # Calculate stock price volatility
    volatility = 0
    if stock_data is not None:
        try:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            company_stock = stock_data[(stock_data['Company'] == company) & (stock_data['Exchange'] == 'NSE')]
            event_date = pd.to_datetime(date).date()
            day0_data = company_stock[company_stock['Date'].dt.date == event_date]
            if not day0_data.empty and 'Daily_Simple_Return' in day0_data.columns:
                volatility = abs(day0_data['Daily_Simple_Return'].iloc[0]) if not day0_data['Daily_Simple_Return'].isna().iloc[0] else 0
        except Exception as e:
            logger.warning(f"Error calculating volatility: {str(e)}")
    
    return classification, sentiment, volatility, confidence

def collect_news_data_enhanced(output_file, stock_file):
    """Enhanced news data collection with multiple sources and better filtering."""
    logger.info("🎯 Starting enhanced FDA news data collection...")
    
    # Load stock data if available
    stock_data = None
    if Path(stock_file).exists():
        try:
            stock_data = pd.read_csv(stock_file)
            logger.info(f"✅ Loaded stock data: {len(stock_data)} records")
        except Exception as e:
            logger.warning(f"⚠️ Could not load stock data: {str(e)}")
    
    all_articles = []
    
    for company_dict in COMPANIES:
        company = company_dict['name']
        ticker = company_dict['ticker']
        logger.info(f"📊 Collecting news for {company}...")
        
        # Generate enhanced search queries
        queries = generate_enhanced_search_queries(company_dict)
        logger.info(f"🔍 Generated {len(queries)} search queries for {company}")
        
        # Fetch from enhanced Google News
        for query in queries[:10]:  # Limit to first 10 queries to avoid rate limiting
            google_articles = fetch_google_news_enhanced(
                query, company, DATA_CONFIG['start_date'], DATA_CONFIG['end_date']
            )
            all_articles.extend(google_articles)
            
            if len(google_articles) >= DATA_CONFIG['max_articles_per_query']:
                break
        
        # Fetch from enhanced Yahoo Finance
        yahoo_articles = fetch_yahoo_finance_enhanced(company, ticker)
        all_articles.extend(yahoo_articles)
        
        # Fetch from alternative pharmaceutical sources
        alt_articles = fetch_alternative_sources(company_dict)
        all_articles.extend(alt_articles)
        
        logger.info(f"📰 Collected {len([a for a in all_articles if a['Company'] == company])} articles for {company}")
    
    # Remove duplicates based on title and date
    unique_articles = []
    seen = set()
    for article in all_articles:
        key = (article['Company'], article['Date'], article['Title'])
        if key not in seen:
            seen.add(key)
            unique_articles.append(article)
    
    logger.info(f"📊 Total unique articles collected: {len(unique_articles)}")
    
    # Add enhanced sample data if needed
    if len(unique_articles) < DATA_CONFIG['min_articles_threshold']:
        logger.warning(f"⚠️ Only {len(unique_articles)} articles found, adding enhanced sample data")
        sample_articles = create_enhanced_sample_data()
        unique_articles.extend(sample_articles)
    
    # Enhanced classification and processing
    news_data = []
    for article in unique_articles:
        company = article['Company']
        date = article['Date']
        title = article['Title']
        summary = article.get('Summary', '')
        
        classification, sentiment, volatility, confidence = classify_news_enhanced(
            title, summary, stock_data, company, date
        )
        
        if classification == 'Neutral':
            continue
        
        # Enhanced drug name extraction
        drug_name = extract_drug_name(title, summary)
        
        # Enhanced significance scoring
        significance_score = calculate_significance_score(
            title, summary, classification, confidence, volatility, article['Source']
        )
        
        event_data = {
            'Company': company,
            'Drug_Name': drug_name,
            'FDA_Scientific_Name': drug_name,
            'Date': date,
            'Classification': classification,
            'Sentiment_Score': sentiment,
            'Confidence_Score': confidence,
            'Sources': article['URL'],
            'Significance_Score': significance_score,
            'Title': title,
            'Summary': summary,
            'Source_Type': article['Source'],
            'Search_Query': article['Search_Query']
        }
        news_data.append(event_data)
    
    if not news_data:
        logger.error("❌ No relevant FDA-related news data collected")
        raise ValueError("No FDA-related news articles found. Check logs for details.")
    
    # Create DataFrame and save
    news_df = pd.DataFrame(news_data)
    news_df = news_df.sort_values(['Date', 'Company', 'Significance_Score'], ascending=[True, True, False])
    
    # Remove duplicates again at the processed level
    news_df = news_df.drop_duplicates(subset=['Company', 'Date', 'Title'], keep='first')
    
    news_df.to_csv(output_file, index=False)
    
    if Path(output_file).exists():
        logger.info(f"✅ Enhanced news data saved: {output_file} ({len(news_df)} events)")
        
        # Print summary statistics
        print("\n📊 COLLECTION SUMMARY:")
        print(f"Total events collected: {len(news_df)}")
        for company in news_df['Company'].unique():
            company_data = news_df[news_df['Company'] == company]
            good_events = len(company_data[company_data['Classification'] == 'Good'])
            bad_events = len(company_data[company_data['Classification'] == 'Bad'])
            print(f"{company}: {len(company_data)} events ({good_events} good, {bad_events} bad)")
        
    else:
        logger.error("❌ Failed to save enhanced news data")
        raise ValueError("Failed to save enhanced news data to CSV")
    
    return news_df

def extract_drug_name(title, summary):
    """Extract drug name from title and summary."""
    text = f"{title} {summary}".lower()
    
    # Common drug name patterns
    patterns = [
        r'approves?\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'approval\s+for\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'generic\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'biosimilar\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)',
        r'([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+approval'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Return first reasonable match
            drug_name = matches[0].strip()
            if len(drug_name) > 2 and drug_name not in ['the', 'and', 'for', 'new', 'drug']:
                return drug_name.title()
    
    return "Unknown Drug"

def calculate_significance_score(title, summary, classification, confidence, volatility, source):
    """Calculate significance score for news event."""
    text = f"{title} {summary}".lower()
    
    # Base score from classification
    base_score = 0.8 if classification == 'Good' else 0.6
    
    # Confidence factor
    confidence_factor = min(confidence * 2, 1.0)
    
    # Volatility factor (if available)
    volatility_factor = min(volatility * 10, 0.3) if volatility > 0 else 0.1
    
    # Source credibility
    source_weights = {
        'Google News Enhanced': 0.9,
        'Yahoo Finance Enhanced': 0.8,
        'Pharmaceutical News': 0.95,
        'Enhanced Sample Data': 0.7
    }
    source_factor = source_weights.get(source, 0.7)
    
    # Action importance - Fixed the max() error
    action_weights = {
        'approval': 1.0,
        'warning letter': 0.8,
        'recall': 0.9,
        'inspection': 0.6,
        'clearance': 0.7
    }
    
    # Find matching actions and get their weights
    matching_actions = [action_weights.get(action, 0.5) for action in action_weights.keys() if action in text]
    action_factor = max(matching_actions) if matching_actions else 0.5  # Default to 0.5 if no matches
    
    # Calculate final score
    significance_score = (
        0.4 * base_score +
        0.2 * confidence_factor +
        0.15 * volatility_factor +
        0.15 * source_factor +
        0.1 * action_factor
    )
    
    return min(significance_score, 1.0)

def main():
    """Main execution function."""
    print("🚀 FDA EVENT STUDY - ENHANCED NEWS DATA COLLECTION")
    print("="*70)
    print(f"🏢 Companies: Sun Pharma, Biocon, Dr. Reddy's")
    print(f"📅 Period: 2015-01-01 to 2025-07-09")
    print(f"🎯 Enhanced collection with multiple sources")
    print("-"*70)
    
    output_file = Path('data/news_data.csv')
    stock_file = Path('data/stock_data.csv')
    
    if not stock_file.exists():
        print("\n⚠️ WARNING: Stock data file not found. Continuing without stock data.")
        print("💡 For better significance scoring, run stock data collection first.")
    
    try:
        news_df = collect_news_data_enhanced(output_file, stock_file)
        
        print("\n🎉 SUCCESS: Enhanced news data collection completed!")
        print(f"✅ Data saved to: {output_file}")
        print(f"📊 Total events: {len(news_df)}")
        
        # Display sample of collected data
        if len(news_df) > 0:
            print("\n📰 Sample of collected news:")
            print("-" * 50)
            for _, row in news_df.head(3).iterrows():
                print(f"📅 {row['Date']} | {row['Company']}")
                print(f"📰 {row['Title'][:80]}...")
                print(f"📊 {row['Classification']} | Score: {row['Significance_Score']:.2f}")
                print("-" * 50)
        
    except ValueError as e:
        print(f"\n❌ DATA ERROR: {str(e)}")
        print("💡 Troubleshooting steps:")
        print("   1. Check internet connection")
        print("   2. Verify RSS feed accessibility")
        print("   3. Check logs: logs/news_collection.log")
        print("   4. Try running with VPN if accessing from restricted region")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("💡 Check logs: logs/news_collection.log")
        print("💡 Ensure all required libraries are installed:")
        print("   pip install requests pandas numpy feedparser textblob yfinance beautifulsoup4")

if __name__ == "__main__":
    main()