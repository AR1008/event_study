"""
Data Processing for FDA Event Study
Combines stock and news data, filters for non-overlapping 11-day event windows, and excludes follow-up articles.
"""
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import numpy as np
from fuzzywuzzy import fuzz

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
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
create_directories()
logger.handlers.append(logging.FileHandler('logs/data_processing.log'))

def is_similar_event(event1, event2, threshold=80):
    """Check if two events are similar based on drug name and title."""
    drug1 = str(event1['Drug_Name']).lower()
    drug2 = str(event2['Drug_Name']).lower()
    title1 = str(event1['Title']).lower()
    title2 = str(event2['Title']).lower()
    
    drug_similarity = fuzz.ratio(drug1, drug2) if drug1 != 'unknown drug' and drug2 != 'unknown drug' else 0
    title_similarity = fuzz.ratio(title1, title2)
    
    return drug_similarity > threshold or title_similarity > threshold

def extract_event_window(stock_data, event_date, company, exchange, min_days=9):
    """Extract 11-day event window from stock data, handling non-trading days."""
    event_date = pd.to_datetime(event_date).date()
    window_start = event_date - timedelta(days=5)
    window_end = event_date + timedelta(days=5)
    
    company_stock = stock_data[
        (stock_data['Company'] == company) & 
        (stock_data['Exchange'] == exchange) & 
        (stock_data['Date'].dt.date >= window_start) & 
        (stock_data['Date'].dt.date <= window_end)
    ].sort_values('Date')
    
    if company_stock.empty:
        logger.warning(f"No stock data for {company} ({exchange}) in window {window_start} to {window_end}")
        return None
    
    all_dates = pd.date_range(window_start, window_end, freq='D')
    trading_days = stock_data[stock_data['Exchange'] == exchange]['Date'].dt.date.unique()
    
    if len(trading_days) == 0:
        logger.warning(f"No trading days found for {exchange} in stock data")
        return None
    
    adjusted_data = []
    missing_dates = []
    for i, day in enumerate(all_dates):
        closest_day = min(trading_days, key=lambda x: abs((x - day.date()).days))
        if abs((closest_day - day.date()).days) > 2:
            missing_dates.append(day.date())
            adjusted_data.append({
                'Date': day,
                'Close': np.nan,
                'Daily_Simple_Return': np.nan,
                'Daily_Log_Return': np.nan,
                'Day': i - 5
            })
        else:
            row = company_stock[company_stock['Date'].dt.date == closest_day]
            if not row.empty:
                adjusted_data.append({
                    'Date': day,
                    'Close': row['Close'].iloc[0],
                    'Daily_Simple_Return': row['Daily_Simple_Return'].iloc[0],
                    'Daily_Log_Return': row['Daily_Log_Return'].iloc[0],
                    'Day': i - 5
                })
            else:
                adjusted_data.append({
                    'Date': day,
                    'Close': np.nan,
                    'Daily_Simple_Return': np.nan,
                    'Daily_Log_Return': np.nan,
                    'Day': i - 5
                })
    
    if len([d for d in adjusted_data if not np.isnan(d['Close'])]) < min_days:
        logger.warning(f"Incomplete window for {company} on {event_date} ({exchange}): {len(adjusted_data)} days. Missing: {missing_dates}")
        return None
    
    return pd.DataFrame(adjusted_data)

def process_data(news_file, stock_file, good_output, bad_output, metadata_output):
    """Process stock and news data, ensuring non-overlapping events and excluding follow-up articles."""
    logger.info("🎯 Starting data processing...")
    
    # Read and validate input files
    try:
        news_df = pd.read_csv(news_file)
        if news_df.empty:
            logger.error("❌ News data is empty")
            raise ValueError("News data is empty")
        news_df['Date'] = pd.to_datetime(news_df['Date'])
    except Exception as e:
        logger.error(f"❌ Error reading news data: {str(e)}")
        raise
    
    try:
        stock_df = pd.read_csv(stock_file)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        if stock_df.empty:
            logger.error("❌ Stock data is empty")
            raise ValueError("Stock data is empty")
        if 'Daily_Simple_Return' not in stock_df.columns or 'Daily_Log_Return' not in stock_df.columns:
            logger.info("Calculating missing returns in stock data")
            stock_df['Daily_Simple_Return'] = stock_df.groupby(['Company', 'Exchange'])['Close'].pct_change()
            stock_df['Daily_Log_Return'] = np.log(stock_df['Close'] / stock_df.groupby(['Company', 'Exchange'])['Close'].shift(1))
    except Exception as e:
        logger.error(f"❌ Error reading stock data: {str(e)}")
        raise
    
    # Validate stock data coverage
    companies = ['Sun Pharma', 'Biocon', 'Dr. Reddy’s']
    exchanges = ['NSE', 'BSE']
    for company in companies:
        for exchange in exchanges:
            company_stock = stock_df[(stock_df['Company'] == company) & (stock_df['Exchange'] == exchange)]
            if company_stock.empty:
                logger.warning(f"⚠️ No stock data for {company} ({exchange}). Run stock_data.py to fetch missing data.")
            else:
                min_date = company_stock['Date'].min().date()
                max_date = company_stock['Date'].max().date()
                if min_date > datetime(2015, 1, 1).date() or max_date < datetime(2025, 7, 9).date():
                    logger.warning(f"⚠️ Incomplete date range for {company} ({exchange}): {min_date} to {max_date}")
    
    # Select non-overlapping events and exclude follow-up articles
    selected_events = []
    for company in companies:
        company_news = news_df[news_df['Company'] == company].sort_values('Date')
        i = 0
        while i < len(company_news):
            event = company_news.iloc[i].copy()
            event_date = pd.to_datetime(event['Date'])
            window_start = event_date - timedelta(days=5)
            window_end = event_date + timedelta(days=5)
            
            # Find overlapping events within 11-day window
            overlapping = company_news[
                (company_news['Date'] >= window_start) & 
                (company_news['Date'] <= window_end) & 
                (company_news.index != company_news.index[i])
            ]
            
            if not overlapping.empty:
                all_overlapping = pd.concat([pd.DataFrame([event]), overlapping])
                selected_event = all_overlapping.loc[all_overlapping['Significance_Score'].idxmax()].to_dict()
                selected_event['Notes'] = f"Selected over {len(overlapping)} overlapping events"
                selected_events.append(selected_event)
                i += len(overlapping) + 1
            else:
                is_follow_up = False
                for prev_event in selected_events:
                    prev_date = pd.to_datetime(prev_event['Date'])
                    if event_date > prev_date + timedelta(days=5) and is_similar_event(event, prev_event):
                        is_follow_up = True
                        logger.info(f"Excluding follow-up event for {company} on {event_date.date()}: {event['Title'][:50]}...")
                        break
                if not is_follow_up:
                    event_dict = event.to_dict()
                    event_dict['Notes'] = ''
                    selected_events.append(event_dict)
                i += 1
    
    selected_events_df = pd.DataFrame(selected_events)
    if selected_events_df.empty:
        logger.error("❌ No valid events after overlap and follow-up handling")
        raise ValueError("No valid events after overlap and follow-up handling")
    
    # Process stock data for selected events
    event_data = []
    for event in selected_events_df.to_dict('records'):
        company = event['Company']
        event_date = pd.to_datetime(event['Date'])
        classification = event['Classification']
        
        for exchange in ['NSE', 'BSE']:
            window_data = extract_event_window(stock_df, event_date, company, exchange, min_days=9)
            if window_data is None:
                logger.warning(f"⚠️ Skipping event for {company} on {event_date.date()} ({exchange}) due to incomplete data")
                continue
            
            row = {
                'Company': company,
                'Exchange': exchange,
                'Drug_Name': event['Drug_Name'],
                'FDA_Scientific_Name': event['FDA_Scientific_Name'],
                'Date': event_date,
                'Classification': classification,
            }
            
            # Add daily returns
            for day in range(-5, 6):
                day_data = window_data[window_data['Day'] == day]
                if not day_data.empty:
                    row[f'Day_{day}_Simple'] = day_data['Daily_Simple_Return'].iloc[0]
                    row[f'Day_{day}_Log'] = day_data['Daily_Log_Return'].iloc[0]
                else:
                    row[f'Day_{day}_Simple'] = np.nan
                    row[f'Day_{day}_Log'] = np.nan
            
            # Calculate cumulative returns
            pre_event = window_data[window_data['Day'].between(-5, -1)]
            post_event = window_data[window_data['Day'].between(1, 5)]
            row['Cumulative_Simple_-5_to_-1'] = pre_event['Daily_Simple_Return'].sum()
            row['Cumulative_Log_-5_to_-1'] = pre_event['Daily_Log_Return'].sum()
            row['Cumulative_Simple_+1_to_+5'] = post_event['Daily_Simple_Return'].sum()
            row['Cumulative_Log_+1_to_+5'] = post_event['Daily_Log_Return'].sum()
            
            event_data.append(row)
    
    event_df = pd.DataFrame(event_data)
    if event_df.empty:
        logger.error("❌ No valid stock data for selected events")
        raise ValueError("No valid stock data for selected events")
    
    # Split into good and bad news
    good_news = event_df[event_df['Classification'] == 'Good']
    bad_news = event_df[event_df['Classification'] == 'Bad']
    
    # Save datasets
    good_news.to_csv(good_output, index=False)
    bad_news.to_csv(bad_output, index=False)
    
    # Save metadata
    metadata = selected_events_df[[
        'Company', 'Drug_Name', 'FDA_Scientific_Name', 'Date', 'Classification',
        'Sentiment_Score', 'Confidence_Score', 'Sources', 'Significance_Score',
        'Title', 'Summary', 'Source_Type', 'Search_Query', 'Notes'
    ]].rename(columns={'Classification': 'Event_Type'})
    metadata.to_csv(metadata_output, index=False)
    
    logger.info(f"✅ Good news saved: {good_output} ({len(good_news)} rows)")
    logger.info(f"✅ Bad news saved: {bad_output} ({len(bad_news)} rows)")
    logger.info(f"✅ Metadata saved: {metadata_output} ({len(metadata)} rows)")
    
    # Print summary
    print("\n📊 PROCESSING SUMMARY:")
    print(f"Total events processed: {len(selected_events_df)}")
    for company in companies:
        company_events = selected_events_df[selected_events_df['Company'] == company]
        good_events = len(company_events[company_events['Classification'] == 'Good'])
        bad_events = len(company_events[company_events['Classification'] == 'Bad'])
        print(f"{company}: {len(company_events)} events ({good_events} good, {bad_events} bad)")

def main():
    """Main execution function."""
    print("🚀 FDA EVENT STUDY - DATA PROCESSING")
    print("="*60)
    print(f"🏢 Companies: Sun Pharma, Biocon, Dr. Reddy’s")
    print(f"📅 Period: 2015-01-01 to 2025-07-09")
    print("-"*60)
    
    news_file = Path('data/news_data.csv')
    stock_file = Path('data/stock_data.csv')
    good_output = Path('data/good_news.csv')
    bad_output = Path('data/bad_news.csv')
    metadata_output = Path('data/metadata.csv')
    
    if not news_file.exists() or not stock_file.exists():
        print("\n❌ ERROR: Input files missing. Run stock_data.py and FDA_news_data.py first.")
        print(f"💡 Expected: {news_file} and {stock_file}")
        return
    
    try:
        process_data(news_file, stock_file, good_output, bad_output, metadata_output)
        print("\n🎉 SUCCESS: Data processing completed!")
        print(f"✅ Good news saved to: {good_output}")
        print(f"✅ Bad news saved to: {bad_output}")
        print(f"✅ Metadata saved to: {metadata_output}")
    except ValueError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("💡 Check logs: logs/data_processing.log")
        print("💡 Troubleshooting steps:")
        print("   1. Ensure stock_data.csv contains data for NSE and BSE for all companies")
        print("   2. Verify date range in stock_data.csv covers 2015-01-01 to 2025-07-09")
        print("   3. Run stock_data.py again if BSE data is missing")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print("💡 Check logs: logs/data_processing.log")
        print("💡 Ensure required libraries are installed: pip install pandas numpy fuzzywuzzy")

if __name__ == "__main__":
    main()