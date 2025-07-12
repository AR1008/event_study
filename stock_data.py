"""
Improved Stock Data Collection for FDA Event Study - NSE ONLY
Focuses exclusively on NSE data for better consistency and reliability.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import requests
from retry import retry
import time
from requests.exceptions import HTTPError, RequestException
import socket

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
logger.handlers.append(logging.FileHandler('logs/stock_collection.log'))

class NSEStockDataCollector:
    """Collects daily stock data for specified companies from NSE only."""
    
    def __init__(self):
        self.start_date = datetime(2015, 1, 1)
        self.end_date = datetime(2025, 7, 9)
        # NSE ONLY - Single source for consistency
        self.companies = {
            'Sun Pharma': 'SUNPHARMA.NS',
            'Biocon': 'BIOCON.NS',
            'Dr. Reddy\'s': 'DRREDDY.NS'  # Proper apostrophe handling
        }
        self.collected_data = {}
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        logger.info(f"🚀 NSE Stock Data Collector Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"📈 Exchange: NSE (National Stock Exchange) only")
    
    def _validate_symbol(self, symbol):
        """Validate NSE symbol availability."""
        try:
            logger.info(f"  Validating symbol: {symbol}")
            time.sleep(2)  # Rate limiting
            ticker = yf.Ticker(symbol)
            
            # Test with a small period first
            test_data = ticker.history(period="5d", timeout=15)
            
            if test_data.empty:
                logger.warning(f"⚠️ No recent data returned for {symbol}")
                return False
            
            logger.info(f"  ✅ Symbol {symbol} validated successfully")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Symbol validation failed for {symbol}: {str(e)}")
            return False
    
    @retry(tries=3, delay=10, backoff=2)
    def download_stock_data(self, symbol, company_name):
        """Download stock data from Yahoo Finance for NSE symbols."""
        logger.info(f"📈 Downloading {company_name} ({symbol}) data...")
        
        # Validate symbol first
        if not self._validate_symbol(symbol):
            logger.error(f"❌ Symbol validation failed for {symbol}")
            return None
        
        try:
            time.sleep(3)  # Rate limiting
            ticker = yf.Ticker(symbol)
            
            # Download historical data
            df = ticker.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                timeout=30,
                interval='1d',
                actions=False  # Don't include dividends/splits for cleaner data
            )
            
            if df.empty:
                logger.error(f"❌ Empty data returned for {symbol}")
                return None
            
            # Check data quality
            if len(df) < 1000:  # Less than ~4 years of data
                logger.warning(f"⚠️ Limited data for {symbol}: {len(df)} records")
            
            # Process the data
            df.reset_index(inplace=True)
            df['Symbol'] = symbol
            df['Company'] = company_name
            df['Exchange'] = 'NSE'
            df['Source'] = 'Yahoo Finance'
            
            # Calculate returns
            df['Daily_Simple_Return'] = df['Close'].pct_change()
            df['Daily_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Calculate volatility (rolling 20-day)
            df['Volatility_20d'] = df['Daily_Simple_Return'].rolling(window=20).std()
            
            # Remove first row (NaN return)
            df = df.dropna(subset=['Daily_Simple_Return'])
            
            # Quality checks
            extreme_returns = df[abs(df['Daily_Simple_Return']) > 0.2]  # >20% daily change
            if len(extreme_returns) > 0:
                logger.info(f"  Found {len(extreme_returns)} extreme return days for {company_name}")
            
            # Select relevant columns
            final_df = df[[
                'Date', 'Company', 'Symbol', 'Exchange', 'Close', 
                'Daily_Simple_Return', 'Daily_Log_Return', 'Volatility_20d', 'Volume'
            ]].copy()
            
            logger.info(f"  ✅ Success: {len(final_df)} records for {company_name}")
            logger.info(f"  📊 Date range: {final_df['Date'].min()} to {final_df['Date'].max()}")
            logger.info(f"  📈 Price range: ₹{final_df['Close'].min():.2f} - ₹{final_df['Close'].max():.2f}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {str(e)}")
            return None
    
    def collect_all_data(self):
        """Collect data for all NSE companies."""
        logger.info("🎯 Starting NSE stock data collection...")
        
        for company_name, symbol in self.companies.items():
            try:
                logger.info(f"\n📊 Processing {company_name}...")
                data = self.download_stock_data(symbol, company_name)
                
                if data is not None:
                    self.collected_data[company_name] = data
                    logger.info(f"✅ {company_name}: {len(data)} records collected")
                    
                    # Print sample statistics
                    returns = data['Daily_Simple_Return']
                    logger.info(f"  📊 Return stats - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
                    logger.info(f"  📊 Max gain: {returns.max():.4f}, Max loss: {returns.min():.4f}")
                else:
                    logger.error(f"❌ Failed to collect data for {company_name}")
                    
                # Sleep between companies to avoid rate limits
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Error collecting {company_name} data: {str(e)}")
    
    def validate_collected_data(self):
        """Validate the completeness and quality of collected data."""
        logger.info("🔍 Validating collected data...")
        
        if not self.collected_data:
            logger.error("❌ No data collected")
            return False
        
        for company_name, df in self.collected_data.items():
            logger.info(f"\n📊 Validating {company_name}:")
            
            # Check date coverage
            date_range = (df['Date'].max() - df['Date'].min()).days
            logger.info(f"  📅 Date coverage: {date_range} days")
            
            # Check for gaps
            df_sorted = df.sort_values('Date')
            date_diffs = df_sorted['Date'].diff().dt.days
            large_gaps = date_diffs[date_diffs > 7]  # More than 7 days
            
            if len(large_gaps) > 0:
                logger.warning(f"  ⚠️ Found {len(large_gaps)} large date gaps")
            
            # Check for missing values
            missing_returns = df['Daily_Simple_Return'].isna().sum()
            if missing_returns > 0:
                logger.warning(f"  ⚠️ Missing returns: {missing_returns}")
            
            # Check data quality
            zero_volume_days = (df['Volume'] == 0).sum()
            if zero_volume_days > 0:
                logger.warning(f"  ⚠️ Zero volume days: {zero_volume_days}")
            
            logger.info(f"  ✅ Validation completed for {company_name}")
        
        return True
    
    def save_data(self):
        """Save collected NSE stock data to CSV."""
        logger.info("💾 Saving NSE stock data...")
        
        try:
            if not self.collected_data:
                logger.error("❌ No data to save")
                return False
            
            # Combine all company data
            all_dataframes = list(self.collected_data.values())
            master_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Sort by date and company
            master_df = master_df.sort_values(['Date', 'Company']).reset_index(drop=True)
            
            # Save to CSV
            filepath = Path('data/stock_data.csv')
            master_df.to_csv(filepath, index=False)
            
            if filepath.exists():
                logger.info(f"✅ NSE stock data saved: {filepath}")
                logger.info(f"📊 Total records: {len(master_df)}")
                
                # Print summary
                print("\n📊 NSE STOCK DATA SUMMARY:")
                print("-" * 50)
                print(f"Total records: {len(master_df):,}")
                print(f"Date range: {master_df['Date'].min()} to {master_df['Date'].max()}")
                print(f"Companies: {', '.join(master_df['Company'].unique())}")
                
                print("\nRecords per company:")
                for company, count in master_df['Company'].value_counts().items():
                    print(f"  {company}: {count:,} trading days")
                
                return True
            else:
                logger.error("❌ File not saved properly")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error saving data: {str(e)}")
            return False
    
    def execute(self):
        """Execute the NSE stock data collection pipeline."""
        try:
            logger.info("🚀 Starting NSE Stock Data Collection Pipeline...")
            
            # Step 1: Collect data
            self.collect_all_data()
            
            if not self.collected_data:
                raise Exception("❌ No data collected from any source")
            
            # Step 2: Validate data
            if not self.validate_collected_data():
                raise Exception("❌ Data validation failed")
            
            # Step 3: Save data
            success = self.save_data()
            
            if success:
                logger.info("🎉 NSE stock data collection completed successfully!")
                return True
            else:
                logger.error("❌ Failed to save data")
                return False
                
        except Exception as e:
            logger.error(f"❌ NSE stock data collection failed: {str(e)}")
            return False
        finally:
            self.session.close()

def main():
    """Main execution function."""
    print("🚀 FDA EVENT STUDY - NSE STOCK DATA COLLECTION")
    print("="*65)
    print("🏢 Companies: Sun Pharma, Biocon, Dr. Reddy's")
    print("📅 Period: 2015-01-01 to 2025-07-09")
    print("📈 Exchange: NSE (National Stock Exchange) ONLY")
    print("🎯 Improved reliability and data quality")
    print("-"*65)
    
    # Initialize collector
    collector = NSEStockDataCollector()
    
    # Execute collection
    success = collector.execute()
    
    if success:
        print("\n🎉 SUCCESS: NSE stock data collection completed!")
        print("✅ Data saved to: data/stock_data.csv")
        print("\n💡 Next steps:")
        print("   1. Run news data collection: python FDA_news_data.py")
        print("   2. Run data processing: python fda_data_processor.py")
    else:
        print("\n❌ FAILED: NSE stock data collection failed")
        print("💡 Troubleshooting steps:")
        print("   1. Check internet connection")
        print("   2. Verify Yahoo Finance API access")
        print("   3. Check logs: logs/stock_collection.log")
        print("   4. Ensure NSE symbols are correct:")
        print("      - SUNPHARMA.NS")
        print("      - BIOCON.NS") 
        print("      - DRREDDY.NS")
        print("   5. Try running with VPN if accessing from restricted region")

if __name__ == "__main__":
    main()