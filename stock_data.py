"""
Stock Data Collection for FDA Event Study
Collects daily stock prices for Sun Pharma, Biocon, and Dr. Reddy's from NSE and BSE (2015-2025).
Adapted from AdvancedStockDataCollector for simplified requirements.
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
from nsepy import get_history
import socket
import os

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
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Add FileHandler after ensuring logs directory exists
create_directories()
logger.handlers.append(logging.FileHandler('logs/stock_collection.log'))

class StockDataCollector:
    """Collects daily stock data for specified companies from NSE and BSE."""
    
    def __init__(self):
        self.start_date = datetime(2015, 1, 1)
        self.end_date = datetime(2025, 7, 9)
        self.companies = {
            'Sun Pharma': ['SUNPHARMA.NS', 'SUNPHARMA.BO'],
            'Biocon': ['BIOCON.NS', 'BIOCON.BO'],
            'Dr. Reddy’s': ['DRREDDY.NS', 'DRREDDY.BO']
        }
        self.collected_data = {}
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com'
        }
        self.proxies = None  # Add proxy config if needed
        logger.info(f"🚀 Stock Data Collector Initialized")
        logger.info(f"📅 Date Range: {self.start_date} to {self.end_date}")
    
    def _check_dns_resolution(self, host):
        """Check if a host resolves correctly."""
        try:
            socket.gethostbyname(host)
            return True
        except socket.gaierror as e:
            logger.error(f"❌ DNS resolution failed for {host}: {str(e)}")
            return False
    
    def _initialize_nse_session(self):
        """Initialize session for NSE India."""
        try:
            if not self._check_dns_resolution('www.nseindia.com'):
                raise Exception("DNS resolution failed")
            response = self.session.get('https://www.nseindia.com', headers=self.headers, proxies=self.proxies, timeout=15)
            response.raise_for_status()
            logger.info("✅ NSE session initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize NSE session: {str(e)}")
    
    def _validate_symbol(self, symbol, source):
        """Validate symbol availability."""
        if source == 'Yahoo Finance':
            try:
                time.sleep(5)
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1d", timeout=15)
                return not df.empty
            except Exception as e:
                logger.warning(f"⚠️ Symbol validation failed for {symbol}: {str(e)}")
                return False
        elif source == 'NSE India':
            try:
                nse_symbol = symbol.replace('.NS', '').replace('.BO', '')
                url = f"https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}"
                response = self.session.get(url, headers=self.headers, proxies=self.proxies, timeout=15)
                response.raise_for_status()
                return True
            except Exception as e:
                logger.warning(f"⚠️ Symbol validation failed for {symbol}: {str(e)}")
                return False
        return True
    
    @retry(tries=3, delay=10, backoff=2)
    def download_stock_data(self, symbol_group, symbol_name):
        """Download stock data from Yahoo Finance or NSE India."""
        logger.info(f"📈 Downloading {symbol_name} data...")
        
        # Try Yahoo Finance
        for symbol in symbol_group:
            if not self._validate_symbol(symbol, 'Yahoo Finance'):
                continue
            try:
                logger.info(f"  Trying Yahoo Finance symbol: {symbol}")
                time.sleep(5)
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    auto_adjust=True,
                    timeout=15
                )
                if df.empty:
                    continue
                if len(df) > 500:
                    df.reset_index(inplace=True)
                    df['Symbol'] = symbol
                    df['Company'] = symbol_name
                    df['Exchange'] = 'NSE' if symbol.endswith('.NS') else 'BSE'
                    df['Source'] = 'Yahoo Finance'
                    df['Daily_Simple_Return'] = df['Close'].pct_change()
                    df['Daily_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
                    logger.info(f"  ✅ Success: {len(df)} records for {symbol}")
                    return df[['Date', 'Company', 'Symbol', 'Exchange', 'Close', 'Daily_Simple_Return', 'Daily_Log_Return']]
            except Exception as e:
                logger.error(f"  ❌ Error downloading {symbol}: {str(e)}")
                continue
        
        # Try NSE India
        try:
            logger.info(f"  Trying NSE India for {symbol_name}")
            df = self.download_nse_data(symbol_name)
            if df is not None and not df.empty and len(df) > 500:
                logger.info(f"  ✅ Success: {len(df)} records from NSE India")
                return df
        except Exception as e:
            logger.error(f"  ❌ Error downloading from NSE India: {str(e)}")
        
        logger.error(f"❌ All sources failed for {symbol_name}")
        return None
    
    @retry(tries=3, delay=10, backoff=2)
    def download_nse_data(self, symbol_name):
        """Fetch stock data from NSE India."""
        try:
            nse_symbol = {
                'Sun Pharma': 'SUNPHARMA',
                'Biocon': 'BIOCON',
                'Dr. Reddy’s': 'DRREDDY'
            }[symbol_name]
            if not self._validate_symbol(nse_symbol, 'NSE India'):
                raise ValueError(f"Invalid symbol {nse_symbol}")
            self._initialize_nse_session()
            df = get_history(
                symbol=nse_symbol,
                start=self.start_date,
                end=self.end_date,
                threads=False
            )
            if df.empty:
                raise ValueError("Empty NSE data")
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'Date',
                'Close': 'Close'
            })
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Symbol'] = nse_symbol
            df['Company'] = symbol_name
            df['Exchange'] = 'NSE'  # Default to NSE for nsepy data
            df['Source'] = 'NSE India'
            df['Daily_Simple_Return'] = df['Close'].pct_change()
            df['Daily_Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            return df[['Date', 'Company', 'Symbol', 'Exchange', 'Close', 'Daily_Simple_Return', 'Daily_Log_Return']]
        except Exception as e:
            logger.error(f"  ❌ Error downloading from NSE: {str(e)}")
            return None
    
    def collect_all_data(self):
        """Collect data for all companies."""
        logger.info("🎯 Starting stock data collection...")
        for symbol_name, symbol_group in self.companies.items():
            try:
                data = self.download_stock_data(symbol_group, symbol_name)
                if data is not None:
                    self.collected_data[symbol_name] = data
                    logger.info(f"✅ {symbol_name}: {len(data)} records collected")
                else:
                    logger.warning(f"⚠️ Failed to collect {symbol_name} data")
            except Exception as e:
                logger.error(f"❌ Error collecting {symbol_name} data: {str(e)}")
        return self.collected_data
    
    def save_data(self):
        """Save collected stock data to CSV."""
        logger.info("💾 Saving stock data...")
        try:
            master_df = pd.concat(self.collected_data.values(), ignore_index=True)
            master_df = master_df.sort_values(['Date', 'Company', 'Exchange']).reset_index(drop=True)
            filepath = Path('data/stock_data.csv')
            master_df.to_csv(filepath, index=False)
            if filepath.exists():
                logger.info(f"✅ Data saved: {filepath} ({len(master_df)} rows)")
                return True
            else:
                logger.error("❌ File not saved properly")
                return False
        except Exception as e:
            logger.error(f"❌ Error saving data: {str(e)}")
            return False
    
    def execute(self):
        """Execute the stock data collection pipeline."""
        try:
            logger.info("🚀 Starting Stock Data Collection Pipeline...")
            self.collect_all_data()
            if not self.collected_data:
                raise Exception("❌ No data collected")
            success = self.save_data()
            if success:
                logger.info("🎉 Stock data collection completed successfully!")
                return True
            else:
                logger.error("❌ Failed to save data")
                return False
        except Exception as e:
            logger.error(f"❌ Stock data collection failed: {str(e)}")
            return False
        finally:
            self.session.close()

def main():
    """Main execution function."""
    print("🚀 FDA EVENT STUDY - STOCK DATA COLLECTION")
    print("="*60)
    print(f"🏢 Companies: Sun Pharma, Biocon, Dr. Reddy’s")
    print(f"📅 Period: 2015-01-01 to 2025-07-09")
    print("-"*60)
    
    collector = StockDataCollector()
    success = collector.execute()
    
    if success:
        print("\n🎉 SUCCESS: Stock data collection completed!")
        print("✅ Data saved to: data/stock_data.csv")
    else:
        print("\n❌ FAILED: Stock data collection failed")
        print("💡 Check logs: logs/stock_collection.log")

if __name__ == "__main__":
    main()
