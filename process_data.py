"""
BIOCON FDA Event Study Data Processor
Comprehensive script for processing FDA news and stock data with advanced filtering and validation.
Handles fuzzy matching, non-overlapping event windows, and missing data gracefully.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz
import re
import sys

warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup logging configuration."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'data_processing.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class BioconFDAEventProcessor:
    """
    Comprehensive processor for BIOCON FDA Event Study.
    Handles data validation, fuzzy matching, event window filtering, and output generation.
    """
    
    def __init__(self):
        """Initialize the processor with configuration."""
        self.companies = ['Sun Pharma', 'Biocon', 'Dr. Reddy\'s']
        self.exchanges = ['NSE', 'BSE']
        self.date_range = {
            'start': '2015-01-01',
            'end': '2025-07-09'
        }
        self.event_window_days = 5  # ±5 days around event
        self.fuzzy_threshold = 80  # 80% similarity threshold
        
        # Data containers
        self.news_data = None
        self.stock_data = None
        self.processed_events = None
        
        # Validation flags
        self.validation_passed = False
        self.data_issues = []
        
        logger.info("🚀 BIOCON FDA Event Study Processor initialized")
        logger.info(f"📅 Analysis period: {self.date_range['start']} to {self.date_range['end']}")
        logger.info(f"🏢 Companies: {', '.join(self.companies)}")
        logger.info(f"📈 Exchanges: {', '.join(self.exchanges)}")
    
    def load_and_validate_data(self) -> bool:
        """Load and validate input datasets."""
        try:
            logger.info("📂 Loading input datasets...")
            
            # Load news data
            news_file = Path('data/news_data.csv')
            if not news_file.exists():
                logger.error(f"❌ News data file not found: {news_file}")
                self.data_issues.append("Missing news_data.csv - Run FDA_news_data.py first")
                return False
            
            self.news_data = pd.read_csv(news_file)
            logger.info(f"✅ Loaded news data: {len(self.news_data)} records")
            
            # Load stock data
            stock_file = Path('data/stock_data.csv')
            if not stock_file.exists():
                logger.error(f"❌ Stock data file not found: {stock_file}")
                self.data_issues.append("Missing stock_data.csv - Run stock_data.py first")
                return False
            
            self.stock_data = pd.read_csv(stock_file)
            logger.info(f"✅ Loaded stock data: {len(self.stock_data)} records")
            
            # Validate data structure and content
            return self._validate_data_structure() and self._validate_data_content()
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            self.data_issues.append(f"Data loading error: {str(e)}")
            return False
    
    def _validate_data_structure(self) -> bool:
        """Validate the structure of loaded datasets."""
        try:
            logger.info("🔍 Validating data structure...")
            
            # Validate news data columns
            required_news_cols = [
                'Company', 'Drug_Name', 'FDA_Scientific_Name', 'Date', 'Classification',
                'Sentiment_Score', 'Confidence_Score', 'Sources', 'Significance_Score',
                'Title', 'Summary', 'Source_Type', 'Search_Query'
            ]
            
            missing_news_cols = [col for col in required_news_cols if col not in self.news_data.columns]
            if missing_news_cols:
                logger.error(f"❌ Missing news columns: {missing_news_cols}")
                self.data_issues.append(f"Missing news columns: {missing_news_cols}")
                return False
            
            # Validate stock data columns
            required_stock_cols = [
                'Date', 'Company', 'Symbol', 'Exchange', 'Close', 
                'Daily_Simple_Return', 'Daily_Log_Return'
            ]
            
            missing_stock_cols = [col for col in required_stock_cols if col not in self.stock_data.columns]
            if missing_stock_cols:
                logger.error(f"❌ Missing stock columns: {missing_stock_cols}")
                self.data_issues.append(f"Missing stock columns: {missing_stock_cols}")
                return False
            
            logger.info("✅ Data structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data structure validation error: {str(e)}")
            self.data_issues.append(f"Structure validation error: {str(e)}")
            return False
    
    def _validate_data_content(self) -> bool:
        """Validate the content and coverage of datasets."""
        try:
            logger.info("🔍 Validating data content and coverage...")
            
            # Convert date columns and ensure timezone consistency
            self.news_data['Date'] = pd.to_datetime(self.news_data['Date']).dt.tz_localize(None)
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.tz_localize(None)
            
            # Validate date ranges
            start_date = pd.to_datetime(self.date_range['start']).tz_localize(None)
            end_date = pd.to_datetime(self.date_range['end']).tz_localize(None)
            
            # Check news data date coverage
            news_min_date = self.news_data['Date'].min()
            news_max_date = self.news_data['Date'].max()
            
            if news_min_date > start_date or news_max_date < end_date:
                logger.warning(f"⚠️ News data coverage: {news_min_date} to {news_max_date}")
                logger.warning(f"Expected: {start_date} to {end_date}")
            
            # Check stock data date coverage
            stock_min_date = self.stock_data['Date'].min()
            stock_max_date = self.stock_data['Date'].max()
            
            if stock_min_date > start_date or stock_max_date < end_date:
                logger.warning(f"⚠️ Stock data coverage: {stock_min_date} to {stock_max_date}")
            
            # Validate companies in both datasets
            news_companies = set(self.news_data['Company'].unique())
            stock_companies = set(self.stock_data['Company'].unique())
            
            logger.info(f"📊 News companies: {sorted(news_companies)}")
            logger.info(f"📊 Stock companies: {sorted(stock_companies)}")
            
            # Check for Dr. Reddy's zero events issue
            company_event_counts = self.news_data['Company'].value_counts()
            logger.info("📈 News event distribution:")
            
            zero_event_companies = []
            for company in self.companies:
                count = company_event_counts.get(company, 0)
                logger.info(f"  {company}: {count} events")
                if count == 0:
                    zero_event_companies.append(company)
            
            if zero_event_companies:
                logger.error(f"❌ Companies with zero events: {zero_event_companies}")
                self.data_issues.append(f"Zero events for: {zero_event_companies} - Re-run FDA_news_data.py")
                
                # Provide specific guidance
                for company in zero_event_companies:
                    if company == "Dr. Reddy's":
                        logger.error("💡 Dr. Reddy's issue: Check company name variations (Dr Reddy, DRL, etc.)")
                        self.data_issues.append("Dr. Reddy's name matching issue - Check aliases in news collection")
            
            # Check classification distribution
            classification_counts = self.news_data['Classification'].value_counts()
            logger.info("📊 Classification distribution:")
            for classification, count in classification_counts.items():
                logger.info(f"  {classification}: {count} events")
            
            if len(classification_counts) == 0:
                logger.error("❌ No valid classifications found")
                self.data_issues.append("No valid Good/Bad classifications")
                return False
            
            # Check stock data availability by exchange
            exchange_counts = self.stock_data['Exchange'].value_counts()
            logger.info("📊 Stock data by exchange:")
            for exchange, count in exchange_counts.items():
                logger.info(f"  {exchange}: {count} records")
            
            # Check for missing exchanges
            for company in self.companies:
                for exchange in self.exchanges:
                    company_exchange_data = self.stock_data[
                        (self.stock_data['Company'] == company) & 
                        (self.stock_data['Exchange'] == exchange)
                    ]
                    
                    if len(company_exchange_data) == 0:
                        logger.warning(f"⚠️ No {exchange} data for {company}")
                        if exchange == 'BSE':
                            logger.info(f"💡 BSE data missing for {company} - will use NSE only")
                        else:
                            logger.error(f"❌ Missing NSE data for {company}")
                            self.data_issues.append(f"Missing NSE data for {company}")
            
            # Set validation flag
            self.validation_passed = len(self.data_issues) == 0
            
            if self.validation_passed:
                logger.info("✅ Data content validation passed")
            else:
                logger.warning(f"⚠️ Data validation completed with {len(self.data_issues)} issues")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data content validation error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.data_issues.append(f"Content validation error: {str(e)}")
            return False
    
    def apply_fuzzy_matching_filter(self) -> pd.DataFrame:
        """Apply fuzzy matching to remove duplicate/follow-up articles."""
        try:
            logger.info("🔍 Applying fuzzy matching filter...")
            
            filtered_news = []
            processed_titles = []
            
            # Sort by date to process chronologically
            sorted_news = self.news_data.sort_values(['Company', 'Date', 'Title']).reset_index(drop=True)
            
            for _, event in sorted_news.iterrows():
                title = str(event['Title']).strip().lower()
                company = event['Company']
                
                # Check similarity with existing titles for the same company
                is_duplicate = False
                
                for existing_title, existing_company in processed_titles:
                    if existing_company == company:
                        similarity = fuzz.ratio(title, existing_title)
                        if similarity >= self.fuzzy_threshold:
                            logger.debug(f"🔄 Duplicate detected ({similarity}%): {title[:50]}...")
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_news.append(event)
                    processed_titles.append((title, company))
                else:
                    logger.debug(f"⚠️ Filtered out duplicate: {event['Title'][:50]}...")
            
            filtered_df = pd.DataFrame(filtered_news)
            
            original_count = len(self.news_data)
            filtered_count = len(filtered_df)
            removed_count = original_count - filtered_count
            
            logger.info(f"📊 Fuzzy matching results:")
            logger.info(f"  Original events: {original_count}")
            logger.info(f"  After filtering: {filtered_count}")
            logger.info(f"  Removed duplicates: {removed_count}")
            
            # Log company-wise results
            for company in self.companies:
                original = len(self.news_data[self.news_data['Company'] == company])
                filtered = len(filtered_df[filtered_df['Company'] == company])
                logger.info(f"  {company}: {original} → {filtered} ({original - filtered} removed)")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Fuzzy matching error: {str(e)}")
            return self.news_data
    
    def filter_non_overlapping_events(self, filtered_news: pd.DataFrame) -> pd.DataFrame:
        """Filter to ensure non-overlapping 11-day event windows."""
        try:
            logger.info("📅 Filtering for non-overlapping event windows...")
            
            non_overlapping_events = []
            
            # Process each company separately
            for company in self.companies:
                company_events = filtered_news[filtered_news['Company'] == company].copy()
                company_events = company_events.sort_values('Date').reset_index(drop=True)
                
                selected_events = []
                last_selected_date = None
                
                for _, event in company_events.iterrows():
                    event_date = event['Date']
                    
                    # Check if this event overlaps with the last selected event
                    if last_selected_date is None:
                        # First event for this company
                        selected_events.append(event)
                        last_selected_date = event_date
                        logger.debug(f"✅ Selected first event for {company}: {event_date}")
                    else:
                        # Check if at least 11 days have passed
                        days_difference = (event_date - last_selected_date).days
                        
                        if days_difference >= 11:  # Non-overlapping 11-day windows
                            selected_events.append(event)
                            last_selected_date = event_date
                            logger.debug(f"✅ Selected event for {company}: {event_date} (gap: {days_difference} days)")
                        else:
                            logger.debug(f"⚠️ Skipped overlapping event for {company}: {event_date} (gap: {days_difference} days)")
                
                non_overlapping_events.extend(selected_events)
                
                logger.info(f"  {company}: {len(company_events)} → {len(selected_events)} events")
            
            result_df = pd.DataFrame(non_overlapping_events)
            
            total_original = len(filtered_news)
            total_filtered = len(result_df)
            
            logger.info(f"📊 Non-overlapping filter results:")
            logger.info(f"  Total events: {total_original} → {total_filtered}")
            logger.info(f"  Removed overlapping: {total_original - total_filtered}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Non-overlapping filter error: {str(e)}")
            return filtered_news
    
    def calculate_event_returns(self, final_events: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and cumulative returns for each event window with advanced market model."""
        try:
            logger.info("📈 Calculating event returns with market model...")
            
            # First calculate market model parameters
            market_params = self._calculate_market_model_parameters()
            
            if not market_params:
                logger.warning("⚠️ No market model parameters calculated, using raw returns")
            
            processed_events = []
            
            for _, event in final_events.iterrows():
                company = event['Company']
                event_date = pd.to_datetime(event['Date']).tz_localize(None)
                
                # Get stock data for this company (prefer NSE, fallback to BSE)
                company_stock = self._get_company_stock_data(company)
                
                if company_stock.empty:
                    logger.warning(f"⚠️ No stock data for {company} - skipping event on {event_date}")
                    continue
                
                # Ensure timezone consistency
                company_stock['Date'] = pd.to_datetime(company_stock['Date']).dt.tz_localize(None)
                
                # Calculate returns for the event window (-5 to +5 days)
                event_returns = self._calculate_window_returns_with_market_model(
                    event_date, company_stock, company, market_params
                )
                
                if event_returns is None:
                    logger.warning(f"⚠️ Could not calculate returns for {company} event on {event_date}")
                    continue
                
                # Combine event data with calculated returns
                event_result = {
                    'Company': company,
                    'Exchange': event_returns['exchange'],
                    'Drug_Name': event.get('Drug_Name', 'Unknown'),
                    'FDA_Scientific_Name': event.get('FDA_Scientific_Name', event.get('Drug_Name', 'Unknown')),
                    'Date': event['Date'],
                    'Classification': event['Classification'],
                    'Sentiment_Score': event.get('Sentiment_Score', 0),
                    'Confidence_Score': event.get('Confidence_Score', 0),
                    'Sources': event.get('Sources', ''),
                    'Significance_Score': event.get('Significance_Score', 0),
                    'Title': event.get('Title', ''),
                    'Summary': event.get('Summary', ''),
                    'Source_Type': event.get('Source_Type', ''),
                    'Search_Query': event.get('Search_Query', ''),
                    'Event_Trading_Date': event_returns['event_trading_date'],
                    'Market_Model_Alpha': event_returns.get('alpha', 0),
                    'Market_Model_Beta': event_returns.get('beta', 1)
                }
                
                # Add return data
                event_result.update(event_returns['returns'])
                
                processed_events.append(event_result)
            
            result_df = pd.DataFrame(processed_events)
            
            logger.info(f"✅ Calculated returns for {len(result_df)} events")
            
            # Log summary by company
            for company in self.companies:
                company_events = result_df[result_df['Company'] == company]
                good_events = len(company_events[company_events['Classification'] == 'Good'])
                bad_events = len(company_events[company_events['Classification'] == 'Bad'])
                logger.info(f"  {company}: {len(company_events)} events ({good_events} good, {bad_events} bad)")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Event returns calculation error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _calculate_market_model_parameters(self, estimation_window: int = 250) -> Dict:
        """Calculate market model parameters for each company."""
        try:
            logger.info("📈 Calculating market model parameters...")
            
            # Create market index (equal-weighted portfolio of all three companies)
            market_returns = []
            
            for date in self.stock_data['Date'].unique():
                date_returns = self.stock_data[
                    self.stock_data['Date'] == date
                ]['Daily_Simple_Return'].dropna()  # Remove NaN values
                
                if len(date_returns) > 0:
                    market_return = date_returns.mean()  # Equal-weighted
                    if not pd.isna(market_return):  # Only add non-NaN returns
                        market_returns.append({
                            'Date': date,
                            'Market_Return': market_return
                        })
            
            market_df = pd.DataFrame(market_returns)
            
            # Calculate parameters for each company
            parameters = {}
            
            for company in self.companies:
                company_data = self.stock_data[self.stock_data['Company'] == company].copy()
                
                if len(company_data) < estimation_window:
                    logger.warning(f"⚠️ Insufficient data for {company}: {len(company_data)} days")
                    continue
                
                # Merge with market returns
                company_data = company_data.merge(market_df, on='Date', how='left')
                # Remove rows with NaN values
                company_data = company_data.dropna(subset=['Daily_Simple_Return', 'Market_Return'])
                
                if len(company_data) < estimation_window:
                    logger.warning(f"⚠️ Insufficient merged data for {company}: {len(company_data)} days")
                    # Use whatever data we have if it's at least 100 days
                    if len(company_data) < 100:
                        continue
                
                # Use the first part of data for estimation (avoid look-ahead bias)
                estimation_data = company_data.head(min(estimation_window, len(company_data)))
                
                # Calculate alpha and beta using OLS regression
                X = estimation_data['Market_Return'].values
                y = estimation_data['Daily_Simple_Return'].values
                
                # Remove any remaining NaN values
                valid_mask = ~(np.isnan(X) | np.isnan(y))
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) < 50:  # Need at least 50 observations
                    logger.warning(f"⚠️ Too few valid observations for {company}: {len(X)}")
                    continue
                
                # Add constant for intercept
                X_with_const = np.column_stack([np.ones(len(X)), X])
                
                # OLS regression: y = alpha + beta * x + error
                try:
                    coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    alpha = coefficients[0]
                    beta = coefficients[1]
                    
                    # Calculate R-squared
                    y_pred = alpha + beta * X
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Calculate residual standard error
                    residuals = y - y_pred
                    residual_std = np.std(residuals)
                    
                    parameters[company] = {
                        'alpha': alpha,
                        'beta': beta,
                        'r_squared': r_squared,
                        'residual_std': residual_std,
                        'estimation_period': len(X),
                        'market_data': market_df
                    }
                    
                    logger.info(f"  {company}: α={alpha:.6f}, β={beta:.4f}, R²={r_squared:.4f} (n={len(X)})")
                    
                except np.linalg.LinAlgError as e:
                    logger.error(f"❌ Linear algebra error for {company}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Could not calculate parameters for {company}: {str(e)}")
                    continue
            
            if parameters:
                logger.info("✅ Market model parameters calculated")
            else:
                logger.warning("⚠️ No market parameters calculated")
            
            return parameters
            
        except Exception as e:
            logger.error(f"❌ Market model calculation error: {str(e)}")
            return {}
    
    def _calculate_window_returns_with_market_model(self, event_date: pd.Timestamp, 
                                                   stock_data: pd.DataFrame, company: str, 
                                                   market_params: Dict) -> Optional[Dict]:
        """Calculate returns for an 11-day event window with market model adjustment."""
        try:
            # Ensure stock data is sorted by date
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            # Find the closest trading day to the event date within ±3 days
            stock_data['date_diff'] = (stock_data['Date'] - event_date).dt.days.abs()
            closest_match = stock_data[stock_data['date_diff'] <= 3]
            
            if closest_match.empty:
                logger.debug(f"⚠️ No trading day within 3 days of {event_date} for {company}")
                return None
            
            # Get the closest trading day
            event_day_idx = closest_match['date_diff'].idxmin()
            event_day_date = stock_data.loc[event_day_idx, 'Date']
            logger.debug(f"📅 Selected trading day {event_day_date} for event on {event_date} (diff: {stock_data.loc[event_day_idx, 'date_diff']} days)")
            
            # Verify the selected date exists in stock data
            if event_day_date not in stock_data['Date'].values:
                logger.debug(f"⚠️ Selected trading day {event_day_date} not found in stock data for {company}")
                return None
            
            # Get market model parameters for this company
            company_params = market_params.get(company, {})
            alpha = company_params.get('alpha', 0)
            beta = company_params.get('beta', 1)
            market_data = company_params.get('market_data', pd.DataFrame())
            
            # Calculate returns for each day in the window
            returns_data = {}
            
            # Get the index of the event day in the stock data
            event_idx = stock_data.index[stock_data['Date'] == event_day_date].tolist()
            if not event_idx:
                logger.debug(f"⚠️ Event day {event_day_date} not found in stock data for {company}")
                return None
            event_idx = event_idx[0]
            
            # Calculate day-by-day returns (abnormal returns if market model available)
            for day_offset in range(-5, 6):  # -5 to +5 days
                target_idx = event_idx + day_offset
                
                if 0 <= target_idx < len(stock_data):
                    row = stock_data.iloc[target_idx]
                    actual_simple_return = row['Daily_Simple_Return']
                    actual_log_return = row['Daily_Log_Return']
                    target_date = row['Date']
                    
                    # Handle NaN values
                    if pd.isna(actual_simple_return) or pd.isna(actual_log_return):
                        simple_return = 0.0
                        log_return = 0.0
                    else:
                        # Calculate abnormal returns if market model is available
                        if not market_data.empty and company in market_params:
                            # Get market return for this date
                            market_return_data = market_data[
                                pd.to_datetime(market_data['Date']).dt.tz_localize(None).dt.date == target_date.date()
                            ]
                            
                            if not market_return_data.empty:
                                market_return = market_return_data['Market_Return'].iloc[0]
                                if not pd.isna(market_return):
                                    # Calculate expected return using market model
                                    expected_return = alpha + beta * market_return
                                    # Calculate abnormal return
                                    simple_return = actual_simple_return - expected_return
                                    # For log returns, use similar adjustment
                                    log_return = actual_log_return - expected_return
                                else:
                                    simple_return = actual_simple_return
                                    log_return = actual_log_return
                            else:
                                simple_return = actual_simple_return
                                log_return = actual_log_return
                        else:
                            # Use raw returns if no market model
                            simple_return = actual_simple_return
                            log_return = actual_log_return
                    
                    if day_offset == 0:
                        returns_data['Day_0_Simple'] = simple_return
                        returns_data['Day_0_Log'] = log_return
                    else:
                        returns_data[f'Day_{day_offset:+d}_Simple'] = simple_return
                        returns_data[f'Day_{day_offset:+d}_Log'] = log_return
                else:
                    # Missing data for this day
                    logger.debug(f"⚠️ No stock data for day offset {day_offset} from {event_day_date} for {company}")
                    if day_offset == 0:
                        returns_data['Day_0_Simple'] = 0.0
                        returns_data['Day_0_Log'] = 0.0
                    else:
                        returns_data[f'Day_{day_offset:+d}_Simple'] = 0.0
                        returns_data[f'Day_{day_offset:+d}_Log'] = 0.0
            
            # Calculate cumulative returns
            # Pre-event cumulative (-5 to -1)
            pre_event_simple = sum([
                returns_data.get(f'Day_{d:+d}_Simple', 0) for d in range(-5, 0)
            ])
            pre_event_log = sum([
                returns_data.get(f'Day_{d:+d}_Log', 0) for d in range(-5, 0)
            ])
            
            returns_data['Cumulative_Simple_-5_to_-1'] = pre_event_simple
            returns_data['Cumulative_Log_-5_to_-1'] = pre_event_log
            
            # Post-event cumulative (+1 to +5)
            post_event_simple = sum([
                returns_data.get(f'Day_{d:+d}_Simple', 0) for d in range(1, 6)
            ])
            post_event_log = sum([
                returns_data.get(f'Day_{d:+d}_Log', 0) for d in range(1, 6)
            ])
            
            returns_data['Cumulative_Simple_+1_to_+5'] = post_event_simple
            returns_data['Cumulative_Log_+1_to_+5'] = post_event_log
            
            # Determine which exchange was used
            exchange = stock_data.iloc[0]['Exchange']
            
            return {
                'returns': returns_data,
                'exchange': exchange,
                'event_trading_date': event_day_date,
                'alpha': alpha,
                'beta': beta
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating window returns with market model for {company} on {event_date}: {str(e)}")
            logger.debug(f"Stock data head:\n{stock_data.head().to_string()}")
            return None
    
    def _get_company_stock_data(self, company: str) -> pd.DataFrame:
        """Get stock data for a company, preferring NSE over BSE."""
        try:
            # Try NSE first
            nse_data = self.stock_data[
                (self.stock_data['Company'] == company) & 
                (self.stock_data['Exchange'] == 'NSE')
            ].copy()
            
            if not nse_data.empty:
                logger.debug(f"📈 Using NSE data for {company}")
                return nse_data
            
            # Fallback to BSE
            bse_data = self.stock_data[
                (self.stock_data['Company'] == company) & 
                (self.stock_data['Exchange'] == 'BSE')
            ].copy()
            
            if not bse_data.empty:
                logger.debug(f"📈 Using BSE data for {company} (NSE not available)")
                return bse_data
            
            logger.warning(f"⚠️ No stock data found for {company}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error getting stock data for {company}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_window_returns(self, event_date: pd.Timestamp, stock_data: pd.DataFrame) -> Optional[Dict]:
        """Calculate returns for an 11-day event window around the event date."""
        try:
            # Ensure stock data is sorted by date
            stock_data = stock_data.sort_values('Date').reset_index(drop=True)
            
            # Find the closest trading day to the event date within ±3 days
            stock_data['date_diff'] = (stock_data['Date'] - event_date).dt.days.abs()
            closest_match = stock_data[stock_data['date_diff'] <= 3]
            
            if closest_match.empty:
                logger.debug(f"⚠️ No trading day within 3 days of {event_date}")
                return None
            
            # Get the closest trading day
            event_day_idx = closest_match['date_diff'].idxmin()
            event_day_date = stock_data.loc[event_day_idx, 'Date']
            
            # Verify the selected date exists in stock data
            if event_day_date not in stock_data['Date'].values:
                logger.debug(f"⚠️ Selected trading day {event_day_date} not found in stock data")
                return None
            
            # Get the index of the event day
            event_idx = stock_data.index[stock_data['Date'] == event_day_date].tolist()
            if not event_idx:
                logger.debug(f"⚠️ Event day {event_day_date} not found in stock data")
                return None
            event_idx = event_idx[0]
            
            # Calculate returns for each day in the window
            returns_data = {}
            
            # Calculate day-by-day returns
            for day_offset in range(-5, 6):  # -5 to +5 days
                target_idx = event_idx + day_offset
                
                if 0 <= target_idx < len(stock_data):
                    row = stock_data.iloc[target_idx]
                    simple_return = row['Daily_Simple_Return']
                    log_return = row['Daily_Log_Return']
                    
                    # Handle NaN values
                    if pd.isna(simple_return) or pd.isna(log_return):
                        simple_return = 0.0
                        log_return = 0.0
                    
                    if day_offset == 0:
                        returns_data['Day_0_Simple'] = simple_return
                        returns_data['Day_0_Log'] = log_return
                    else:
                        returns_data[f'Day_{day_offset:+d}_Simple'] = simple_return
                        returns_data[f'Day_{day_offset:+d}_Log'] = log_return
                else:
                    # Missing data for this day
                    logger.debug(f"⚠️ No stock data for day offset {day_offset} from {event_day_date}")
                    if day_offset == 0:
                        returns_data['Day_0_Simple'] = 0.0
                        returns_data['Day_0_Log'] = 0.0
                    else:
                        returns_data[f'Day_{day_offset:+d}_Simple'] = 0.0
                        returns_data[f'Day_{day_offset:+d}_Log'] = 0.0
            
            # Calculate cumulative returns
            # Pre-event cumulative (-5 to -1)
            pre_event_simple = sum([
                returns_data.get(f'Day_{d:+d}_Simple', 0) for d in range(-5, 0)
            ])
            pre_event_log = sum([
                returns_data.get(f'Day_{d:+d}_Log', 0) for d in range(-5, 0)
            ])
            
            returns_data['Cumulative_Simple_-5_to_-1'] = pre_event_simple
            returns_data['Cumulative_Log_-5_to_-1'] = pre_event_log
            
            # Post-event cumulative (+1 to +5)
            post_event_simple = sum([
                returns_data.get(f'Day_{d:+d}_Simple', 0) for d in range(1, 6)
            ])
            post_event_log = sum([
                returns_data.get(f'Day_{d:+d}_Log', 0) for d in range(1, 6)
            ])
            
            returns_data['Cumulative_Simple_+1_to_+5'] = post_event_simple
            returns_data['Cumulative_Log_+1_to_+5'] = post_event_log
            
            # Determine which exchange was used
            exchange = stock_data.iloc[0]['Exchange']
            
            return {
                'returns': returns_data,
                'exchange': exchange,
                'event_trading_date': event_day_date
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating window returns: {str(e)}")
            return None
    
    def save_output_files(self, processed_events: pd.DataFrame) -> bool:
        """Save the three required output files."""
        try:
            logger.info("💾 Saving output files...")
            
            # Create outputs directory
            output_dir = Path('outputs')
            output_dir.mkdir(exist_ok=True)
            
            # Define the exact column order for output files
            output_columns = [
                'Company', 'Exchange', 'Drug_Name', 'FDA_Scientific_Name', 'Date', 'Classification',
                'Day_-5_Simple', 'Day_-5_Log', 'Day_-4_Simple', 'Day_-4_Log', 
                'Day_-3_Simple', 'Day_-3_Log', 'Day_-2_Simple', 'Day_-2_Log',
                'Day_-1_Simple', 'Day_-1_Log', 'Day_0_Simple', 'Day_0_Log',
                'Day_+1_Simple', 'Day_+1_Log', 'Day_+2_Simple', 'Day_+2_Log',
                'Day_+3_Simple', 'Day_+3_Log', 'Day_+4_Simple', 'Day_+4_Log',
                'Day_+5_Simple', 'Day_+5_Log', 'Cumulative_Simple_-5_to_-1', 'Cumulative_Log_-5_to_-1',
                'Cumulative_Simple_+1_to_+5', 'Cumulative_Log_+1_to_+5'
            ]
            
            # Ensure all columns exist
            for col in output_columns:
                if col not in processed_events.columns:
                    processed_events[col] = None
            
            # Select columns in correct order
            output_data = processed_events[output_columns].copy()
            
            # 1. Save good_news.csv
            good_news = output_data[output_data['Classification'] == 'Good']
            good_news_file = output_dir / 'good_news.csv'
            good_news.to_csv(good_news_file, index=False)
            logger.info(f"✅ Saved good_news.csv: {len(good_news)} events")
            
            # 2. Save bad_news.csv
            bad_news = output_data[output_data['Classification'] == 'Bad']
            bad_news_file = output_dir / 'bad_news.csv'
            bad_news.to_csv(bad_news_file, index=False)
            logger.info(f"✅ Saved bad_news.csv: {len(bad_news)} events")
            
            # 3. Create and save metadata.csv
            self._create_metadata_file(processed_events, output_dir)
            
            # Print final summary
            self._print_final_summary(processed_events)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving output files: {str(e)}")
            return False
    
    def _create_metadata_file(self, processed_events: pd.DataFrame, output_dir: Path):
        """Create metadata.csv with comprehensive analysis information."""
        try:
            # Create metadata based on news_data.csv structure but rename Classification to Event_Type
            metadata_records = []
            
            for _, event in processed_events.iterrows():
                metadata_record = {
                    'Company': event['Company'],
                    'Drug_Name': event['Drug_Name'],
                    'FDA_Scientific_Name': event['FDA_Scientific_Name'],
                    'Date': event['Date'],
                    'Event_Type': event['Classification'],  # Renamed from Classification
                    'Sentiment_Score': event.get('Sentiment_Score', 0),
                    'Confidence_Score': event.get('Confidence_Score', 0),
                    'Sources': event.get('Sources', ''),
                    'Significance_Score': event.get('Significance_Score', 0),
                    'Title': event.get('Title', ''),
                    'Summary': event.get('Summary', ''),
                    'Source_Type': event.get('Source_Type', ''),
                    'Search_Query': event.get('Search_Query', ''),
                    'Exchange_Used': event.get('Exchange', 'NSE'),
                    'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Fuzzy_Threshold_Used': self.fuzzy_threshold,
                    'Event_Window_Days': f'±{self.event_window_days}',
                    'Data_Issues': '; '.join(self.data_issues) if self.data_issues else 'None'
                }
                metadata_records.append(metadata_record)
            
            metadata_df = pd.DataFrame(metadata_records)
            metadata_file = output_dir / 'metadata.csv'
            metadata_df.to_csv(metadata_file, index=False)
            logger.info(f"✅ Saved metadata.csv: {len(metadata_df)} records")
            
        except Exception as e:
            logger.error(f"❌ Error creating metadata file: {str(e)}")
    
    def perform_statistical_analysis(self, processed_events: pd.DataFrame) -> Dict:
        """Perform comprehensive statistical analysis on the processed events."""
        try:
            logger.info("📊 Performing statistical analysis...")
            
            if processed_events.empty:
                logger.error("❌ No processed events available for analysis")
                return {}
            
            # Import scipy for statistical tests
            try:
                from scipy import stats
            except ImportError:
                logger.warning("⚠️ scipy not available, skipping statistical tests")
                return {}
            
            test_results = {}
            
            # Define event windows for analysis
            windows = {
                'pre_event': ['Cumulative_Simple_-5_to_-1', 'Cumulative_Log_-5_to_-1'],
                'event_day': ['Day_0_Simple', 'Day_0_Log'],
                'post_event': ['Cumulative_Simple_+1_to_+5', 'Cumulative_Log_+1_to_+5']
            }
            
            # Overall tests for each window
            for window_name, [simple_col, log_col] in windows.items():
                if simple_col in processed_events.columns:
                    returns = processed_events[simple_col].dropna()
                    
                    if len(returns) > 0:
                        # One-sample t-test (H0: mean return = 0)
                        t_stat, p_value = stats.ttest_1samp(returns, 0)
                        
                        test_results[f'{window_name}_overall'] = {
                            'n_events': len(returns),
                            'mean_return': returns.mean(),
                            'std_return': returns.std(),
                            'median_return': returns.median(),
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'positive_percent': (returns > 0).mean() * 100
                        }
            
            # Tests by classification (Good vs Bad news)
            for classification in ['Good', 'Bad']:
                class_data = processed_events[processed_events['Classification'] == classification]
                
                for window_name, [simple_col, log_col] in windows.items():
                    if simple_col in class_data.columns:
                        returns = class_data[simple_col].dropna()
                        
                        if len(returns) > 0:
                            # One-sample t-test
                            t_stat, p_value = stats.ttest_1samp(returns, 0)
                            
                            test_results[f'{window_name}_{classification.lower()}'] = {
                                'n_events': len(returns),
                                'mean_return': returns.mean(),
                                'std_return': returns.std(),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            
            # Two-sample t-test (Good vs Bad news comparison)
            for window_name, [simple_col, log_col] in windows.items():
                good_returns = processed_events[
                    processed_events['Classification'] == 'Good'
                ][simple_col].dropna()
                
                bad_returns = processed_events[
                    processed_events['Classification'] == 'Bad'
                ][simple_col].dropna()
                
                if len(good_returns) > 0 and len(bad_returns) > 0:
                    t_stat, p_value = stats.ttest_ind(good_returns, bad_returns)
                    
                    test_results[f'{window_name}_good_vs_bad'] = {
                        'n_good': len(good_returns),
                        'n_bad': len(bad_returns),
                        'mean_good': good_returns.mean(),
                        'mean_bad': bad_returns.mean(),
                        'mean_difference': good_returns.mean() - bad_returns.mean(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            # Tests by company
            for company in self.companies:
                company_data = processed_events[processed_events['Company'] == company]
                
                if len(company_data) == 0:
                    continue
                
                for window_name, [simple_col, log_col] in windows.items():
                    if simple_col in company_data.columns:
                        returns = company_data[simple_col].dropna()
                        
                        if len(returns) > 0:
                            # One-sample t-test
                            t_stat, p_value = stats.ttest_1samp(returns, 0)
                            
                            company_clean = company.replace("'", "").replace(".", "").replace(" ", "_")
                            
                            test_results[f'{window_name}_{company_clean}'] = {
                                'n_events': len(returns),
                                'mean_return': returns.mean(),
                                'std_return': returns.std(),
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            
            logger.info("✅ Statistical analysis completed")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Statistical analysis error: {str(e)}")
            return {}
    
    def create_visualizations(self, processed_events: pd.DataFrame):
        """Create comprehensive visualizations for the analysis."""
        try:
            logger.info("📊 Creating visualizations...")
            
            if processed_events.empty:
                logger.error("❌ No processed events available for visualization")
                return
            
            # Check if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
            except ImportError:
                logger.warning("⚠️ matplotlib/seaborn not available, skipping visualizations")
                return
            
            # Create charts directory
            charts_dir = Path('outputs/charts')
            charts_dir.mkdir(exist_ok=True)
            
            # Set style
            plt.style.use('default')
            
            # 1. Event Day Returns Distribution
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Event Day Returns Distribution by Classification', fontsize=16, fontweight='bold')
            
            # Simple returns
            good_returns = processed_events[processed_events['Classification'] == 'Good']['Day_0_Simple'].dropna()
            bad_returns = processed_events[processed_events['Classification'] == 'Bad']['Day_0_Simple'].dropna()
            
            if len(good_returns) > 0:
                axes[0].hist(good_returns, alpha=0.7, label=f'Good News ({len(good_returns)})', bins=15, color='green')
            if len(bad_returns) > 0:
                axes[0].hist(bad_returns, alpha=0.7, label=f'Bad News ({len(bad_returns)})', bins=15, color='red')
            
            axes[0].set_title('Simple Returns')
            axes[0].set_xlabel('Event Day Return')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Log returns
            good_log_returns = processed_events[processed_events['Classification'] == 'Good']['Day_0_Log'].dropna()
            bad_log_returns = processed_events[processed_events['Classification'] == 'Bad']['Day_0_Log'].dropna()
            
            if len(good_log_returns) > 0:
                axes[1].hist(good_log_returns, alpha=0.7, label=f'Good News ({len(good_log_returns)})', bins=15, color='green')
            if len(bad_log_returns) > 0:
                axes[1].hist(bad_log_returns, alpha=0.7, label=f'Bad News ({len(bad_log_returns)})', bins=15, color='red')
            
            axes[1].set_title('Log Returns')
            axes[1].set_xlabel('Event Day Log Return')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'event_day_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Company-wise Analysis
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Event Day Returns by Company', fontsize=16, fontweight='bold')
            
            for i, company in enumerate(self.companies):
                company_data = processed_events[processed_events['Company'] == company]
                
                if len(company_data) > 0:
                    good_data = company_data[company_data['Classification'] == 'Good']['Day_0_Simple'].dropna()
                    bad_data = company_data[company_data['Classification'] == 'Bad']['Day_0_Simple'].dropna()
                    
                    ax = axes[i]
                    
                    if len(good_data) > 0:
                        ax.scatter(range(len(good_data)), good_data, alpha=0.7, 
                                 label=f'Good ({len(good_data)})', color='green', s=50)
                    if len(bad_data) > 0:
                        ax.scatter(range(len(bad_data)), bad_data, alpha=0.7, 
                                 label=f'Bad ({len(bad_data)})', color='red', s=50)
                    
                    ax.set_title(company)
                    ax.set_xlabel('Event Number')
                    ax.set_ylabel('Event Day Return')
                    ax.legend()
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.grid(True, alpha=0.3)
                else:
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                               transform=axes[i].transAxes, fontsize=14)
                    axes[i].set_title(company)
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'company_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Cumulative Returns Comparison
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Cumulative Returns: Pre-Event vs Post-Event', fontsize=16, fontweight='bold')
            
            # Pre-event returns
            pre_good = processed_events[processed_events['Classification'] == 'Good']['Cumulative_Simple_-5_to_-1'].dropna()
            pre_bad = processed_events[processed_events['Classification'] == 'Bad']['Cumulative_Simple_-5_to_-1'].dropna()
            
            data_pre = []
            labels_pre = []
            if len(pre_good) > 0:
                data_pre.append(pre_good)
                labels_pre.append('Good News')
            if len(pre_bad) > 0:
                data_pre.append(pre_bad)
                labels_pre.append('Bad News')
            
            if data_pre:
                box_plot = axes[0].boxplot(data_pre, labels=labels_pre, patch_artist=True)
                colors = ['lightgreen', 'lightcoral']
                for patch, color in zip(box_plot['boxes'], colors[:len(data_pre)]):
                    patch.set_facecolor(color)
            
            axes[0].set_title('Pre-Event Returns (-5 to -1)')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0].grid(True, alpha=0.3)
            
            # Post-event returns
            post_good = processed_events[processed_events['Classification'] == 'Good']['Cumulative_Simple_+1_to_+5'].dropna()
            post_bad = processed_events[processed_events['Classification'] == 'Bad']['Cumulative_Simple_+1_to_+5'].dropna()
            
            data_post = []
            labels_post = []
            if len(post_good) > 0:
                data_post.append(post_good)
                labels_post.append('Good News')
            if len(post_bad) > 0:
                data_post.append(post_bad)
                labels_post.append('Bad News')
            
            if data_post:
                box_plot = axes[1].boxplot(data_post, labels=labels_post, patch_artist=True)
                colors = ['lightgreen', 'lightcoral']
                for patch, color in zip(box_plot['boxes'], colors[:len(data_post)]):
                    patch.set_facecolor(color)
            
            axes[1].set_title('Post-Event Returns (+1 to +5)')
            axes[1].set_ylabel('Cumulative Return')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(charts_dir / 'cumulative_returns_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Timeline Analysis
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Convert dates to datetime for plotting
            processed_events_plot = processed_events.copy()
            processed_events_plot['Date'] = pd.to_datetime(processed_events_plot['Date'])
            
            good_events = processed_events_plot[processed_events_plot['Classification'] == 'Good']
            bad_events = processed_events_plot[processed_events_plot['Classification'] == 'Bad']
            
            if len(good_events) > 0:
                ax.scatter(good_events['Date'], good_events['Day_0_Simple'], 
                          alpha=0.7, label='Good News', color='green', s=60)
            
            if len(bad_events) > 0:
                ax.scatter(bad_events['Date'], bad_events['Day_0_Simple'], 
                          alpha=0.7, label='Bad News', color='red', s=60)
            
            ax.set_title('FDA Events Timeline: Event Day Returns', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Event Day Return')
            ax.legend()
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(charts_dir / 'timeline_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Visualizations created and saved to outputs/charts/")
            
        except Exception as e:
            logger.error(f"❌ Visualization error: {str(e)}")
    
    def generate_comprehensive_report(self, processed_events: pd.DataFrame, 
                                    statistical_results: Dict) -> str:
        """Generate a comprehensive analysis report."""
        try:
            logger.info("📋 Generating comprehensive analysis report...")
            
            report = []
            report.append("=" * 80)
            report.append("BIOCON FDA EVENT STUDY - COMPREHENSIVE ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Analysis Period: {self.date_range['start']} to {self.date_range['end']}")
            report.append(f"Fuzzy Matching Threshold: {self.fuzzy_threshold}%")
            report.append(f"Event Window: ±{self.event_window_days} days")
            report.append("")
            
            # Data Overview
            report.append("📊 DATA OVERVIEW")
            report.append("-" * 50)
            report.append(f"Total processed events: {len(processed_events)}")
            
            # Company distribution
            for company in self.companies:
                company_events = processed_events[processed_events['Company'] == company]
                good_events = len(company_events[company_events['Classification'] == 'Good'])
                bad_events = len(company_events[company_events['Classification'] == 'Bad'])
                report.append(f"{company}: {len(company_events)} events ({good_events} good, {bad_events} bad)")
            
            report.append("")
            
            # Statistical Results
            if statistical_results:
                report.append("📊 STATISTICAL ANALYSIS RESULTS")
                report.append("-" * 50)
                
                # Overall results
                report.append("Overall Results:")
                for test_name, results in statistical_results.items():
                    if 'overall' in test_name:
                        window = test_name.replace('_overall', '').replace('_', ' ').title()
                        report.append(f"  {window}:")
                        report.append(f"    Events: {results['n_events']}")
                        report.append(f"    Mean Return: {results['mean_return']:.4f} ({results['mean_return']*100:.2f}%)")
                        report.append(f"    t-statistic: {results['t_statistic']:.4f}")
                        report.append(f"    p-value: {results['p_value']:.4f}")
                        report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
                        report.append("")
                
                # Good vs Bad comparison
                report.append("Good vs Bad News Comparison:")
                for test_name, results in statistical_results.items():
                    if 'good_vs_bad' in test_name:
                        window = test_name.replace('_good_vs_bad', '').replace('_', ' ').title()
                        report.append(f"  {window}:")
                        report.append(f"    Good news return: {results['mean_good']:.4f} ({results['mean_good']*100:.2f}%)")
                        report.append(f"    Bad news return: {results['mean_bad']:.4f} ({results['mean_bad']*100:.2f}%)")
                        report.append(f"    Difference: {results['mean_difference']:.4f}")
                        report.append(f"    t-statistic: {results['t_statistic']:.4f}")
                        report.append(f"    p-value: {results['p_value']:.4f}")
                        report.append(f"    Significant: {'Yes' if results['significant'] else 'No'}")
                        report.append("")
            
            # Key Findings
            report.append("🔍 KEY FINDINGS")
            report.append("-" * 50)
            
            if statistical_results:
                # Find significant results
                significant_results = []
                for test_name, results in statistical_results.items():
                    if results.get('significant', False):
                        significant_results.append((test_name, results))
                
                if significant_results:
                    report.append("Significant findings:")
                    for test_name, results in significant_results[:5]:
                        mean_return = results.get('mean_return', results.get('mean_difference', 0))
                        report.append(f"  • {test_name}: Return = {mean_return:.4f} (p = {results['p_value']:.4f})")
                else:
                    report.append("• No statistically significant abnormal returns found")
                
                # Event day analysis
                event_day_results = statistical_results.get('event_day_overall')
                if event_day_results:
                    report.append(f"• Event day average impact: {event_day_results['mean_return']*100:.2f}%")
            
            # Data Quality Issues
            if self.data_issues:
                report.append("")
                report.append("⚠️ DATA QUALITY ISSUES")
                report.append("-" * 50)
                for i, issue in enumerate(self.data_issues, 1):
                    report.append(f"{i}. {issue}")
            
            report.append("")
            report.append("=" * 80)
            
            report_text = "\n".join(report)
            
            # Save report
            report_file = Path('outputs/comprehensive_analysis_report.txt')
            with open(report_file, 'w') as f:
                f.write(report_text)
            
            logger.info(f"✅ Comprehensive report saved to: {report_file}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"❌ Report generation error: {str(e)}")
            return ""
    
    def _save_additional_analysis_files(self, processed_events: pd.DataFrame, statistical_results: Dict):
        """Save additional analysis files for comprehensive study."""
        try:
            logger.info("💾 Saving additional analysis files...")
            
            # Create additional outputs directory
            analysis_dir = Path('outputs/analysis')
            analysis_dir.mkdir(exist_ok=True)
            
            # Save detailed event results
            detailed_results = processed_events.copy()
            detailed_file = analysis_dir / 'detailed_event_results.csv'
            detailed_results.to_csv(detailed_file, index=False)
            logger.info(f"✅ Detailed results saved: {detailed_file}")
            
            # Save statistical test results
            if statistical_results:
                stats_rows = []
                for test_name, results in statistical_results.items():
                    row = {'test_name': test_name}
                    row.update(results)
                    stats_rows.append(row)
                
                stats_df = pd.DataFrame(stats_rows)
                stats_file = analysis_dir / 'statistical_test_results.csv'
                stats_df.to_csv(stats_file, index=False)
                logger.info(f"✅ Statistical tests saved: {stats_file}")
            
            # Save summary statistics by company
            summary_stats = []
            for company in self.companies:
                company_data = processed_events[processed_events['Company'] == company]
                
                for classification in ['Good', 'Bad']:
                    class_data = company_data[company_data['Classification'] == classification]
                    
                    if len(class_data) > 0:
                        event_day_returns = class_data['Day_0_Simple'].dropna()
                        
                        if len(event_day_returns) > 0:
                            summary_stats.append({
                                'Company': company,
                                'Classification': classification,
                                'Event_Count': len(class_data),
                                'Mean_Event_Day_Return': event_day_returns.mean(),
                                'Median_Event_Day_Return': event_day_returns.median(),
                                'Std_Event_Day_Return': event_day_returns.std(),
                                'Min_Event_Day_Return': event_day_returns.min(),
                                'Max_Event_Day_Return': event_day_returns.max(),
                                'Positive_Return_Percent': (event_day_returns > 0).mean() * 100,
                                'Mean_Pre_Event_Return': class_data['Cumulative_Simple_-5_to_-1'].mean(),
                                'Mean_Post_Event_Return': class_data['Cumulative_Simple_+1_to_+5'].mean()
                            })
            
            summary_df = pd.DataFrame(summary_stats)
            summary_file = analysis_dir / 'company_summary_statistics.csv'
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"✅ Summary statistics saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving additional analysis files: {str(e)}")
    
    def _print_final_summary(self, processed_events: pd.DataFrame):
        """Print final summary of processed events."""
        print("\n" + "="*80)
        print("📊 BIOCON FDA EVENT STUDY - PROCESSING SUMMARY")
        print("="*80)
        print(f"📅 Analysis Period: {self.date_range['start']} to {self.date_range['end']}")
        print(f"🔍 Fuzzy Matching Threshold: {self.fuzzy_threshold}%")
        print(f"📅 Event Window: ±{self.event_window_days} days")
        print(f"📊 Total Processed Events: {len(processed_events)}")
        print(f"🎯 Market Model: Applied for abnormal return calculation")
        
        print("\n📈 EVENTS BY COMPANY:")
        print("-" * 50)
        
        total_good = 0
        total_bad = 0
        
        for company in self.companies:
            company_events = processed_events[processed_events['Company'] == company]
            good_events = len(company_events[company_events['Classification'] == 'Good'])
            bad_events = len(company_events[company_events['Classification'] == 'Bad'])
            
            total_good += good_events
            total_bad += bad_events
            
            # Calculate average event day return
            event_day_returns = company_events['Day_0_Simple'].dropna()
            avg_return = event_day_returns.mean() if len(event_day_returns) > 0 else 0
            
            print(f"{company}: {len(company_events)} events ({good_events} good, {bad_events} bad)")
            print(f"  Average event day return: {avg_return:.4f} ({avg_return*100:.2f}%)")
        
        print(f"\nTOTAL: {len(processed_events)} events ({total_good} good, {total_bad} bad)")
        
        # Calculate overall statistics
        all_event_day_returns = processed_events['Day_0_Simple'].dropna()
        if len(all_event_day_returns) > 0:
            print(f"Overall average event day return: {all_event_day_returns.mean():.4f} ({all_event_day_returns.mean()*100:.2f}%)")
            print(f"Positive return events: {(all_event_day_returns > 0).sum()} ({(all_event_day_returns > 0).mean()*100:.1f}%)")
        
        if self.data_issues:
            print("\n⚠️ DATA ISSUES DETECTED:")
            print("-" * 50)
            for i, issue in enumerate(self.data_issues, 1):
                print(f"{i}. {issue}")
        
        print("\n📁 OUTPUT FILES:")
        print("-" * 50)
        print("✅ outputs/good_news.csv - Good FDA events with returns")
        print("✅ outputs/bad_news.csv - Bad FDA events with returns")
        print("✅ outputs/metadata.csv - Event metadata and analysis info")
        print("✅ outputs/comprehensive_analysis_report.txt - Full analysis report")
        print("✅ outputs/charts/ - Visualizations and charts")
        print("✅ outputs/analysis/ - Detailed analysis files")
        
        if self.data_issues:
            print("\n💡 TROUBLESHOOTING STEPS:")
            print("-" * 50)
            if any("Dr. Reddy's" in issue for issue in self.data_issues):
                print("1. For Dr. Reddy's zero events:")
                print("   - Re-run: python FDA_news_data.py")
                print("   - Check company name variations (Dr Reddy, DRL, etc.)")
                print("   - Verify search terms and aliases")
            
            if any("stock_data" in issue for issue in self.data_issues):
                print("2. For missing stock data:")
                print("   - Re-run: python stock_data.py")
                print("   - Check NSE/BSE ticker symbols")
                print("   - Verify date range coverage")
            
            if any("news_data" in issue for issue in self.data_issues):
                print("3. For missing news data:")
                print("   - Re-run: python FDA_news_data.py")
                print("   - Check internet connection")
                print("   - Verify RSS feed accessibility")
        else:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("📊 Data quality is excellent for comprehensive analysis")
        
        print("\n" + "="*80)
    
    def _print_troubleshooting_guide(self):
        """Print comprehensive troubleshooting guide."""
        print("\n" + "="*80)
        print("🛠️ TROUBLESHOOTING GUIDE")
        print("="*80)
        
        print("\n📋 COMMON ISSUES AND SOLUTIONS:")
        print("-" * 50)
        
        print("1. Missing input files:")
        print("   ❌ FileNotFoundError: data/news_data.csv")
        print("   💡 Solution: Run 'python FDA_news_data.py' first")
        print("   ❌ FileNotFoundError: data/stock_data.csv")
        print("   💡 Solution: Run 'python stock_data.py' first")
        
        print("\n2. Dr. Reddy's zero events:")
        print("   ❌ Dr. Reddy's: 0 events")
        print("   💡 Solutions:")
        print("      - Check company name variations in FDA_news_data.py")
        print("      - Verify aliases: ['Dr Reddy', 'DRL', 'Dr. Reddy\\'s']")
        print("      - Review search terms and regex patterns")
        print("      - Check logs for name matching debug info")
        
        print("\n3. Missing stock data:")
        print("   ❌ No NSE/BSE data for company")
        print("   💡 Solutions:")
        print("      - Verify ticker symbols (SUNPHARMA.NS, BIOCON.NS, DRREDDY.NS)")
        print("      - Check internet connection and Yahoo Finance access")
        print("      - Review date range in stock_data.py")
        print("      - Check for trading holidays/weekends")
        
        print("\n4. Data quality issues:")
        print("   ❌ High number of filtered events")
        print("   💡 Solutions:")
        print("      - Adjust fuzzy matching threshold (default: 80%)")
        print("      - Review event window overlap logic")
        print("      - Check for duplicate news sources")
        
        print("\n📦 REQUIRED LIBRARIES:")
        print("-" * 50)
        print("pip install pandas numpy fuzzywuzzy python-levenshtein")
        
        print("\n📁 EXPECTED FILE STRUCTURE:")
        print("-" * 50)
        print("project/")
        print("├── data/")
        print("│   ├── news_data.csv")
        print("│   └── stock_data.csv")
        print("├── logs/")
        print("│   └── data_processing.log")
        print("├── outputs/")
        print("│   ├── good_news.csv")
        print("│   ├── bad_news.csv")
        print("│   └── metadata.csv")
        print("├── FDA_news_data.py")
        print("├── stock_data.py")
        print("└── process_data.py")
        
        print("\n🔍 DATA VALIDATION CHECKLIST:")
        print("-" * 50)
        print("□ All three companies have news events")
        print("□ Date range covers 2015-01-01 to 2025-07-09")
        print("□ Both Good and Bad classifications present")
        print("□ NSE stock data available for all companies")
        print("□ Stock returns calculated properly")
        print("□ No excessive duplicate filtering")
        
        print("\n" + "="*80)

    def run_complete_processing(self) -> bool:
        """Run the complete processing pipeline."""
        try:
            # Step 1: Load and validate data
            if not self.load_and_validate_data():
                logger.error("❌ Data loading/validation failed")
                return False
            
            # Step 2: Apply fuzzy matching to remove duplicates
            filtered_news = self.apply_fuzzy_matching_filter()
            
            # Step 3: Filter for non-overlapping events
            final_events = self.filter_non_overlapping_events(filtered_news)
            
            # Step 4: Calculate returns
            self.processed_events = self.calculate_event_returns(final_events)
            
            if self.processed_events.empty:
                logger.error("❌ No processed events after returns calculation")
                return False
            
            # Step 5: Perform statistical analysis
            statistical_results = self.perform_statistical_analysis(self.processed_events)
            
            # Step 6: Create visualizations
            self.create_visualizations(self.processed_events)
            
            # Step 7: Save output files
            if not self.save_output_files(self.processed_events):
                logger.error("❌ Failed to save output files")
                return False
            
            # Step 8: Save additional analysis files
            self._save_additional_analysis_files(self.processed_events, statistical_results)
            
            # Step 9: Generate comprehensive report
            self.generate_comprehensive_report(self.processed_events, statistical_results)
            
            logger.info("✅ Complete processing pipeline executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False

def main():
    """Main execution function."""
    print("🚀 BIOCON FDA EVENT STUDY - DATA PROCESSOR")
    print("="*70)
    print("📊 Advanced processing with fuzzy matching and validation")
    print("🏢 Companies: Sun Pharma, Biocon, Dr. Reddy's")
    print("📅 Period: 2015-01-01 to 2025-07-09")
    print("🔍 Features: Fuzzy matching, non-overlapping windows, BSE fallback")
    print("-"*70)
    
    # Check required libraries
    try:
        import fuzzywuzzy
        logger.info("✅ All required libraries available")
    except ImportError as e:
        print(f"\n❌ Missing required library: {e}")
        print("💡 Install with: pip install fuzzywuzzy python-levenshtein")
        return False
    
    # Initialize and run processor
    processor = BioconFDAEventProcessor()
    success = processor.run_complete_processing()
    
    if success:
        print("\n🎉 SUCCESS: BIOCON FDA Event Study processing completed!")
        print("\n📁 Check these output files:")
        print("   📄 outputs/good_news.csv")
        print("   📄 outputs/bad_news.csv")
        print("   📄 outputs/metadata.csv")
        print("\n📋 Review logs/data_processing.log for detailed processing info")
    else:
        print("\n❌ PROCESSING FAILED")
        print("💡 Check logs/data_processing.log for error details")
        print("💡 Review the troubleshooting guide above")
        print("💡 Ensure input files exist in data/ directory")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)