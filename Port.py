import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import os
import plotly.io as pio
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

def create_session_with_retries():
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

### ---- Class Initialization ---- ###
class StockAnalysis:
    def __init__(self, tickers, time_frame):
        self.tickers = tickers
        self.time_frame = time_frame
        self.initial_investment = 1000000  # Fixed baseline portfolio value
        self.risk_free_rate = self.get_risk_free_rate()
        self.sim_runs = self.determine_simulation_runs()
        self.start_date, self.end_date = self.calculate_dates()
        self.data = self.get_all_stock_data()
        self.adjusted_closing_prices_df = self.data.filter(regex='_Adj_Close$', axis=1)
        self.daily_returns_df, self.weekly_returns_df = self.calculate_daily_returns()
        self.benchmark_tickers = ["^GSPC", "^DJI", "QQQ", "^IXIC"]
        self.benchmark_data = self.get_benchmark_data()
        # Set up Seaborn color palette
        self.colors = sns.color_palette("husl", n_colors=max(len(self.tickers), 10))

    @staticmethod
    def get_sp500_tickers():
        """Return a curated list of 50 major S&P 500 stocks."""
        return [
            # Original 20 stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "JNJ", "V",
            "PG", "XOM", "BAC", "MA", "UNH", "HD", "CVX", "MRK", "PFE", "KO",
            # Additional 30 stocks
            "TSLA", "WMT", "DIS", "CSCO", "ADBE", "NFLX", "INTC", "VZ", "CMCSA", "PEP",
            "COST", "ABT", "TMO", "MCD", "ACN", "DHR", "IBM", "LIN", "TXN", "LOW",
            "QCOM", "PM", "UPS", "MS", "RTX", "GS", "BLK", "AMD", "CAT", "DE"
        ]

    @staticmethod
    def get_random_portfolio():
        """Generate a random portfolio of 4-12 stocks from S&P 500."""
        try:
            # Get S&P 500 tickers
            sp500_tickers = StockAnalysis.get_sp500_tickers()
            if not sp500_tickers:
                raise ValueError("Failed to fetch S&P 500 tickers")
            
            # Randomly select number of stocks (4-12)
            num_stocks = random.randint(4, 12)
            num_stocks = min(num_stocks, len(sp500_tickers))  # Ensure we don't exceed available tickers
            
            # Randomly select tickers
            selected_tickers = random.sample(sp500_tickers, num_stocks)
            
            print(f"\nRandomly selected {num_stocks} stocks from available pool:")
            for i, ticker in enumerate(selected_tickers, 1):
                print(f"{i}. {ticker}")
            
            return selected_tickers
        except Exception as e:
            print(f"Error generating random portfolio: {e}")
            default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
            print(f"Using default portfolio: {', '.join(default_tickers)}")
            return default_tickers

### ---- Utility Functions ---- ###
    def get_risk_free_rate(self):
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=30)
        ten_year_rate = web.DataReader('DGS10', 'fred', start_date, end_date)
        risk_free_rate = ten_year_rate.iloc[-1].values[0] / 100  # Convert to decimal
        return risk_free_rate

    def determine_simulation_runs(self):
        num_stocks = len(self.tickers)
        if num_stocks < 5:
            return 8000
        elif 5 <= num_stocks <= 15:
            return 12000
        else:
            return 15000

    def calculate_dates(self):
        end_date = dt.datetime.now()
        if self.time_frame == 1:
            start_date = end_date - dt.timedelta(days=365)
        elif self.time_frame == 3:
            start_date = end_date - dt.timedelta(days=3 * 365)
        elif self.time_frame == 5:
            start_date = end_date - dt.timedelta(days=5 * 365)
        elif self.time_frame == 10:
            start_date = end_date - dt.timedelta(days=10 * 365)
        else:
            print("Invalid selection. Defaulting to 1 year.")
            start_date = end_date - dt.timedelta(days=365)
        return start_date, end_date

### ---- Data Retrieval Functions ---- ###
    def get_all_stock_data(self):  # Renamed method
        data_frames = []
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            stock_history = stock.history(start=self.start_date, end=self.end_date)
            if 'Dividends' not in stock_history.columns:
                stock_history['Dividends'] = 0.0
            stock_history['Adj Close'] = stock_history['Close'] * (1 + stock_history['Dividends'].cumsum() / stock_history['Close'])
            df = stock_history[['Open', 'High', 'Low', 'Close', 'Adj Close']].rename(columns={
                'Open': f'{ticker.strip()}_Open',
                'High': f'{ticker.strip()}_High',
                'Low': f'{ticker.strip()}_Low',
                'Close': f'{ticker.strip()}_Close',
                'Adj Close': f'{ticker.strip()}_Adj_Close'
            })
            data_frames.append(df)
        combined_data = pd.concat(data_frames, axis=1)
        return combined_data

    def get_benchmark_data(self):
        data_frames = []
        for ticker in self.benchmark_tickers:
            benchmark = yf.Ticker(ticker)
            benchmark_history = benchmark.history(start=self.start_date, end=self.end_date)
            df = benchmark_history[['Close']].rename(columns={
                'Close': f'{ticker.strip()}_Close'
            })
            data_frames.append(df)
        combined_benchmark_data = pd.concat(data_frames, axis=1)
        return combined_benchmark_data

    def calculate_daily_returns(self):
        """Calculate both daily and weekly returns."""
        daily_returns = self.adjusted_closing_prices_df.pct_change().dropna()
        # Resample to weekly (end of week) and calculate returns
        weekly_prices = self.adjusted_closing_prices_df.resample('W').last()
        weekly_returns = weekly_prices.pct_change().dropna()
        return daily_returns, weekly_returns

### ---- Analysis and Plotting Functions ---- ###
    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=True)

    def plot_financial_data(self, data, title, y_label):
        fig = go.Figure()
        for column in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[column],
                mode='lines',
                name=column
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=y_label,
            legend_title="Legend",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        fig.show()

    def plot_stock_prices(self):
        self.plot_financial_data(self.adjusted_closing_prices_df, "Unscaled Adjusted Closing Prices", "Adjusted Close Price")

    def plot_scaled_prices(self):
        scaled_prices_df = self.price_scaling(self.adjusted_closing_prices_df)
        self.plot_financial_data(scaled_prices_df, "Scaled Adjusted Closing Prices", "Scaled Price")

    def plot_stock_prices_with_smas(self):
        sma_days = [100, 200]
        sma_dataframes = []
        for sma in sma_days:
            sma_df = self.adjusted_closing_prices_df.apply(lambda x: x.rolling(window=sma).mean())
            sma_df.columns = [f"{col.split('_Adj_Close')[0]}_{sma}SMA" for col in self.adjusted_closing_prices_df.columns]
            sma_dataframes.append(sma_df)
        sma_combined = pd.concat(sma_dataframes, axis=1)
        adjusted_closing_prices_with_sma = pd.concat([self.adjusted_closing_prices_df, sma_combined], axis=1)
        fig = go.Figure()
        for column in adjusted_closing_prices_with_sma.filter(regex='_Adj_Close$', axis=1).columns:
            stock_ticker = column.split('_Adj_Close')[0]
            fig.add_trace(go.Scatter(
                x=adjusted_closing_prices_with_sma.index,
                y=adjusted_closing_prices_with_sma[column],
                mode='lines',
                name=f'{stock_ticker} Adjusted Close',
                line=dict(width=2)
            ))
            for sma in sma_days:
                fig.add_trace(go.Scatter(
                    x=adjusted_closing_prices_with_sma.index,
                    y=adjusted_closing_prices_with_sma[f'{stock_ticker}_{sma}SMA'],
                    mode='lines',
                    name=f'{stock_ticker} {sma}-Day SMA',
                    line=dict(dash='dot', width=1.5),
                    opacity=0.6
                ))
        fig.update_layout(
            title="Stock Prices with SMAs (100-Day and 200-Day)",
            xaxis_title="Date",
            yaxis_title="Adjusted Close Price",
            legend_title="Stock Tickers and Indicators",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        fig.show()

    def plot_correlation_matrix(self):
        correlation_matrix = self.daily_returns_df.corr()
        
        # Set up the matplotlib figure
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        
        # Create heatmap with original crest palette
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='crest',  # Changed back to crest
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": .8}
        )
        
        plt.title("Correlation Matrix of Daily Returns", pad=20)
        plt.tight_layout()
        plt.show()

    def price_scaling(self, raw_prices_df):
        scaled_prices_df = raw_prices_df.copy()
        for column in raw_prices_df.columns:
            scaled_prices_df[column] = raw_prices_df[column] / raw_prices_df[column].iloc[0]
        return scaled_prices_df

### ---- Portfolio Management Functions ---- ###
    def generate_portfolio_weights(self, n):
        weights = np.random.random(n)
        weights /= weights.sum()
        return weights

    def generate_equal_weights(self, n):
        return np.array([1/n] * n)

    def asset_allocation(self, df, weights):
        portfolio_df = pd.DataFrame(index=df.index)
        scaled_df = self.price_scaling(df)
        for i, stock in enumerate(scaled_df.columns):
            portfolio_df[stock] = scaled_df[stock] * weights[i]
        portfolio_df['Portfolio Value'] = portfolio_df.sum(axis=1)
        portfolio_df['Portfolio Daily Return [%]'] = portfolio_df['Portfolio Value'].pct_change(1) * 100
        portfolio_df.replace(np.nan, 0, inplace=True)
        return portfolio_df

    def simulation_engine(self, df, weights):
        portfolio_df = self.asset_allocation(df, weights)
        final_value = portfolio_df['Portfolio Value'].iloc[-1]
        initial_value = portfolio_df['Portfolio Value'].iloc[0]
        return_on_investment = ((final_value - initial_value) / initial_value) * 100
        portfolio_daily_return_df = portfolio_df.drop(columns=['Portfolio Value', 'Portfolio Daily Return [%]']).pct_change()
        expected_portfolio_return = np.sum(weights * portfolio_daily_return_df.mean()) * 252
        covariance = portfolio_daily_return_df.cov() * 252
        expected_volatility = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        sharpe_ratio = max(0, (expected_portfolio_return - self.risk_free_rate) / expected_volatility)  # Set negative values to 0
        return expected_portfolio_return, expected_volatility, sharpe_ratio, final_value, return_on_investment

    def monte_carlo_simulation(self):
        n = self.adjusted_closing_prices_df.shape[1]
        weights_runs = np.zeros((self.sim_runs, n))
        sharpe_ratio_runs = np.zeros(self.sim_runs)
        expected_portfolio_returns_runs = np.zeros(self.sim_runs)
        volatility_runs = np.zeros(self.sim_runs)
        for i in range(self.sim_runs):
            weights = self.generate_portfolio_weights(n)
            weights_runs[i, :] = weights
            (expected_portfolio_returns_runs[i], 
            volatility_runs[i], 
            sharpe_ratio_runs[i], 
            _, _) = self.simulation_engine(self.adjusted_closing_prices_df, weights)
            if (i + 1) % 1000 == 0 or i == self.sim_runs - 1:
                print(f"Simulation Run {i + 1}/{self.sim_runs}")
        simulation_results = pd.DataFrame({
            "Expected Return (%)": expected_portfolio_returns_runs * 100,
            "Volatility (%)": volatility_runs * 100,
            "Sharpe Ratio": sharpe_ratio_runs,
        })
        weights_df = pd.DataFrame(weights_runs, columns=self.adjusted_closing_prices_df.columns)
        combined_results = pd.concat([simulation_results, weights_df], axis=1)
        return combined_results, volatility_runs, expected_portfolio_returns_runs, sharpe_ratio_runs

### ---- Plotting Simulation Results ---- ###
    def plot_simulation_results(self, volatility_runs, expected_portfolio_returns_runs, sharpe_ratio_runs):
        sim_out_df = pd.DataFrame({
            'Volatility': volatility_runs.tolist(),
            'Portfolio_Return': expected_portfolio_returns_runs.tolist(),
            'Sharpe_Ratio': sharpe_ratio_runs.tolist()
        })
        
        # Get Seaborn colors
        colors = self.get_seaborn_colors(10)  # Get 10 colors for the colorscale
        rgb_colors = [f'rgb({",".join([str(int(x*255)) for x in color])})' for color in colors]
        
        # Identify the portfolio with the highest Sharpe ratio
        optimal_index = sim_out_df['Sharpe_Ratio'].idxmax()
        optimal_volatility = sim_out_df.loc[optimal_index, 'Volatility']
        optimal_portfolio_return = sim_out_df.loc[optimal_index, 'Portfolio_Return']

        # Normalize Sharpe ratio for marker size (ensure positive values)
        min_sharpe = sim_out_df['Sharpe_Ratio'].min()
        max_sharpe = sim_out_df['Sharpe_Ratio'].max()
        normalized_sharpe = (sim_out_df['Sharpe_Ratio'] - min_sharpe) / (max_sharpe - min_sharpe)
        marker_sizes = 10 + (normalized_sharpe * 20)  # Scale between 10 and 30

        # Create a scatter plot with Seaborn colorscale
        fig = px.scatter(
            sim_out_df,
            x='Volatility',
            y='Portfolio_Return',
            color='Sharpe_Ratio',
            size=marker_sizes,  # Use normalized sizes
            hover_data=['Sharpe_Ratio'],
            color_continuous_scale=rgb_colors,
            labels={
                'Volatility': 'Portfolio Volatility (%)',
                'Portfolio_Return': 'Expected Return (%)',
                'Sharpe_Ratio': 'Sharpe Ratio'
            },
            title="Monte Carlo Simulation: Portfolio Volatility vs Expected Return"
        )

        # Highlight the optimal portfolio point
        fig.add_trace(go.Scatter(
            x=[optimal_volatility],
            y=[optimal_portfolio_return],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=25, color='red', symbol='x')
        ))
        
        # Calculate and add the Capital Allocation Line (CAL)
        cal_x = np.linspace(0, max(volatility_runs), 100)
        cal_y = self.risk_free_rate + cal_x * (optimal_portfolio_return - self.risk_free_rate) / optimal_volatility

        fig.add_trace(go.Scatter(
            x=cal_x,
            y=cal_y,
            mode='lines',
            name='Capital Allocation Line (CAL)',
            line=dict(color=rgb_colors[0], dash='dash', width=2)
        ))

        # Customize the chart layout
        fig.update_layout(
            coloraxis_colorbar=dict(title='Sharpe Ratio', dtick=5),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(t=50, b=50, l=50, r=50)
        )

        fig.show()

### ---- Portfolio Optimization ---- ###
    def optimize_portfolio(self):
        mean_returns = self.daily_returns_df.mean()
        cov_matrix = self.daily_returns_df.cov()

        def portfolio_performance(weights):
            returns = np.sum(weights * mean_returns) * 252
            std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            return returns, std_dev

        def neg_sharpe_ratio(weights):
            p_returns, p_std_dev = portfolio_performance(weights)
            return -(p_returns - self.risk_free_rate) / p_std_dev

        initial_guess = np.array(len(self.tickers) * [1. / len(self.tickers)])
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))

        optimized_result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized_result.x
        expected_return, volatility = portfolio_performance(optimal_weights)
        return optimal_weights, expected_return, volatility

    def plot_portfolio_vs_benchmarks(self, optimal_portfolio_df, equal_weighted_portfolio_df):
        benchmark_scaled = self.price_scaling(self.benchmark_data)
        optimal_portfolio_scaled = optimal_portfolio_df['Portfolio Value'] / optimal_portfolio_df['Portfolio Value'].iloc[0]
        equal_weighted_portfolio_scaled = equal_weighted_portfolio_df['Portfolio Value'] / equal_weighted_portfolio_df['Portfolio Value'].iloc[0]
        combined_df = benchmark_scaled.copy()
        combined_df['Optimal Portfolio'] = optimal_portfolio_scaled
        combined_df['Equal Weighted Portfolio'] = equal_weighted_portfolio_scaled
        self.plot_financial_data(combined_df, "Portfolio vs Benchmark Indices", "Scaled Value")

### ---- Stock Projection Data ---- ###
    def get_stock_data(self, ticker):
        # Fetch data for the given ticker
        stock = yf.Ticker(ticker)
        
        # Get the recent closing price
        stock_history = stock.history(period='1d')
        if stock_history.empty:
            raise ValueError(f"No price data found for {ticker}. The stock might be delisted.")
        recent_close_price = stock_history['Close'].iloc[0]
        
        # Get the TTM EPS (Trailing Twelve Months Earnings Per Share)
        ttm_eps = stock.info.get('trailingEps', None)
        if ttm_eps is None:
            raise ValueError(f"No TTM EPS data available for {ticker}.")
        
        # Get the beta
        beta = stock.info.get('beta', None)
        if beta is None:
            raise ValueError(f"No beta data available for {ticker}.")
        
        return recent_close_price, ttm_eps, beta

    def get_eps_growth_rate_yahoo(self, ticker):
        url = f"https://finance.yahoo.com/quote/{ticker}/analysis"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        growth_rate = None
        try:
            growth_rate_elem = soup.find("td", string="Next 5 Years (per annum)").find_next_sibling("td")
            growth_rate = float(growth_rate_elem.text.strip('%')) / 100
        except AttributeError:
            raise ValueError(f"Could not find EPS growth rate for {ticker} on Yahoo Finance.")
        return growth_rate

    def get_eps_growth_rate_finviz(self, ticker):
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        growth_rate = None
        try:
            growth_rate_elem = soup.find(string="EPS next 5Y").find_next("b")
            growth_rate = float(growth_rate_elem.text.strip('%')) / 100
        except AttributeError:
            raise ValueError(f"Could not find EPS growth rate for {ticker} on Finviz.")
        return growth_rate

    def get_eps_growth_rate_alpha_vantage(self, ticker):
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        
        growth_rate = None
        try:
            growth_rate = float(data['EPS'] / data['EPS'])
        except (KeyError, TypeError):
            raise ValueError(f"Could not find EPS growth rate for {ticker} using Alpha Vantage.")
        return growth_rate

    def get_historical_pe_ratio(self, ticker):
        stock = yf.Ticker(ticker)
        hist = stock.history(period='10y')
        hist['PE'] = hist['Close'] / stock.info['trailingEps']
        return hist['PE'].mean()

    def get_industry_pe_ratio(self, ticker):
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        industry_pe = None
        try:
            industry_pe_elem = soup.find(string="P/E").find_next("b")
            industry_pe = float(industry_pe_elem.text.strip())
        except AttributeError:
            raise ValueError(f"Could not find industry P/E ratio for {ticker} on Finviz.")
        return industry_pe

    def get_ttm_fcf(self, ticker):
        """Get TTM Free Cash Flow per share using multiple methods with retry logic."""
        session = create_session_with_retries()
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Method 1: Try Finviz P/FCF ratio
                try:
                    url = f"https://finviz.com/quote.ashx?t={ticker}"
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1"
                    }
                    
                    response = session.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Find P/FCF ratio - try multiple possible text variations
                    pfcf_variations = ["P/FCF", "Price/Free Cash Flow", "Price to Free Cash Flow"]
                    pfcf_elem = None
                    
                    for variation in pfcf_variations:
                        pfcf_elem = soup.find(string=lambda text: text and variation in text)
                        if pfcf_elem:
                            break
                    
                    if not pfcf_elem:
                        # Try finding by class name as fallback
                        pfcf_elem = soup.find("td", class_="snapshot-td2", string=lambda text: text and "P/FCF" in text)
                    
                    if pfcf_elem:
                        # Get the next element containing the value
                        pfcf_value = pfcf_elem.find_next("td", class_="snapshot-td2")
                        if pfcf_value:
                            # Clean and convert the value
                            pfcf_text = pfcf_value.text.strip()
                            if pfcf_text != "N/A" and pfcf_text != "-":
                                try:
                                    pfcf_ratio = float(pfcf_text)
                                    # Get current stock price
                                    stock = yf.Ticker(ticker)
                                    current_price = stock.history(period='1d')['Close'].iloc[0]
                                    fcf_per_share = current_price / pfcf_ratio
                                    fcf_yield = 1 / pfcf_ratio
                                    return fcf_per_share, fcf_yield
                                except (ValueError, IndexError):
                                    pass
                except Exception as e:
                    print(f"Finviz P/FCF method failed: {str(e)}")
                
                # Method 2: Try Yahoo Finance cash flow data
                try:
                    stock = yf.Ticker(ticker)
                    cash_flow = stock.quarterly_cashflow
                    
                    # Get last 4 quarters
                    operating_cash_flow = cash_flow.loc['Operating Cash Flow'][:4].sum()
                    capital_expenditures = cash_flow.loc['Capital Expenditure'][:4].sum()
                    
                    # Calculate FCF
                    fcf = operating_cash_flow + capital_expenditures  # CapEx is negative
                    
                    # Get shares outstanding
                    shares = stock.info.get('sharesOutstanding')
                    if shares:
                        fcf_per_share = fcf / shares
                        current_price = stock.history(period='1d')['Close'].iloc[0]
                        fcf_yield = fcf_per_share / current_price
                        
                        # Add warning for negative FCF
                        if fcf_per_share < 0:
                            print(f"\nWARNING: {ticker} has negative FCF per share: ${fcf_per_share:.2f}")
                            print("This may indicate financial distress or significant investment phase.")
                        
                        return fcf_per_share, fcf_yield
                except Exception as e:
                    print(f"Yahoo Finance cash flow method failed: {str(e)}")
                
                # Method 3: Try levered FCF from Yahoo Finance
                try:
                    levered_fcf = stock.info.get('freeCashflow')
                    shares = stock.info.get('sharesOutstanding')
                    if levered_fcf and shares:
                        fcf_per_share = levered_fcf / shares
                        current_price = stock.history(period='1d')['Close'].iloc[0]
                        fcf_yield = fcf_per_share / current_price
                        
                        # Add warning for negative FCF
                        if fcf_per_share < 0:
                            print(f"\nWARNING: {ticker} has negative FCF per share: ${fcf_per_share:.2f}")
                            print("This may indicate financial distress or significant investment phase.")
                        
                        return fcf_per_share, fcf_yield
                except Exception as e:
                    print(f"Yahoo Finance levered FCF method failed: {str(e)}")
                
                # If all methods fail, raise an error
                raise ValueError("Could not calculate FCF using any available method")
                
            except requests.RequestException as e:
                attempt += 1
                if attempt == max_attempts:
                    raise ValueError(f"Failed to fetch data after {max_attempts} attempts: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                raise ValueError(f"Error calculating FCF for {ticker}: {str(e)}")

    def get_fcf_growth_rate(self, ticker):
        """Get FCF growth rate using multiple methods with improved error handling and retries."""
        session = create_session_with_retries()
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                growth_rates = []
                weights = []
                
                # Method 1: Try Finviz first (primary source)
                try:
                    url = f"https://finviz.com/quote.ashx?t={ticker}"
                    headers = {"User-Agent": "Mozilla/5.0"}
                    response = session.get(url, headers=headers, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Get EPS growth rate
                    eps_growth_elem = soup.find(string="EPS next 5Y")
                    if eps_growth_elem:
                        eps_growth_text = eps_growth_elem.find_next("b").text.strip('%')
                        if eps_growth_text != "-" and eps_growth_text != "N/A":
                            eps_growth_rate = float(eps_growth_text) / 100
                            growth_rates.append(eps_growth_rate)
                            weights.append(0.4)  # Higher weight for EPS growth
                    
                    # Get Sales Y/Y TTM growth rate
                    sales_growth_elem = soup.find(string="Sales Y/Y TTM")
                    if sales_growth_elem:
                        sales_growth_text = sales_growth_elem.find_next("b").text.strip('%')
                        if sales_growth_text != "-" and sales_growth_text != "N/A":
                            sales_growth_rate = float(sales_growth_text) / 100
                            growth_rates.append(sales_growth_rate)
                            weights.append(0.3)  # Medium weight for sales growth
                except Exception as e:
                    print(f"Finviz data retrieval failed: {str(e)}")
                
                # Method 2: Calculate historical FCF growth using quarterly data
                try:
                    stock = yf.Ticker(ticker)
                    for _ in range(3):  # Retry logic for yfinance
                        try:
                            quarterly_cash_flow = stock.quarterly_cashflow
                            break
                        except Exception:
                            time.sleep(1)
                    
                    if not quarterly_cash_flow.empty and len(quarterly_cash_flow.columns) >= 8:
                        operating_cash_flow = quarterly_cash_flow.loc['Operating Cash Flow']
                        capital_expenditures = quarterly_cash_flow.loc['Capital Expenditure']
                        quarterly_fcf = operating_cash_flow + capital_expenditures
                        
                        current_ttm_fcf = quarterly_fcf[:4].sum()
                        previous_ttm_fcf = quarterly_fcf[4:8].sum()
                        
                        if previous_ttm_fcf != 0:
                            historical_growth = (current_ttm_fcf / previous_ttm_fcf) - 1
                            growth_rates.append(historical_growth)
                            weights.append(0.3)  # Medium weight for historical FCF growth
                except Exception as e:
                    print(f"Historical FCF calculation failed: {str(e)}")
                
                if growth_rates:
                    # Normalize weights
                    weights = [w/sum(weights) for w in weights]
                    weighted_growth = sum(r * w for r, w in zip(growth_rates, weights))
                    
                    # Add warning for negative growth
                    if weighted_growth < 0:
                        print(f"\nWARNING: {ticker} has negative growth rate: {weighted_growth:.2%}")
                        print("This may indicate declining business performance.")
                    
                    return weighted_growth
                
                # If no growth rates were successfully calculated, use a conservative default
                print(f"\nWARNING: Could not calculate growth rate for {ticker}")
                print("Using conservative default growth rate of 5%")
                return 0.05
                
            except requests.RequestException as e:
                attempt += 1
                if attempt == max_attempts:
                    print(f"Failed to fetch data after {max_attempts} attempts: {str(e)}")
                    return 0.05  # Conservative default growth rate
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Error in FCF growth rate calculation: {str(e)}")
                return 0.05  # Conservative default growth rate

    def get_industry_fcf_yield(self, ticker):
        """Get industry average FCF yield."""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        try:
            # Try to get industry FCF yield
            sector = soup.find(string="Sector").find_next("b").text
            industry = soup.find(string="Industry").find_next("b").text
            
            # Default industry average FCF yields based on sector
            sector_yields = {
                "Technology": 0.04,
                "Healthcare": 0.045,
                "Consumer Defensive": 0.05,
                "Consumer Cyclical": 0.055,
                "Industrial": 0.06,
                "Financial": 0.065,
                "Basic Materials": 0.07,
                "Energy": 0.075,
                "Real Estate": 0.08,
                "Utilities": 0.085,
                "Communication Services": 0.05
            }
            
            return sector_yields.get(sector, 0.05)  # Default to 5% if sector not found
        except Exception:
            return 0.05  # Default to 5% if unable to determine industry

    def calculate_fcf_future_price(self, ttm_fcf_per_share, fcf_growth_rate, future_fcf_yield, years=5):
        """Calculate future stock price using FCF per share valuation model."""
        try:
            # Calculate future FCF per share
            future_fcf_per_share = ttm_fcf_per_share * ((1 + fcf_growth_rate) ** years)
            
            # Calculate future stock price using expected FCF yield
            # Price = FCF per share / expected yield
            future_price = future_fcf_per_share / future_fcf_yield
            
            return future_price
            
        except Exception as e:
            print(f"Error in FCF calculation: {str(e)}")
            return None

    def calculate_future_price(self, ttm_eps, eps_growth_rate, future_pe_ratio, years=5):
        # Calculate the future EPS
        future_eps = ttm_eps * ((1 + eps_growth_rate) ** years)
        
        # Calculate the future stock price
        future_price = future_eps * future_pe_ratio
        
        return future_price

    def calculate_cagr(self, current_price, future_price, years=5):
        return (future_price / current_price) ** (1 / years) - 1

    def calculate_required_rate_of_return(self, beta, risk_free_rate, market_premium=0.0433):
        return risk_free_rate + beta * market_premium

    def calculate_present_value(self, future_value, required_rate_of_return, years=5):
        return future_value / ((1 + required_rate_of_return) ** years)

    def plot_future_price(self, tickers, recent_close_prices, future_prices, current_year, current_quarter, required_rate_of_return):
        for ticker, recent_close_price, future_price in zip(tickers, recent_close_prices, future_prices):
            # Get historical data (3 years)
            historical_data = self.data.filter(regex=f'{ticker}_Adj_Close$', axis=1)
            historical_dates = historical_data.index
            historical_prices = historical_data.iloc[:, 0].values

            # Set up the x-axis labels for future projections (quarters)
            future_quarters = [f"{year} Q{quarter}" for year in range(current_year, current_year + 6) for quarter in range(1, 5)]
            future_quarters = future_quarters[:21]  # Limit to 5 years + 1 quarter
            
            # Generate dates for future quarters
            future_dates = pd.date_range(start=self.end_date, periods=len(future_quarters), freq='QE')

            # Generate the future prices for each quarter
            cagr = self.calculate_cagr(recent_close_price, future_price)
            future_prices = [recent_close_price * ((1 + cagr) ** (i / 4)) for i in range(len(future_quarters))]
            present_values = [self.calculate_present_value(price, required_rate_of_return, i / 4) for i, price in enumerate(future_prices)]

            # Create the plot
            fig = go.Figure()

            # Add historical prices
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_prices,
                mode='lines',
                name=f'{ticker} Historical Price',
                line=dict(color='blue')
            ))

            # Add vertical line at current date using shapes
            fig.add_shape(
                type="line",
                x0=self.end_date,
                x1=self.end_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color="gray",
                    dash="dash",
                )
            )

            # Add annotation for projection start
            fig.add_annotation(
                x=self.end_date,
                y=1,
                yref="paper",
                text="Projection Start",
                showarrow=False,
                yshift=10
            )

            # Add projected prices
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode='lines',
                name=f'{ticker} Projected Price',
                line=dict(color='green')
            ))

            # Add present values
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=present_values,
                mode='lines',
                name=f'{ticker} Present Value',
                line=dict(color='red')
            ))

            fig.update_layout(
                title=f"Historical and Projected Stock Price for {ticker}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=True,
                hovermode='x unified'
            )
            
            # Show the chart
            fig.show()

    def display_projection_table(self, tickers, recent_close_prices, future_prices, required_rate_of_return, current_year):
        for ticker, recent_close_price, future_price in zip(tickers, recent_close_prices, future_prices):
            # Generate the future prices and present values for each year
            years = list(range(current_year, current_year + 6))
            future_prices = [recent_close_price * ((1 + self.calculate_cagr(recent_close_price, future_price)) ** (year - current_year)) for year in years]
            present_values = [self.calculate_present_value(price, required_rate_of_return, year - current_year) for year, price in zip(years, future_prices)]
            expected_returns = [(pv / recent_close_price) - 1 for pv in present_values]

            # Create the table
            table_data = {
                "Year": years,
                "Projected Price": [f"${price:.2f}" for price in future_prices],
                "Present Value": [f"${pv:.2f}" for pv in present_values],
                "Expected Return": [f"{er:.2%}" for er in expected_returns]
            }
            table_df = pd.DataFrame(table_data)

            # Display the table
            print(f"\nProjection Table for {ticker}:")
            print(table_df)

### ---- Future Portfolio Optimization ---- ###
    def calculate_future_portfolio_metrics(self, tickers, recent_close_prices, weighted_future_prices, eps_growth_rates):
        # Calculate expected annual returns from all models separately
        pe_returns = []
        fcf_returns = []
        advanced_returns = []
        
        for ticker, recent_price, weighted_price in zip(tickers, recent_close_prices, weighted_future_prices):
            # Get PE model projection
            historical_pe = self.get_historical_pe_ratio(ticker)
            industry_pe = self.get_industry_pe_ratio(ticker)
            future_pe = (historical_pe + industry_pe) / 2
            ttm_eps = self.get_stock_data(ticker)[1]
            eps_growth = next((rate for t, rate in zip(tickers, eps_growth_rates) if t == ticker), 0)
            pe_future_price = self.calculate_future_price(ttm_eps, eps_growth, future_pe)
            
            # Get FCF model projection
            try:
                ttm_fcf_per_share, _ = self.get_ttm_fcf(ticker)
                fcf_growth_rate = self.get_fcf_growth_rate(ticker)
                future_fcf_yield = self.get_weighted_fcf_yield(ticker, print_results=False)
                fcf_future_price = self.calculate_fcf_future_price(ttm_fcf_per_share, fcf_growth_rate, future_fcf_yield)
                fcf_return = (fcf_future_price/recent_price)**(1/5) - 1 if fcf_future_price else None
            except Exception:
                fcf_return = None
            
            # Get Advanced model projection
            confidence_range = self.calculate_advanced_price_projection(
                ticker, self.data, ttm_eps, eps_growth, self.get_stock_data(ticker)[2]
            )
            advanced_future_price = confidence_range['base_case']
            
            # Calculate returns for all models
            pe_return = (pe_future_price/recent_price)**(1/5) - 1
            advanced_return = (advanced_future_price/recent_price)**(1/5) - 1
            
            pe_returns.append(pe_return)
            fcf_returns.append(fcf_return if fcf_return is not None else pe_return)  # Use PE return as fallback
            advanced_returns.append(advanced_return)
        
        # Convert to numpy arrays
        pe_returns = np.array(pe_returns)
        fcf_returns = np.array(fcf_returns)
        advanced_returns = np.array(advanced_returns)
        
        # Calculate weighted average returns (equal weights for all three models)
        weighted_returns = (pe_returns * 0.333 + fcf_returns * 0.333 + advanced_returns * 0.334)
        
        # Get historical volatility (risk)
        historical_volatility = np.array([self.daily_returns_df[f"{ticker}_Adj_Close"].std() * np.sqrt(252) for ticker in tickers])
        
        # Calculate growth-adjusted Sharpe ratios using weighted returns
        growth_adjusted_sharpe = (weighted_returns - self.risk_free_rate) / historical_volatility
        
        return pe_returns, fcf_returns, advanced_returns, weighted_returns, historical_volatility, growth_adjusted_sharpe

    def optimize_future_portfolio(self, tickers, pe_returns, fcf_returns, advanced_returns, weighted_returns, historical_volatility):
        n = len(tickers)
        
        def future_portfolio_performance(weights):
            returns = np.sum(weights * weighted_returns)  # Use weighted returns for optimization
            risk = np.sqrt(np.dot(weights.T, np.dot(np.diag(historical_volatility**2), weights)))
            return returns, risk
        
        def neg_future_sharpe_ratio(weights):
            p_returns, p_risk = future_portfolio_performance(weights)
            return -(p_returns - self.risk_free_rate) / p_risk if p_risk > 0 else -999
        
        initial_weights = np.array([1/n] * n)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        optimized_result = minimize(neg_future_sharpe_ratio, initial_weights, method='SLSQP', 
                                  bounds=bounds, constraints=constraints)
        
        optimal_weights = optimized_result.x
        expected_return, expected_risk = future_portfolio_performance(optimal_weights)
        future_sharpe_ratio = (expected_return - self.risk_free_rate) / expected_risk
        
        return optimal_weights, expected_return, expected_risk, future_sharpe_ratio

    def display_future_portfolio_analysis(self, tickers, current_weights, future_weights, 
                                        pe_returns, fcf_returns, advanced_returns, weighted_returns, historical_volatility):
        # Create comparison DataFrame with all model returns
        comparison_data = {
            'Current Weight': [f"{w:.2%}" for w in current_weights],
            'Future Weight': [f"{w:.2%}" for w in future_weights],
            'PE Model Return': [f"{r:.2%}" for r in pe_returns],
            'FCF Model Return': [f"{r:.2%}" for r in fcf_returns],
            'Advanced Model Return': [f"{r:.2%}" for r in advanced_returns],
            'Weighted Return': [f"{r:.2%}" for r in weighted_returns],
            'Historical Volatility': [f"{v:.2%}" for v in historical_volatility]
        }
        comparison_df = pd.DataFrame(comparison_data, index=tickers)
        
        # Break the DataFrame display into chunks
        df_str = comparison_df.to_string()
        df_lines = df_str.split('\n')
        
        print("\n" + "="*80)
        print("5-YEAR PORTFOLIO OPTIMIZATION MODELING")
        print("="*80)
        
        print("\nPortfolio Allocation and Returns Comparison:")
        print("-" * 80)
        
        # Print DataFrame in chunks
        chunk_size = 30
        for i in range(0, len(df_lines), chunk_size):
            print('\n'.join(df_lines[i:i + chunk_size]))
            if i + chunk_size < len(df_lines):
                print("\n" + "-" * 40 + " Continued " + "-" * 40 + "\n")
        
        print("\n" + "-" * 80)
        print("\nCurrent Portfolio Metrics:")
        print("-" * 40)
        
        # Calculate current portfolio metrics
        current_pe_return = np.sum(current_weights * pe_returns)
        current_fcf_return = np.sum(current_weights * fcf_returns)
        current_adv_return = np.sum(current_weights * advanced_returns)
        current_weighted_return = np.sum(current_weights * weighted_returns)
        current_portfolio_risk = np.sqrt(np.dot(current_weights.T, np.dot(np.diag(historical_volatility**2), current_weights)))
        current_sharpe = (current_weighted_return - self.risk_free_rate) / current_portfolio_risk
        
        # Display current metrics in chunks
        print(f"PE Model Expected Return: {current_pe_return:.2%}")
        print(f"FCF Model Expected Return: {current_fcf_return:.2%}")
        print(f"Advanced Model Expected Return: {current_adv_return:.2%}")
        print(f"Weighted Expected Return: {current_weighted_return:.2%}")
        print(f"Expected Annual Volatility: {current_portfolio_risk:.2%}")
        print(f"Projected Sharpe Ratio: {current_sharpe:.2f}")
        
        print("\n" + "-" * 80)
        print("\nOptimized Future Portfolio Metrics:")
        print("-" * 40)
        
        # Calculate future portfolio metrics
        future_pe_return = np.sum(future_weights * pe_returns)
        future_fcf_return = np.sum(future_weights * fcf_returns)
        future_adv_return = np.sum(future_weights * advanced_returns)
        future_weighted_return = np.sum(future_weights * weighted_returns)
        future_portfolio_risk = np.sqrt(np.dot(future_weights.T, np.dot(np.diag(historical_volatility**2), future_weights)))
        future_sharpe = (future_weighted_return - self.risk_free_rate) / future_portfolio_risk
        
        # Display future metrics in chunks
        print(f"PE Model Expected Return: {future_pe_return:.2%}")
        print(f"FCF Model Expected Return: {future_fcf_return:.2%}")
        print(f"Advanced Model Expected Return: {future_adv_return:.2%}")
        print(f"Weighted Expected Return: {future_weighted_return:.2%}")
        print(f"Expected Annual Volatility: {future_portfolio_risk:.2%}")
        print(f"Projected Sharpe Ratio: {future_sharpe:.2f}")
        
        print("\n" + "-" * 80)
        
        # Create visualization of weight changes
        self.plot_weight_comparison(tickers, current_weights, future_weights)

    def plot_weight_comparison(self, tickers, current_weights, future_weights):
        fig = go.Figure()
        
        # Add current weights
        fig.add_trace(go.Bar(
            name='Current Weights',
            x=tickers,
            y=current_weights,
            marker_color='blue'
        ))
        
        # Add future weights
        fig.add_trace(go.Bar(
            name='Future Weights',
            x=tickers,
            y=future_weights,
            marker_color='green'
        ))
        
        # Update layout
        fig.update_layout(
            title="Portfolio Weight Comparison: Current vs Future",
            xaxis_title="Stocks",
            yaxis_title="Weight",
            barmode='group',
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Update y-axis to show percentages
        fig.update_yaxes(tickformat='.0%')
        
        fig.show()

### ---- Advanced Price Projection Models ---- ###
    def calculate_technical_indicators(self, ticker_data):
        # Calculate technical indicators
        data = ticker_data.copy()
        
        # Moving averages
        data['SMA_50'] = ticker_data.rolling(window=50).mean()
        data['SMA_200'] = ticker_data.rolling(window=200).mean()
        
        # Momentum indicators
        data['ROC'] = ticker_data.pct_change(periods=20)  # 20-day Rate of Change
        data['RSI'] = self.calculate_rsi(ticker_data)
        
        # Volatility
        data['Volatility'] = ticker_data.rolling(window=20).std()
        
        return data

    def calculate_rsi(self, data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_advanced_price_projection(self, ticker, historical_data, ttm_eps, eps_growth_rate, beta):
        # Get the data ready
        ticker_data = historical_data[f'{ticker}_Adj_Close']
        tech_indicators = self.calculate_technical_indicators(ticker_data)
        
        # 1. Technical Analysis Projection
        current_price = ticker_data.iloc[-1]
        sma_50 = tech_indicators['SMA_50'].iloc[-1]
        sma_200 = tech_indicators['SMA_200'].iloc[-1]
        momentum = tech_indicators['ROC'].iloc[-1]
        rsi = tech_indicators['RSI'].iloc[-1]
        
        # Technical trend factor (bounded)
        tech_trend = np.clip((sma_50 / sma_200 - 1) + (momentum * 0.5) + ((rsi - 50) / 100), -0.2, 0.2)

        # 2. Fundamental Analysis
        fundamental_growth = (1 + eps_growth_rate) ** 5
        market_risk_premium = 0.0433  # Market risk premium
        required_return = self.risk_free_rate + beta * market_risk_premium
        
        # 3. Market Sentiment Adjustment (bounded)
        sentiment_factor = np.clip(1 + (tech_trend * 0.2), 0.8, 1.2)
        
        # 4. Volatility Adjustment
        volatility = tech_indicators['Volatility'].iloc[-1]
        vol_adjustment = np.clip(1 + (volatility * beta), 0.8, 1.2)
        
        # Combine all factors for final projection
        base_projection = current_price * fundamental_growth
        technical_projection = current_price * (1 + tech_trend) ** 5
        
        # Weighted average of different projections
        final_projection = (
            base_projection * 0.4 +      # Fundamental weight
            technical_projection * 0.3 +  # Technical weight
            (base_projection * sentiment_factor) * 0.2 +  # Sentiment-adjusted weight
            (base_projection * vol_adjustment) * 0.1      # Volatility-adjusted weight
        )
        
        # Calculate more reasonable confidence intervals
        annual_volatility = tech_indicators['Volatility'].std() * np.sqrt(252)
        z_score = 1.96  # 95% confidence interval
        
        # Calculate confidence bounds with reasonable limits
        volatility_factor = annual_volatility * z_score / np.sqrt(5)  # Adjust for 5-year projection
        upper_bound = final_projection * (1 + volatility_factor)
        lower_bound = final_projection * (1 - volatility_factor)
        
        # Ensure bounds are reasonable (within ±50% of base projection)
        upper_bound = min(upper_bound, final_projection * 1.5)
        lower_bound = max(lower_bound, final_projection * 0.5)
        
        confidence_range = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'base_case': final_projection
        }
        
        return confidence_range

    def plot_advanced_projections(self, ticker, historical_data, confidence_range, recent_close_price, pe_future_price, fcf_future_price, required_rate_of_return):
        # Get 5 years of historical data
        current_date = pd.Timestamp(self.end_date).tz_localize(None)
        historical_start = current_date - dt.timedelta(days=5*365)
        
        # Convert index to timezone-naive if it's timezone-aware
        historical_data_index = historical_data.index.tz_localize(None) if historical_data.index.tz is not None else historical_data.index
        historical_data = historical_data.copy()
        historical_data.index = historical_data_index
        
        historical_data_filtered = historical_data[historical_start:current_date]
        
        future_dates = pd.date_range(start=current_date, periods=21, freq='QE')
        
        # Get the most recent closing price from historical data
        current_price = historical_data_filtered[f'{ticker}_Adj_Close'].iloc[-1]
        
        # Create projection lines starting from current price
        base_line = np.linspace(current_price, 
                              confidence_range['base_case'], len(future_dates))
        lower_line = np.linspace(current_price, 
                               confidence_range['lower_bound'], len(future_dates))
        upper_line = np.linspace(current_price, 
                               confidence_range['upper_bound'], len(future_dates))
        
        # Calculate PE-based projections from current price
        pe_cagr = self.calculate_cagr(current_price, pe_future_price)
        pe_projections = [current_price * ((1 + pe_cagr) ** (i / 4)) for i in range(len(future_dates))]
        
        # Calculate FCF-based projections if available from current price
        fcf_projections = None
        if fcf_future_price is not None:
            fcf_cagr = self.calculate_cagr(current_price, fcf_future_price)
            fcf_projections = [current_price * ((1 + fcf_cagr) ** (i / 4)) for i in range(len(future_dates))]
        
        # Create the plot
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical_data_filtered.index,
            y=historical_data_filtered[f'{ticker}_Adj_Close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue'),
            hovertemplate='<b>Historical Price</b><br>' +
                          'Date: %{x}<br>' +
                          'Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Plot technical/fundamental base projection
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=base_line,
            mode='lines',
            name='Technical/Fundamental Projection',
            line=dict(color='green', dash='dash'),
            hovertemplate='<b>Tech/Fund Projection</b><br>' +
                          'Date: %{x}<br>' +
                          'Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Plot PE-based projection
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=pe_projections,
            mode='lines',
            name='PE-Based Projection',
            line=dict(color='yellow'),
            hovertemplate='<b>PE-Based Projection</b><br>' +
                          'Date: %{x}<br>' +
                          'Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Plot FCF-based projection if available
        if fcf_projections is not None:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=fcf_projections,
                mode='lines',
                name='FCF-Based Projection',
                line=dict(color='purple'),
                hovertemplate='<b>FCF-Based Projection</b><br>' +
                              'Date: %{x}<br>' +
                              'Price: $%{y:.2f}<extra></extra>'
            ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_line,
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(128,128,128,0.2)'),
            hovertemplate='<b>Upper Bound</b><br>' +
                          'Date: %{x}<br>' +
                          'Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_line,
            mode='lines',
            name='Lower Bound',
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)',
            line=dict(color='rgba(128,128,128,0.2)'),
            hovertemplate='<b>Lower Bound</b><br>' +
                          'Date: %{x}<br>' +
                          'Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Add vertical line at current date
        fig.add_shape(
            type="line",
            x0=current_date,
            x1=current_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", dash="dash")
        )
        
        # Add annotation for projection start
        fig.add_annotation(
            x=current_date,
            y=1,
            yref="paper",
            text="Projection Start",
            showarrow=False,
            yshift=10
        )
        
        # Update layout with hover modifications
        fig.update_layout(
            title=f"Comprehensive Price Projections for {ticker}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            # Modify hover style
            hoverlabel=dict(
                bgcolor='rgba(50,50,50,0.9)',  # Dark semi-transparent background
                font_size=12,
                font_family="Arial",
                font_color='white',  # White text
                bordercolor='rgba(255,255,255,0.3)'  # Light border
            )
        )
        
        fig.show()

    def get_seaborn_colors(self, n_colors):
        """Get a list of Seaborn colors for plotting."""
        return sns.color_palette("husl", n_colors=n_colors)

    def plot_daily_returns_boxplot(self):
        """Plot weekly returns distribution with non-linear scale for outliers."""
        fig = go.Figure()
        colors = self.get_seaborn_colors(len(self.weekly_returns_df.columns))
        
        def transform_returns(returns):
            """Transform returns to emphasize -10% to 10% range while condensing outliers."""
            returns = returns * 100  # Convert to percentage
            mask_positive = returns > 10
            mask_negative = returns < -10
            mask_middle = ~(mask_positive | mask_negative)
            
            # Keep middle range as is
            transformed = returns.copy()
            
            # Transform values above 10%
            transformed[mask_positive] = 10 + np.log1p(returns[mask_positive] - 10)
            
            # Transform values below -10%
            transformed[mask_negative] = -10 - np.log1p(-returns[mask_negative] - 10)
            
            return transformed
        
        for i, column in enumerate(self.weekly_returns_df.columns):
            ticker = column.split('_')[0]  # Extract ticker from column name
            rgb_color = f'rgb({",".join([str(int(x*255)) for x in colors[i]])})'
            
            # Transform the returns
            transformed_returns = transform_returns(self.weekly_returns_df[column])
            
            fig.add_trace(go.Box(
                y=transformed_returns,
                name=ticker,
                boxpoints='outliers',
                marker_color=rgb_color,
                line_color='white'
            ))
        
        # Create custom tick values and labels
        tick_values = []
        tick_labels = []
        
        # Add ticks for the condensed negative region
        for val in [-30, -20]:
            transformed_val = -10 - np.log1p(-val - 10)
            tick_values.append(transformed_val)
            tick_labels.append(f"{val}%")
        
        # Add ticks for the linear region
        for val in [-10, -5, 0, 5, 10]:
            tick_values.append(val)
            tick_labels.append(f"{val}%")
        
        # Add ticks for the condensed positive region
        for val in [20, 30]:
            transformed_val = 10 + np.log1p(val - 10)
            tick_values.append(transformed_val)
            tick_labels.append(f"{val}%")
        
        fig.update_layout(
            title="Weekly Returns Distribution by Stock (Non-linear Scale)",
            yaxis=dict(
                title="Weekly Returns (%)",
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_labels,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.5)',
                zerolinewidth=1
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        # Add horizontal lines at ±10%
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(self.weekly_returns_df.columns) - 0.5,
            y0=10,
            y1=10,
            line=dict(color="rgba(255, 255, 255, 0.3)", dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(self.weekly_returns_df.columns) - 0.5,
            y0=-10,
            y1=-10,
            line=dict(color="rgba(255, 255, 255, 0.3)", dash="dash")
        )
        
        fig.show()

    def get_fcf_yield_from_cash_flow(self, ticker):
        """Calculate FCF Yield using actual cash flow statement data."""
        try:
            stock = yf.Ticker(ticker)
            cash_flow = stock.quarterly_cashflow
            
            # Get last 4 quarters
            operating_cash_flow = cash_flow.loc['Operating Cash Flow'][:4].sum()
            capital_expenditures = cash_flow.loc['Capital Expenditure'][:4].sum()
            
            # Calculate FCF
            fcf = operating_cash_flow + capital_expenditures  # CapEx is negative
            
            # Get market cap
            market_cap = stock.info.get('marketCap')
            
            # Calculate yield
            fcf_yield = fcf / market_cap
            
            return fcf_yield
            
        except Exception as e:
            raise ValueError(f"Error calculating FCF yield from cash flow: {str(e)}")

    def get_industry_peers(self, ticker):
        """Get peer companies in the same industry."""
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find industry
            industry_elem = soup.find(string="Industry")
            if not industry_elem:
                raise ValueError("Could not find industry")
            
            industry = industry_elem.find_next("b").text
            
            # Search for peers in the same industry
            peers_url = f"https://finviz.com/screener.ashx?v=111&f=ind_{industry.replace(' ', '')}"
            response = requests.get(peers_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract peer tickers
            peer_tickers = []
            ticker_elems = soup.find_all("a", class_="screener-link-primary")
            for elem in ticker_elems:
                peer_ticker = elem.text.strip()
                if peer_ticker != ticker:  # Exclude the original ticker
                    peer_tickers.append(peer_ticker)
            
            return peer_tickers
            
        except Exception as e:
            raise ValueError(f"Error finding industry peers: {str(e)}")

    def get_enhanced_industry_fcf_yield(self, ticker):
        """Get enhanced industry FCF yield with peer comparison."""
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Get sector and industry
            sector = soup.find(string="Sector").find_next("b").text
            industry = soup.find(string="Industry").find_next("b").text
            
            # Get peer tickers in same industry
            peer_tickers = self.get_industry_peers(ticker)
            
            # Calculate average FCF yield of peers
            peer_yields = []
            for peer in peer_tickers[:5]:  # Limit to 5 peers
                try:
                    _, peer_yield = self.get_ttm_fcf(peer)
                    peer_yields.append(peer_yield)
                except:
                    continue
            
            if peer_yields:
                return sum(peer_yields) / len(peer_yields)
            
            # Fallback to sector averages if peer calculation fails
            return self.get_industry_fcf_yield(ticker)
            
        except Exception:
            return 0.05  # Default to 5% if unable to determine

    def get_fcf_yield_from_levered_fcf(self, ticker):
        """Calculate FCF Yield using levered free cash flow."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get levered free cash flow from Yahoo Finance
            levered_fcf = stock.info.get('freeCashflow')
            market_cap = stock.info.get('marketCap')
            
            if levered_fcf and market_cap:
                return levered_fcf / market_cap
            
            raise ValueError("Could not get levered FCF or market cap")
            
        except Exception as e:
            raise ValueError(f"Error calculating FCF yield from levered FCF: {str(e)}")

    def get_weighted_fcf_yield(self, ticker, print_results=False):
        """Calculate FCF Yield using multiple methods and weight them."""
        try:
            yields = []
            weights = []
            method_results = []
            
            # Method 1: Direct P/FCF method
            try:
                _, yield1 = self.get_ttm_fcf(ticker)
                yields.append(yield1)
                weights.append(0.4)  # Highest weight for direct calculation
                method_results.append({
                    'name': "Method 1 - Direct P/FCF",
                    'yield': yield1
                })
            except Exception as e:
                print(f"Direct P/FCF method failed: {str(e)}")
            
            # Method 2: Cash flow statement method
            try:
                yield2 = self.get_fcf_yield_from_cash_flow(ticker)
                yields.append(yield2)
                weights.append(0.3)
                method_results.append({
                    'name': "Method 2 - Cash Flow Statement",
                    'yield': yield2
                })
            except Exception as e:
                print(f"Cash flow statement method failed: {str(e)}")
            
            # Method 3: Levered FCF method
            try:
                yield3 = self.get_fcf_yield_from_levered_fcf(ticker)
                yields.append(yield3)
                weights.append(0.2)
                method_results.append({
                    'name': "Method 3 - Levered FCF",
                    'yield': yield3
                })
            except Exception as e:
                print(f"Levered FCF method failed: {str(e)}")
            
            # Method 4: Industry average method
            try:
                yield4 = self.get_enhanced_industry_fcf_yield(ticker)
                yields.append(yield4)
                weights.append(0.1)  # Lowest weight for industry average
                method_results.append({
                    'name': "Method 4 - Industry Average",
                    'yield': yield4
                })
            except Exception as e:
                print(f"Industry average method failed: {str(e)}")
            
            if yields:
                # Print results from each method only if requested
                if print_results:
                    print("\nFCF Yield Calculation Methods:")
                    for method in method_results:
                        print(f"{method['name']}: {method['yield']:.2%}")
                
                # Normalize weights
                weights = [w/sum(weights) for w in weights]
                # Calculate weighted average
                weighted_yield = sum(y * w for y, w in zip(yields, weights))
                if print_results:
                    print(f"\nWeighted Average FCF Yield: {weighted_yield:.2%}")
                return weighted_yield
            
            raise ValueError("Could not calculate FCF yield using any method")
            
        except Exception as e:
            raise ValueError(f"Error calculating weighted FCF yield: {str(e)}")

### ---- Main Execution ---- ###
if __name__ == "__main__":
    while True:
        print("\nStock Portfolio Analysis Tool")
        print("-" * 40)
        print("Enter stock tickers (comma-separated) or '1' for random S&P 500 selection")
        print("-" * 40)
        
        tickers_input = input("Enter your choice: ").strip()
        
        if tickers_input == "1":
            tickers = StockAnalysis.get_random_portfolio()
            if not tickers:
                print("\nError generating random portfolio.")
                print("Using default major stocks instead:")
                tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
                print(", ".join(tickers))
        else:
            tickers = [ticker.strip() for ticker in tickers_input.split(",")]
            tickers = [ticker.upper() for ticker in tickers if ticker]
        
        if not tickers:
            print("\nNo valid tickers provided. Please try again.")
            continue
            
        try:
            print("\nInitializing analysis...")
            time_frame = 5  # Default to 5 years
            stock_analysis = StockAnalysis(tickers, time_frame)
            break
        except Exception as e:
            print(f"\nError initializing analysis: {e}")
            print("Please try again with different tickers.")
            continue
    
    print("\n" + "="*80)
    print("STOCK PORTFOLIO ANALYSIS")
    print("="*80)
    
    print("\nSelected Stocks for Analysis:")
    print("-" * 40)
    for i, ticker in enumerate(tickers, 1):
        print(f"{i}. {ticker}")
    
    print("\n" + "-"*40)
    print("MARKET CONDITIONS & ANALYSIS PARAMETERS")
    print("-"*40)
    print(f"Analysis Period: {stock_analysis.start_date.strftime('%Y-%m-%d')} to {stock_analysis.end_date.strftime('%Y-%m-%d')}")
    print(f"Risk-free Rate: {stock_analysis.risk_free_rate:.2%}")
    print(f"Market Risk Premium: 4.33%")

    try:
        print("\n" + "-"*80)
        print("INDIVIDUAL STOCK ANALYSIS")
        print("-"*80)
        
        # Initialize lists to store results
        recent_close_prices = []
        ttm_eps_list = []
        betas = []
        pe_future_prices = []
        eps_growth_rates = []
        future_pe_ratios = []
        required_returns = []
        weighted_future_prices = []
        
        # Process each ticker
        for ticker in tickers:
            try:
                print(f"\nAnalyzing {ticker}...")
                print("="*50)

                # Get current stock data
                recent_close_price, ttm_eps, beta = stock_analysis.get_stock_data(ticker)
                required_rate_of_return = stock_analysis.calculate_required_rate_of_return(beta, stock_analysis.risk_free_rate)

                # Current Price and Basic Metrics
                print("\nBASIC METRICS")
                print("-" * 40)
                print(f"Current Price: ${recent_close_price:.2f}")
                print(f"TTM EPS: ${ttm_eps:.2f}")
                if ttm_eps < 0:
                    print("WARNING: Negative EPS indicates current losses")
                print(f"Beta: {beta:.2f}")
                print(f"Required Return (CAPM): {required_rate_of_return:.2%}")

                # Free Cash Flow Analysis
                print("\nFREE CASH FLOW ANALYSIS")
                print("-" * 40)
                try:
                    ttm_fcf_per_share, current_fcf_yield = stock_analysis.get_ttm_fcf(ticker)
                    weighted_fcf_yield = stock_analysis.get_weighted_fcf_yield(ticker, print_results=False)
                    print(f"FCF per Share: ${ttm_fcf_per_share:.2f}")
                    if ttm_fcf_per_share < 0:
                        print("WARNING: Negative FCF indicates cash burn or significant investment phase")
                    print(f"P/FCF Ratio: {recent_close_price/ttm_fcf_per_share:.2f}")
                    print(f"FCF Yield: {current_fcf_yield*100:.2f}%")
                except Exception as e:
                    print(f"FCF Data Unavailable: {str(e)}")
                    print("Note: FCF-based valuation will not be available")

                # Growth Analysis
                print("\nGROWTH ANALYSIS")
                print("-" * 40)
                try:
                    fcf_growth_rate = stock_analysis.get_fcf_growth_rate(ticker)
                    eps_growth_rate = stock_analysis.get_eps_growth_rate_finviz(ticker)
                    print(f"FCF Growth Rate: {fcf_growth_rate:.2%}")
                    print(f"EPS Growth Rate: {eps_growth_rate:.2%}")
                    if fcf_growth_rate < 0 or eps_growth_rate < 0:
                        print("WARNING: Negative growth rates indicate declining performance")
                except ValueError as e:
                    print(f"Growth Rate Data Unavailable: {str(e)}")
                    print("Using conservative growth assumptions")

                # Valuation Metrics
                print("\nVALUATION METRICS")
                print("-" * 40)
                try:
                    historical_pe_ratio = stock_analysis.get_historical_pe_ratio(ticker)
                    industry_pe_ratio = stock_analysis.get_industry_pe_ratio(ticker)
                    future_pe_ratio = (historical_pe_ratio + industry_pe_ratio) / 2
                    future_fcf_yield = stock_analysis.get_weighted_fcf_yield(ticker, print_results=False)

                    print(f"Historical P/E: {historical_pe_ratio:.2f}")
                    print(f"Industry P/E: {industry_pe_ratio:.2f}")
                    print(f"Estimated Future P/E: {future_pe_ratio:.2f}")
                    
                    if ttm_eps < 0:
                        print("Note: P/E ratio is not meaningful for companies with negative earnings")
                except Exception as e:
                    print(f"Valuation metrics unavailable: {str(e)}")
                    print("Using industry average metrics")

                # Price Projections
                print("\nPRICE PROJECTIONS")
                print("-" * 40)
                try:
                    # Initialize variables for different models
                    pe_future_price = None
                    fcf_future_price = None
                    confidence_range = None
                    weighted_future_price = None
                    
                    # Calculate PE-based projection only if EPS is positive
                    if ttm_eps > 0:
                        try:
                            pe_future_price = stock_analysis.calculate_future_price(
                                ttm_eps, eps_growth_rate, future_pe_ratio)
                        except Exception as e:
                            print(f"PE-based projection not available: {str(e)}")
                    else:
                        print("Note: PE-based projection omitted due to negative EPS")
                    
                    # Calculate FCF-based projection only if FCF per share is positive
                    if ttm_fcf_per_share > 0:
                        try:
                            fcf_future_price = stock_analysis.calculate_fcf_future_price(
                                ttm_fcf_per_share, fcf_growth_rate, future_fcf_yield)
                        except Exception as e:
                            print(f"FCF-based projection not available: {str(e)}")
                    else:
                        print("Note: FCF-based projection omitted due to negative FCF per share")
                    
                    # Calculate advanced model projection
                    try:
                        confidence_range = stock_analysis.calculate_advanced_price_projection(
                            ticker, stock_analysis.data, ttm_eps, eps_growth_rate, beta
                        )
                    except Exception as e:
                        print(f"Advanced model projection not available: {str(e)}")
                    
                    # Calculate weighted average using available models
                    available_models = []
                    weights = []
                    
                    if pe_future_price is not None and pe_future_price > 0:
                        available_models.append(pe_future_price)
                        weights.append(0.4)
                    
                    if fcf_future_price is not None and fcf_future_price > 0:
                        available_models.append(fcf_future_price)
                        weights.append(0.3)
                    
                    if confidence_range is not None:
                        available_models.append(confidence_range['base_case'])
                        weights.append(0.3)
                    
                    if available_models:
                        # Normalize weights
                        weights = [w/sum(weights) for w in weights]
                        weighted_future_price = sum(p * w for p, w in zip(available_models, weights))
                        
                        # Print available projections
                        print("\nAvailable Price Projections:")
                        if pe_future_price is not None and pe_future_price > 0:
                            print(f"PE-Based Price Target: ${pe_future_price:.2f}")
                        if fcf_future_price is not None and fcf_future_price > 0:
                            print(f"FCF-Based Price Target: ${fcf_future_price:.2f}")
                        if confidence_range is not None:
                            print(f"Advanced Model Target: ${confidence_range['base_case']:.2f}")
                        
                        print(f"\nWeighted Average Target: ${weighted_future_price:.2f}")
                        print(f"Expected CAGR: {stock_analysis.calculate_cagr(recent_close_price, weighted_future_price):.2%}")
                        
                        print("\nGenerating price projections chart...")
                        stock_analysis.plot_advanced_projections(
                            ticker, 
                            stock_analysis.data, 
                            confidence_range,
                            recent_close_price,
                            pe_future_price,
                            fcf_future_price,
                            required_rate_of_return
                        )
                    else:
                        print("\nWARNING: No valid price projections available")
                        print("This may be due to negative metrics or calculation errors")
                        print("Consider using alternative valuation methods or waiting for improved financial metrics")
                    
                except Exception as e:
                    print(f"Error generating price projections: {str(e)}")
                    print("Note: Price projections may be unreliable for companies with negative metrics")
                
                # Store results
                recent_close_prices.append(recent_close_price)
                ttm_eps_list.append(ttm_eps)
                betas.append(beta)
                pe_future_prices.append(pe_future_price)
                weighted_future_prices.append(weighted_future_price)
                eps_growth_rates.append(eps_growth_rate)
                future_pe_ratios.append(future_pe_ratio)
                
            except ValueError as e:
                print(f"\nError processing {ticker}: {e}")
                print("Note: Some metrics may be unavailable or unreliable")
                continue

        print("\n" + "-"*80)
        print("PORTFOLIO ANALYSIS")
        print("-"*80)
        
        print("\nGenerating correlation matrix...")
        stock_analysis.plot_correlation_matrix()
        
        print("\nGenerating daily returns box plot...")
        stock_analysis.plot_daily_returns_boxplot()
        
        print(f"\nRunning Monte Carlo simulation...")
        print(f"Number of iterations: {stock_analysis.sim_runs:,}")
        sim_results, volatility_runs, expected_portfolio_returns_runs, sharpe_ratio_runs = (
            stock_analysis.monte_carlo_simulation())
        stock_analysis.plot_simulation_results(
            volatility_runs, expected_portfolio_returns_runs, sharpe_ratio_runs)

        print("\nOptimizing current portfolio...")
        optimal_weights, expected_return, volatility = stock_analysis.optimize_portfolio()
        
        print("\nOptimal Portfolio Allocation:")
        print("-" * 50)
        print(f"{'Ticker':<10} {'Weight':>10}")
        print("-" * 50)
        
        # Print weights in chunks
        for i, (ticker, weight) in enumerate(zip(tickers, optimal_weights)):
            print(f"{ticker:<10} {weight:>10.2%}")
            if (i + 1) % 10 == 0 and i + 1 < len(tickers):
                print("-" * 50)
        
        print("-" * 50)
        print(f"{'Total':<10} {'100.00%':>10}")
        
        print("\nPortfolio Metrics:")
        print("-" * 30)
        print(f"Expected Annual Return: {expected_return:.2%}")
        print(f"Expected Annual Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {(expected_return - stock_analysis.risk_free_rate) / volatility:.2f}")

        print("\nGenerating portfolio performance comparison...")
        optimal_portfolio_df = stock_analysis.asset_allocation(
            stock_analysis.adjusted_closing_prices_df, optimal_weights)
        equal_weights = stock_analysis.generate_equal_weights(len(tickers))
        equal_weighted_portfolio_df = stock_analysis.asset_allocation(
            stock_analysis.adjusted_closing_prices_df, equal_weights)
        stock_analysis.plot_portfolio_vs_benchmarks(
            optimal_portfolio_df, equal_weighted_portfolio_df)

        print("\n" + "-"*80)
        print("FUTURE PORTFOLIO OPTIMIZATION")
        print("-"*80)

        print("\nCalculating future portfolio metrics...")
        pe_returns, fcf_returns, advanced_returns, weighted_returns, historical_volatility, growth_adjusted_sharpe = (
            stock_analysis.calculate_future_portfolio_metrics(
                tickers, recent_close_prices, weighted_future_prices, eps_growth_rates))
        
        print("\nOptimizing future portfolio weights...")
        future_weights, future_return, future_risk, future_sharpe = stock_analysis.optimize_future_portfolio(
            tickers, pe_returns, fcf_returns, advanced_returns, weighted_returns, historical_volatility)
        
        stock_analysis.display_future_portfolio_analysis(
            tickers, optimal_weights, future_weights, pe_returns, fcf_returns, advanced_returns, 
            weighted_returns, historical_volatility)

    except ValueError as e:
        print(f"\nError: {e}")
        
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80) 