import pandas as pd
import numpy as np

def load_and_preprocess_orders(csv_path):
    """
    Loads raw orders from a CSV containing 'Date' and 'Quantity',
    and aggregates them into daily and monthly time series.
    """
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Ensure we have the correct columns
        if 'Date' not in df.columns or 'Quantity' not in df.columns:
            raise ValueError("CSV must contain 'Date' and 'Quantity' columns.")
            
        # Convert Date column to datetime objects
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Set Date as the index for time-series resampling
        df.set_index('Date', inplace=True)
        
        # Resample to Daily sum
        daily_sales = df['Quantity'].resample('D').sum().fillna(0)
        
        # Resample to Monthly sum
        monthly_sales = df['Quantity'].resample('ME').sum().fillna(0)
        
        return daily_sales.values, monthly_sales.values
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None

def generate_sample_orders_csv(filename='orders.csv', days=150):
    """
    Generates a sample orders.csv with random quantities across dates.
    """
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days*24, freq='h') # hourly orders
    # Simulate some hours having no orders, some having 1-5
    quantities = np.where(np.random.random(len(dates)) > 0.5, np.random.randint(1, 5, size=len(dates)), 0)
    
    # Filter out 0 quantities to make it look like a real order export
    valid_orders = quantities > 0
    df = pd.DataFrame({'Date': dates[valid_orders], 'Quantity': quantities[valid_orders]})
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {len(df)} sample orders spanning {days} days.")

if __name__ == "__main__":
    generate_sample_orders_csv()
    daily, monthly = load_and_preprocess_orders('orders.csv')
    print("\nData Processing Test:")
    print(f"Daily Array Shape: {daily.shape}")
    print(f"Monthly Array Shape: {monthly.shape}")
    print(f"First 5 Daily Sales: {daily[:5]}")
    print(f"Monthly Sales: {monthly}")
