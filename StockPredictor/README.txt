# E-Commerce Inventory & Sales Predictor

This program is designed to forecast e-commerce stock and sales using a dual-model forecasting strategy. It separates the problem into macro-inventory planning and micro-sales execution to maximize prediction accuracy across hundreds of product categories.

## How It Works

The system utilizes a Dual-Model Strategy:
1. **Macro-Inventory Planning (ARIMA)**: Uses an Autoregressive Integrated Moving Average (ARIMA) model to forecast monthly inventory levels. It dynamically selects parameters to capture long-term trends and structural shifts in relatively stable data.
2. **Micro-Sales Execution (LSTM)**: Uses a Long Short-Term Memory (LSTM) recurrent neural network to forecast daily sales volumes. It excels at mapping complex, non-linear dependencies and high-frequency fluctuations like flash sales.
3. **Strategic Tiering**: The program categorizes products into 6 performance tiers (Class 1 to Class 6) based on forecasted sales volume, allowing businesses to prioritize high-volume revenue drivers over slow-moving stock.

## Inputs
- **Monthly Inventory Data**: Historical monthly stock levels for various product categories.
- **Daily Sales Data**: Historical daily transactional sales volumes for the same categories.

## Outputs
- **Monthly Inventory Forecasts**: Expected macro inventory levels over the upcoming months.
- **Daily Sales Forecasts**: Daily expected micro sales volumes, capturing sudden spikes and drops.
- **Category Tiering (Class 1-6)**: A classification of each product category to inform strategic supply-chain decisions.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the program from the command line:

```bash
python stockpredictor.py
```
