# Quantitative_Finance
This project builds a technical-indicator dataset for equities and trains baseline ML models to classify forward returns (positive, negative, or neutral) for a chosen horizon.

The workflow:
1. Scrape or load equity symbols.
2. Download historical prices for each symbol.
3. Compute technical indicators and labels.
4. Merge per-symbol CSVs into a single dataset.
5. Train and evaluate baseline classification models.

## Requirements
- Python 3.x
- Install dependencies from requirements.txt

### Poetry (recommended)
- Install Poetry and run: poetry install
- Run scripts with: poetry run python <script>

## Project Files
- Technical_Indicators.py: Downloads data and computes indicators/labels.
- Merge_All_CSV.py: Merges multiple CSVs into a single dataset.
- Quantitative_finance.py: Trains and compares baseline models.
- EquityCollection_UsingR.R: (Optional) Scrape equity symbols from ADVFN.

## Setup
1. Create a folder named data at the project root.
2. Place StockSymbols_All.csv in data/.
3. Run Technical_Indicators.py.
4. The script creates data/equities_IT_TS/preprocessed and data/equities_IT_TS/processed.
5. Copy Merge_All_CSV.py into each of those folders and run it to create main.csv.
6. Use the merged CSV as your dataset for modeling.
7. Run Quantitative_finance.py and adjust feature selection as needed.

### Example
- python Quantitative_finance.py --data main.csv --label "15d Label" --test-size 0.25
- python Quantitative_finance.py --data main.csv --label "15d Label" --time-split --date-col Date --test-size 0.25 --log-level INFO

## Notes
- Data collection uses Yahoo as the source via pandas_datareader; some tickers may fail or be delisted.
- The date range is set in Technical_Indicators.py and can be adjusted.
- The label uses a forward return threshold and a fixed horizon (currently 15 days); adjust the threshold/horizon for your use case.
- Use --time-split to avoid lookahead bias; requires a Date column in the dataset.

## Poster
See QuantitativeFinance_Poster.pdf for a performance comparison across models.
