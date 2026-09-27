import pandas as pd

df = pd.read_parquet("data/training/model_lstm_cnn/test_predictions.parquet")

print(df.head())