# Surf-Height-Prediction
Use an LSTM to predict next step significant wave height at Mooloolaba Beach.

## Prep
You will need to download the data file from: https://www.kaggle.com/jolasa/waves-measuring-buoys-data-mooloolaba/download

## Build/Run
 docker build -t surf-pred .
 docker run surf-pred

## Results
| forecast_horizon | dataset | rMSE (cm) |
|---|---|---|
| 1 | Training | 7 |
| 1 | Testing | 8 |
| 24 | Training | 22 |
| 24 | Testing | 28 |