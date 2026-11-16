# EV Battery State-of-Charge (SOC) Estimation (BMW i3)

Estimating the state of charge of a BMW i3 battery pack from four easily measured signals, using a CNN-GRU deep learning model in PyTorch. This work extends an earlier Random Forest replication of Sulaiman & Mustaffa (2024) by adding temporal context.

## Problem

SOC cannot be measured directly with a sensor; it has to be inferred. Given pack voltage, pack current, battery temperature and ambient temperature recorded during real driving, the model estimates the manufacturer's SOC on **completely unseen trips**.

## Data

- BMW i3 real driving-cycle data: *Battery and Heating Data in Real Driving Cycles* (IEEE DataPort, doi: 10.21227/6jr9-5235).
- 60 training trips and 10 independent test trips (trips 61-70).
- The CSV files are **not included** in this repo. Download them and place them in the same folder as the script, named:
  - `Final_60trips_Training_BMW_Data(2).csv`
  - `Final_10trips_Testing_BMW_Data(1).csv`

## Method

- Trip boundaries are detected where the time column resets (60 train / 10 test trips).
- Inputs: voltage, current, battery temperature, ambient temperature. Time is not used as a feature.
- Last 6 training trips (55-60) are held out for validation; features are standardised using the fitting trips only.
- Sequences of 20 consecutive observations are built within each trip (never crossing a trip boundary).
- Model: 2 x Conv1D (32 channels, kernel 3, BatchNorm, GELU) -> 1-layer GRU (48 units) -> dropout + dense head.
- Training: AdamW, learning rate 0.001, MSE loss, batch size 1024, 4 epochs, seed 42. The best validation checkpoint is used for testing.

## Results (independent test trips 61-70)

| Metric | CNN-GRU |
|---|---|
| RMSE | 5.04 % |
| MAE | 3.80 % |
| Max absolute error | 27.91 % |
| Error std dev | 4.86 % |
| R2 | 0.907 |

For reference, my earlier Random Forest replication (25 trees, same test trips) averaged 5.43 % RMSE and 4.02 % MAE.

## Limitations

- Only 4 epochs were run, so the learning curve does not show full convergence.
- The model needs 20 observations of history, so the first 19 observations of each trip are not predicted.
- The largest errors (about 28 %) occur on a low-SOC trip that starts below the lowest SOC seen in training.
- The target is the manufacturer-reported SOC, not an independent ground truth.

## Run it

```
pip install -r requirements.txt
python Assessment10_CNN_GRU_Source_Code.py
```

Outputs (metrics, predictions, plots, saved model) are written to `assessment10_cnn_gru_outputs/`.

## Reference

M. H. Sulaiman and Z. Mustaffa, "State of charge estimation for electric vehicles using random forest," *Green Energy and Intelligent Transportation*, vol. 3, 100177, 2024.
