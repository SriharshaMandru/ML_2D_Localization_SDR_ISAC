# DeepAoA Localization Final ML Complete

## Raw data structure

TX side has one reference file per condition:

```text
data/raw_iq/tx/850MHz/1M/(1,3)/-60/tx_ref.csv
```

RX side has five repetitions per condition:

```text
data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep01.csv
data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep02.csv
data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep03.csv
data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep04.csv
data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep05.csv
```

RX CSV columns: `I1,Q1,I2,Q2`

TX CSV columns: `I,Q` or `I_tx,Q_tx`

## Run

```bash
pip install -r requirements.txt
python3 run_full_pipeline.py
```
