# CNN-LSTM Driver Drowsiness Detection

This project uses the same style as the HAR/SLR hybrid architecture projects, but for driver drowsiness detection.

## Architecture

- CNN encoder per frame (`TimeDistributed` Conv2D blocks)
- LSTM layer 1 for temporal modeling
- LSTM layer 2 for sequence summarization
-  Dense classifier with Softmax output

<img width="697" height="638" alt="image" src="https://github.com/user-attachments/assets/cf1dde72-d355-4cad-a9af-99f98a560da9" />

## Input

- Default expected shape: `(seq_len, img_size, img_size, channels)`
- Current config: `(16, 48, 48, 1)`
- Classes:
  - `0`: Alert
  - `1`: Drowsy

## Data Options

### 1) Use your own data (recommended)
Put an NPZ file here:

- `data/drowsiness_sequences.npz`

The file must contain:

- `X`: shape `(samples, 16, 48, 48, 1)`
- `y`: shape `(samples,)`, integer labels in `{0,1}`

### 2) No dataset provided
If the NPZ file does not exist, the script automatically generates a synthetic balanced dataset so you can start immediately.

## Run

```bash
cd D:\Desktop\CNN-LSTM-Driver-Drowsiness
pip install -r requirements.txt
python main.py
```

<img width="944" height="443" alt="image" src="https://github.com/user-attachments/assets/343b77a1-f96d-4ce4-9e17-f388eceef087" />


## Outputs

All outputs are saved to `artifacts/`:

- `best_cnn_lstm_driver_drowsiness.keras`
- `final_cnn_lstm_driver_drowsiness.keras`
- `history.json`
- `results.json`
