# Seven-state HRS resilience PINN

This repository contains the code used to train the gradient-calibrated PINN,
predict the probabilities of states S0-S6, and calculate the resilience indices
of a hydrogen refueling station.

The aggregated transition-rate parameters were estimated from anonymized fault
records of an operating hydrogen refueling station in China. The raw records are
not included because they contain confidential operational information.

## Environment

- Ubuntu 20.04
- Python 3.8
- TensorFlow 2.9.0
- CUDA 11.2
- NVIDIA A100 PCIe GPU with 40 GB memory

Install the Python packages with:

```bash
pip install -r requirements.txt
```

Run the default case with:

```bash
python pinn_7state_resilience.py
```

The default calculation uses 80,000 training iterations, 8,000 nominal residual
points, and an assessment period of 150,000 h. The main settings can be changed
through the environment variables listed at the beginning of the Python script.

## Outputs

The script writes the following files to the `results` directory:

- `state_probabilities.csv`
- `resilience_indices.csv`
- `resilience_summary.csv`
- `main_pinn_training_loss.csv`
- `robustness_pinn_training_loss.csv`
