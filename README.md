# Quantum Autoencoder for Quantum Data Compression

This project implements a Quantum Autoencoder (QAE) to compress quantum states into a lower-dimensional latent space and reconstruct them with high fidelity. It explores various ansätze, entanglement levels, and noise robustness.

## Project Structure

- `src/`: Core implementation of the QAE, including circuit construction, training, and metrics.
- `experiments/`: Scripts to run specific experiments (Ansatz comparison, Entanglement study, Noise robustness, Baselines).
- `results/`: Directory where experiment plots and logs are saved.

## Installation

1. **Clone the repository** (if applicable).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is organized into separate experiment scripts. Run them from the root directory using `python`.

### Experiment 1: Ansatz Comparison
Compares different variational circuits (RealAmplitudes, EfficientSU2, HardwareEfficient) on product states.
```bash
python experiments/exp1_ansatz_comparison.py
```
**Output**: `results/exp1_ansatz_comparison_*.png`

### Experiment 2: Entanglement Study
Evaluates compression performance on Product, W, and GHZ states.
```bash
python experiments/exp2_entanglement_study.py
```
**Output**: `results/exp2_entanglement_study.png`

### Experiment 3: Noise Robustness
Tests the QAE under realistic noise models (simulated IBM hardware).
```bash
python experiments/exp3_noise_robustness.py
```
**Output**: `results/exp3_noise.png`

### Experiment 5: Baseline Comparison
Compares QAE against classical PCA and Random Unitary baselines.
```bash
python experiments/exp5_baselines.py
```
**Output**: `results/exp5_baselines.png`

## Key Constraints & Configuration

- **Max Iterations**: Training is limited to 200 iterations (`maxiter=200`).
- **Circuit Depth**: Checked to be < 100 gates for the chosen ansatz.
- **Qubit Usage**: Designed to fit within 10 qubits (typically n=4 or 6).

## Authors
Gokul Chaluvadi, Ayush Kumar Lnu, Denique Black
Virginia Commonwealth University
