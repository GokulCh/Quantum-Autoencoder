import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src import get_state_circuit, get_ansatz, QAECircuit, QAETrainer, compute_reconstruction_fidelity, ClassicalBaselines

def run_experiment():
    """
    Runs Experiment 5: Baseline Comparison.
    
    Compares the Quantum Autoencoder against classical baselines (PCA, Random Unitary).
    """
    print("Running Experiment 5: Baseline Comparison")
    
    n_qubits = 4
    k_qubits = 2
    n_train = 50
    n_test = 20
    
    # Use GHZ states as they are good for QAE
    state_type = 'ghz'
    
    print(f"Generating {state_type} data...")
    train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
    test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
    
    # 1. QAE
    print("Running QAE...")
    ansatz = get_ansatz('RealAmplitudes', n_qubits)
    qae = QAECircuit(n_qubits, k_qubits, ansatz)
    trainer = QAETrainer(qae, maxiter=200)
    result = trainer.train(train_states)
    
    encoder = qae.get_encoder().assign_parameters(result.x)
    decoder = qae.get_decoder().assign_parameters(result.x)
    
    qae_fidelities = []
    for state in test_states:
        fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
        qae_fidelities.append(fid)
    avg_qae = np.mean(qae_fidelities)
    print(f"QAE Fidelity: {avg_qae}")
    
    # 2. PCA
    print("Running PCA...")
    # PCA needs statevectors
    avg_pca = ClassicalBaselines.run_pca(test_states, k_qubits)
    print(f"PCA Fidelity: {avg_pca}")
    
    # 3. Random Unitary
    print("Running Random Unitary...")
    avg_random = ClassicalBaselines.run_random_unitary(test_states, n_qubits, k_qubits)
    print(f"Random Unitary Fidelity: {avg_random}")
    
    # Plot
    labels = ['QAE', 'PCA', 'Random']
    values = [avg_qae, avg_pca, avg_random]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['blue', 'green', 'gray'])
    plt.ylabel('Average Fidelity')
    plt.title(f'Baseline Comparison ({state_type}, n={n_qubits}->{k_qubits})')
    plt.ylim(0, 1.0)
    plt.savefig('results/exp5_baselines.png')
    print("Results saved to results/exp5_baselines.png")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    run_experiment()
