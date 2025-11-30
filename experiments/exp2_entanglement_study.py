import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity

def run_experiment():
    """
    Runs Experiment 2: Entanglement Study.
    
    Evaluates the compression performance on different types of entangled states (Product, W, GHZ).
    """
    print("Running Experiment 2: Entanglement Study")
    
    n_qubits = 4
    k_qubits = 2
    n_train = 20
    n_test = 10
    
    state_types = ['product', 'w', 'ghz']
    results = {}
    
    # Use best ansatz from Exp 1 (assuming RealAmplitudes for now as it's simpler)
    ansatz_name = 'RealAmplitudes'
    
    for state_type in state_types:
        print(f"Testing on {state_type} states...")
        
        # Generate Data
        train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
        test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
        
        ansatz = get_ansatz(ansatz_name, n_qubits)
        qae = QAECircuit(n_qubits, k_qubits, ansatz)
        trainer = QAETrainer(qae)
        
        # Train
        result = trainer.train(train_states)
        opt_params = result.x
        
        # Evaluate
        fidelities = []
        encoder = qae.get_encoder().assign_parameters(opt_params)
        decoder = qae.get_decoder().assign_parameters(opt_params)
        
        for state in test_states:
            fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
            fidelities.append(fid)
            
        avg_fid = np.mean(fidelities)
        print(f"Average Test Fidelity ({state_type}): {avg_fid}")
        results[state_type] = avg_fid
        
    # Plot results
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(names, values, color='orange')
    plt.ylabel('Average Reconstruction Fidelity')
    plt.title(f'Entanglement Study (n={n_qubits}, k={k_qubits})')
    plt.ylim(0, 1.0)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp2_entanglement_study.png')
    print("Results saved to results/exp2_entanglement_study.png")

if __name__ == "__main__":
    run_experiment()
