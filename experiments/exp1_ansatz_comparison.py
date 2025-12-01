import numpy as np
import matplotlib.pyplot as plt
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity

def run_experiment():
    """Runs Experiment 1: Ansatz Comparison."""

    print("Running Experiment 1: Ansatz Comparison")
    
    configurations = [
        (4, 2),
        (6, 3),
        (6, 2)
    ]
    
    n_train = 20
    n_test = 10
    
    ansatzes = ['RealAmplitudes', 'EfficientSU2', 'HardwareEfficient']
    
    for n_qubits, k_qubits in configurations:
        print(f"\n--- Running for n={n_qubits}, k={k_qubits} ---")
        
        # Generate Data (Product States)
        print("Generating data...")
        train_states = [get_state_circuit('product', n_qubits) for _ in range(n_train)]
        test_states = [get_state_circuit('product', n_qubits) for _ in range(n_test)]
        
        results = {}
        
        for name in ansatzes:
            print(f"Training with {name}...")
            try:
                ansatz = get_ansatz(name, n_qubits)
                qae = QAECircuit(n_qubits, k_qubits, ansatz)
                trainer = QAETrainer(qae)
                
                # Train
                result = trainer.train(train_states)
                opt_params = result.x
                print(f"Final Loss ({name}): {result.fun}")
                
                # Evaluate
                fidelities = []
                encoder = qae.get_encoder().assign_parameters(opt_params)
                decoder = qae.get_decoder().assign_parameters(opt_params)
                
                for state in test_states:
                    fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
                    fidelities.append(fid)
                    
                avg_fid = np.mean(fidelities)
                print(f"Average Test Fidelity ({name}): {avg_fid}")
                results[name] = avg_fid
            except Exception as e:
                print(f"Failed to run {name}: {e}")
                results[name] = 0.0
            
        # Plot results for this configuration
        names = list(results.keys())
        values = list(results.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, values)
        plt.ylabel('Average Reconstruction Fidelity')
        plt.title(f'Ansatz Comparison (n={n_qubits}, k={k_qubits})')
        plt.ylim(0, 1.0)
        
        # Ensure results directory exists
        os.makedirs('results/plots', exist_ok=True)
        filename = f'results/plots/exp1_ansatz_comparison_n{n_qubits}_k{k_qubits}.png'
        plt.savefig(filename)
        print(f"Results saved to {filename}")
        plt.close()

if __name__ == "__main__":
    run_experiment()
