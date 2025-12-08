import numpy as np
import os
import matplotlib.pyplot as plt
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity
from src.classical_baselines import ClassicalBaselines
from src.logging import ExperimentLogger

def run_experiment():
    """Runs Experiment 5: Baseline Comparison with enhanced logging."""
    
    experiment_name = "exp5_baselines"
    logger = ExperimentLogger(experiment_name)
    print(f"Running {experiment_name}...")
    
    n_qubits = 4
    k_qubits = 2
    n_train = 20
    n_test = 10
    state_type = 'ghz'
    
    logger.log_config({
        "n_qubits": n_qubits,
        "k_qubits": k_qubits,
        "n_train": n_train,
        "n_test": n_test,
        "state_type": state_type
    })

    print(f"Generating {state_type} data...")
    train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
    test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
    
    results = {}
    
    # 1. QAE
    print("Running QAE...")
    try:
        ansatz = get_ansatz('RealAmplitudes', n_qubits)
        qae = QAECircuit(n_qubits, k_qubits, ansatz)
        trainer = QAETrainer(qae)
        result = trainer.train(train_states)
        
        encoder = qae.get_encoder().assign_parameters(result.x)
        decoder = qae.get_decoder().assign_parameters(result.x)
        
        qae_fidelities = []
        for state in test_states:
            fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
            qae_fidelities.append(fid)
        avg_qae = np.mean(qae_fidelities)
        std_qae = np.std(qae_fidelities)
        print(f"QAE Fidelity: {avg_qae:.4f}")
        
        results['QAE'] = {"mean": avg_qae, "std": std_qae}
        logger.log_result("QAE", {"avg_fidelity": avg_qae, "std_fidelity": std_qae, "fidelities": qae_fidelities})
        
    except Exception as e:
        print(f"QAE Failed: {e}")
        logger.log_result("QAE", {"error": str(e)})

    # 2. PCA
    print("Running PCA...")
    try:
        # Note: ClassicalBaselines.run_pca returns a single float (avg fidelity). 
        # Ideally we'd want the list, but we'll stick to the existing API or just use the float.
        # Looking at previous view_file, it returns avg.
        avg_pca = ClassicalBaselines.run_pca(test_states, k_qubits)
        print(f"PCA Fidelity: {avg_pca:.4f}")
        results['PCA'] = {"mean": avg_pca, "std": 0.0}
        logger.log_result("PCA", {"avg_fidelity": avg_pca})
    except Exception as e:
         print(f"PCA Failed: {e}")

    # 3. Random Unitary
    print("Running Random Unitary...")
    try:
        avg_random = ClassicalBaselines.run_random_unitary(test_states, n_qubits, k_qubits)
        print(f"Random Unitary Fidelity: {avg_random:.4f}")
        results['Random'] = {"mean": avg_random, "std": 0.0}
        logger.log_result("Random", {"avg_fidelity": avg_random})
    except Exception as e:
        print(f"Random Failed: {e}")
        
    logger.save()
    
    # Plotting
    labels = list(results.keys())
    values = [results[l]['mean'] for l in labels]
    # some might not have std
    errors = [results[l].get('std', 0.0) for l in labels]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, yerr=errors, capsize=5, color=['blue', 'green', 'gray'], alpha=0.8)
    plt.ylabel('Average Fidelity')
    plt.title(f'Baseline Comparison ({state_type}, n={n_qubits}->{k_qubits})')
    plt.ylim(0, 1.1)
    
    os.makedirs('results/plots', exist_ok=True)
    filename = f'results/plots/{experiment_name}.png'
    plt.savefig(filename)
    print(f"Results saved to {filename}")
    plt.close()

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    run_experiment()
