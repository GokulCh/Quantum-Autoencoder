import numpy as np
import matplotlib.pyplot as plt
import os
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity
from src.logging import ExperimentLogger

def run_experiment():
    """Runs Experiment 2: Entanglement Study with enhanced logging."""

    experiment_name = "exp2_entanglement_study"
    logger = ExperimentLogger(experiment_name)
    print(f"Running {experiment_name}...")
    
    n_qubits = 4
    k_qubits = 2
    n_train = 20
    n_test = 10
    
    state_types = ['product', 'w', 'ghz']
    ansatz_name = 'RealAmplitudes'
    
    logger.log_config({
        "n_qubits": n_qubits,
        "k_qubits": k_qubits,
        "n_train": n_train,
        "n_test": n_test,
        "state_types": state_types,
        "ansatz": ansatz_name
    })
    
    results = {}
    
    for state_type in state_types:
        print(f"Testing on {state_type} states...")
        
        try:
            # Generate Data
            train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
            test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
            
            ansatz = get_ansatz(ansatz_name, n_qubits)
            qae = QAECircuit(n_qubits, k_qubits, ansatz)
            trainer = QAETrainer(qae)
            
            # Train
            result = trainer.train(train_states)
            opt_params = result.x
            loss = result.fun
            training_time = getattr(trainer, 'training_time', 0.0)
            
            # Evaluate
            fidelities = []
            encoder = qae.get_encoder().assign_parameters(opt_params)
            decoder = qae.get_decoder().assign_parameters(opt_params)
            
            for state in test_states:
                fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
                fidelities.append(fid)
                
            avg_fid = np.mean(fidelities)
            std_fid = np.std(fidelities)
            print(f"Average Test Fidelity ({state_type}): {avg_fid:.4f} +/- {std_fid:.4f}")
            
            run_data = {
                "final_loss": loss,
                "avg_fidelity": avg_fid,
                "std_fidelity": std_fid,
                "fidelities": fidelities,
                "loss_history": getattr(trainer, 'loss_history', []),
                "training_time": training_time
            }
            logger.log_result(state_type, run_data)
            results[state_type] = run_data
            
        except Exception as e:
            print(f"Failed to run {state_type}: {e}")
            logger.log_result(state_type, {"error": str(e)})

    logger.save()
        
    # Plot results
    names = []
    means = []
    stds = []
    
    for name, data in results.items():
        if data and "avg_fidelity" in data:
            names.append(name)
            means.append(data["avg_fidelity"])
            stds.append(data["std_fidelity"])
    
    plt.figure(figsize=(8, 6))
    plt.bar(names, means, yerr=stds, capsize=5, color='orange', alpha=0.8)
    plt.ylabel('Average Reconstruction Fidelity')
    plt.title(f'Entanglement Study (n={n_qubits}, k={k_qubits})')
    plt.ylim(0, 1.1)
    
    os.makedirs('results/plots', exist_ok=True)
    filename = f'results/plots/{experiment_name}.png'
    plt.savefig(filename)
    print(f"Results saved to {filename}")
    plt.close()

if __name__ == "__main__":
    run_experiment()
