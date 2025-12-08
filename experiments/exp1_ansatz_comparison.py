import numpy as np
import matplotlib.pyplot as plt
import os
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity
from src.logging import ExperimentLogger
from src.utils import set_seed, get_rng

def run_experiment(seed: int = None):
    """Runs Experiment 1: Ansatz Comparison with enhanced logging."""

    set_seed(seed)
    rng = get_rng(seed)

    experiment_name = "exp1_ansatz_comparison"
    logger = ExperimentLogger(experiment_name)
    print(f"Running {experiment_name}...")
    
    configurations = [
        (4, 2),
        (6, 3),
        (6, 2)
    ]
    
    n_train = 20
    n_test = 10
    ansatzes = ['RealAmplitudes', 'EfficientSU2', 'HardwareEfficient']
    
    logger.log_config({
        "configurations": configurations,
        "n_train": n_train,
        "n_test": n_test,
        "ansatzes": ansatzes,
        "seed": seed
    })
    
    all_results = {} # hierarchical structure for plotting later

    for n_qubits, k_qubits in configurations:
        config_key = f"n{n_qubits}_k{k_qubits}"
        print(f"\n--- Running for {config_key} ---")
        
        all_results[config_key] = {}

        # Generate Data (Product States)
        print("Generating data...")
        train_states = [get_state_circuit('product', n_qubits, rng=rng) for _ in range(n_train)]
        test_states = [get_state_circuit('product', n_qubits, rng=rng) for _ in range(n_test)]
        
        for name in ansatzes:
            print(f"Training with {name}...")
            try:
                ansatz = get_ansatz(name, n_qubits)
                qae = QAECircuit(n_qubits, k_qubits, ansatz)
                trainer = QAETrainer(qae)
                
                # Train
                result = trainer.train(train_states)
                opt_params = result.x
                loss = result.fun
                training_time = getattr(trainer, 'training_time', 0.0)
                loss_history = getattr(trainer, 'loss_history', [])

                print(f"Final Loss ({name}): {loss:.4f}, Time: {training_time:.2f}s")
                
                # Evaluate
                fidelities = []
                encoder = qae.get_encoder().assign_parameters(opt_params)
                decoder = qae.get_decoder().assign_parameters(opt_params)
                
                for state in test_states:
                    fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
                    fidelities.append(fid)
                    
                avg_fid = np.mean(fidelities)
                std_fid = np.std(fidelities)
                print(f"Average Test Fidelity ({name}): {avg_fid:.4f} +/- {std_fid:.4f}")
                
                # Log detailed results
                run_data = {
                    "final_loss": loss,
                    "avg_fidelity": avg_fid,
                    "std_fidelity": std_fid,
                    "fidelities": fidelities,
                    "loss_history": loss_history,
                    "training_time": training_time
                }
                
                logger.log_result(f"{config_key}_{name}", run_data)
                all_results[config_key][name] = run_data

            except Exception as e:
                print(f"Failed to run {name}: {e}")
                logger.log_result(f"{config_key}_{name}", {"error": str(e)})
                all_results[config_key][name] = None
            
        # Plot results for this configuration
        plot_results(n_qubits, k_qubits, all_results[config_key], experiment_name)

    logger.save()

def plot_results(n, k, results, experiment_name):
    """Generates plots for a specific configuration."""
    names = []
    means = []
    stds = []
    
    # 1. Bar Chart with Error Bars
    for name, data in results.items():
        if data and "avg_fidelity" in data:
            names.append(name)
            means.append(data["avg_fidelity"])
            stds.append(data["std_fidelity"])
    
    if not names:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Average Reconstruction Fidelity')
    plt.title(f'Ansatz Comparison (n={n}, k={k})')
    plt.ylim(0, 1.1)
    
    os.makedirs('results/plots', exist_ok=True)
    filename_bar = f'results/plots/{experiment_name}_n{n}_k{k}_fidelity.png'
    plt.savefig(filename_bar)
    plt.close()
    
    # 2. Loss History Plot
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        if data and "loss_history" in data:
            plt.plot(data["loss_history"], label=name)
            
    plt.xlabel('Iteration')
    plt.ylabel('Loss (1 - Fidelity)')
    plt.title(f'Training Convergence (n={n}, k={k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename_loss = f'results/plots/{experiment_name}_n{n}_k{k}_loss.png'
    plt.savefig(filename_loss)
    plt.close()
    print(f"Plots saved to {filename_bar} and {filename_loss}")

if __name__ == "__main__":
    run_experiment()
