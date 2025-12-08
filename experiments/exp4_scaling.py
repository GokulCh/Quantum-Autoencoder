import numpy as np
import os
import matplotlib.pyplot as plt
from qiskit import transpile
from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity
from src.logging import ExperimentLogger

def run_experiment():
    """Runs Experiment 4: Scaling with improved metrics."""
    experiment_name = "exp4_scaling"
    logger = ExperimentLogger(experiment_name)
    print(f"Running {experiment_name}...")
    
    # Scaling configs (n, k)
    configs = [
        (4, 2),
        (6, 3),
        (8, 4)
    ]
    
    n_train = 20
    n_test = 10
    state_type = 'ghz'
    ansatz_name = 'RealAmplitudes'
    
    logger.log_config({
        "configs": configs,
        "n_train": n_train,
        "n_test": n_test,
        "state_type": state_type,
        "ansatz": ansatz_name
    })
    
    results = {}
    
    for n_qubits, k_qubits in configs:
        config_key = f"n{n_qubits}_k{k_qubits}"
        print(f"\n--- Running Scaling Analysis for n={n_qubits}, k={k_qubits} ---")
        
        try:
            print(f"Generating {state_type} data...")
            train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
            test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
            
            ansatz = get_ansatz(ansatz_name, n_qubits)
            qae = QAECircuit(n_qubits, k_qubits, ansatz)
            
            # Check Depth
            basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
            t_ansatz = transpile(ansatz, basis_gates=basis_gates)
            depth = t_ansatz.depth()
            print(f"Ansatz Depth: {depth}")
            
            trainer = QAETrainer(qae)
            
            # Train
            result = trainer.train(train_states)
            opt_params = result.x
            training_time = getattr(trainer, 'training_time', 0.0)
            
            print(f"Final Loss: {result.fun:.4f}, Time: {training_time:.2f}s")
            
            # Evaluate
            fidelities = []
            encoder = qae.get_encoder().assign_parameters(opt_params)
            decoder = qae.get_decoder().assign_parameters(opt_params)
            
            for state in test_states:
                fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
                fidelities.append(fid)
                
            avg_fid = np.mean(fidelities)
            std_fid = np.std(fidelities)
            print(f"Average Test Fidelity: {avg_fid:.4f}")
            
            run_data = {
                "n_qubits": n_qubits,
                "k_qubits": k_qubits,
                "depth": depth,
                "training_time": training_time,
                "avg_fidelity": avg_fid,
                "std_fidelity": std_fid,
                "final_loss": result.fun
            }
            logger.log_result(config_key, run_data)
            results[config_key] = run_data
            
        except Exception as e:
            print(f"Failed for {config_key}: {e}")
            logger.log_result(config_key, {"error": str(e)})
            
    logger.save()
    
    # Plotting
    ns = []
    times = []
    fids = []
    depths = []
    
    for key, data in results.items():
        if data and "avg_fidelity" in data:
            ns.append(data["n_qubits"])
            times.append(data["training_time"])
            fids.append(data["avg_fidelity"])
            depths.append(data["depth"])
            
    if ns:
        # Plot 1: Fidelity vs n
        plt.figure(figsize=(8, 6))
        plt.plot(ns, fids, 'o-', label='Fidelity', color='blue')
        plt.xlabel('Number of Qubits (n)')
        plt.ylabel('Average Fidelity')
        plt.title('Scaling: Fidelity vs System Size')
        plt.ylim(0, 1.1)
        plt.grid(True)
        filename_fid = f'results/plots/{experiment_name}_fidelity.png'
        plt.savefig(filename_fid)
        plt.close()
        
        # Plot 2: Time vs n
        plt.figure(figsize=(8, 6))
        plt.plot(ns, times, 's-', label='Time', color='red')
        plt.xlabel('Number of Qubits (n)')
        plt.ylabel('Training Time (s)')
        plt.title('Scaling: Training Time vs System Size')
        plt.grid(True)
        filename_time = f'results/plots/{experiment_name}_time.png'
        plt.savefig(filename_time)
        plt.close()
        print(f"Plots saved to {filename_fid} and {filename_time}")

if __name__ == "__main__":
    os.makedirs('results/plots', exist_ok=True)
    run_experiment()
