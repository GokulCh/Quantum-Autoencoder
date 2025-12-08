import numpy as np
import matplotlib.pyplot as plt
import os
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile

try:
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeManilaV2
    except ImportError:
        print("Warning: FakeManilaV2 not found. Using ideal simulator.")
        FakeManilaV2 = None

from src.state_prep import get_state_circuit
from src.ansatze import get_ansatz
from src.qae_circuit import QAECircuit
from src.training import QAETrainer
from src.metrics import compute_reconstruction_fidelity
from src.logging import ExperimentLogger

def run_experiment():
    experiment_name = "exp3_noise_robustness"
    logger = ExperimentLogger(experiment_name)
    print(f"Running {experiment_name}...")
    
    # Setup Noise Model
    backend = None
    noise_model_name = "Ideal"
    
    if FakeManilaV2 is not None:
        backend = FakeManilaV2()
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        basis_gates = backend.configuration().basis_gates
        noise_model_name = backend.name
        print(f"Using noise model from {backend.name}")
    else:
        print("FakeManilaV2 not found. Using generic noise model (Depolarizing error).")
        from qiskit_aer.noise import depolarizing_error
        
        noise_model = NoiseModel()
        error_1 = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'h', 'ry'])
        error_2 = depolarizing_error(0.02, 2)
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'ecr'])
        
        coupling_map = None
        basis_gates = ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'h', 'ry', 'cx', 'cz', 'ecr']
        noise_model_name = "Generic Depolarizing"

    n_qubits = 4
    k_qubits = 2
    n_train = 10
    n_test = 5
    
    logger.log_config({
        "n_qubits": n_qubits,
        "k_qubits": k_qubits,
        "n_train": n_train,
        "n_test": n_test,
        "noise_model": noise_model_name,
        "ansatz": "RealAmplitudes"
    })
    
    train_states = [get_state_circuit('product', n_qubits) for _ in range(n_train)]
    test_states = [get_state_circuit('product', n_qubits) for _ in range(n_test)]
    
    ansatz = get_ansatz('RealAmplitudes', n_qubits)
    qae = QAECircuit(n_qubits, k_qubits, ansatz)
    trainer = QAETrainer(qae)
    
    # Train (Ideal)
    print("Training on ideal simulator...")
    result = trainer.train(train_states)
    opt_params = result.x
    training_time = getattr(trainer, 'training_time', 0.0)
    
    # Evaluate Ideal
    print("Evaluating on Ideal...")
    encoder = qae.get_encoder().assign_parameters(opt_params)
    decoder = qae.get_decoder().assign_parameters(opt_params)
    
    fidelities_ideal = []
    for state in test_states:
        fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
        fidelities_ideal.append(fid)
    avg_ideal = np.mean(fidelities_ideal)
    std_ideal = np.std(fidelities_ideal)
    print(f"Ideal Fidelity: {avg_ideal:.4f}")
    
    logger.log_result("ideal", {
        "avg_fidelity": avg_ideal,
        "std_fidelity": std_ideal,
        "fidelities": fidelities_ideal,
        "loss_history": getattr(trainer, 'loss_history', []),
        "training_time": training_time
    })
    
    # Evaluate Noisy
    print("Evaluating on Noisy Simulator...")
    noisy_sim = AerSimulator(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
    
    fidelities_noisy = []
    
    for state in test_states:
        inv_state = state.inverse()
        fid_circ = state.compose(encoder).compose(decoder).compose(inv_state)
        fid_circ.measure_all()
        
        t_circ = transpile(fid_circ, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates)
        
        job = noisy_sim.run(t_circ, shots=1024)
        counts = job.result().get_counts()
        
        zero_count = counts.get('0'*n_qubits, 0)
        fid = zero_count / 1024.0
        fidelities_noisy.append(fid)
        
    avg_noisy = np.mean(fidelities_noisy)
    std_noisy = np.std(fidelities_noisy)
    print(f"Noisy Fidelity: {avg_noisy:.4f}")
    
    logger.log_result("noisy", {
        "avg_fidelity": avg_noisy,
        "std_fidelity": std_noisy,
        "fidelities": fidelities_noisy
    })
    
    logger.save()
    
    # Plot
    plt.figure(figsize=(8, 6))
    labels = ['Ideal', 'Noisy']
    means = [avg_ideal, avg_noisy]
    stds = [std_ideal, std_noisy]
    
    plt.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'red'], alpha=0.8)
    plt.ylabel('Average Fidelity')
    plt.title(f'Noise Robustness ({noise_model_name})')
    plt.ylim(0, 1.1)
    
    os.makedirs('results/plots', exist_ok=True)
    filename = f'results/plots/{experiment_name}.png'
    plt.savefig(filename)
    print(f"Results saved to {filename}")
    plt.close()

if __name__ == "__main__":
    run_experiment()
