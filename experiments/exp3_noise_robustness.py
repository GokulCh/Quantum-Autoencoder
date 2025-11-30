import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile

# Try importing FakeManilaV2 from various locations to be robust
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

def run_experiment():
    print("Running Experiment 3: Noise Robustness")
    
    # Setup Noise Model
    backend = None
    if FakeManilaV2 is not None:
        backend = FakeManilaV2()
        noise_model = NoiseModel.from_backend(backend)
        coupling_map = backend.configuration().coupling_map
        basis_gates = backend.configuration().basis_gates
        print(f"Using noise model from {backend.name}")
    else:
        print("FakeManilaV2 not found. Using generic noise model (Depolarizing error).")
        from qiskit_aer.noise import depolarizing_error
        
        noise_model = NoiseModel()
        # Add 1% depolarizing error to all single qubit gates
        error_1 = depolarizing_error(0.01, 1)
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'h', 'ry'])
        
        # Add 2% depolarizing error to all 2-qubit gates
        error_2 = depolarizing_error(0.02, 2)
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'ecr'])
        
        coupling_map = None # All-to-all connectivity for generic
        basis_gates = ['u1', 'u2', 'u3', 'rz', 'sx', 'x', 'h', 'ry', 'cx', 'cz', 'ecr']

    
    n_qubits = 4
    k_qubits = 2
    n_train = 10
    n_test = 5 # Small number for speed
    
    train_states = [get_state_circuit('product', n_qubits) for _ in range(n_train)]
    test_states = [get_state_circuit('product', n_qubits) for _ in range(n_test)]
    
    ansatz = get_ansatz('RealAmplitudes', n_qubits)
    qae = QAECircuit(n_qubits, k_qubits, ansatz)
    trainer = QAETrainer(qae)
    
    print("Training on ideal simulator...")
    result = trainer.train(train_states)
    opt_params = result.x
    
    # Evaluate Ideal
    print("Evaluating on Ideal...")
    encoder = qae.get_encoder().assign_parameters(opt_params)
    decoder = qae.get_decoder().assign_parameters(opt_params)
    
    fidelities_ideal = []
    for state in test_states:
        fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
        fidelities_ideal.append(fid)
    avg_ideal = np.mean(fidelities_ideal)
    print(f"Ideal Fidelity: {avg_ideal}")
    
    # Evaluate Noisy
    print("Evaluating on Noisy Simulator...")
    noisy_sim = AerSimulator(noise_model=noise_model, coupling_map=coupling_map, basis_gates=basis_gates)
    
    fidelities_noisy = []
    
    for state in test_states:
        # Full circuit: State -> Encoder -> Decoder -> Inverse State -> Measure
        # If perfect, we should measure |0...0>
        
        inv_state = state.inverse()
        fid_circ = state.compose(encoder).compose(decoder).compose(inv_state)
        fid_circ.measure_all()
        
        # Transpile for backend
        t_circ = transpile(fid_circ, backend=backend, coupling_map=coupling_map, basis_gates=basis_gates)
        
        # Run
        job = noisy_sim.run(t_circ, shots=1024)
        counts = job.result().get_counts()
        
        # Probability of '0000'
        zero_count = counts.get('0'*n_qubits, 0)
        fid = zero_count / 1024.0
        fidelities_noisy.append(fid)
        
    avg_noisy = np.mean(fidelities_noisy)
    print(f"Noisy Fidelity: {avg_noisy}")
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.bar(['Ideal', 'Noisy'], [avg_ideal, avg_noisy], color=['blue', 'red'])
    plt.ylabel('Average Fidelity')
    plt.title('Noise Robustness (FakeManilaV2)')
    plt.ylim(0, 1.0)
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/exp3_noise.png')
    print("Results saved to results/exp3_noise.png")

if __name__ == "__main__":
    run_experiment()
