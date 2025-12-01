import numpy as np
import matplotlib.pyplot as plt
from src import get_state_circuit, get_ansatz, QAECircuit, QAETrainer, compute_reconstruction_fidelity

def run_experiment():
    """Runs Experiment 4: Scaling."""

    print("Running Experiment 4: Scaling (n=6 -> k=3)")
    
    n_qubits = 6
    k_qubits = 3
    n_train = 20
    n_test = 10
    
    state_type = 'ghz'
    
    print(f"Generating {state_type} data for n={n_qubits}...")
    train_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_train)]
    test_states = [get_state_circuit(state_type, n_qubits) for _ in range(n_test)]
    
    ansatz_name = 'RealAmplitudes'
    print(f"Training with {ansatz_name}...")
    
    ansatz = get_ansatz(ansatz_name, n_qubits)
    qae = QAECircuit(n_qubits, k_qubits, ansatz)
    

    # Check Depth
    from qiskit import transpile
    basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
    t_ansatz = transpile(ansatz, basis_gates=basis_gates)
    depth = t_ansatz.depth()
    print(f"Ansatz Depth (n={n_qubits}): {depth}")
    
    if depth > 100:
        print("WARNING: Circuit depth exceeds 100 gates!")
    else:
        print("Constraint Check: Depth < 100 passed.")
        
    trainer = QAETrainer(qae)
    
    # Train
    result = trainer.train(train_states)
    opt_params = result.x
    print(f"Final Loss: {result.fun}")
    
    # Evaluate
    fidelities = []
    encoder = qae.get_encoder().assign_parameters(opt_params)
    decoder = qae.get_decoder().assign_parameters(opt_params)
    
    for state in test_states:
        fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
        fidelities.append(fid)
        
    avg_fid = np.mean(fidelities)
    print(f"Average Test Fidelity (n=6, GHZ): {avg_fid}")
    
    # Save results
    with open('results/exp4_scaling.txt', 'w') as f:
        f.write(f"Ansatz: {ansatz_name}\n")
        f.write(f"Depth: {depth}\n")
        f.write(f"Fidelity: {avg_fid}\n")
    print("Results saved to results/exp4_scaling.txt")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    run_experiment()
