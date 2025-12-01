from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector, partial_trace, DensityMatrix

def compute_reconstruction_fidelity(original_circuit: QuantumCircuit, encoder: QuantumCircuit, decoder: QuantumCircuit, n_qubits: int, k_qubits: int) -> float:
    """Computes the fidelity between the original state and the reconstructed state."""

    # Get original statevector
    sv_original = Statevector.from_instruction(original_circuit)
    
    # Construct full circuit for encoding
    # Original -> Encoder
    full_circ = original_circuit.compose(encoder)
    
    # Get density matrix after encoding
    rho_encoded = DensityMatrix.from_instruction(full_circ)
    
    # Trace out trash qubits (indices k to n-1)
    trash_qubits = list(range(k_qubits, n_qubits))
    rho_latent = partial_trace(rho_encoded, trash_qubits)
    
    # Prepare fresh ancilla state |0...0> for trash qubits
    rho_trash_reset = DensityMatrix.from_label('0' * (n_qubits - k_qubits))
    rho_reconstructed_input = rho_trash_reset.tensor(rho_latent)
    
    # Apply Decoder
    rho_final = rho_reconstructed_input.evolve(decoder)
    
    # Compute fidelity
    fid = state_fidelity(sv_original, rho_final)
    
    return fid
