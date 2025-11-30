from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity, Statevector, partial_trace, DensityMatrix

def compute_reconstruction_fidelity(original_circuit: QuantumCircuit, encoder: QuantumCircuit, decoder: QuantumCircuit, n_qubits: int, k_qubits: int) -> float:
    """
    Computes the fidelity between the original state and the reconstructed state.
    
    Args:
        original_circuit (QuantumCircuit): Circuit preparing the input state.
        encoder (QuantumCircuit): The trained encoder.
        decoder (QuantumCircuit): The trained decoder.
        n_qubits (int): Total number of qubits.
        k_qubits (int): Number of latent qubits.
        
    Returns:
        float: The state fidelity between original and reconstructed states.
    """
    # 1. Get original statevector
    sv_original = Statevector.from_instruction(original_circuit)
    
    # 2. Construct full circuit for encoding
    # Original -> Encoder
    full_circ = original_circuit.compose(encoder)
    
    # 3. Get density matrix after encoding
    rho_encoded = DensityMatrix.from_instruction(full_circ)
    
    # 4. Trace out trash qubits (indices k to n-1)
    trash_qubits = list(range(k_qubits, n_qubits))
    rho_latent = partial_trace(rho_encoded, trash_qubits)
    
    # 5. Prepare fresh ancilla state |0...0> for trash qubits
    # We need to tensor rho_latent with |0><0|^(n-k)
    # Note: partial_trace returns a DensityMatrix of size 2^k.
    # We need to expand it back to 2^n.
    # The tensor product order matters. Qiskit uses little-endian (qubit 0 is rightmost).
    # But partial_trace and tensor methods usually handle this if we are careful.
    # If we traced out the *last* qubits (higher indices), then rho_latent corresponds to the *first* qubits (lower indices).
    # So we want rho_total = rho_latent (qubits 0..k-1) (tensor) |0><0| (qubits k..n-1).
    
    rho_trash_reset = DensityMatrix.from_label('0' * (n_qubits - k_qubits))
    # Qiskit tensor product: A.tensor(B) = A (kron) B
    # We want |trash> (kron) |latent> to match q_n...q_k...q_0
    # rho_trash corresponds to higher indices, rho_latent to lower indices.
    rho_reconstructed_input = rho_trash_reset.tensor(rho_latent)
    
    # 6. Apply Decoder
    # We can evolve the density matrix by the decoder unitary
    rho_final = rho_reconstructed_input.evolve(decoder)
    
    # 7. Compute fidelity
    fid = state_fidelity(sv_original, rho_final)
    
    return fid
