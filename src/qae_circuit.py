from qiskit import QuantumCircuit
from typing import List

class QAECircuit:

    def __init__(self, n_qubits: int, k_qubits: int, ansatz: QuantumCircuit):
        """Initialize the Quantum Autoencoder Circuit."""

        self.n_qubits: int = n_qubits
        self.k_qubits: int = k_qubits
        self.ansatz: QuantumCircuit = ansatz
        
        self.trash_qubits: List[int] = list(range(k_qubits, n_qubits))
        self.latent_qubits: List[int] = list(range(k_qubits))
        
    def compose_training_circuit(self, input_state_circuit: QuantumCircuit) -> QuantumCircuit:
        """Creates the circuit for training the autoencoder using the SWAP test / Fidelity check."""

        # State Preparation
        qc = QuantumCircuit(self.n_qubits, len(self.trash_qubits))
        qc.compose(input_state_circuit, inplace=True)
        
        # Encoder (Ansatz)
        qc.compose(self.ansatz, inplace=True)
        
        # Measurement of Trash Qubits
        qc.measure(self.trash_qubits, range(len(self.trash_qubits)))
        
        return qc
        
    def get_encoder(self) -> QuantumCircuit:
        """Returns the encoder circuit (ansatz)."""
        return self.ansatz
    
    def get_decoder(self) -> QuantumCircuit:
        """Returns the decoder circuit (inverse of ansatz)."""
        return self.ansatz.inverse()
