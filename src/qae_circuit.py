from qiskit import QuantumCircuit
from typing import List

class QAECircuit:
    """
    A class to construct and manage the Quantum Autoencoder (QAE) circuit.
    
    This class handles the creation of the training circuit (including the SWAP test)
    and provides access to the encoder and decoder components.
    """

    def __init__(self, n_qubits: int, k_qubits: int, ansatz: QuantumCircuit):
        """
        Initialize the Quantum Autoencoder Circuit.

        Args:
            n_qubits (int): Total number of qubits for the input state.
            k_qubits (int): Number of latent qubits (compressed size).
            ansatz (QuantumCircuit): The parameterized circuit to use as the Encoder.
                                     This circuit must have `n_qubits`.
        """
        self.n_qubits: int = n_qubits
        self.k_qubits: int = k_qubits
        self.ansatz: QuantumCircuit = ansatz
        
        # Trash qubits are the ones we discard (measure/reset)
        # We keep the first k qubits as latent, and discard the last n-k
        self.trash_qubits: List[int] = list(range(k_qubits, n_qubits))
        self.latent_qubits: List[int] = list(range(k_qubits))
        
    def compose_training_circuit(self, input_state_circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Creates the circuit for training the autoencoder using the SWAP test / Fidelity check.

        The training objective is to maximize the overlap between the trash state and |0...0>.
        This is equivalent to minimizing the probability of measuring '1' on any of the trash qubits.

        Args:
            input_state_circuit (QuantumCircuit): Circuit preparing the input state.

        Returns:
            QuantumCircuit: The full training circuit (State Prep + Encoder + Measure Trash).
        """
        
        # 1. State Preparation
        # We need a circuit with n_qubits (for data) + potentially aux qubits if we were doing a real SWAP test
        # But here we are doing the "Direct Fidelity" method where we just measure trash qubits in Z basis.
        # If trash qubits are |0...0>, compression is perfect (assuming unitary encoder).
        qc = QuantumCircuit(self.n_qubits, len(self.trash_qubits))
        qc.compose(input_state_circuit, inplace=True)
        
        # 2. Encoder (Ansatz)
        qc.compose(self.ansatz, inplace=True)
        
        # 3. Measurement of Trash Qubits
        # We measure the trash qubits to compute the cost function
        qc.measure(self.trash_qubits, range(len(self.trash_qubits)))
        
        return qc
        
    def get_encoder(self) -> QuantumCircuit:
        """
        Returns the encoder circuit (ansatz).
        
        Returns:
            QuantumCircuit: The encoder circuit.
        """
        return self.ansatz
    
    def get_decoder(self) -> QuantumCircuit:
        """
        Returns the decoder circuit (inverse of ansatz).
        
        Returns:
            QuantumCircuit: The decoder circuit.
        """
        return self.ansatz.inverse()
