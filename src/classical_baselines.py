import numpy as np
from sklearn.decomposition import PCA
from qiskit.quantum_info import Statevector, state_fidelity
from typing import List, Union
from qiskit import QuantumCircuit

class ClassicalBaselines:
    """
    A class providing classical baseline methods for comparison with the Quantum Autoencoder.
    """
    @staticmethod
    def run_pca(input_states: List[Union[Statevector, QuantumCircuit]], k_latent: int) -> float:
        """
        Runs Principal Component Analysis (PCA) on the input quantum states (amplitudes) as a classical baseline.
        
        Since quantum states are complex vectors of size 2^n, and PCA works on real vectors,
        we treat the real and imaginary parts as separate features, effectively doubling the input dimension.
        We then reduce the dimension to match the degrees of freedom of the k latent qubits.

        Args:
            input_states (List[Union[Statevector, QuantumCircuit]]): List of input states to compress.
            k_latent (int): Number of latent qubits in the quantum autoencoder. 
                            The target PCA dimension is derived from this to ensure fair comparison.
                            Target dimension = 2 * 2^k (real components).

        Returns:
            float: The average reconstruction fidelity.
        """
        # Convert circuits to statevectors (amplitudes)
        data = []
        for state in input_states:
            if not isinstance(state, Statevector):
                sv = Statevector.from_instruction(state)
            else:
                sv = state
            data.append(np.array(sv))
            
        X = np.array(data)
        
        # PCA in sklearn doesn't support complex numbers.
        # We can treat real and imaginary parts as separate features.
        # Input shape: (n_samples, 2^n) complex
        # New shape: (n_samples, 2 * 2^n) real
        X_real = np.real(X)
        X_imag = np.imag(X)
        X_combined = np.hstack([X_real, X_imag])
        
        # PCA target dimension logic:
        # QAE compresses to k qubits => 2^k complex amplitudes.
        # So the latent space size is 2^k complex numbers.
        # In real terms, that's 2 * 2^k real numbers.
        n_components = 2 * (2**k_latent)
        
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        n_samples, n_features = X_combined.shape
        n_components = min(n_components, n_samples, n_features)
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_combined)
        X_reconstructed_combined = pca.inverse_transform(X_reduced)
        
        # Reconstruct complex vectors
        n_original_features = X.shape[1]
        X_reconstructed = X_reconstructed_combined[:, :n_original_features] + 1j * X_reconstructed_combined[:, n_original_features:]
        
        # Compute average fidelity
        fidelities = []
        for i in range(len(X)):
            # Normalize reconstructed vector (PCA doesn't preserve norm exactly)
            recon = X_reconstructed[i]
            norm = np.linalg.norm(recon)
            if norm > 1e-9:
                recon = recon / norm
            
            # Compute fidelity |<psi|recon>|^2
            # Since these are complex vectors:
            overlap = np.abs(np.dot(X[i].conj(), recon))**2
            fidelities.append(overlap)
            
        return np.mean(fidelities)

    @staticmethod
    def run_random_unitary(input_states: List[QuantumCircuit], n_qubits: int, k_qubits: int) -> float:
        """
        Runs a Random Unitary baseline.
        
        This method uses a random unitary matrix as the encoder and its inverse as the decoder.
        It serves as a "zero-knowledge" baseline to see how well a random compression works.

        Args:
            input_states (List[QuantumCircuit]): List of input states.
            n_qubits (int): Total number of qubits.
            k_qubits (int): Number of latent qubits.

        Returns:
            float: The average reconstruction fidelity.
        """
        from qiskit.quantum_info import random_unitary
        from src.metrics import compute_reconstruction_fidelity
        from qiskit import QuantumCircuit
        
        # Generate one random unitary
        u = random_unitary(2**n_qubits)
        encoder = QuantumCircuit(n_qubits)
        encoder.append(u, range(n_qubits))
        decoder = encoder.inverse()
        
        fidelities = []
        for state in input_states:
            fid = compute_reconstruction_fidelity(state, encoder, decoder, n_qubits, k_qubits)
            fidelities.append(fid)
            
        return np.mean(fidelities)
