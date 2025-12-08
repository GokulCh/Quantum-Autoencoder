import numpy as np
from qiskit import QuantumCircuit

def get_product_state(n_qubits: int, theta: float = None, rng: np.random.Generator = None) -> QuantumCircuit:
    """Generates a product state from a 1-parameter family."""
    
    qc = QuantumCircuit(n_qubits)
    if rng is None:
        rng = np.random.default_rng()
        
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        angle = theta * (i + 1) / n_qubits
        qc.ry(angle, i)
    
    return qc

def get_ghz_state(n_qubits: int, theta: float = None, rng: np.random.Generator = None) -> QuantumCircuit:
    """Generates a rotated GHZ state."""
    
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
        
    if rng is None:
        rng = np.random.default_rng()
        
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        qc.ry(theta, i)
        
    return qc

def get_w_state(n_qubits: int, theta: float = None, rng: np.random.Generator = None) -> QuantumCircuit:
    """Generates a rotated W state."""

    vector = np.zeros(2**n_qubits)
    for i in range(n_qubits):
        vector[1 << i] = 1.0
    vector /= np.linalg.norm(vector)
    
    qc = QuantumCircuit(n_qubits)
    qc.prepare_state(vector)
    
    if rng is None:
        rng = np.random.default_rng()

    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        qc.ry(theta, i)
        
    return qc

def get_state_circuit(state_type: str, n_qubits: int, rng: np.random.Generator = None) -> QuantumCircuit:
    """Factory function to get the state preparation circuit."""
    
    if state_type.lower() == 'product':
        return get_product_state(n_qubits, rng=rng)
    elif state_type.lower() == 'ghz':
        return get_ghz_state(n_qubits, rng=rng)
    elif state_type.lower() == 'w':
        return get_w_state(n_qubits, rng=rng)
    else:
        raise ValueError(f"Unknown state type: {state_type}")
