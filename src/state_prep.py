import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

def get_product_state(n_qubits: int, theta: float = None) -> QuantumCircuit:
    """
    Generates a product state from a 1-parameter family.
    |psi(theta)> = product_i Ry(theta * (i+1)) |0>
    
    Args:
        n_qubits (int): Number of qubits.
        theta (float, optional): Rotation parameter. If None, chosen randomly.
        
    Returns:
        QuantumCircuit: Circuit preparing the product state.
    """
    qc = QuantumCircuit(n_qubits)
    rng = np.random.default_rng()
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        # Use a fixed relation between qubits based on theta
        # This ensures the state lives in a 1D manifold
        angle = theta * (i + 1) / n_qubits
        qc.ry(angle, i)
    return qc

def get_ghz_state(n_qubits: int, theta: float = None) -> QuantumCircuit:
    """
    Generates a rotated GHZ state.
    |psi(theta)> = Ry(theta)^n |GHZ>
    
    Args:
        n_qubits (int): Number of qubits.
        theta (float, optional): Rotation parameter. If None, chosen randomly.
        
    Returns:
        QuantumCircuit: Circuit preparing the rotated GHZ state.
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i+1)
        
    rng = np.random.default_rng()
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        qc.ry(theta, i)
        
    return qc

def get_w_state(n_qubits: int, theta: float = None) -> QuantumCircuit:
    """
    Generates a rotated W state.
    
    Args:
        n_qubits (int): Number of qubits.
        theta (float, optional): Rotation parameter. If None, chosen randomly.
        
    Returns:
        QuantumCircuit: Circuit preparing the rotated W state.
    """
    # W state construction
    vector = np.zeros(2**n_qubits)
    for i in range(n_qubits):
        vector[1 << i] = 1.0
    vector /= np.linalg.norm(vector)
    
    qc = QuantumCircuit(n_qubits)
    qc.prepare_state(vector)
    
    rng = np.random.default_rng()
    if theta is None:
        theta = rng.uniform(0, 2 * np.pi)
        
    for i in range(n_qubits):
        qc.ry(theta, i)
        
    return qc

def get_state_circuit(state_type: str, n_qubits: int) -> QuantumCircuit:
    """
    Factory function to get the state preparation circuit.
    
    Args:
        state_type (str): 'product', 'ghz', or 'w'.
        n_qubits (int): Number of qubits.
        
    Returns:
        QuantumCircuit: The requested state preparation circuit.
        
    Raises:
        ValueError: If the state type is unknown.
    """
    if state_type.lower() == 'product':
        return get_product_state(n_qubits)
    elif state_type.lower() == 'ghz':
        return get_ghz_state(n_qubits)
    elif state_type.lower() == 'w':
        return get_w_state(n_qubits)
    else:
        raise ValueError(f"Unknown state type: {state_type}")
