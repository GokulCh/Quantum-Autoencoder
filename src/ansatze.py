from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2

def get_ansatz(name: str, n_qubits: int, reps: int = 5) -> QuantumCircuit:
    """Returns the requested ansatz circuit for the Quantum Autoencoder."""

    if name == 'RealAmplitudes':
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement='linear')
    elif name == 'EfficientSU2':
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear')
    elif name == 'HardwareEfficient':
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear', su2_gates=['ry', 'rz'])
    else:
        raise ValueError(f"Unknown ansatz: {name}")
