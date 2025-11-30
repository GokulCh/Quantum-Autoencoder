from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2

def get_ansatz(name: str, n_qubits: int, reps: int = 5) -> QuantumCircuit:
    """
    Returns the requested ansatz circuit for the Quantum Autoencoder.

    Args:
        name (str): The name of the ansatz to use. Options are:
                    - 'RealAmplitudes': Uses Qiskit's RealAmplitudes circuit.
                    - 'EfficientSU2': Uses Qiskit's EfficientSU2 circuit.
                    - 'HardwareEfficient': A hardware-efficient ansatz tailored for IBM devices,
                      using 'ry', 'rz' rotations and 'cx' entangling gates.
        n_qubits (int): The number of qubits in the ansatz.
        reps (int): The number of repetitions (layers) of the ansatz. Defaults to 3.

    Returns:
        QuantumCircuit: The constructed parameterized quantum circuit.

    Raises:
        ValueError: If the provided ansatz name is not recognized.
    """
    if name == 'RealAmplitudes':
        return RealAmplitudes(num_qubits=n_qubits, reps=reps, entanglement='linear')
    elif name == 'EfficientSU2':
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear')
    elif name == 'HardwareEfficient':
        # Hardware-efficient ansatz typically uses single-qubit rotations and CNOTs
        # EfficientSU2 with 'linear' entanglement is a good approximation for IBM hardware
        # We customize the gates to ['ry', 'rz'] and 'cx' which are native/efficient on many superconducting backends
        return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement='linear', su2_gates=['ry', 'rz'])
    else:
        raise ValueError(f"Unknown ansatz: {name}")
