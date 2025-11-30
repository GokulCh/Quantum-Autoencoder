# src/ansatze/real_amplitudes.py

from qiskit.circuit import QuantumCircuit, ParameterVector

class RealAmplitudesAnsatz:
    """
    Simple variational ansatz for QAE experiments.
    Matches the RealAmplitudes template but implemented manually
    so the project is 100% reproducible.
    """

    def __init__(self, num_qubits: int, reps: int = 3, entanglement="linear"):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement

        # Total parameters: 2 * num_qubits * reps
        self.theta = ParameterVector("θ", length=2 * num_qubits * reps)

    def _entangle(self, qc: QuantumCircuit):
        if self.entanglement == "linear":
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cx(i, j)

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        idx = 0

        for _ in range(self.reps):
            # Rotation layer (RX)
            for q in range(self.num_qubits):
                qc.rx(self.theta[idx], q)
                idx += 1

            # Rotation layer (RZ)
            for q in range(self.num_qubits):
                qc.rz(self.theta[idx], q)
                idx += 1

            # Entanglement block
            self._entangle(qc)

        return qc



# src/ansatze/efficient_su2.py

from qiskit.circuit import QuantumCircuit, ParameterVector

class EfficientSU2Ansatz:
    """
    EfficientSU2-style ansatz used widely in QAE and VQE studies.
    More expressive but harder to train.
    """

    def __init__(self, num_qubits: int, reps: int = 3, entanglement="linear"):
        self.num_qubits = num_qubits
        self.reps = reps
        self.entanglement = entanglement

        # Each layer has 3 rotations per qubit
        self.theta = ParameterVector("φ", length=3 * num_qubits * reps)

    def _entangle(self, qc: QuantumCircuit):
        if self.entanglement == "linear":
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        elif self.entanglement == "circular":
            for i in range(self.num_qubits):
                qc.cx(i, (i + 1) % self.num_qubits)
        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    qc.cx(i, j)

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        idx = 0

        for _ in range(self.reps):
            for q in range(self.num_qubits):
                qc.rx(self.theta[idx], q)
                qc.ry(self.theta[idx + 1], q)
                qc.rz(self.theta[idx + 2], q)
                idx += 3

            self._entangle(qc)

        return qc



# src/ansatze/hardware_efficient.py

from qiskit.circuit import QuantumCircuit, ParameterVector

class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz reflecting realistic IBM device connectivity.
    Used for Experiment 3 (noise robustness).
    """

    def __init__(self, num_qubits: int, reps: int = 3):
        self.num_qubits = num_qubits
        self.reps = reps

        # Two rotations per layer per qubit
        self.theta = ParameterVector("α", length=2 * num_qubits * reps)

        # Simplified heavy-hex connectivity map
        # (device subset of FakeManila)
        self.connectivity = [(0, 1), (1, 2), (2, 3), (1, 4)] if num_qubits >= 5 else \
                            [(i, i + 1) for i in range(num_qubits - 1)]

    def build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        idx = 0

        for _ in range(self.reps):
            for q in range(self.num_qubits):
                qc.rx(self.theta[idx], q)
                qc.rz(self.theta[idx + 1], q)
                idx += 2

            for (c, t) in self.connectivity:
                if c < self.num_qubits and t < self.num_qubits:
                    qc.cx(c, t)

        return qc
