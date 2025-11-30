import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize, OptimizeResult
from typing import List, Optional
from .qae_circuit import QAECircuit

class QAETrainer:
    """
    A class to handle the training process of the Quantum Autoencoder.
    """
    def __init__(self, qae_circuit: QAECircuit, maxiter: int = 1000):
        """
        Initialize the QAE Trainer.
        
        Args:
            qae_circuit (QAECircuit): The QAE circuit object containing the ansatz and configuration.
            maxiter (int): Maximum number of optimization iterations for the COBYLA optimizer.
                           Defaults to 200.
        """
        self.qae: QAECircuit = qae_circuit
        self.maxiter: int = maxiter
        self.loss_history: List[float] = []
        
    def train(self, input_states: List[QuantumCircuit], initial_point: Optional[np.ndarray] = None) -> OptimizeResult:
        """
        Trains the Quantum Autoencoder using the provided input states.
        
        The training minimizes the loss function, which is defined as the probability of measuring
        '1' on the trash qubits (i.e., maximizing the overlap with |0...0> on trash qubits).
        
        Args:
            input_states (List[QuantumCircuit]): List of QuantumCircuits preparing the input states to be compressed.
            initial_point (Optional[np.ndarray]): Initial parameters for the ansatz. If None, random parameters are used.
            
        Returns:
            OptimizeResult: The result of the optimization process (from scipy.optimize).
        """
        self.loss_history = []
        
        # Use Qiskit's Sampler primitive
        sampler = Sampler()
        
        def objective_function(params: np.ndarray) -> float:
            cost = 0.0
            # Batch circuits for efficiency
            circuits_to_run = []
            for state_circ in input_states:
                # Create full circuit using QAE helper
                full_circ = self.qae.compose_training_circuit(state_circ)
                circuits_to_run.append(full_circ)
            
            # Bind parameters
            bound_circuits = [c.assign_parameters(params) for c in circuits_to_run]
            
            # Run sampler
            job = sampler.run(bound_circuits)
            result = job.result()
            
            # Calculate cost: Probability of NOT being 0...0
            # We want trash to be |0...0>, so we penalize any '1's.
            # Actually, simply sum of probabilities of all bitstrings != '0...0'
            
            for i, quasi_dist in enumerate(result.quasi_dists):
                # Probability of '0'*len(trash)
                prob_zero = quasi_dist.get(0, 0.0) 
                # Cost is 1 - prob_zero (we want prob_zero to be 1)
                cost += (1.0 - prob_zero)
                
            avg_cost = cost / len(input_states)
            self.loss_history.append(avg_cost)
            return avg_cost

        # Initial parameters
        if initial_point is None:
            initial_point = np.random.random(self.qae.ansatz.num_parameters)
            
        # Optimize using Scipy
        # COBYLA is a gradient-free optimizer, often better for noisy quantum landscapes
        result = minimize(objective_function, x0=initial_point, method='COBYLA', options={'maxiter': self.maxiter})
        
        return result
