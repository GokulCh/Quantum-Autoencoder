import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize, OptimizeResult
from typing import List, Optional
from .qae_circuit import QAECircuit

class QAETrainer:
    def __init__(self, qae_circuit: QAECircuit, maxiter: int = 1000):
        """Initialize the QAE Trainer."""
   
        self.qae: QAECircuit = qae_circuit
        self.maxiter: int = maxiter
        self.loss_history: List[float] = []
        
    def train(self, input_states: List[QuantumCircuit], initial_point: Optional[np.ndarray] = None) -> OptimizeResult:
        """Trains the Quantum Autoencoder using the provided input states."""

        self.loss_history = []
        
        sampler = Sampler()
        
        def objective_function(params: np.ndarray) -> float:
            cost = 0.0
            circuits_to_run = []
            for state_circ in input_states:
                full_circ = self.qae.compose_training_circuit(state_circ)
                circuits_to_run.append(full_circ)
            
            bound_circuits = [c.assign_parameters(params) for c in circuits_to_run]
            
            job = sampler.run(bound_circuits)
            result = job.result()
     
            for i, quasi_dist in enumerate(result.quasi_dists):
                prob_zero = quasi_dist.get(0, 0.0) 
                cost += (1.0 - prob_zero)
                
            avg_cost = cost / len(input_states)
            self.loss_history.append(avg_cost)
            return avg_cost

        if initial_point is None:
            initial_point = np.random.random(self.qae.ansatz.num_parameters)
            
        result = minimize(objective_function, x0=initial_point, method='COBYLA', options={'maxiter': self.maxiter})
        
        return result
