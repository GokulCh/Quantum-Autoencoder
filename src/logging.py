import json
import os
import time
from typing import Any, Dict, Optional

class ExperimentLogger:
    def __init__(self, experiment_name: str, results_dir: str = "results/data"):
        """
        Initializes the ExperimentLogger.

        Args:
            experiment_name (str): The name of the experiment (used for filenames).
            results_dir (str): output directory for JSON files.
        """
        self.experiment_name = experiment_name
        self.results_dir = results_dir
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.data = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "config": {},
            "results": {}
        }
        
    def log_config(self, config: Dict[str, Any]):
        """Logs configuration parameters."""
        self.data["config"].update(config)

    def log_result(self, key: str, value: Any):
        """Logs a result metric."""
        self.data["results"][key] = value

    def save(self):
        """Saves the logged data to a JSON file."""
        os.makedirs(self.results_dir, exist_ok=True)
        filename = f"{self.experiment_name}_{self.timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy types to python types for JSON serialization
        def default_converter(o):
            if hasattr(o, 'item'): 
                return o.item()
            if hasattr(o, 'tolist'):
                return o.tolist()
            return str(o)

        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4, default=default_converter)
        
        print(f"Experiment data saved to {filepath}")
        return filepath
