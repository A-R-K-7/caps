import abc
from typing import Dict, Any

class Environment(abc.ABC):
    """
    Standard interface for evaluating CAPS receiver outputs.
    """
    
    @abc.abstractmethod
    def reset(self, task_idx: int) -> str:
        """
        Resets the environment for the given task index.
        Returns the initial task description.
        """
        pass
        
    @abc.abstractmethod
    def evaluate(self, receiver_output: str) -> bool:
        """
        Evaluates the receiver's output to determine if the task was successful.
        Task success must NOT depend on the communication policy.
        """
        pass
        
    @abc.abstractmethod
    def ground_truth(self) -> Any:
        """
        Returns the ground truth answer or expected state.
        """
        pass
        
    @abc.abstractmethod
    def metrics(self) -> Dict[str, Any]:
        """
        Returns any environment-specific metrics for the current episode.
        """
        pass
