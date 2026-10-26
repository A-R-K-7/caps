import abc
from caps.policy.models import PlannerState, Action
from caps.planner.agent import PlannerAgent
from caps.receiver.agent import ReceiverAgent
from caps.utils.llm import BaseLLMClient
from caps.evaluator.attribution import SelfReflectionEvaluator
from caps.evaluator.reward import RewardEngine
from caps.storage.db import ReplayBuffer

class Policy(abc.ABC):
    @abc.abstractmethod
    def select(self, state: PlannerState) -> Action:
        pass
        
    @abc.abstractmethod
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        pass

class Transition:
    def __init__(self, state, action, result, available_cius, voi_scores=None, reward=0.0):
        self.state = state
        self.action = action
        self.result = result
        self.available_cius = available_cius
        self.voi_scores = voi_scores or {}
        self.reward = reward

class CAPSEnv:
    def __init__(self, llm_client: BaseLLMClient, temperature: float = None, seed: int = None):
        self.planner = PlannerAgent(llm_client, temperature, seed)
        self.receiver = ReceiverAgent(llm_client, temperature, seed)
        self.evaluator = SelfReflectionEvaluator(llm_client, temperature, seed)
        self.reward_engine = RewardEngine()
        self.replay_buffer = ReplayBuffer()
        
    def step(self, task: str, raw_context: str, policy: Policy) -> Transition:
        """
        Executes a full MDP step for the CAPS framework, EXCLUDING reward and objective task success.
        The caller (ExperimentRunner) is responsible for evaluating task success, calculating reward,
        updating the policy, and saving to the replay buffer.
        """
        # 1. Planner Phase
        state, available_cius = self.planner.extract_state(task, raw_context)
        
        # 2. Policy Action Phase
        action = policy.select(state)
        selected_cius = [ciu for ciu in available_cius if ciu.id in action.selected_ciu_ids]
        
        # 3. Receiver Execution Phase
        result = self.receiver.execute(task, selected_cius)
        
        # 4. Evaluator Phase (Self-Reflection VOI, used as a fast baseline)
        voi_scores = self.evaluator.evaluate(task, selected_cius, result.answer)
        
        return Transition(state, action, result, available_cius, voi_scores)

