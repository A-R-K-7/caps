import random
from caps.engine.env import Policy
from caps.policy.models import PlannerState, Action

class AlwaysSendEverythingPolicy(Policy):
    """
    V1: Selects all available CIUs, ignoring token budgets.
    """
    def select(self, state: PlannerState) -> Action:
        selected_ids = [ciu.id for ciu in state.available_cius]
        # Equal dummy scores for all
        scores = {ciu.id: 1.0 for ciu in state.available_cius}
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)
        
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        # Baseline policy does not learn
        pass

class RandomPolicy(Policy):
    """
    V2: Randomly selects a subset of CIUs.
    Does not enforce token budget, simply introduces variance.
    """
    def __init__(self, select_prob: float = 0.5):
        self.select_prob = select_prob
        
    def select(self, state: PlannerState) -> Action:
        selected_ids = []
        scores = {}
        for ciu in state.available_cius:
            score = random.random()
            scores[ciu.id] = score
            if score < self.select_prob:
                selected_ids.append(ciu.id)
                
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)
        
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        # Baseline policy does not learn
        pass

class TokenBudgetGreedyPolicy(Policy):
    """
    V3: Greedily selects CIUs until the token budget is exhausted.
    Currently selects them in the order they are presented.
    Future iterations can sort by heuristic score before greedy selection.
    """
    def select(self, state: PlannerState) -> Action:
        selected_ids = []
        scores = {}
        current_tokens = 0
        
        for idx, ciu in enumerate(state.available_cius):
            # Give higher scores to items earlier in the list to represent ranking
            score = 1.0 / (idx + 1)
            scores[ciu.id] = score
            
            if current_tokens + ciu.token_count <= state.token_budget:
                selected_ids.append(ciu.id)
                current_tokens += ciu.token_count
                
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)
        
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        # Baseline policy does not learn
        pass

import numpy as np


class TopKPolicy(Policy):
    """
    Always send the first K CIUs, up to the token budget.
    """
    def __init__(self, k: int = 5):
        self.k = k

    def select(self, state: PlannerState) -> Action:
        selected_ids = []
        scores = {}
        current_tokens = 0
        
        for idx, ciu in enumerate(state.available_cius):
            if idx >= self.k:
                break
                
            score = 1.0 / (idx + 1)
            scores[ciu.id] = score
            
            if current_tokens + ciu.token_count <= state.token_budget:
                selected_ids.append(ciu.id)
                current_tokens += ciu.token_count
                
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)
        
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        pass

class EmbeddingSimilarityPolicy(Policy):
    """
    Compute similarity between task embedding and CIU embedding.
    Send highest similarity CIUs up to token budget.
    """
    def select(self, state: PlannerState) -> Action:
        selected_ids = []
        scores = {}
        
        if not state.task_embedding:
            return TokenBudgetGreedyPolicy().select(state)
            
        task_emb = np.array(state.task_embedding)
        task_norm = np.linalg.norm(task_emb)
        
        for ciu in state.available_cius:
            if not ciu.embedding:
                scores[ciu.id] = 0.0
                continue
                
            ciu_emb = np.array(ciu.embedding)
            ciu_norm = np.linalg.norm(ciu_emb)
            
            if task_norm > 0 and ciu_norm > 0:
                similarity = np.dot(task_emb, ciu_emb) / (task_norm * ciu_norm)
            else:
                similarity = 0.0
                
            scores[ciu.id] = float(similarity)
            
        sorted_cius = sorted(state.available_cius, key=lambda c: scores[c.id], reverse=True)
        
        current_tokens = 0
        for ciu in sorted_cius:
            if current_tokens + ciu.token_count <= state.token_budget:
                selected_ids.append(ciu.id)
                current_tokens += ciu.token_count
                
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)

    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        pass

class HeuristicPolicy(Policy):
    """
    Hand-written communication rules prioritizing certain types.
    """
    def __init__(self):
        self.type_scores = {
            "task": 1.0,
            "stack_trace": 0.9,
            "source_code": 0.8,
            "tests": 0.7,
            "metadata": 0.2,
            "readme": 0.1,
            "license": 0.0
        }
        self.default_score = 0.5
        
    def select(self, state: PlannerState) -> Action:
        selected_ids = []
        scores = {}
        
        for ciu in state.available_cius:
            ctype = ciu.type.lower()
            score = self.type_scores.get(ctype, self.default_score)
            scores[ciu.id] = score
            
        sorted_cius = sorted(state.available_cius, key=lambda c: scores[c.id], reverse=True)
        
        current_tokens = 0
        for ciu in sorted_cius:
            if current_tokens + ciu.token_count <= state.token_budget:
                selected_ids.append(ciu.id)
                current_tokens += ciu.token_count
                
        return Action(selected_ciu_ids=selected_ids, ciu_scores=scores)
        
    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        pass
