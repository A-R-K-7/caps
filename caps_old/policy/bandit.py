import numpy as np
from typing import Dict, List, Tuple
from caps.engine.env import Policy
from caps.policy.models import PlannerState, Action
from caps.ciu.models import CIU

class MOLinUCBPolicy(Policy):
    """
    Multi-Objective Contextual Bandit (MO-LinUCB) with Pareto Front Selection.
    Optimizes for both VOI (utilization) and global reward (task success - cost).
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_dim = None
        
        # Dual matrices for VOI and Reward
        self.A_voi = None
        self.A_voi_inv = None
        self.b_voi = None
        
        self.A_reward = None
        self.A_reward_inv = None
        self.b_reward = None

    def _init_matrices(self, feature_dim: int):
        self.feature_dim = feature_dim
        
        self.A_voi = np.eye(feature_dim)
        self.A_voi_inv = np.eye(feature_dim)
        self.b_voi = np.zeros((feature_dim, 1))
        
        self.A_reward = np.eye(feature_dim)
        self.A_reward_inv = np.eye(feature_dim)
        self.b_reward = np.zeros((feature_dim, 1))

    def _get_features(self, state: PlannerState, ciu: CIU) -> np.ndarray:
        # Construct feature vector z
        task_emb = state.task_embedding or []
        ciu_emb = ciu.embedding or []
        
        # Scalar features
        scalars = [
            float(state.planner_confidence),
            float(state.token_budget) / 10000.0, # scale it
            float(state.latency_budget)
        ]
        
        z = np.array(task_emb + ciu_emb + scalars).reshape(-1, 1)
        
        if self.feature_dim is None:
            self._init_matrices(z.shape[0])
            
        return z

    def select(self, state: PlannerState) -> Action:
        if not state.available_cius:
            return Action(selected_ciu_ids=[], ciu_scores={})
            
        # Ensure matrices are initialized
        # We do this by getting features for the first CIU
        self._get_features(state, state.available_cius[0])
        
        theta_voi = self.A_voi_inv @ self.b_voi
        theta_reward = self.A_reward_inv @ self.b_reward
        
        # 1. Compute UCB bounds for both objectives
        ucb_tuples: Dict[str, Tuple[float, float]] = {}
        for ciu in state.available_cius:
            z = self._get_features(state, ciu)
            
            # VOI Model
            p_voi = (theta_voi.T @ z)[0, 0] + self.alpha * np.sqrt((z.T @ self.A_voi_inv @ z)[0, 0])
            
            # Reward Model
            p_reward = (theta_reward.T @ z)[0, 0] + self.alpha * np.sqrt((z.T @ self.A_reward_inv @ z)[0, 0])
            
            ucb_tuples[ciu.id] = (float(p_voi), float(p_reward))
            
        # 2. Non-dominated sorting (Pareto fronts)
        # We want to maximize both p_voi and p_reward
        fronts = self._non_dominated_sort(state.available_cius, ucb_tuples)
        
        # 3. Selection
        selected_ids = []
        current_tokens = 0
        
        for front in fronts:
            # Sort within front by sum of normalized scores or just pick sequentially
            # For simplicity, we just pick sequentially from the front until budget exhausted
            for ciu in front:
                if current_tokens + ciu.token_count <= state.token_budget:
                    selected_ids.append(ciu.id)
                    current_tokens += ciu.token_count
                    
        # Action requires scalar scores, we can just return the sum of the UCBs or the scalarized value
        ciu_scores = {c.id: (ucb_tuples[c.id][0] + ucb_tuples[c.id][1]) for c in state.available_cius}
        
        return Action(selected_ciu_ids=selected_ids, ciu_scores=ciu_scores)
        
    def _non_dominated_sort(self, cius: List[CIU], ucb_tuples: Dict[str, Tuple[float, float]]) -> List[List[CIU]]:
        """
        Organizes CIUs into Pareto fronts.
        fronts[0] is the non-dominated set. fronts[1] is non-dominated when fronts[0] is removed, etc.
        """
        domination_count = {ciu.id: 0 for ciu in cius}
        dominated_by = {ciu.id: [] for ciu in cius}
        
        fronts = [[]]
        
        for i, p in enumerate(cius):
            p_voi, p_reward = ucb_tuples[p.id]
            for j, q in enumerate(cius):
                if i == j:
                    continue
                q_voi, q_reward = ucb_tuples[q.id]
                
                # Check if p dominates q
                # p dominates q if it is >= in all objectives and > in at least one
                if (p_voi >= q_voi and p_reward >= q_reward) and (p_voi > q_voi or p_reward > q_reward):
                    dominated_by[p.id].append(q.id)
                elif (q_voi >= p_voi and q_reward >= p_reward) and (q_voi > p_voi or q_reward > p_reward):
                    domination_count[p.id] += 1
                    
            if domination_count[p.id] == 0:
                fronts[0].append(p)
                
        # Build subsequent fronts
        i = 0
        ciu_dict = {ciu.id: ciu for ciu in cius}
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q_id in dominated_by[p.id]:
                    domination_count[q_id] -= 1
                    if domination_count[q_id] == 0:
                        next_front.append(ciu_dict[q_id])
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break
                
        return fronts

    def update(self, state: PlannerState, action: Action, reward: float, **kwargs):
        if not action.selected_ciu_ids:
            return
            
        voi_scores = kwargs.get("voi_scores", {})
        
        for ciu in state.available_cius:
            if ciu.id in action.selected_ciu_ids:
                z = self._get_features(state, ciu)
                
                # Update VOI Model
                v_score = voi_scores.get(ciu.id, 0.0)
                self.A_voi += z @ z.T
                self.b_voi += v_score * z
                
                # Update Reward Model
                self.A_reward += z @ z.T
                self.b_reward += reward * z
                
        # Recompute inverses
        if self.A_voi is not None:
            self.A_voi_inv = np.linalg.inv(self.A_voi)
        if self.A_reward is not None:
            self.A_reward_inv = np.linalg.inv(self.A_reward)
