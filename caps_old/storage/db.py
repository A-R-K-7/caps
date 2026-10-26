from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from configs.settings import settings

Base = declarative_base()

class ReplayTransition(Base):
    __tablename__ = 'replay_buffer'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(String, nullable=False)
    task_id = Column(String, nullable=True)
    policy_name = Column(String, nullable=True)
    
    # State features
    available_ciu_ids = Column(JSON, nullable=False)
    token_budget = Column(Integer, nullable=False)
    
    # Action
    selected_ciu_ids = Column(JSON, nullable=False)
    
    # Result
    asked_clarification = Column(Boolean, nullable=False)
    receiver_answer = Column(String, nullable=False)
    task_success = Column(Boolean, nullable=False)
    env_metrics = Column(JSON, nullable=True)
    
    # Evaluator & Reward
    voi_scores = Column(JSON, nullable=False)
    oracle_results = Column(JSON, nullable=True)
    reward = Column(Float, nullable=False)
    
    # Metadata
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ReplayBuffer:
    def __init__(self):
        self.engine = create_engine(settings.sqlite_db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def add_transition(self, task: str, available_ciu_ids: list, token_budget: int,
                       selected_ciu_ids: list, asked_clarification: bool, receiver_answer: str,
                       task_success: bool, voi_scores: dict, reward: float,
                       task_id: str = None, policy_name: str = None, 
                       env_metrics: dict = None, oracle_results: dict = None):
        
        transition = ReplayTransition(
            task=task,
            task_id=task_id,
            policy_name=policy_name,
            available_ciu_ids=available_ciu_ids,
            token_budget=token_budget,
            selected_ciu_ids=selected_ciu_ids,
            asked_clarification=asked_clarification,
            receiver_answer=receiver_answer,
            task_success=task_success,
            env_metrics=env_metrics,
            voi_scores=voi_scores,
            oracle_results=oracle_results,
            reward=reward
        )
        
        with self.Session() as session:
            session.add(transition)
            session.commit()
