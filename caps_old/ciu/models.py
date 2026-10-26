from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uuid

class CIU(BaseModel):
    """
    Candidate Information Unit.
    Represents a discrete piece of information that can be communicated to the Receiver.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    content: str
    token_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
