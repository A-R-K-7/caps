from sentence_transformers import SentenceTransformer
from caps.ciu.models import CIU
from configs.settings import settings

class CIUEmbedder:
    """
    Handles embedding generation for Candidate Information Units.
    """
    def __init__(self):
        # Only loads the model into memory when instantiated
        self.model = SentenceTransformer(settings.embedding_model)

    def embed(self, cius: list[CIU]) -> list[CIU]:
        """
        Takes a list of CIUs, computes their embeddings, and updates them in-place.
        """
        if not cius:
            return cius
            
        contents = [ciu.content for ciu in cius]
        embeddings = self.model.encode(contents).tolist()
        
        for ciu, emb in zip(cius, embeddings):
            ciu.embedding = emb
            
        return cius
