from pathlib import Path
import clip
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple, List
from tqdm import tqdm

class VisualSearch:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.image_vectors = None
        self.frame_indices = None
        
    def encode_video(self, video_path: Union[str, Path]) -> None:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.image_vectors = torch.zeros((frame_count, 512), device=self.device)
        self.frame_indices = []
        
        for i in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.image_vectors[i] = self.model.encode_image(image)
            self.frame_indices.append(i)
        cap.release()
        
    def encode_image(self, image_path: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        else:
            image = image_path
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(image_input)
            
    def encode_text(self, text: str) -> torch.Tensor:
        text_token = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(text_token)
    
    def search(self, 
               text_query: str = None,
               image_query: Union[str, Path, Image.Image] = None,
               weight_text: float = 0.5,
               top_k: int = 5
               ) -> Tuple[List[int], List[float]]:
        """
        Search using text, image, or both.
        
        Args:
            text_query: Optional text description
            image_query: Optional image query
            weight_text: Weight for text similarity (0 to 1)
            top_k: Number of results to return
        """
        if self.image_vectors is None:
            raise ValueError("No video has been encoded yet. Call encode_video first.")
        
        if text_query is None and image_query is None:
            raise ValueError("Provide at least one of text_query or image_query")
            
        similarities = torch.zeros_like(self.image_vectors[:, 0])
        
        if text_query is not None and image_query is not None:
            # Combine text and image similarities
            text_vector = self.encode_text(text_query)
            image_vector = self.encode_image(image_query)
            
            text_similarities = torch.cosine_similarity(self.image_vectors, text_vector)
            image_similarities = torch.cosine_similarity(self.image_vectors, image_vector)
            
            similarities = weight_text * text_similarities + (1 - weight_text) * image_similarities
        elif text_query is not None:
            query_vector = self.encode_text(text_query)
            similarities = torch.cosine_similarity(self.image_vectors, query_vector)
        else:
            query_vector = self.encode_image(image_query)
            similarities = torch.cosine_similarity(self.image_vectors, query_vector)
            
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(similarities)))
        return (
            [self.frame_indices[i] for i in top_indices.cpu().numpy()],
            top_similarities.cpu().numpy().tolist()
        )
    
    def get_frame(self, video_path: Union[str, Path], frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Could not extract frame {frame_idx}")