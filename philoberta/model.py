from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class PHILOBERTA(nn.Module):
    def __init__(self, base_model: str = "bert-base-multilingual-cased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.tokenizer = BertTokenizer.from_pretrained(base_model)

        # Additional projection layer for temporal masking
        self.temporal_proj = nn.Linear(768, 768)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Apply temporal projection
        temporal_embed = self.temporal_proj(cls_output)
        return temporal_embed

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        self.eval()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                embeddings = self.forward(encoded.input_ids, encoded.attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def cross_similarity(
        self,
        greek_term: str,
        latin_term: str,
        greek_contexts: List[str],
        latin_contexts: List[str],
    ) -> float:
        """Compute cross-lingual similarity between Greek and Latin terms"""
        greek_embeds = self.get_embeddings(greek_contexts)
        latin_embeds = self.get_embeddings(latin_contexts)

        # Average embeddings
        greek_avg = greek_embeds.mean(axis=0)
        latin_avg = latin_embeds.mean(axis=0)

        # Compute angular similarity
        dot_product = np.dot(greek_avg, latin_avg)
        norms = np.linalg.norm(greek_avg) * np.linalg.norm(latin_avg)
        cos_sim = dot_product / norms

        # Convert to angular similarity
        angular_sim = 1 - (2 / np.pi) * np.arccos(np.clip(cos_sim, -1.0, 1.0))
        return float(angular_sim)
