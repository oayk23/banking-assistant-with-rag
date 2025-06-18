from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch
import torch.nn.functional as F
from pathlib import Path

class Embedder:
    def __init__(self,model_path:Path | str,device:str = "cuda"):
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def embed(self,texts):
        encoded_input = self.tokenizer(texts,padding=True,truncation=True,return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == "__main__":
    embedder_path = Path("./artifacts/embedder")
    embedder = Embedder(embedder_path)
    query = "This is a text."
    embedded_sentence = embedder.embed(query)
    print(embedded_sentence.shape)