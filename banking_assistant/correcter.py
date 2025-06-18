from transformers.models.bart import BartForConditionalGeneration,BartTokenizer
from typing import List
from pathlib import Path

class Correcter:
    def __init__(self,path:Path|str,device:str = "cuda"):
        self.model = BartForConditionalGeneration.from_pretrained(path).to(device) # type: ignore
        self.tokenizer = BartTokenizer.from_pretrained(path)
    
    def correct(self,texts:List[str]):
        assert isinstance(texts,list), "Texts parameter should be a list!"
        input_ids = self.tokenizer.batch_encode_plus(texts,return_tensors="pt",padding=True,truncation=True).to(self.model.device)
        outputs = self.model.generate(
            **input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True,
        )
        corrected_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected_text

if __name__ == "__main__":
    correcter_path = Path("./artifacts/correcter")
    correcter = Correcter(correcter_path)
    text = ["Thisis a bda sentencee"]
    corrected_text = correcter.correct(text)
    print(corrected_text)
