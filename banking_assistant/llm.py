from transformers.generation.streamers import TextIteratorStreamer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import threading
import torch
from pathlib import Path

class LLM:
    def __init__(self,path,device = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.max_new_tokens = 2048
    
    def build_rag_prompt(self,prompt,retrieved_docs):
        
        context = "\n".join([f"- {doc}" for doc in retrieved_docs])
        system_prompt = f"""You are a helpful and polite banking assistant.

You MUST use the context below to answer the question.
You are NOT allowed to use any external knowledge or make assumptions.
If the answer is not in the context, say: "Sorry, I couldn't find enough information in the documents."

Use exact wording from the documents where possible.

Context:
{context}
"""
        user_prompt = f"{prompt}"
        messages = [{"role":"user","content":system_prompt},{"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False 
        )
        return text
    
    def generate_and_stream(self,query,retrieved_docs):
        prompt = self.build_rag_prompt(query,retrieved_docs)
        inputs = self.tokenizer(prompt,return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            #temperature=0.0,
            #top_p=1.0,
            repetition_penalty=1.1,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for token in streamer:
            yield token

if __name__ == "__main__":
    llm_path = Path("./artifacts/llm")
    llm = LLM(llm_path)
    query = "Who is the founder of Tesla Inc. ?"
    retrieved_docs = [
        "Tesla was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning as Tesla Motors. Its name is a tribute to inventor and electrical engineer Nikola Tesla. In February 2004, Elon Musk led Tesla's first funding round and became the company's chairman; in 2008, he was named chief executive officer. In 2008, the company began production of its first car model, the Roadster sports car, followed by the Model S sedan in 2012, the Model X SUV in 2015, the Model 3 sedan in 2017, the Model Y crossover in 2020, the Tesla Semi truck in 2022 and the Cybertruck pickup truck in 2023.",
        "The company was incorporated as Tesla Motors, Inc. on July 1, 2003, by Martin Eberhard and Marc Tarpenning."
    ]
    for token in llm.generate_and_stream(query,retrieved_docs):
        print(token,end="",flush=True)
