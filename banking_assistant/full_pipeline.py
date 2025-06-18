import joblib
import os
from pathlib import Path
import faiss

from .correcter import Correcter
from .embedder import Embedder
from .llm import LLM

class FullPipeline:
    def __init__(
            self,
            artifacts_path:Path,
            indexes_path:Path,
            device = "cuda"
    ):
        correcter_path = os.path.join(artifacts_path,"correcter")
        self.correcter = Correcter(correcter_path)
        classifier_path = os.path.join(artifacts_path,"classifier")
        with open(os.path.join(classifier_path,"classifier.pkl"),"rb") as pipe_fp:
            self.classifier = joblib.load(pipe_fp)
        
        embedder_path = os.path.join(artifacts_path,"embedder")
        self.embedder = Embedder(embedder_path,device=device)

        llm_path = os.path.join(artifacts_path,"llm")
        self.llm = LLM(llm_path,device)
        self.indexes_path = indexes_path

    def pipe(self,text):

        corrected_text = self.correcter.correct([text])
        intent = self.classifier.predict(corrected_text)[0]
        index = faiss.read_index(os.path.join(self.indexes_path,intent,"index.index"))
        docs_path = os.path.join(self.indexes_path,intent,"docs.pkl")
        with open(docs_path,"rb") as docs_fp:
            docs = joblib.load(docs_fp)
        
        corrected_text_embeds = self.embedder.embed(corrected_text).cpu().numpy()
        _, I = index.search(corrected_text_embeds, k=3)
        retrieved_docs = [docs[i] for i in I[0]]
        return self.llm.generate_and_stream(corrected_text,retrieved_docs)

if __name__ == "__main__":
    artifacts_path = Path("./artifacts")
    indexes_path = Path("./indexes")
    pipe = FullPipeline(artifacts_path,indexes_path)

    query = "How can i delete my accont"
    for token in pipe.pipe(query):
        print(token,end="",flush=True)