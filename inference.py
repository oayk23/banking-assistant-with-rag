from banking_assistant.full_pipeline import FullPipeline
from pathlib import Path

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--query",type=str,required=True,help="Question you want to ask the assistant.")

    args = parser.parse_args()

    pipeline = FullPipeline(Path("artifacts"),Path("indexes"),"cuda")

    for token in pipeline.pipe(args.query):
        print(token,end="",flush=True)