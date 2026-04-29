"""Generate BERT embeddings for molecular descriptions."""

import pickle

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@hydra.main(
    version_base=None,
    config_path="../conf/features",
    config_name="baseline_description_embeddings",
)
def main(cfg: DictConfig) -> None:
    # Configuration
    MAX_TOKEN_LENGTH = cfg.max_token_length
    data_path = cfg.data_path
    output_path = cfg.output_path

    # Load BERT model
    print("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")

    # Process each split
    for split in ["train", "validation"]:
        print(f"\nProcessing {split}...")

        # Load graphs from pkl file
        pkl_path = f"{data_path}/{split}_graphs.pkl"
        print(f"Loading from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs")

        # Generate embeddings
        ids = []
        embeddings = []

        for graph in tqdm(graphs, total=len(graphs)):
            # Get description from graph
            description = graph.description

            # Tokenize
            inputs = tokenizer(
                description,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKEN_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embedding
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            ids.append(graph.id)
            embeddings.append(embedding)

        # Save to CSV
        result = pd.DataFrame(
            {"ID": ids, "embedding": [",".join(map(str, emb)) for emb in embeddings]}
        )
        save_path = f"{output_path}/baseline_{split}_embeddings.csv"
        result.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
