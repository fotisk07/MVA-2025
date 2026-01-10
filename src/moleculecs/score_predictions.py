from pathlib import Path

import hydra
import pandas as pd
import sacrebleu
import torch
from bert_score import score as bert_score
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moleculecs.data_utils import (
    PreprocessedGraphDataset,
    collate_fn,
    load_descriptions_from_graphs,
    load_id2emb,
)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="score",
)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hydra_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ckpt_path = Path(cfg.model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    task = hydra.utils.instantiate(cfg.task)

    # Load train embeddings to infer dimension
    train_emb_dict = load_id2emb(cfg.train_emb_csv)
    train_ids = list(train_emb_dict.keys())
    train_text_embeddings = torch.stack([train_emb_dict[i] for i in train_ids])
    emb_dim = train_text_embeddings.size(1)

    model = hydra.utils.instantiate(cfg.model, out_dim=emb_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Loaded model from {ckpt_path}")

    val_ds = PreprocessedGraphDataset(cfg.val_graphs)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Ground-truth descriptions
    val_id2desc = load_descriptions_from_graphs(cfg.val_graphs)

    print("Running generation on validation set...")
    predictions = task.predict(
        model=model,
        test_loader=val_loader,
        train_text_embeddings=train_text_embeddings,
        train_ids=train_ids,
        train_id2desc=load_descriptions_from_graphs(cfg.train_graphs),
        device=device,
    )
    df = pd.DataFrame(predictions)
    hypotheses = df["description"].tolist()
    references = [val_id2desc[i] for i in df["ID"]]

    # --------------------------------------------------
    # BLEU-4
    # --------------------------------------------------
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    bleu_score = bleu.score

    # --------------------------------------------------
    # BERTScore (RoBERTa-base)
    # --------------------------------------------------
    P, R, F1 = bert_score(
        hypotheses,
        references,
        model_type="seyonec/ChemBERTa-zinc-base-v1",
        num_layers=6,
        lang="en",
        rescale_with_baseline=True,
    )
    bert_f1 = F1.mean().item()

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    print("\n===== LOCAL VALIDATION SCORES =====")
    print(f"BLEU-4       : {bleu_score:.2f}")
    print(f"BERTScore F1 : {bert_f1:.4f}")
    print("==================================")

    out_csv = hydra_dir / "val_predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved validation predictions to {out_csv}")


if __name__ == "__main__":
    main()
