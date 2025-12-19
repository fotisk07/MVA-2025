import torch
import torch.nn.functional as F


class GraphTextEmbeddingRetrievalBaseline:
    """
    Baseline task:
    - Graph → text embedding alignment
    - MSE loss in shared embedding space
    - Retrieval-based evaluation (nearest neighbor)
    """

    def __init__(self):
        pass

    def train_epoch(self, model, loader, optimizer, device):
        """
        One training epoch.
        """
        model.train()
        total_loss, total = 0.0, 0

        for graphs, text_emb in loader:
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)

            graph_emb = model(graphs)
            text_emb = F.normalize(text_emb, dim=-1)

            loss = F.mse_loss(graph_emb, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = graphs.num_graphs
            total_loss += loss.item() * bs
            total += bs

        return total_loss / total

    @torch.no_grad()
    def evaluate(self, model, loader, device):
        """
        Retrieval-based evaluation.
        """
        model.eval()

        all_graph_emb, all_text_emb = [], []

        for graphs, text_emb in loader:
            graphs = graphs.to(device)
            text_emb = text_emb.to(device)

            all_graph_emb.append(model(graphs))
            all_text_emb.append(F.normalize(text_emb, dim=-1))

        graph_emb = torch.cat(all_graph_emb, dim=0)
        text_emb = torch.cat(all_text_emb, dim=0)

        # Cosine similarity via dot product (embeddings are normalized)
        sims = text_emb @ graph_emb.t()
        ranks = sims.argsort(dim=-1, descending=True)

        N = text_emb.size(0)
        correct = torch.arange(N, device=sims.device)

        pos = (ranks == correct.unsqueeze(1)).nonzero()[:, 1] + 1

        results = {
            "MRR": (1.0 / pos.float()).mean().item(),
        }

        for k in (1, 5, 10):
            results[f"R@{k}"] = (pos <= k).float().mean().item()

        return results

    @torch.no_grad()
    def predict(
        self,
        model,
        test_loader,
        train_text_embeddings,
        train_ids,
        device,
    ):
        """
        Predict by nearest-neighbor retrieval.

        Args:
            model: trained graph encoder
            test_loader: DataLoader yielding test graphs
            train_text_embeddings: (N_train, D) tensor
            train_ids: list of train IDs aligned with embeddings
            device: torch device

        Returns:
            list of retrieved train IDs (one per test sample)
        """
        model.eval()

        train_text_embeddings = F.normalize(train_text_embeddings.to(device), dim=-1)

        retrieved_ids = []

        for graphs in test_loader:
            graphs = graphs.to(device)
            test_graph_emb = model(graphs)

            sims = test_graph_emb @ train_text_embeddings.t()
            nn_indices = sims.argmax(dim=-1)

            for idx in nn_indices.tolist():
                retrieved_ids.append(train_ids[idx])

        return retrieved_ids
