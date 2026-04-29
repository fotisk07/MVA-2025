import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from class_model_gnn import StudentModel

BATCH_SIZE = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_epochs = 200


###############################
########## Data ###############
###############################

# Train Dataset
train_dataset = PPI(root="", split="train")
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# Val Dataset
val_dataset = PPI(root="", split="val")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# Test Dataset
test_dataset = PPI(root="", split="test")
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Number of features and classes
n_features, n_classes = train_dataset[0].x.shape[1], train_dataset[0].y.shape[1]

#############################
#############################


def evaluate(model, device, dataloader):
    score_list_batch = []

    model.eval()
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        output = model(batch.x, batch.edge_index)
        predict = np.where(output.detach().cpu().numpy() >= 0, 1, 0)
        score = f1_score(batch.y.cpu().numpy(), predict, average="micro")
        score_list_batch.append(score)

    return np.array(score_list_batch).mean()


def train(model, loss_fcn, device, optimizer, max_epochs, train_dataloader, val_dataloader):
    best_score = 0.0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        losses = []

        for train_batch in train_dataloader:
            optimizer.zero_grad()
            train_batch = train_batch.to(device)

            logits = model(train_batch.x, train_batch.edge_index)
            loss = loss_fcn(logits, train_batch.y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch + 1:04d} | Loss: {np.mean(losses):.4f}")

        if epoch % 5 == 0:
            val_f1 = evaluate(model, device, val_dataloader)
            print(f"Val F1: {val_f1:.4f}")

            if val_f1 > best_score:
                best_score = val_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # restore best model
    model.load_state_dict(best_state)


print("Device: ", device)

model = StudentModel().to(device)
loss_fcn = nn.BCEWithLogitsLoss()

### DEFINE OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

train(
    model,
    loss_fcn,
    device,
    optimizer,
    max_epochs,
    train_dataloader,
    val_dataloader,
)

torch.save(model.state_dict(), "model.pth")
