from drsa.functions import event_time_loss, event_rate_loss
from drsa.model import DRSA
import torch
import torch.nn as nn
import torch.optim as optim

# generating random data
batch_size, seq_len, n_features = (64, 25, 10)
def data_gen(batch_size, seq_len, n_features):
    samples = []
    for _ in range(batch_size):
        sample = torch.cat([torch.normal(mean=torch.arange(1., float(seq_len)+1)).unsqueeze(-1) for _ in range(n_features)], dim=-1)
        samples.append(sample.unsqueeze(0))
    return torch.cat(samples, dim=0)
data = data_gen(batch_size, seq_len, n_features)

# generating random embedding for each sequence
n_embeddings = 10
embedding_idx = torch.mul(
    torch.ones(batch_size, seq_len, 1),
    torch.randint(low=0, high=n_embeddings, size=(batch_size, 1, 1)),
)

# concatenating embeddings and features
X = torch.cat([embedding_idx, data], dim=-1)

# instantiating embedding parameters
embedding_size = 5
embeddings = torch.nn.Embedding(n_embeddings, embedding_size)

# instantiating model
model = DRSA(
    n_features=n_features + 1,  # +1 for the embeddings
    hidden_dim=2,
    n_layers=1,
    embeddings=[embeddings],
)

# defining training loop
def training_loop(X, optimizer, alpha, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X)

        # weighted average of survival analysis losses
        evt_loss = event_time_loss(preds)
        evr_loss = event_rate_loss(preds)
        loss = (alpha * evt_loss) + ((1 - alpha) * evr_loss)

        # updating parameters
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch: {epoch} - loss: {round(loss.item(), 4)}")

# running training loop
optimizer = optim.Adam(model.parameters())
training_loop(X, optimizer, alpha=0.5, epochs=1001)