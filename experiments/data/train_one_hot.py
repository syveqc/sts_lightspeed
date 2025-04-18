from flax import nnx
import jax
import orbax.checkpoint as ocp
import jax.numpy as jnp
import numpy as np
import time as time
import sys
import os

class Model(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(371, 512, rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(512, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.2, rngs=rngs)
        self.linear2 = nnx.Linear(512, 512, rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(512, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.2, rngs=rngs)
        self.linear3 = nnx.Linear(512, 512, rngs=rngs)
        self.batch_norm3 = nnx.BatchNorm(512, rngs=rngs)
        self.dropout3 = nnx.Dropout(0.2, rngs=rngs)
        self.linear4 = nnx.Linear(512, 512, rngs=rngs)
        self.linear5 = nnx.Linear(512, 512, rngs=rngs)
        self.linear6 = nnx.Linear(512, dout, rngs=rngs)

    def __call__(self, x):
        x = jax.nn.one_hot(x.squeeze(1), 371)
        x = nnx.relu(self.linear1(x))
        x = self.dropout1(self.batch_norm1(x))
        x = nnx.relu(self.linear2(x))
        x = self.dropout2(self.batch_norm2(x))
        x = nnx.relu(self.linear3(x))
        x = self.dropout3(self.batch_norm3(x))
        x = nnx.relu(self.linear4(x))
        x = nnx.relu(self.linear5(x))
        return self.linear6(x)  # type: ignore

if __name__=='__main__':
    from generate_card_embeddings import train_with_model
    input_size = 20
    model = Model(1, input_size, rngs=nnx.Rngs(0))  # eager initialization
    dist = lambda x,y: jnp.linalg.norm(x-y, axis=1)
    batch_size = 2048
    train_steps = 10000
    learning_rate = 1e-3

    losses = train_with_model(model, dist, batch_size, train_steps, learning_rate)

    loss_dir = sys.argv[2]
    if loss_dir.endswith('/'):
        loss_dir = loss_dir[:-1]
    np.save(f"{loss_dir}/one_hot_{batch_size}_{input_size}_{learning_rate}_{int(time.time())}.npy", np.array(losses))  # type: ignore

    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    _, _, state = nnx.split(model, nnx.RngState, ...)

    checkpoint_dir = sys.argv[1]
    if checkpoint_dir.endswith('/'):
        checkpoint_dir = checkpoint_dir[:-1]

    checkpoint_location = f"{checkpoint_dir}/one_hot"
    if os.path.exists(checkpoint_location):
        print('checkpoint already exists, skipping...')
    else:
        checkpointer.save(checkpoint_location, state)
