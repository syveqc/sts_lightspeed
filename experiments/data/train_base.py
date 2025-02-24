from flax import nnx
import orbax.checkpoint as ocp
import jax.numpy as jnp
from generate_card_embeddings import train_with_model
import numpy as np
import time as time
import sys

class Model(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, 512, rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(512, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.2, rngs=rngs)
        self.linear2 = nnx.Linear(512, 512, rngs=rngs)
        self.linear3 = nnx.Linear(512, 512, rngs=rngs)
        self.linear4 = nnx.Linear(512, 512, rngs=rngs)
        self.linear5 = nnx.Linear(512, 512, rngs=rngs)
        self.linear6 = nnx.Linear(512, dout, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = self.dropout1(self.batch_norm1(x))
        x = nnx.relu(self.linear2(x))
        x = nnx.relu(self.linear3(x))
        x = nnx.relu(self.linear4(x))
        x = nnx.relu(self.linear5(x))
        return self.linear6(x)  # type: ignore

input_size = 20
model = Model(1, input_size, rngs=nnx.Rngs(0))  # eager initialization
dist = lambda x,y: jnp.linalg.norm(x-y, axis=1)
batch_size = 2048
train_steps = 1000

losses = train_with_model(model, dist, batch_size, train_steps)

np.save(f"base_{batch_size}_{input_size}_{int(time.time())}.npy", np.array(losses))  # type: ignore

ckpt_dir = ocp.test_utils.erase_and_create_empty(sys.argv[1])
checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
_, _, state = nnx.split(model, nnx.RngState, ...)
checkpointer.save(ckpt_dir / 'base', state)
