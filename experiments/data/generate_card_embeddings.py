from flax import nnx
import jax
import jax.numpy as jnp
import optax
import h5py

def train_with_model(model, dist, batch_size, train_steps):
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing

    @nnx.jit  # automatic state management for JAX transforms
    def train_step(model, optimizer, a1, s01, s11, a2, s02, s12):
        def loss_fn(model):
            embedding1 = model(a1)
            embedding2 = model(a2)
            dist_emb = dist(embedding1, embedding2)
            dist_s0 = dist(s01, s02)
            dist_s1 = dist(s11, s12)
            # jax.debug.print('dist_emb: {x}', x=dist_emb.mean())
            # jax.debug.print('dist_s0: {x}', x=dist_s0.mean())
            # jax.debug.print('dist_s1: {x}', x=dist_s1.mean())
            loss = nnx.relu(1-dist_s0/2)*(dist_s1-dist_emb)**2
            return loss.mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        # jax.debug.print('grads: {x}', x=grads)
        optimizer.update(grads)  # in-place updates

        return loss

    with h5py.File('dataset.hdf5', 'r') as f:
        s0 = f['s0'][:]  # type: ignore
        s1 = f['s1'][:]  # type: ignore
        card_ids = f['card_ids'][:] # type: ignore

    max_entries = jnp.maximum(jnp.max(jnp.abs(s0), 0), jnp.max(jnp.abs(s1), 0))  # type: ignore
    s0 = jnp.nan_to_num(s0/max_entries)
    s1 = jnp.nan_to_num(s1/max_entries)

    key = jax.random.key(0)

    losses = []
    for i in range(train_steps):
        key, split = jax.random.split(key)
        a1 = jax.random.choice(split, card_ids, (batch_size,1)) # type: ignore
        s11 = jax.random.choice(split, s0, (batch_size,)) # type: ignore
        s01 = jax.random.choice(split, s1, (batch_size,)) # type: ignore
        key, split = jax.random.split(key)
        a2 = jax.random.choice(split, card_ids, (batch_size,1)) # type: ignore
        s12 = jax.random.choice(split, s0, (batch_size,)) # type: ignore
        s02 = jax.random.choice(split, s1, (batch_size,)) # type: ignore

        loss = train_step(model, optimizer, a1, s01, s11, a2, s02, s12)
        losses.append(loss)
        print(f"{i}: {loss}")

    return losses
