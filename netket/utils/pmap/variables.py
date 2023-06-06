import jax

n_devices = len(jax.devices())
n_local_devices = jax.local_device_count()
rank = 0

n_nodes = len(jax.devices())//n_local_devices


