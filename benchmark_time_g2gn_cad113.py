import numpy as np
from qoop.compilation.qsp import QuantumCompilation
from qoop.core.ansatz import g2gn, zxz_layer
from qoop.core.state import specific
from qoop.backend import constant
import time
for num_layers in range(1, 10):
    loss_valuess = []
    num_qubits = 5
    state = np.random.uniform(low = 0, high = 2*np.pi, size = 2**num_qubits)

    optimizers = ['sgd', 'adam', 'qng_fubini_study', 'qng_qfim', 'qng_adam']

    times = []
    num_steps = 5
    for optimizer in optimizers:
        start = time.time()
        compiler = QuantumCompilation(
            u = g2gn(num_qubits, num_layers),
            vdagger = specific(state).inverse(),
            optimizer = optimizer,
            metrics_func = [
                'loss_basic', 
                'compilation_trace_fidelities'
            ]
        )

        compiler.fit(
            num_steps = num_steps, 
            verbose=1
        )
        loss_valuess.append(compiler.metrics['loss_basic'])
        times.append(time.time() - start)
    np.savetxt(f'./data/time/cad113/times_{num_qubits}qubit_{num_layers}layer_{g2gn.__name__}.txt', np.round(np.array(times)/num_steps, 4))
    print(f'TIME_{num_qubits}qubit_{num_layers}layer_{g2gn.__name__}', np.round(np.array(times)/num_steps, 4))
