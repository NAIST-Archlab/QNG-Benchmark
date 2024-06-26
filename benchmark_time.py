import numpy as np
from qoop.compilation.qsp import QuantumCompilation
from qoop.core.ansatz import g2gn
from qoop.core.state import specific
from qoop.backend import constant
import time

num_qubits = 2
for num_layers in range(2, 10):
    
    state = np.random.uniform(low = 0, high = 2*np.pi, size = 2**num_qubits)

    optimizers = ['sgd', 'adam', 'qng_fubini_study', 'qng_qfim', 'qng_adam']

    times = []
    num_steps = 50
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
        )
        #plot figure
        times.append(time.time() - start)
    print(f'num_layers: {num_layers}')
    np.savetxt(f'./data/times/times_{num_qubits}qubit_{num_layers}layer.txt', np.round(np.array(times)/num_steps, 4))
    print(np.round(np.array(times)/num_steps, 4))
