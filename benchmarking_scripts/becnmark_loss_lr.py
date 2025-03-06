import numpy as np
import sys
sys.path.insert(0, '..')
from qoop.compilation.qsp import QuantumCompilation
from qoop.core.ansatz import g2gn
from qoop.core.state import specific
from qoop.backend import constant
import time


num_qubits = 5
num_layers = 1
lrs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
for lr in lrs:

    state = np.random.uniform(low = 0, high = 2*np.pi, size = 2**num_qubits)

    optimizers = ['sgd', 'adam', 'qng_fubini_study', 'qng_qfim', 'qng_adam']
    constant.LEARNING_RATE = lr
    loss = []
    num_steps = 100
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
        loss.append(compiler.metrics['loss_basic'][-1])
    print(f'lr: {constant.LEARNING_RATE}')
    np.savetxt(f'../data/loss_lr/loss_5qubit_lr{lr}.txt', loss)