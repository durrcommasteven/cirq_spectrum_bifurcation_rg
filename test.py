import cirq

from cirq_bifurcation_rg import *

hamiltonian_terms = [0.5**i * cirq.X(q) for i, q in enumerate(cirq.LineQubit.range(25))]

hamiltonian_terms += [0.8**i * cirq.Z(q1) * cirq.Z(q2) for i, q1, q2 in zip(range(25), cirq.LineQubit.range(25), cirq.LineQubit.range(1, 26))]

hamiltonian = sum(hamiltonian_terms)

sbrg = BifurcationRG(hamiltonian)

sbrg.diagonalize()

print(sbrg.hamiltonian)
print(sbrg.residual_terms)
print(sbrg.effective_terms)
print(sbrg.hamiltonian == hamiltonian)

clifford_diagonalized_ham = sbrg.apply_clifford_diagonalization(hamiltonian)

print(clifford_diagonalized_ham)

easy_energy_variance(
		hamiltonian_generator= lambda : hamiltonian,
		reps = 2,
		growth_rate = 2,
)