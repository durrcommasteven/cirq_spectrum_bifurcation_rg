#import tensorflow as tf
#import tensorflow_quantum as tfq
from typing import Callable, List, Optional, Tuple, Union
import cirq
import copy
#from cirq import protocols, qis, value
#import sympy
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy import stats


class BifurcationRG:
	def __init__(
			self,
			hamiltonian: cirq.PauliSum,
			growth_rate: int = 2,
	):
		"""
		Hamiltonian contains all the terms, and all of the operators
		these are in the form of lists of terms

		hamiltonian.operators : a list of operators
		each operator is represented by a tuple (immutable, so we can use it as a dict key)

		eg: ident ident ident Sx Sx ident
		=> (0,0,0,1,1,0)

		hamiltonian.coefficients = a list of floats, one for each nonzero operator
		"""

		# First map the hamiltonian to LineQubits
		self.original_qubits = hamiltonian.qubits
		self.line_qubits = cirq.LineQubit.range(len(self.original_qubits))

		self.hamiltonian = hamiltonian.with_qubits(*self.line_qubits)
		self.effective_terms = cirq.PauliSum()
		self.residual_terms = cirq.PauliSum()

		self.growth_rate = growth_rate

		# number of spins in use
		self.length = len(self.line_qubits)

		# a list of the clifford rotations, the schrieffer-wolff rotations, and
		# the number of diagonalization steps already enacted, n
		self.clifford_rotations = []
		self.schrieffer_wolff_transformations = []
		self.thouless_parameters = []
		self.N = 0

		self.update_res_eff_hams()

	def clifford_rotate(
			self,
			leading_residual_term: cirq.PauliString,
			target_position: int,
	) -> cirq.PauliString:
		""" Author: P.S.
		Update the list of operators and coefficients to correspond to the rotated version
		Append clifford Rotation R to the list
		Inputs
		------
			op_index: (int) This is the index corresponding to the largest h, which we want to bring to the form [0..0]3[0..0]
			target_position: (int) denotes the location of the sigma operator which we want to rotate so that it is 0.
		Returns
		------
			None
		"""
		# print(self.hamiltonian.operators[op_index], target_position)
		assert self.is_leading_residual_term_of_correct_form(
			leading_residual_term,
			target_position,
		), ("The term to be rotated is not of the form [lambda][mu] with lambdas being 0's or 3's only", leading_residual_term)

		# Obtain the necessary clifford rotation
		clifford = self.get_clifford_operations(
			leading_residual_term,
			target_position,
		)

		assert type(clifford) == list, f"clifford should be a list of suboperations. Recieved {clifford}"

		print("heres the term and clifford")
		#print(target_position)
		#print(leading_residual_term)
		print("this is the clifford", clifford)

		# record the clifford rotation used
		self.clifford_rotations.append(clifford)

		length_before_clifford = len(self.hamiltonian)

		# Apply the clifford rotation to all terms
		self.hamiltonian = self.conjugate_with_clifford(
			pauli_object=self.hamiltonian,
			clifford=clifford,
		)
		#assert len(self.hamiltonian) == length_before_clifford, "term number does not match"

		for term in self.residual_terms:
			for i in range(self.N):
				assert term.get(cirq.LineQubit(i), None) in {None, cirq.Z}, (
					"before rotation residual terms has some bad term: ", term, " self.N = ", self.N)

		self.residual_terms = self.conjugate_with_clifford(
			pauli_object=self.residual_terms,
			clifford=clifford,
		)

		everbad = False
		for term in self.residual_terms:
			for i in range(self.N):
				goodbool = term.get(cirq.LineQubit(i), None) in {None, cirq.Z}
				if not goodbool:
					print("residual terms has some bad term: ", term, " self.N = ", self.N)
					everbad=True

		assert not everbad

		self.effective_terms = self.conjugate_with_clifford(
			pauli_object=self.effective_terms,
			clifford=clifford,
		)

		conjugated_leading_residual_term = self.conjugate_with_clifford(
			pauli_object=leading_residual_term,
			clifford=clifford,
		)

		#print(conjugated_leading_residual_term)
		assert conjugated_leading_residual_term.equal_up_to_coefficient(
			cirq.Z(cirq.LineQubit(target_position))
		), "Clifford does not appropriately rotate the leading residual term"

		return conjugated_leading_residual_term

	def is_leading_residual_term_of_correct_form(
			self,
			leading_term: cirq.PauliString,
			target_position: int
	) -> bool:
		""" Author: P.S.
		Checks if the term to be brought to the form [0..0]3[0..0] is of the form
		[lambda][mu], whith lambdas being only 0s and 3s

		Parameters
		----------
		op_index : int
			the index of the operator
		target_position : dictionary
			the intended location at which we want there to be a 3


		Returns
		-------
		is_correct_type: Bool
			true if it is of the correct form, else false.
		"""

		# print("leading order op", operator_tuple, self.hamiltonian.coefficients[op_index])

		#print("some problem here", leading_term, target_position)

		for qubit, operation in sorted(leading_term.items(), key=lambda x: x[0]):

			#print(qubit.x, target_position)

			if qubit.x >= target_position:
				return True

			if operation in {cirq.X, cirq.Y}:
				return False

		return True

	def c4_rotation(
			self,
			pauli_string: cirq.PauliString,
	) -> cirq.PauliSum:
		"""

		:param pauli_string:
		:return:
		"""
		return (1 + 1j * pauli_string) / np.sqrt(2)

	def conjugate_transpose(
			self,
			pauli_object: Union[cirq.PauliString, cirq.PauliSum]
	) -> Union[cirq.PauliString, cirq.PauliSum]:
		"""

		:param pauli_object:
		:return:
		"""
		if type(pauli_object) == cirq.PauliString:
			pauli_sum = cirq.PauliSum.from_pauli_strings([pauli_object])
		else:
			assert type(pauli_object) == cirq.PauliSum, "pauli_object must either be PauliSum or PauliString"
			pauli_sum = pauli_object

		conjugate_transpose_pauli_sum = cirq.PauliSum()
		for term in pauli_sum:
			coefficient = term.coefficient

			conjugate_transpose_pauli_sum -= 1j * np.imag(coefficient) * term / coefficient
			conjugate_transpose_pauli_sum += np.real(coefficient) * term / coefficient

		if type(pauli_object) == cirq.PauliString:
			return next(iter(conjugate_transpose_pauli_sum))

		return conjugate_transpose_pauli_sum

	def conjugate_with_clifford(
			self,
			pauli_object: Union[cirq.PauliString, cirq.PauliSum],
			clifford: List[Union[cirq.PauliSum, cirq.GateOperation]],
	) -> Union[cirq.PauliString, cirq.PauliSum]:
		"""
		The final list of all cliffords is a list of sub-lists.
		Each sublist contains operations listed in reverse order (to match the notation of cirq)
		This function applies the sub-list operations.

		Args:
			pauli_object: The object we'll be conjugating
			clifford: the List of PauliSums or Gate Operations which we'll be applying
				to the pauli object

		Returns:

		"""
		# TODO: I know this is suboptimal, byut for some reason conjugated_by isnt working for me

		if type(pauli_object) == cirq.PauliString:
			pauli_sum = cirq.PauliSum.from_pauli_strings([pauli_object])
		else:
			assert type(pauli_object) == cirq.PauliSum, "pauli_object must either be PauliSum or PauliString"
			pauli_sum = pauli_object

		if not bool(pauli_object):
			return cirq.PauliSum(cirq.LinearDict({}))

		rotated_pauli_sum = pauli_sum.copy()

		# We do not reverse the order to match the `conjugated_by` notation
		for operation in clifford[::-1]:

			if type(operation) == cirq.PauliSum:

				operation_conjugate = self.conjugate_transpose(operation)
				rotated_pauli_sum = operation_conjugate * rotated_pauli_sum * operation

			else:
				assert type(operation) == cirq.GateOperation

				rotated_pauli_sum = sum(
					[term.conjugated_by(operation) for term in rotated_pauli_sum]
				)

			#print(pauli_object, bool(pauli_object))
			#print(rotated_pauli_sum)
			#print(pauli_sum)

			if len(rotated_pauli_sum) != len(pauli_sum):
				pass
				#assert len(rotated_pauli_sum) == len(pauli_sum), (
				#	"error in clifford rotation: number of pauli strings has changed")

		if type(pauli_object) == cirq.PauliString:
			return next(iter(rotated_pauli_sum))

		return rotated_pauli_sum

	def get_clifford_operations(
			self,
			leading_residual_term: cirq.PauliString,
			target_position: int,
	) -> List[Union[cirq.PauliSum, cirq.GateOperation]]:
		"""
		Go through the procedure of finding the clifford operation to rotate
		the leading residual term of the hamiltonian.

		"""

		unit_leading_residual_term = leading_residual_term / leading_residual_term.coefficient

		# appropriate leading terms here have terms after or at the intent position

		clifford = []

		# Note below we assume that operations are returned in order of their sorted qubits

		# Check whether X or Y is present after or at the intent qubit position

		first_x_or_y_qubit_operation_tuple = None
		z_qubit_operation_tuples = []

		for qubit, operation in sorted(unit_leading_residual_term.items()):
			if qubit.x >= target_position:
				if operation in {cirq.X, cirq.Y} and first_x_or_y_qubit_operation_tuple is None:
					first_x_or_y_qubit_operation_tuple = (qubit, operation)
				#else:
				#	z_qubit_operation_tuples.append(
				#		(qubit, operation)
				#	)
			if operation == cirq.Z:
				z_qubit_operation_tuples.append(
					(qubit, operation)
				)

			#elif operation == cirq.Z:
			#	z_qubit_operation_tuples.append(
			#		(qubit, operation)
			#	)

		assert (first_x_or_y_qubit_operation_tuple is not None) or z_qubit_operation_tuples, (
			"Leading term is not an appropriate residual term"
		)

		if first_x_or_y_qubit_operation_tuple is not None:
			if first_x_or_y_qubit_operation_tuple[1] == cirq.X:
				# Rc4(- \sigma^([\lambda] 2 [\nu]))
				# Then swap to intent qubit position
				# TODO: figure out an elegant way to implement the c4 rotation gate
				#   For now, just use multiplication
				new_pauli_string = -1 * (-1j * cirq.Z(first_x_or_y_qubit_operation_tuple[0])) * unit_leading_residual_term
				clifford.append(self.c4_rotation(new_pauli_string))

				if first_x_or_y_qubit_operation_tuple[0].x != target_position:
					swap = cirq.SWAP(
						first_x_or_y_qubit_operation_tuple[0], cirq.LineQubit(target_position)
					)

					clifford.append(swap)

			elif first_x_or_y_qubit_operation_tuple[1] == cirq.Y:
				# Rc4( \sigma^([\lambda] 1 [\nu]))
				# Then swap to intent qubit position
				new_pauli_string = (1j * cirq.Z(first_x_or_y_qubit_operation_tuple[0])) * unit_leading_residual_term
				clifford.append(self.c4_rotation(new_pauli_string))

				if first_x_or_y_qubit_operation_tuple[0].x != target_position:
					swap = cirq.SWAP(
						first_x_or_y_qubit_operation_tuple[0], cirq.LineQubit(target_position)
					)

					clifford.append(swap)

		else:
			# There is one or multiple Z's
			if len(z_qubit_operation_tuples) == 1:
				print("there is one z", len(z_qubit_operation_tuples), z_qubit_operation_tuples)
				# Swap to place this at the intent qubit position

				if z_qubit_operation_tuples[0][0].x != target_position:

					swap = cirq.SWAP(
						z_qubit_operation_tuples[0][0], cirq.LineQubit(target_position)
					)

					clifford.append(swap)

			else:
				# There are multiple Zs
				print("there are multiple zs")
				z_positions = {qubit_operation_tuple[0].x for qubit_operation_tuple in z_qubit_operation_tuples}

				if target_position in z_positions:
					# Perform double Rc4 on \sigma^([\lambda] 3 [\nu])
					# first Rc4( \sigma^([\lambda] 2 [\nu]))
					# then Rc4( -\sigma^([0...] 2 [0...]))
					first_new_pauli_string = (1j * cirq.X(cirq.LineQubit(target_position))) * unit_leading_residual_term
					second_new_pauli_string = -1 * cirq.Y(cirq.LineQubit(target_position))

					clifford.append(self.c4_rotation(first_new_pauli_string))
					clifford.append(self.c4_rotation(second_new_pauli_string))

				else:
					# Apply these Rc4 operations on the final 3 present (here \nu are at most zeros)
					# Perform double Rc4 on \sigma^([\lambda] 3 [\nu])
					# first Rc4( \sigma^([\lambda] 2 [\nu]))
					# then Rc4( -\sigma^([0...] 2 [0...]))
					# finally swap the Z here to the intent qubit position

					first_new_pauli_string = (1j * cirq.X(z_qubit_operation_tuples[-1][0])) * unit_leading_residual_term
					second_new_pauli_string = -1 * cirq.Y(z_qubit_operation_tuples[-1][0])

					clifford.append(self.c4_rotation(first_new_pauli_string))
					clifford.append(self.c4_rotation(second_new_pauli_string))

					#(z_qubit_operation_tuples[-1][0], cirq.LineQubit(target_position))

					print(z_qubit_operation_tuples)

					swap = cirq.SWAP(
						z_qubit_operation_tuples[-1][0], cirq.LineQubit(target_position)
					)

					clifford.append(swap)

		return clifford[::-1]

	def is_effective(
			self,
			term: cirq.PauliString,
	) -> bool:
		"""
		Argument: op
			an operator tuple
		returns: Bool
			whether the operator is diagonalized

		this function tests whether a term is
		ready to be added to the residual hamiltonian

		we determine a term belongs in the residual Ham if its first self.N
		elements are either 3 or 0, and all remaining terms are 0

		IS THIS TRUE?
		"""
		#if self.N == 0:
		#	return False

		for qubit, operation in term.items():
			if qubit.x <= self.N:
				if operation in {cirq.X, cirq.Y}:
					return False
			else:
				return False

		return True

	def split_up_operators(
			self,
			term: cirq.PauliString
	) -> Tuple[cirq.PauliSum, cirq.PauliSum]:
		"""
		Argument: term
			a tuple indicating an operator corresponding to H0, this function identifies
			the operators currently used in the hamiltonian which commute (Delta), and
			those which anticommute (Sigma) with H0

			We assert that op is not the identity. This case should be taken care of
			outside of this function

		Returns: Delta_ops, Sigma_ops
			the list of operators which do commute with op
			followed by a list of the operators which anticommute
		"""
		# First check the op is of the correct form
		assert list(term.values()) == [cirq.Z], "op must contain one and only one Pauli Z operator"

		intent_qubit = next(iter(term.keys()))

		commuting_terms = cirq.PauliSum()
		anticommuting_terms = cirq.PauliSum()

		for term in self.hamiltonian:

			intent_qubit_op = term.get(intent_qubit)

			if intent_qubit_op is None or intent_qubit_op == cirq.Z:
				# commutes
				commuting_terms += term
			else:
				# anticommutes
				anticommuting_terms += term

		return commuting_terms, anticommuting_terms

	def schrieffer_wolff_rotate(
			self,
			h0_term: cirq.PauliString,
	):
		"""
		Argument: term
				the leading order operator, already appropriately rotated by a clifford rotation

		given the leading order operator's index, op_index, enact schrieffer wolff,
		modify the hamiltonian accordingly, and record the transformation.

		Take note of new terms that arise, and add them to the hamiltonian appropriately
		(a branching cutoff may also be applied)

		op is assumed not to be the identity, since the relevant rotation would be trivial

		following the paper's notation, we indicate commuting operators by Delta
		and indicate anticommuting operators by Sigma

		record the new thouless parameter
		"""

		# First check the op is of the correct form
		assert list(h0_term.values()) == [cirq.Z], "op must contain one and only one Pauli Z operator"

		delta_terms, sigma_terms = self.split_up_operators(h0_term)

		sigma_coefficients = np.array([sigma_term.coefficient for sigma_term in sigma_terms])

		"""
		record the new thouless parameter
		"""
		self.thouless_parameters.append(
			np.mean(
				np.log(10 ** -10 + np.abs(sigma_coefficients) / np.abs(h0_term.coefficient))
			)
		)

		if not sigma_terms:
			# then we have no sigmas to worry about for this value of self.n
			# Do nothing

			self.schrieffer_wolff_transformations.append(
				(h0_term, cirq.PauliSum())
			)

			return

		"""
		The rotation itself is given by 

		(h_3 := op_coeff)

		S = exp(-H0 . Sigma / 2 h_3^2)

		therefore S can be uniquely identified by its log:
		-H0.Sigma / 2 h_3^2
		
		We record this operation by recording (H0, Sigma)
		"""

		self.schrieffer_wolff_transformations.append(
			(h0_term, sigma_terms)
		)

		"""
		now that we have recorded which transformation is being enacted, we apply
		the actual rotation (up to order 1/h_3)

		H0 + Delta + Sigma => H0 + Delta + H0.(Sigma^2)/(2*h_3**2) 
		"""

		# H0.(Sigma^2)/(2*h_3**2)

		h0_sigma_squared_terms = h0_term * sigma_terms ** 2 / (2 * h0_term.coefficient ** 2)

		"""
		We only wish to take a limited number of the newly generated terms

		following the procedure of the paper, we select the largest magnitude num_keep

		(num_keep = self.growth_rate*len(Sigma_ops))

		new terms

		NOTE: Consider adding a caveat here -- really the only problematic terms here are
		those which are not already in the hamiltonian. We could add those terms already included 
		immediately, and deal with NEW terms afterwards. This isn't what's done in the original paper
		but it would likely improve the process (potentially to a very small degree though)
		"""

		sorted_terms = []
		for term in h0_sigma_squared_terms:
			# Check whether this term is already present in the Hamiltonian
			# If so, adding it contributes no complexity
			term_key = frozenset(zip(term.keys(), term.values()))
			if term_key in self.hamiltonian._linear_dict:
				self.hamiltonian += term

			# Otherwise, we'll add the largest terms
			sorted_terms.append(term)

		sorted_terms.sort(key=lambda x: abs(x.coefficient), reverse=True)

		num_keep = int(round(self.growth_rate * len(sigma_terms)))

		# Subtract sigma_terms and add the largest num_keep sorted_terms


		for sigma_term in sigma_terms:
			if sigma_term not in self.hamiltonian:
				assert False
		print("last sigma term", sigma_terms)
		self.hamiltonian -= sigma_terms



		# Add all pauli_strings which are already present
		remaining_terms = []
		for term in sorted_terms:
			if term in self.hamiltonian._linear_dict.keys():
				self.hamiltonian+=term
			else:
				remaining_terms.append(term)

		self.hamiltonian += cirq.PauliSum.from_pauli_strings(
			remaining_terms[:num_keep]
		)

	def update_res_eff_hams(self):
		"""
		Here we update the residual and effective hamiltonians

		first, the residual_ops and effective_ops need to be
		updated with the newly introduced terms from the schrieffer-wolff
		transformation
		"""

		self.effective_terms = cirq.PauliSum()
		self.residual_terms = cirq.PauliSum()

		for term in self.hamiltonian:
			if self.is_effective(term):
				self.effective_terms += term
			else:
				self.residual_terms += term

	def get_leading_residual_term(self) -> cirq.PauliString:
		"""

		"""
		assert self.residual_terms, "residual_terms should not be empty"

		"""
		if self.N=0, we are just beginning, and the max residual term is the max term
		"""
		if self.N == 0:
			return max(self.hamiltonian, key=lambda x: abs(x.coefficient))

		return max(self.residual_terms, key=lambda x: abs(x.coefficient))

		"""leading_residual_term = None

		for term in self.residual_terms:

			if leading_residual_term is None:
				leading_residual_term = term

			elif abs(term.coefficient) > abs(leading_residual_term.coefficient):
				leading_residual_term = term

		return leading_residual_term"""

	def apply_spectrum_bifurcation(self):
		"""
		Apply the actual spectrum bifurcation implementation

		1) identify the leading order residual term

		2) clifford rotate

		3) Schrieffer-Wolff rotate

		4) update residual and effective hamiltonians

		5) increase self.n by 1
		"""

		# Check whether we've run out of residual terms
		if not self.residual_terms:
			self.N = self.length
			return

		assert self.N < self.length, "Spectrum Bifurcation already complete"

		print("were here")

		max_residual_term = self.get_leading_residual_term()

		print("max_residual_term", max_residual_term)
		print("N", self.N)
		rotated_max_residual_term = self.clifford_rotate(
			max_residual_term,
			self.N,
		)

		self.schrieffer_wolff_rotate(
			rotated_max_residual_term
		)

		self.update_res_eff_hams()
		self.N += 1

	def diagonalize(self):
		"""
		apply SBRG until the hamiltonian is fully diagonalized
		"""
		op_num = len(self.clifford_rotations)

		print("diagonalizing")

		assert op_num == len(
			self.schrieffer_wolff_transformations), f"lens dont match: {op_num}, {len(self.clifford_rotations)}"

		assert self.N < self.length, "Spectrum Bifurcation already complete"

		while self.N < self.length:
			self.apply_spectrum_bifurcation()

	def give_energy(self, liom_pattern):
		"""
		take a liom pattern and output the corresponding energies
		"""
		assert len(liom_pattern) == self.length, "incorrect liom lengths"

		implied_state = [(1, 0) if el == 1 else (0, 1) for el in liom_pattern]
		qubit_map = dict(zip(self.line_qubits, range(self.length)))

		return self.hamiltonian.expectation_from_wavefunction(
			state=implied_state,
			qubit_map=qubit_map,
		)

	def give_cliffords_with_preserved_locality(
			self,
			cliffords_list: List[Union[cirq.PauliSum, cirq.GateOperation]],
	) ->  List[Union[cirq.PauliSum, cirq.GateOperation]]:
		"""

		:param cliffords_list:
		:return:
		"""
		swaps = []
		non_swap_cliffords = []

		for idx, clifford in enumerate(cliffords_list):
			current_swaps = []
			current_c4_rotations = []
			for operation in clifford:
				if type(operation) == cirq.GateOperation and operation.gate == cirq.SWAP:
					current_swaps.append(
						operation
					)
				else:
					current_c4_rotations.append(
						operation
					)

			swaps.append(current_swaps)
			non_swap_cliffords.append(current_c4_rotations)

		localized_cliffords_list = []
		for i, c4_rotations in enumerate(non_swap_cliffords):
			for j in range(i):
				swap = swaps[j]

				c4_rotations = list(
					map(
						lambda pauli_sum: self.conjugate_with_clifford(
							pauli_object=pauli_sum,
							clifford=swap,
						),
						c4_rotations
					)
				)

			localized_cliffords_list.append(c4_rotations)

		return localized_cliffords_list

	def invert_cliffords(
			self,
			cliffords_list : List[List[Union[cirq.PauliSum, cirq.GateOperation]]] = None,
	) -> Optional[List[List[Union[cirq.PauliSum, cirq.GateOperation]]]]:
		"""

		:param cliffords_list:
		:return:
		"""
		inverted_cliffords_list = []

		for clifford in cliffords_list[::-1]:
			reversed_clifford = []

			for operation in clifford[::-1]:
				if type(operation) == cirq.PauliSum:
					reversed_clifford.append(self.conjugate_transpose(operation))
				else:
					reversed_clifford.append(operation)

			inverted_cliffords_list.append(reversed_clifford)

		return inverted_cliffords_list

	def apply_clifford_diagonalization(
			self,
			hamiltonian: cirq.PauliSum,
			preserve_locality: bool = False,
			cliffords_list: Optional[List[List[Union[cirq.PauliSum, cirq.GateOperation]]]] = None,
	) -> cirq.PauliSum:
		"""
		Assuming that a diagonalization has already been computed,
		apply it to a new hamiltonian, ham

		Args:
			hamiltonian: a hamiltonian instance

			clifford_only: Bool. whether we only apply clifford rotations (ignoring schrieffer-wolff)

			preserve_location: Bool. If true, location is preserved in clifford rotations

		Returns:
			ham: another instance of a hamitlonian,
				this time with the diagonalization procedure applied
		"""

		op_num = len(self.clifford_rotations)
		#print(op_num, len(self.schrieffer_wolff_transformations))
		assert op_num == len(self.schrieffer_wolff_transformations)

		assert len(hamiltonian.qubits) <= self.length, "hamiltonian lengths do not match"

		rotated_hamiltonian = hamiltonian.copy()

		if cliffords_list is None:
			cliffords_to_apply = copy.deepcopy(self.clifford_rotations)
		else:
			cliffords_to_apply = cliffords_list

		if preserve_locality:
			# Then we remove swaps and modify c4 rotations accordingly
			cliffords_to_apply = self.give_cliffords_with_preserved_locality(cliffords_to_apply)

		for clifford in cliffords_to_apply:
			rotated_hamiltonian = self.conjugate_with_clifford(
				pauli_object=rotated_hamiltonian,
				clifford=clifford,
			)

			assert len(rotated_hamiltonian) == len(hamiltonian), (
				"error in clifford rotation: number of pauli strings has changed")

		return rotated_hamiltonian

	def apply_clifford_and_sw_diagonalization(
			self,
			pauli_object: Union[cirq.PauliSum, cirq.PauliString],
			order_cutoff: Optional[int] = 2,
			sw_cutoff: Optional[Union[int, float]] = None,
			cliffords_list: Optional[List[List[Union[cirq.PauliSum, cirq.GateOperation]]]] = None,
			sw_list: Optional[List[Tuple[cirq.PauliString, cirq.PauliSum]]] = None,
			maintain_positions: bool = False,
	) -> cirq.PauliSum:
		"""
		Here we apply both clifford and schrieffer wolff transformations to diagonalize a term.

		When we apply SW rotations, we generate terms up to order 4 in |Sigma|/|H0|. order_cutoff indicates
		what order we should keep here. None indicates that we keep all orders

		the sw list we use rotates using terms of order |Sigma|/|H0|. Ideally the magnitude of this small parameter
		drops with SBRG iterations. sw_cutoff tells us how many of these sw rotations to perform.
		None -> apply all
		an integer tells us the number to apply (keeping the nth largest of these)
		a float in (0, 1) tells us the fraction to apply

		Args:
			hamiltonian:
			order_cutoff:
			sw_cutoff:
			cliffords_list:
			sw_list:

		Returns:

		"""

		numerical_cutoff = 10**-15

		if cliffords_list is None:
			cliffords_to_apply = copy.deepcopy(self.clifford_rotations)
		else:
			cliffords_to_apply = cliffords_list

		if sw_list is None:
			sw_to_apply = copy.deepcopy(self.schrieffer_wolff_transformations)
		else:
			sw_to_apply = sw_list

		# select the sw_transofrmations
		if sw_cutoff is None:
			sw_count = len(sw_to_apply)
		elif 0 <= sw_cutoff < 1:
			sw_count = int(round(sw_cutoff*len(sw_to_apply)))
		else:
			assert int(sw_cutoff) == sw_cutoff, "sw_to_apply not of correct form"
			sw_count = int(sw_cutoff)

		for el in sw_to_apply:
			print(el)
			print(len(el))
			print("here we are")
			print(sigma_h0_ratio(el[0], el[1]))

			print("------------------")

		if sw_count>0:
			indices_and_sw = sorted(
				list(enumerate(sw_to_apply)),
				key=lambda idx_h0_sigma: sigma_h0_ratio(idx_h0_sigma[-1][0], idx_h0_sigma[-1][1]),
				reverse=True
			)[:sw_count]

			indices, _ = zip(*indices_and_sw)

			sw_to_apply = [sw if i in indices else None for i, sw in enumerate(sw_to_apply)]
		else:
			sw_to_apply = [None for sw in sw_to_apply]

		# combine cliffords and sw

		rotated_pauli_object = pauli_object.copy()

		for clifford, sw in zip(cliffords_to_apply, sw_to_apply):
			rotated_pauli_object = self.conjugate_with_clifford(
				pauli_object=rotated_pauli_object,
				clifford=clifford,
			)

			if sw is not None:
				h0, sigma = sw

				rotated_pauli_object = sw_rotate(
					pauli_object=rotated_pauli_object,
					h0=h0,
					sigma=sigma,
					order_cutoff=order_cutoff
				)

			rotated_pauli_object = np.sum(
				[term for term in rotated_pauli_object if np.abs(term.coefficient)>numerical_cutoff]
			)

		if maintain_positions:
			# Then unrotate all swaps
			unrotated_qubits = unrotate_swaps(cliffords_to_apply, qubits=self.hamiltonian.qubits)

			old_new_mapping = dict(zip(self.hamiltonian.qubits, unrotated_qubits))

			new_qubits = [old_new_mapping[q] for q in rotated_pauli_object.qubits]

			rotated_pauli_object = rotated_pauli_object.with_qubits(*new_qubits)

		return rotated_pauli_object


def unrotate_swaps(
	cliffords_list: List[List[Union[cirq.PauliSum, cirq.GateOperation]]],
	qubits: List[cirq.Qid],
) -> List[cirq.Qid]:
	"""
	go through cliffords to identify the overal positional swaps which are occurring
	return a list of qubits such that by applying
	plaulisum.with_qubits(*qubits)

	Args:
		cliffords_list:
		qubits:

	Returns:

	"""
	swapped_qubits = [qubit for qubit in qubits]
	swapped_qubit_indices = {qubit: idx for idx, qubit in enumerate(swapped_qubits)}

	swap_cliffords = []

	for idx, clifford in enumerate(cliffords_list):
		current_swaps = []
		for operation in clifford:
			if type(operation) == cirq.GateOperation and operation.gate == cirq.SWAP:
				current_swaps.append(
					operation
				)

		swap_cliffords.append(current_swaps)

	for swaps in swap_cliffords:
		for swap in swaps:
			q1_idx = swapped_qubit_indices[swap.qubits[0]]
			q2_idx = swapped_qubit_indices[swap.qubits[1]]

			swapped_qubits[q1_idx], swapped_qubits[q2_idx] = swapped_qubits[q2_idx], swapped_qubits[q1_idx]

	return swapped_qubits








def sw_rotate(
		pauli_object,
		h0,
		sigma,
		order_cutoff,
) -> cirq.PauliSum:
	"""

	Args:
		pauli_object:
		h0:
		sigma:

	Returns:

	"""
	output_terms = cirq.PauliSum()+pauli_object

	if order_cutoff >= 1:
		# include order 1 terms
		order_1 = -(pauli_object*h0*sigma + sigma*h0*pauli_object)/(2 * np.abs(h0.coefficient)**2)

		output_terms += order_1

	if order_cutoff >= 2:
		# include order 2 terms
		order_2 = (pauli_object*sigma*sigma + sigma*sigma*pauli_object)/(8 * np.abs(h0.coefficient)**2)

		order_2 += (sigma*h0*pauli_object*h0*sigma)/(4 * np.abs(h0.coefficient)**4)

		output_terms += order_2

	if order_cutoff >= 3:
		# include order 3 terms
		order_3 = (sigma*h0*pauli_object*sigma*sigma + sigma*sigma*h0*pauli_object*sigma)/(16 * np.abs(h0.coefficient)**4)

		output_terms += order_3

	if order_cutoff >= 4:
		# include order 4 terms

		output_terms += (sigma*sigma*pauli_object*sigma*sigma)/(64 * np.abs(h0.coefficient)**4)

	return output_terms


def sigma_h0_ratio(h0, sigma) -> float:
	"""



	Args:
		sigma:
		h0:

	Returns:

	"""

	print("h0", h0)
	print("sigma", sigma)

	sigma_coefficients = np.array([term.coefficient for term in sigma])

	ratio = np.mean(
		np.log(10 ** -10 + np.abs(sigma_coefficients) / np.abs(h0.coefficient))
	)

	return ratio

def n_body_term_scale(
		hamiltonian_generator: Callable[[], cirq.PauliSum],
		reps: int,
		growth_rate: int = 2,
) -> List[float]:
	"""
	"""
	term_length_coefficients = [[] for _ in range(N+1)]

	for rep in range(reps):
		hamiltonian = hamiltonian_generator()

		sb_object = BifurcationRG(hamiltonian=hamiltonian, growth_rate=growth_rate)

		sb_object.diagonalize()

		for term in sb_object.hamiltonian:
			term_length_coefficients[len(term)].append(term.coefficient ** 2)

	#now average over each list, and apply log
	for i, data in enumerate(term_length_coefficients):
		if len(data)>0:
			term_length_coefficients[i] = np.log(np.sqrt(np.mean(data)))
		else:
			term_length_coefficients[i] = 10**-10

	return term_length_coefficients


def easy_energy_variance(
		hamiltonian_generator: Callable[[], cirq.PauliSum],
		reps: int,
		growth_rate: int = 2,
):
	"""
	determine the variance in the energy
	This is obtained by
	1) initializing H
	2) obtain H^2
	3) diagonalize H
	4) Applying the obtained diagonalization procedure to H^2
	We wish to obtain <H^2> - <H>^2.
	We obtain the first value using the trace of the diagonalized H^2
	We can obtain the second term efficiently as well, but for now we'll sample
	5) take the trace of H^2 to obtain <H^2>
	6) take sum_t <t|H|t>^2
	Returns:
		exp_ham_squared: a list of exact <H^2> values
		squared_exp_ham: a list of approximated <H>^2 values
		normalized_energy_variances: a list of (<H^2> -<H>^2)/<H^2>
	"""

	exp_ham_squared = []
	squared_exp_ham = []
	normalized_energy_variances = []

	for realization in range(reps):
		hamiltonian = hamiltonian_generator()
		sb_object = BifurcationRG(
			hamiltonian=hamiltonian.copy(),
			growth_rate=growth_rate)

		sb_object.diagonalize()

		#apply only the rotation procedure to the hamiltonian
		clifford_diagonalized_hamiltonian = sb_object.apply_clifford_diagonalization(
			hamiltonian
		)

		#now collect diagonal and off diagonal elements

		all_coefficients = 0
		all_diagonal_coefficients = 0

		for term in clifford_diagonalized_hamiltonian:
			coefficient = float(np.abs(term.coefficient))

			if not term.values() and not set(term.values()).issubset({cirq.X, cirq.Y}):
				all_diagonal_coefficients += coefficient ** 2

			all_coefficients += coefficient ** 2

		normalized_energy_variances.append((all_coefficients-all_diagonal_coefficients)/all_coefficients)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances

class coefficient_distribution(stats.rv_continuous):
	"""
	Define a class to sample from the distribution
	defined in section III B of
	https://arxiv.org/pdf/1508.03635.pdf
	"""

	def _pdf(self, x, param0, gamma0):

		#return param0 * np.exp(-param0*x)

		term1 = 1/(x*gamma0)

		term2 = (x/param0)**(1/gamma0)

		return term1*term2



def give_dist(
		n: float,
		j0: float,
		k0: float,
		h0: float,
		gamma0: float,
) -> Tuple[np.array, np.array, np.array]:
	"""
	Generate Js, Ks, and hs. Each of which is an array
	of length n
	"""

	#An instance to produce the coefficients
	#a=0 is the minimum
	param_dist = coefficient_distribution(a=0)

	Js = []
	Ks = []
	hs = []

	#select values according to each distribution.
	#if they are greater than their upper bound, re-select

	for _ in range(n):
		#select J
		if j0>0:
			J = param_dist.rvs(param0=j0, gamma0=gamma0, size=1)
			while J>j0:
				J = param_dist.rvs(param0=j0, gamma0=gamma0, size=1)
		else:
			J=0

		#select K
		if k0>0:
			K = param_dist.rvs(param0=k0, gamma0=gamma0, size=1)
			while K>k0:
				K = param_dist.rvs(param0=k0, gamma0=gamma0, size=1)
		else:
			K=0

		#select h
		if h0>0:
			h = param_dist.rvs(param0=h0, gamma0=gamma0, size=1)
			while h>h0:
				h = param_dist.rvs(param0=h0, gamma0=gamma0, size=1)
		else:
			h=0

		Js.append(J)
		Ks.append(K)
		hs.append(h)

	Js = np.squeeze(np.array(Js))
	Ks = np.squeeze(np.array(Ks))
	hs = np.squeeze(np.array(hs))

	return Js, Ks, hs


def give_tfim_1d_hamiltonian_generator(
		n: float,
		j0: float,
		k0: float,
		h0: float,
		gamma0: float,
		periodic: bool = False,
) -> Callable[[], cirq.PauliSum]:
	"""
	"""

	def give_hamiltonian() -> cirq.PauliSum:
		"""

		:return:
		"""
		js, ks, hs = give_dist(
			n=n,
			j0=j0,
			k0=k0,
			h0=h0,
			gamma0=gamma0,
		)

		if not periodic:
			# kill the element wrapping around the lattice
			js[-1] = 0.0
			ks[-1] = 0.0

		hamiltonian = cirq.PauliSum()

		for index, j in enumerate(js):
			current_index = index
			next_index = (index + 1) % n

			term = cirq.PauliString(
				cirq.X(cirq.LineQubit(current_index)) * cirq.X(cirq.LineQubit(next_index)),
				j
			)

			hamiltonian += term

		for index, h in enumerate(hs):
			term = cirq.PauliString(
				cirq.Z(cirq.LineQubit(index)),
				h,
			)

			hamiltonian += term

		for index, k in enumerate(ks):
			current_index = index
			next_index = (index + 1) % n

			term = cirq.PauliString(
				cirq.Z(cirq.LineQubit(current_index)) * cirq.Z(cirq.LineQubit(next_index)),
				k,
			)

			hamiltonian += term

		return hamiltonian

	return give_hamiltonian


def place_object_for_visualization(
		positions: np.array,
		object_x_range: Tuple[int, int],
) -> Tuple[np.array, int]:
	"""
	Compute the height to optimally place an object, such that it doesn't intersect
	any other object. Record the new position, and return both the new positions, and
	the computed height.

	Args:
		positions:
		object_x_range:

	Returns:

	"""
	height = -1
	intersects = True

	while intersects:
		height += 1

		# check for intersection
		intersects = np.max(positions[object_x_range[0]:object_x_range[-1]+1, height]) == 1

	# at this point, we know we have no intersection
	next_positions = positions.copy()
	next_positions[object_x_range[0]:object_x_range[-1]+1, height] = 1

	return next_positions, height


def visualize_hamiltonian(
		hamiltonian: cirq.PauliSum,
		mag_range: Optional[Tuple[float, float]] = None,
		dpi=100,
		legend=False,
		use_colors=False,
		use_letters=True,
):
	"""
	Visualize the links between terms in the hamiltonian.

	Args:
		hamiltonian:

	Returns:
		fig, axes indicating the links
	"""
	fig, ax = plt.subplots(dpi=dpi)

	if use_colors:
		color_dict = {cirq.X: "tab:red", cirq.Y: "tab:green", cirq.Z: "tab:blue"}
	else:
		color_dict = {cirq.X: "white", cirq.Y: "white", cirq.Z: "white"}
	if use_letters:
		marker_dict = {cirq.X: "$X$", cirq.Y: "$Y$", cirq.Z: "$Z$"}
	else:
		marker_dict = {cirq.X: ".", cirq.Y: ".", cirq.Z: "."}

	hamiltonian_terms = sorted([term for term in hamiltonian if term.values()], key=len)
	all_coefficients = sorted([abs(abs(term.coefficient)) for term in hamiltonian if term.values()])

	"""
	Determine the maximum and minimum magnitudes of the terms
	"""

	if mag_range is None:
		minimum_mag = all_coefficients[0]
		maximum_mag = all_coefficients[-1]
	else:
		minimum_mag = min(
			all_coefficients[0],
			mag_range[0],
		)
		maximum_mag = max(
			all_coefficients[-1],
			mag_range[-1],
		)

	alpha_offset = .1
	delta_height = .4

	"""
	Now for each term, assign an alpha, a position, and a marker style
	"""
	qubit_x_positions = [qubit.x for qubit in hamiltonian.qubits]
	all_positions = np.zeros((max(qubit_x_positions)+1, len(hamiltonian_terms)))

	for term in hamiltonian_terms:
		abs_coefficient = np.abs(term.coefficient)

		alpha_value = alpha_offset + (1-alpha_offset) * (
				abs_coefficient - minimum_mag)/(
				maximum_mag - minimum_mag)

		# ensure alphas are in correct range
		alpha_value = min(max(0, alpha_value), 1)

		colors = [color_dict[operation] for operation in term.values()]

		term_x_positions = [qubit.x for qubit in term.qubits]

		term_range = (min(term_x_positions), max(term_x_positions))

		all_positions, height = place_object_for_visualization(
			positions=all_positions,
			object_x_range=term_range,
		)

		scaled_height = height * delta_height

		# Now update the figure
		if len(term_x_positions) == 1:
			ax.scatter(
				x=term_x_positions[0],
				y=scaled_height,
				s=25,
				alpha=alpha_value,
				color='black',
				marker=marker_dict[next(iter(term.values()))],
			)
		else:
			ax.hlines(
				y=scaled_height,
				xmin=min(term_x_positions),
				xmax=max(term_x_positions),
				linewidth=14,
				alpha=alpha_value,
				color='black',
			).set_capstyle('round')

			for index, (qubit, operation) in enumerate(term.items()):
				ax.scatter(
					x=qubit.x,
					y=scaled_height,
					s=25,
					alpha=1,
					color=colors[index],
					zorder=len(hamiltonian)+1,
					marker=marker_dict[operation],
				)

	ax.set_xticks([qubit.x for qubit in hamiltonian.qubits])
	ax.set_yticks([])

	if legend:
		fig.legend()

	return fig, ax


def combine_unitaries(
		unitaries : List[Union[cirq.PauliSum, cirq.GateOperation]],
) -> cirq.PauliSum:
	"""
	The total list of cliffords is a list of lists.
	each sub-list makes up one clifford operation whose purpose it is to
	diagonalize the current leading order residual term.

	combine_unitaries is a function to turn a sub-list into a paulisum

	Args:
		unitaries: A list of Paulisums or Gate Operations.
			To fit with the convention of cirq, these are listed in the reverse
			that they are applied

	Returns:
		A PauliSum which has the same effect as the list of unitaries.
	"""
	combined_unitaries = cirq.PauliSum(cirq.LinearDict({frozenset(): (1+0j)}))

	#paulis = [cirq.PauliSum(cirq.LinearDict({frozenset(): (1+0j)})), cirq.X, cirq.Y, cirq.Z]

	paulis = [cirq.X, cirq.Y, cirq.Z]

	assert type(unitaries) == list, (
		f"unitaries should be a list of cliffords. Recieved {unitaries}")

	for unitary in reversed(unitaries):

		assert type(unitary) in {cirq.GateOperation, cirq.PauliSum}, (
			f"unitary is of incorrect type. Recieved {unitary}")

		if type(unitary) == cirq.GateOperation:
			# Then its a swap
			qubit_1, qubit_2 = unitary.qubits
			# TODO: why does this not break? i have qubit_1 twice
			current_swap = sum([op(qubit_1)*op(qubit_2)/2 for op in paulis]) + 1/2
			combined_unitaries = combined_unitaries * current_swap
		else:
			# it is a pauli sum
			combined_unitaries = combined_unitaries * unitary

	return combined_unitaries


def combine_all_unitaries(
		clifford_list
) -> cirq.PauliSum:
	"""
	We'll go through each clifford list, turn each into a paulisum.
	Combine the entire one into one unitary

	Args:
		clifford_list:

	Returns:
		cirq.PauliSum representing the complete unitary U
	"""

	output = cirq.PauliSum()+1

	for clifford_sub_list in clifford_list:
		output = output * combine_unitaries(clifford_sub_list)

	return output


def obtain_eigensystem_given_cliffords(
		hamiltonian: cirq.PauliSum,
):
	"""
	We'll want to compare how our diagonalization does with the correct one.
	We can do this by applying our cliffords to outer-products of exact
	eigenvectors

	we want to obtain |< \psi_n | U | e_n > |^2

	U^t | \psi_n > < \psi_n | U ~> | e'_n > < e'_n |

	we want to know how closely | e'_n > compares to | e_n >

	and so we want
	< e_n | e'_n > < e'_n | e_n > = < e_n | U^t | \psi_n > < \psi_n | U | e_n >

	simply look at the nth diagonal element
	we'll try to find the maximum diagonal element for each.

	we'll do this for all 2^n vectors and compare the inner product

	Args:
		cliffords_list:

	Returns:

	"""
	hamiltonian_matrix = hamiltonian.matrix()

	hamiltonian_length = hamiltonian_matrix.shape[0]

	sbrg_object = BifurcationRG(
		hamiltonian=hamiltonian,
	)

	sbrg_object.diagonalize()

	diagonalized_hamiltonian = sbrg_object.apply_clifford_diagonalization(
		hamiltonian=hamiltonian.copy(),
	)

	"""
	Now obtain the true diagonalization
	"""
	unitary_pauli_sum = combine_all_unitaries(sbrg_object.clifford_rotations)

	unitary_matrix = unitary_pauli_sum.matrix()

	w, v = LA.eigh(hamiltonian_matrix)

	overlaps = []

	for i in range(hamiltonian_length):
		psi = v[:, i]

		outer_product_matrix = np.outer(psi.transpose().conjugate(), psi)

		print("psi shape", psi.shape)
		print("psi outer shape", outer_product_matrix.shape)

		print("unitary matrix shape", unitary_matrix.shape)


		diag_outer_product_matrix = unitary_matrix.transpose().conjugate() @ outer_product_matrix @ unitary_matrix

		diag_elements = np.diagonal(
			diag_outer_product_matrix
		)

		overlaps.append(
			sorted(zip(diag_elements, range(hamiltonian_length)), reverse=True)
		)

	"""
	Now we want to find the optimal overlap ordering 
	this involves identifying each eigenvector with a diagonal 
	matrix index. 
	
	we start by identifying elements which have maximal overlap at the same
	diagonal index -- (can this ever happen?) I think it might not be
	"""

	matches = [[] for _ in range(hamiltonian_length)]

	for overlap in overlaps:
		max_index = overlap[0][-1]

		matches[max_index].append(overlap)

	"""
	collect those elements with more than a single max element
	"""
	repeats = []

	for i, match in enumerate(matches):
		if len(match) > 1:
			repeats.append((i, match))

	if repeats:
		"""
		Then we have to deal with this
		
		for now we'll remove these
		"""
		print("repeated match")
		matches = [match_list[:1] for match_list in matches if match_list]


		#assert False

	overlaps = [match_list[0][0][0] for match_list in matches]

	corresponding_indices = [match_list[0][0][1] for match_list in matches]

	energies = w[corresponding_indices]

	return energies, overlaps


def compare_eigenvalues(
		hamiltonian: cirq.PauliSum,
		growth_rate: int = 2,
) -> Tuple[np.array, np.array]:
	"""
	Here we'll compare the implied spectrum of the hamiltonian
	to its exact spectrum.

	specifically, we'll only apply the cliffords to get the diagonal
	terms.

	Args:
		hamiltonian:

	Returns:
		a tuple of two arrays. The first for the exact spectrum, the second for the
		implied spectrum
	"""

	hamiltonian_matrix = hamiltonian.matrix()

	w, v = LA.eigh(hamiltonian_matrix)

	exact_energies = sorted(np.real(w))

	sbrg_object = BifurcationRG(
		hamiltonian=hamiltonian.copy(),
		growth_rate=growth_rate,
	)

	sbrg_object.diagonalize()

	rotated_hamiltonian = sbrg_object.apply_clifford_diagonalization(
		hamiltonian=hamiltonian
	)

	pruned_rotated_hamiltonian = cirq.PauliSum()

	for term in rotated_hamiltonian:
		if set(term.values()).issubset({cirq.Z}):
			pruned_rotated_hamiltonian += term

	implied_energies_0 = sorted(np.real(np.diag(sbrg_object.hamiltonian)))

	"""
	Now extract the implied spectrum 
	"""
	implied_matrix = pruned_rotated_hamiltonian.matrix()

	"""
	check if diagonal
	"""

	implied_energies = sorted(np.real(np.diag(implied_matrix)))

	return exact_energies, implied_energies, implied_energies_0
























def pauli_object_is_zero(
		pauli_object
):
	pass


def test_clifford_inversion(
		hamiltonian: cirq.PauliSum
) -> bool:
	"""
	Test that the clifford which diagonalize a hamiltonian
	can be inverted correctly.

	Args:
		hamiltonian: A paulisum to which we apply sbrg

	Returns:
		bool, whether or not the inversion worked.
	"""
	sbrg_object = BifurcationRG(hamiltonian.copy())

	sbrg_object.diagonalize()

	rotated_hamiltonian = sbrg_object.apply_clifford_diagonalization(
		hamiltonian.copy(),
		cliffords_list=sbrg_object.clifford_rotations,
	)
	#print("eff ham", sbrg_object.effective_terms)
	#print("res ham", sbrg_object.residual_terms)


	#print("original ham", hamiltonian)
	#print("rotated_hamiltonian", rotated_hamiltonian)

	cliffords = sbrg_object.clifford_rotations

	inverted_cliffords = sbrg_object.invert_cliffords(sbrg_object.clifford_rotations)

	unrotated_rotated_hamiltonian = sbrg_object.apply_clifford_diagonalization(
		rotated_hamiltonian.copy(),
		cliffords_list=inverted_cliffords,
	)

	#print("unrotated_rotated_hamiltonian", unrotated_rotated_hamiltonian)

	rerotated_hamiltonian = sbrg_object.apply_clifford_diagonalization(
		unrotated_rotated_hamiltonian.copy(),
		cliffords_list=cliffords,
	)

	#print("rerotated_hamiltonian", rerotated_hamiltonian)# - unrotated_rotated_hamiltonian)

	#print("comparing")
	#print(rotated_hamiltonian)
	#print(rerotated_hamiltonian, rotated_hamiltonian == rerotated_hamiltonian)
	#print(" unrot ")
	#print(hamiltonian , unrotated_rotated_hamiltonian, hamiltonian == unrotated_rotated_hamiltonian)

	rotated_terms = sorted([term for term in rotated_hamiltonian], key = lambda x: abs(x.coefficient))
	rerotated_terms = sorted([term for term in rerotated_hamiltonian], key=lambda x: abs(x.coefficient))

	original_terms = sorted([term for term in hamiltonian], key=lambda x: abs(x.coefficient))
	unrotated_rotated_terms = sorted([term for term in unrotated_rotated_hamiltonian], key=lambda x: abs(x.coefficient))

	for term_1, term_2 in zip(rotated_terms, rerotated_terms):
		assert term_1._qubit_pauli_map == term_2._qubit_pauli_map

		difference = (term_1.coefficient - term_2.coefficient)/(abs(term_1.coefficient) + abs(term_2.coefficient))
		assert abs(difference) < 10 ** -5

	for term_1, term_2 in zip(original_terms, unrotated_rotated_terms):
		assert term_1._qubit_pauli_map == term_2._qubit_pauli_map

		difference = (term_1.coefficient - term_2.coefficient) / (abs(term_1.coefficient) + abs(term_2.coefficient))
		assert abs(difference) < 10 ** -5

	return True


def test_diagonalization():
	"""
	Here we want to test that diagonalization doesnt run into any hiccups
	We'll do this by repeatedly running diagonalization until we find a problem.

	If we encounter a problem, return the hamiltonian.
	Otherwise, return None

	Returns:
		Optional[cirq.PauliSum]
	"""
	ham_generator = give_tfim_1d_hamiltonian_generator(
		n=10,
		j0=1,
		k0=1,
		h0=1,
		gamma0=4,
		periodic=True,
	)

	max_reps = 100

	for rep in range(max_reps):
		ham = ham_generator()

		sbrg_object = BifurcationRG(
			hamiltonian=ham.copy()
		)

		try:
			sbrg_object.diagonalize()
		except:
			return ham

	return None



def compare_diagonalization():
	"""
	Here we want to compare the two operations:

	diagonalizing a hamiltonian, obtaining the effective hamiltonian

	taking the full hamiltonian, applying the cliffords, removing non-diaognals

	In general, I dont think these should match. this is because the
	SW transformaitons have

	H \Sigma^2 / (2 h_3^2)

	Returns:
		None
	"""
	ham_generator = give_tfim_1d_hamiltonian_generator(
		n=10,
		j0=1,
		k0=1,
		h0=1,
		gamma0=4,
		periodic=True,
	)

	reps = 10

	for rep in range(reps):
		ham = ham_generator()

		sbrg_object = BifurcationRG(
			hamiltonian=ham.copy()
		)

		sbrg_object.diagonalize()

		"""
		now we apply diagonalization to the original 
		hamiltonian
		"""
		diagonalized_ham = sbrg_object.apply_clifford_diagonalization(
			ham
		)

		pruned_diagonalized_ham = cirq.PauliSum()
		for term in diagonalized_ham:

			operations = set(term.values())

			print("operations", operations, operations.intersection({cirq.X, cirq.Y}))

			if not operations.intersection({cirq.X, cirq.Y}):
				print("here")
				pruned_diagonalized_ham += term

		hamiltonian_difference = sbrg_object.hamiltonian - pruned_diagonalized_ham

		max_coeff = max(
			[np.abs(term.coefficient) for term in hamiltonian_difference]
		)

		if max_coeff > 10**-10:
			print("hamiltonians do not match")
			sorted_terms_1 = sorted(
				[t for t in pruned_diagonalized_ham],
				key=lambda x: np.abs(x.coefficient),
			)

			sorted_terms_2 = sorted(
				[t for t in sbrg_object.hamiltonian],
				key=lambda x: np.abs(x.coefficient),
			)

			print("pruned diagonal terms")
			for term in sorted_terms_1:
				print(term)

			print("sbrg_object.hamiltonian terms")
			for term in sorted_terms_2:
				print(term)


def mbl_level_statistic(
		hamiltonian: cirq.PauliSum
) -> float:
	"""
	Diagonalize the hamiltonian and compute the level statistics.

	r_i = min(s_i, s_{i-1}) / max(s_i, s_{i-1})

	s_i = E_{i+1} - E_i

	Args:
		hamiltonian: A PauliSum representing a Hamiltonian

	Returns:
		the average of r_i, the level statistic (a float)
	"""

	w, v = LA.eigh(
		hamiltonian.matrix()
	)

	# ensure it's sorted
	w = sorted(w)

	gaps = [w[i+1] - w[i] for i in range(len(w)-1)]

	rs = []

	for gap1, gap2 in zip(gaps[:-1], gaps[1:]):
		rs.append(min(gap1, gap2)/max(gap1, gap2))

	return float(np.mean(rs))











