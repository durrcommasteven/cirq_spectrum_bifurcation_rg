from Master_Spectrum_Bifurcation_Tools import *
from two_d_lattice_ham import *
from scipy import stats

"""
generate their hamiltonian
"""

"""
Define the distribution
"""

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


def test_coefficient_distribution():
	"""
	Confirm that the coefficient distribution is as expected
	
	make logarithmically spaced bins
	"""

	bin_number = 10**2

	reps = 10**5

	"""
	Now we'll test things out 
	"""

	param0 = 1

	#bins = np.logspace(-10, np.exp())


def give_dist(N, J0, K0, h0, gamma0):
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

	for _ in range(N):
		#select J
		J = param_dist.rvs(param0 = J0, gamma0 = gamma0, size = 1)
		while J>J0:
			J = param_dist.rvs(param0 = J0, gamma0 = gamma0, size = 1)

		#select K
		K = param_dist.rvs(param0 = K0, gamma0 = gamma0, size = 1)
		while K>K0:
			K = param_dist.rvs(param0 = K0, gamma0 = gamma0, size = 1)

		#select h
		h = param_dist.rvs(param0 = h0, gamma0 = gamma0, size = 1)
		while h>h0:
			h = param_dist.rvs(param0 = h0, gamma0 = gamma0, size = 1)

		Js.append(J)
		Ks.append(K)
		hs.append(h)

	Js = np.array(Js)
	Ks = np.array(Ks)
	hs = np.array(hs)

	return Js, Ks, hs



def give_dist_1(N, J0, K0, h0, Gamma0, bins = 10**5):

	Js = np.linspace(10**-10, J0, bins)
	Ks = np.linspace(10**-10, K0, bins)
	hs = np.linspace(10**-10, h0, bins)

	Jprobs = ((Js/J0)**(1/Gamma0))/(Gamma0*Js)
	Kprobs = ((Ks/K0)**(1/Gamma0))/(Gamma0*Ks)
	hprobs = ((hs/h0)**(1/Gamma0))/(Gamma0*hs)

	Jprobs = Jprobs/np.sum(Jprobs)
	Kprobs = Kprobs/np.sum(Kprobs)
	hprobs = hprobs/np.sum(hprobs)

	#plt.plot(Jprobs)
	#plt.yscale("log")
	#plt.show()
	

	Selected_Js = []
	Selected_Ks = []
	Selected_hs = []

	for _ in range(N):
		j = np.random.choice(Js, p=Jprobs)
		k = np.random.choice(Ks, p=Kprobs)
		h = np.random.choice(hs, p=hprobs)

		Selected_Js.append(j)
		Selected_Ks.append(k)
		Selected_hs.append(h)

	return np.array(Selected_Js), np.array(Selected_Ks), np.array(Selected_hs)



"""
compare to exact diagonalization

We do this by explicitly reconstructing the states and obtaining 
<psi_{exact} | psi_{reconstructed}>

note 
|tau> = (SW_N R_N)...(SW_1 R_1)|psi_{exact}> 

and so 

| psi_{reconstructed}> = (R_1^{-1} SW_1^{-1}) ... (R_N^{-1} SW_N^{-1}) |tau>


Note to diagonalize we applied 
<psi|H|psi>

<psi| Exp[S] Exp[-S] H Exp[S] Exp[-S] |psi>

<tau| H_{diag} |tau>

Therefore to rebuild |psi> from |tau> we must apply Exp[S]

During diagonalization we could take S = -H0.Sigma /(2 h_0^2) + S2

Where S2 was such that 

(H0 /(2 h_0^2)).[Sigma, Delta] + O(h_0^2) = [S2, H0] 

This term vanished in the expression for S^t H S precisely due to the above constraint on S2

However when we apply it to states, no such cancellation will occur. 

We could leave out S2, but if we want to be more accurate in our reconstruction of states, we might want to include it

THEREFORE

to first order in 1/h0

Exp[S] = 1 - H0.Sigma /(2 h_0^2) + O(h0^{-2})
"""



def reconstruct_eigenstate(SB_object, liom_pattern):
	"""
	Given a fully diagonalized SB_object, reconstruct the eigenstate corresponding to a 
	liom_pattern
	"""


	return None



def term_length_1(operator):
	"""
	given an operator tuple, give the spatial length. This is equivalent to the 
	length of the operator minus the length of the longest subsequence of 0's

	Args:
		operator: a tuple of 

	"""

	longest_zeros_length = 0
	current_zeros_length = 0

	for i in operator:
		if i==0:
			current_zeros_length+=1
		if current_zeros_length> longest_zeros_length:
			longest_zeros_length=current_zeros_length
		if i!=0:
			current_zeros_length=0

	#now take care of the ends 
	front_zeros = 0
	back_zeros = 0

	index = 0
	while operator[index] == 0:
		front_zeros+=1
		index+=1

	index = -1
	while operator[index] == 0:
		back_zeros+=1
		index-=1

	longest_zeros_length = max(longest_zeros_length, front_zeros+back_zeros)

	return len(operator) - longest_zeros_length


def term_length(operator):
	"""
	given an operator tuple, give the number of non-identity operators

	Args:
		operator: a tuple

	"""

	n_body = 0
	for p in operator:
		if p!=0:
			n_body+=1

	return n_body

def n_body_term_scale(gamma0, reps, N = 64, growth_rate = 2):
	"""

	"""
	term_length_coefficients = [[] for _ in range(N+1)]

	for rep in range(reps):

		Js, hs, Ks = give_dist(N, 1, 1, 1, gamma0)

		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)

		SB.diagonalize()

		for op, coeff in zip(SB.ham.operators, SB.ham.coefficients):
			term_length_coefficients[term_length(op)].append(coeff**2)




	print(term_length_coefficients)


	#now average over each list, and apply log
	for i, data in enumerate(term_length_coefficients):
		if len(data)>0:
			term_length_coefficients[i] = np.log(np.sqrt(np.mean(data)))
		else:
			term_length_coefficients[i] = 10**-10

	return term_length_coefficients



"""
Obtain variance of spectrum for larger systems
"""
def easy_energy_variance(realization_reps, N, J0, K0, h0, Gamma0, growth_rate = 2, check = False, clifford_only = False):
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

	for realization in range(realization_reps):

		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)

		#obtain the hamiltonian
		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)

		SB.diagonalize()

		#apply only the rotation procedure to the hamiltonian
		ham = copy.deepcopy(
			SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only = True))

		#now collect diagonal and off diagonal elements

		all_coeffs = 0
		all_diags = 0

		for op, i in ham.operator_dict.items():
			coef = float(np.abs(ham.coefficients[i]))

			if set(op) == {0, 3}:
				all_diags+=coef**2

			all_coeffs+=coef**2

		normalized_energy_variances.append((all_coeffs-all_diags)/all_coeffs)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances



def comparison_easy_energy_variance(realization_reps, N, J0, K0, h0, Gamma0, growth_rate = 2, check = False, clifford_only = False):
	"""
	compare the variance in the energy. 
	Do this by obtaining a diagonalization for one realization
	then applying it to a totally different realization.



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

	for realization in range(realization_reps):

		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)

		#obtain the hamiltonian
		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)

		SB.diagonalize()

		#apply only the rotation procedure to the hamiltonian
		#this time male a completely new realization for comparison
		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)
		ham = copy.deepcopy(
			SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only = True))

		#now collect diagonal and off diagonal elements

		all_coeffs = 0
		all_diags = 0

		for op, i in ham.operator_dict.items():
			coef = float(np.abs(ham.coefficients[i]))

			if set(op) == {0, 3}:
				all_diags+=coef**2

			all_coeffs+=coef**2

		normalized_energy_variances.append((all_coeffs-all_diags)/all_coeffs)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances








def energy_variance(realization_reps, N, J0, K0, h0, Gamma0, growth_rate = 2, check = False, clifford_only = False):
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

	for realization in range(realization_reps):

		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)

		#obtain the hamiltonian
		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)
		
		#get the squared hamiltonian
		squared_ops, squared_coeffs = SB.multiply_terms(tfim.operators, tfim.coefficients, tfim.operators, tfim.coefficients)

		squared_tfim = TFIM_ham_1d(Js, hs, Ks)

		squared_tfim.operators = []
		squared_tfim.coefficients = []
		squared_tfim.operator_dict = dict()

		for op, coef in zip(squared_ops, squared_coeffs):
			squared_tfim.add_operator(op, coef)

		#diagonalize H
		SB.diagonalize()

		#apply diagonalization to H^2 
		ham_squared = SB.apply_diagonalization(squared_tfim, clifford_only = clifford_only)

		#now take the trace of ham_squared
		#only identity elements contribute
		ham_squared_mean_trace = 0.0
		for op, coeff in zip(ham_squared.operators, ham_squared.coefficients):
			if sum(op)==0:
				#then it is the identity -- this is the only term with a nonzero trace
				ham_squared_mean_trace+= np.sum(coeff.real)

		ham_squared_mean_trace = np.divide(ham_squared_mean_trace, 2**SB.length)

		#now randomly sample eigenstates
		#Do we only care about the clifford steps?

		if clifford_only:
			#apply only the rotation procedure to the hamiltonian
			ham = SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only = True)
			SB.ham = ham

		#now evaluate
		H_exp_squared = 0.0

		"""
		note sum_t <t|H|t>^2 should be equivalent to 

		sum |a_{diag}|^2
		"""
		for op, i in SB.ham.operator_dict.items():
			if set(op) == {0, 3}:
				#then it's a diagonal element, get the coefficient

				diag_coeff = SB.ham.coefficients[i]

				H_exp_squared += np.sum(np.abs(diag_coeff)**2)

		H_exp_squared = np.divide(H_exp_squared, 2**SB.length)

		#now record our results
		normalized_energy_variance = (ham_squared_mean_trace - H_exp_squared)/ham_squared_mean_trace
		exp_ham_squared.append(ham_squared_mean_trace)
		squared_exp_ham.append(H_exp_squared)
		normalized_energy_variances.append(normalized_energy_variance)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances


def sampled_energy_variance(realization_reps, states_to_compare_per_rep, N, J0, K0, h0, Gamma0, growth_rate = 2, check = False, clifford_only = False):
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

	6) sample sum_t <t|H|t>^2


	Returns:
		exp_ham_squared: a list of exact <H^2> values

		squared_exp_ham: a list of approximated <H>^2 values

		normalized_energy_variances: a list of (<H^2> -<H>^2)/<H^2> 
	"""

	exp_ham_squared = []
	squared_exp_ham = []
	normalized_energy_variances = []

	for realization in range(realization_reps):

		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)

		#obtain the hamiltonian
		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)
		
		#get the squared hamiltonian
		squared_ops, squared_coeffs = SB.multiply_terms(tfim.operators, tfim.coefficients, tfim.operators, tfim.coefficients)

		squared_tfim = TFIM_ham_1d(Js, hs, Ks)

		squared_tfim.operators = []
		squared_tfim.coefficients = []
		squared_tfim.operator_dict = dict()

		for op, coef in zip(squared_ops, squared_coeffs):
			squared_tfim.add_operator(op, coef)

		#diagonalize H
		SB.diagonalize()

		#apply diagonalization to H^2 
		ham_squared = SB.apply_diagonalization(squared_tfim, clifford_only = clifford_only)

		#now take the trace of ham_squared
		#only identity elements contribute
		ham_squared_mean_trace = 0
		for op, coeff in zip(ham_squared.operators, ham_squared.coefficients):
			if sum(op)==0:
				#then it is the identity -- this is the only term with a nonzero trace
				ham_squared_mean_trace+=coeff

		ham_squared_mean_trace /= 2**SB.length

		#now randomly sample eigenstates
		#Do we only care about the clifford steps?

		if clifford_only:
			#apply only the rotation procedure to the hamiltonian
			ham = SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only = True)
			SB.ham = ham

		#now evaluate
		H_exp_squared = 0

		"""
		note sum_t <t|H|t>^2 should be equivalent to 

		sum |a_{diag}|^2
		"""
		for op, i in SB.ham.operator_dict.items():
			if set(op) == {0, 3}:
				#then it's a diagonal element, get the coefficient

				diag_coeff = SB.ham.coefficients[i]

				H_exp_squared += np.abs(diag_coeff)**2

		H_exp_squared /= 2**SB.length

		#now record our results
		normalized_energy_variance = (ham_squared_mean_trace - H_exp_squared)/ham_squared_mean_trace
		exp_ham_squared.append(ham_squared_mean_trace)
		squared_exp_ham.append(H_exp_squared)
		normalized_energy_variances.append(normalized_energy_variance)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances

def easy_energy_variance_bose_hubbard_2d(realization_reps, states_to_compare_per_rep, N, J, W, growth_rate = 2, check = False, clifford_only = True):
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

	6) randomly sample energies from H to obtain <H>^2


	Returns:
		exp_ham_squared: a list of exact <H^2> values

		squared_exp_ham: a list of approximated <H>^2 values

		normalized_energy_variances: a list of (<H^2> -<H>^2)/<H^2> 
	"""

	exp_ham_squared = []
	squared_exp_ham = []
	normalized_energy_variances = []

	for realization in range(realization_reps):
		#generate the on-site pootentials
		deltas = (np.random.rand(N, N)*2 -1)*W
		#generate the hams
		bh_ham = Bose_Hubbard_2d(J, deltas)

		SB = BifurcationRG(bh_ham, growth_rate=growth_rate)

		#diagonalize H
		SB.diagonalize()

		ham = copy.deepcopy(
			SB.apply_diagonalization(Bose_Hubbard_2d(J, deltas), clifford_only = True))

		#now collect diagonal and off diagonal elements

		all_coeffs = 0
		all_diags = 0

		for op, i in ham.operator_dict.items():
			coef = float(np.abs(ham.coefficients[i]))

			if set(op) == {0, 3}:
				all_diags+=coef**2

			all_coeffs+=coef**2

		normalized_energy_variances.append((all_coeffs-all_diags)/all_coeffs)
		squared_exp_ham.append(all_diags)
		exp_ham_squared.append(all_coeffs)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances

def energy_variance_bose_hubbard_2d(realization_reps, states_to_compare_per_rep, N, J, W, growth_rate = 2, check = False, clifford_only = False):
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

	6) randomly sample energies from H to obtain <H>^2


	Returns:
		exp_ham_squared: a list of exact <H^2> values

		squared_exp_ham: a list of approximated <H>^2 values

		normalized_energy_variances: a list of (<H^2> -<H>^2)/<H^2> 
	"""

	exp_ham_squared = []
	squared_exp_ham = []
	normalized_energy_variances = []

	for realization in range(realization_reps):
		#generate the on-site pootentials
		deltas = (np.random.rand(N, N)*2 -1)*W
		#generate the hams
		bh_ham = Bose_Hubbard_2d(J, deltas)

		SB = BifurcationRG(bh_ham, growth_rate=growth_rate)
		
		#get the squared hamiltonian
		squared_ops, squared_coeffs = SB.multiply_terms(bh_ham.operators, bh_ham.coefficients, bh_ham.operators, bh_ham.coefficients)

		squared_bh_ham = Bose_Hubbard_2d(J, deltas)

		squared_bh_ham.operators = []
		squared_bh_ham.coefficients = []
		squared_bh_ham.operator_dict = dict()

		for op, coef in zip(squared_ops, squared_coeffs):
			squared_bh_ham.add_operator(op, coef)

		#diagonalize H
		SB.diagonalize()

		#apply diagonalization to H^2 
		ham_squared = SB.apply_diagonalization(squared_bh_ham, clifford_only = clifford_only)

		#now take the trace of ham_squared
		#only identity elements contribute
		ham_squared_mean_trace = 0
		for op, coeff in zip(ham_squared.operators, ham_squared.coefficients):
			if sum(op)==0:
				ham_squared_mean_trace+=coeff

		ham_squared_mean_trace /= 2**SB.length

		#now randomly sample eigenstates
		#Do we only care about the clifford steps?

		if clifford_only:
			#apply only the rotation procedure to the hamiltonian
			ham = SB.apply_diagonalization(Bose_Hubbard_2d(J, deltas), clifford_only = True)
			SB.ham = ham

		#now randomly sample states
		H_exp_squared = 0

		for state in range(states_to_compare_per_rep):
			liom_pattern = 2*(np.random.rand(N**2)>0.5) - 1

			ham_energy = SB.give_energy(liom_pattern)

			H_exp_squared += ham_energy**2

		H_exp_squared /= states_to_compare_per_rep

		#now record our results
		normalized_energy_variance = (ham_squared_mean_trace - H_exp_squared)/ham_squared_mean_trace
		exp_ham_squared.append(ham_squared_mean_trace)
		squared_exp_ham.append(H_exp_squared)
		normalized_energy_variances.append(normalized_energy_variance)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances


class TFIM_ham_1d(object):
	def __init__(self, Js, hs, Ks, periodic=True):
		"""
		Js, hs, Ks: the lists of parameters

		using the paper's notation, we take
		J_i <=> J_i sigma_i^1 sigma_{i+1}^1
		K_i <=> K_i sigma_i^3 sigma_{i+1}^3
		h_i <=> h_i sigma_i^3

		Here we take the parameters and create the list of operators
		"""
		assert (len(Js) == len(hs) == len(Ks)), 'unequal coefficient lengths'

		if not periodic:
			# kill the element wrapping around the lattice
			Js[-1] = 0.0
			Ks[-1] = 0.0

		self.Js = Js
		self.hs = hs
		self.Ks = Ks

		self.length = len(self.Js)

		"""
		now we build the operators
		"""
		self.operators = []
		self.coefficients = []

		for index, j in enumerate(self.Js):
			current_index = index
			next_index = (index + 1) % self.length
			op = tuple(1 if i in (current_index, next_index) else 0 for i in range(self.length))

			coefficient = j

			self.operators.append(op)
			self.coefficients.append(coefficient)

		for index, h in enumerate(self.hs):
			current_index = index
			op = tuple(3 if i == current_index else 0 for i in range(self.length))

			coefficient = h

			self.operators.append(op)
			self.coefficients.append(coefficient)

		for index, k in enumerate(self.Ks):
			current_index = index
			next_index = (index + 1) % self.length
			op = tuple(3 if i in (current_index, next_index) else 0 for i in range(self.length))

			coefficient = k

			self.operators.append(op)
			self.coefficients.append(coefficient)

		"""
		The above defined operator and coefficient lists allow us to identify 
		an operator given an index

		now we define a dictionary to identify the index given an operator

		we define operator_dict, containing the indices of all operators
		"""
		self.operator_dict = dict()

		for index, operator in enumerate(self.operators):
			self.operator_dict[operator] = index

		"""
		Finally, we need to track the operators which are diagonalized, and those which are not. 
		The indices of diagonalized operators are included in the "effective ops" dictionary
		The indices of remaining operators are included in the "residual ops" dictionary
		"""

		self.effective_ops = dict()
		self.residual_ops = self.operator_dict.copy()
		# TODO: fix this all and format it

def sampled_energy_variance_bose_hubbard_2d(realization_reps, states_to_compare_per_rep, N, J, W, growth_rate = 2, check = False, clifford_only = False):
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

	6) randomly sample energies from H to obtain <H>^2


	Returns:
		exp_ham_squared: a list of exact <H^2> values

		squared_exp_ham: a list of approximated <H>^2 values

		normalized_energy_variances: a list of (<H^2> -<H>^2)/<H^2> 
	"""

	exp_ham_squared = []
	squared_exp_ham = []
	normalized_energy_variances = []

	for realization in range(realization_reps):
		#generate the on-site pootentials
		deltas = (np.random.rand(N, N)*2 -1)*W
		#generate the hams
		bh_ham = Bose_Hubbard_2d(J, deltas)

		SB = BifurcationRG(bh_ham, growth_rate=growth_rate)
		
		#get the squared hamiltonian
		squared_ops, squared_coeffs = SB.multiply_terms(bh_ham.operators, bh_ham.coefficients, bh_ham.operators, bh_ham.coefficients)

		squared_bh_ham = Bose_Hubbard_2d(J, deltas)

		squared_bh_ham.operators = []
		squared_bh_ham.coefficients = []
		squared_bh_ham.operator_dict = dict()

		for op, coef in zip(squared_ops, squared_coeffs):
			squared_bh_ham.add_operator(op, coef)

		#diagonalize H
		SB.diagonalize()

		#apply diagonalization to H^2 
		ham_squared = SB.apply_diagonalization(squared_bh_ham, clifford_only = clifford_only)

		#now take the trace of ham_squared
		#only identity elements contribute
		ham_squared_mean_trace = 0
		for op, coeff in zip(ham_squared.operators, ham_squared.coefficients):
			if sum(op)==0:
				ham_squared_mean_trace+=coeff

		ham_squared_mean_trace /= 2**SB.length

		#now randomly sample eigenstates
		#Do we only care about the clifford steps?

		if clifford_only:
			#apply only the rotation procedure to the hamiltonian
			ham = SB.apply_diagonalization(Bose_Hubbard_2d(J, deltas), clifford_only = True)
			SB.ham = ham

		#now randomly sample states
		H_exp_squared = 0

		for state in range(states_to_compare_per_rep):
			liom_pattern = 2*(np.random.rand(N**2)>0.5) - 1

			ham_energy = SB.give_energy(liom_pattern)

			H_exp_squared += ham_energy**2

		H_exp_squared /= states_to_compare_per_rep

		#now record our results
		normalized_energy_variance = (ham_squared_mean_trace - H_exp_squared)/ham_squared_mean_trace
		exp_ham_squared.append(ham_squared_mean_trace)
		squared_exp_ham.append(H_exp_squared)
		normalized_energy_variances.append(normalized_energy_variance)

	return exp_ham_squared, squared_exp_ham, normalized_energy_variances


def energy_variance_1(realization_reps, states_to_compare_per_rep, N, J0, K0, h0, Gamma0, growth_rate = 2, check = False):
	"""
	determine the variance in the energy
	"""

	normed_energy_variance_list = []

	for realization in range(realization_reps):

		Js, hs, Ks = give_dist(N, J0, K0, h0, Gamma0)

		tfim = TFIM_ham_1d(Js, hs, Ks)

		SB = BifurcationRG(tfim, growth_rate=growth_rate)

		"""
		get the squared hamiltonian
		"""
		squared_ops, squared_coeffs = SB.multiply_terms(tfim.operators, tfim.coefficients, tfim.operators, tfim.coefficients)

		squared_tfim = TFIM_ham_1d(Js, hs, Ks)

		squared_tfim.operators = []
		squared_tfim.coefficients = []
		squared_tfim.operator_dict = dict()

		for op, coef in zip(squared_ops, squared_coeffs):
			squared_tfim.add_operator(op, coef)

		squared_SB = BifurcationRG(squared_tfim, growth_rate=2)

		squared_SB.update_res_eff_hams()

		if check:
			#explicitly build these two matrices
			#check that the latter is the square of the former

			assert SB.length<16

			orig_eigs = SB.ham.exactly_diagonalize()
			orig_squared = sorted(np.array(orig_eigs)**2)

			squared_eigs = squared_SB.ham.exactly_diagonalize()

			assert (np.max(np.abs(squared_eigs-orig_squared)/np.abs(orig_squared)))<10**-10

		#hjk
		"""
		Now select lioms and compute squares
		"""
		SB.diagonalize()
		squared_SB.diagonalize()

		energy_variance_numerator = 0.0

		energy_variance_denominator = 0.0

		for state in range(states_to_compare_per_rep):
			liom_pattern = 2*(np.random.rand(N)>0.5) - 1

			ham_energy = SB.give_energy(liom_pattern)
			squared_ham_energy = squared_SB.give_energy(liom_pattern)

			energy_variance_numerator += (squared_ham_energy - ham_energy**2)

			energy_variance_denominator += squared_ham_energy

		normed_energy_variance_list.append(energy_variance_numerator / energy_variance_denominator)

	"""
	finally return the mean
	"""

	return np.mean(normed_energy_variance_list)






			

"""
apply sbrg once
"""

def test_application(terms, coeffs, growth_rate, only_clifford = False):
	"""
	
	"""	
	#check that the inputs are appropriate
	assert len(terms) == len(coeffs)
	
	length = len(terms[0])
	for term in terms:
		assert len(term) == length

	#construct the hamiltonian
	Js = np.zeros(length)
	hs = np.zeros(length)
	Ks = np.zeros(length)

	obj = TFIM_ham_1d(Js, hs, Ks, periodic = False)

	for i, term in enumerate(terms):
		coeff = coeffs[i]

		obj.add_operator(term, coeff)

	#now add it to the bifurcation rg operators
	sbrg = BifurcationRG(obj, growth_rate)

	#update it
	sbrg.update_res_eff_hams()

	#apply sbrg step by step
	max_residual_index = sbrg.ham.max_residual_index()

	sbrg.clifford_rotate(max_residual_index, sbrg.N)

	if not only_clifford:
		max_residual_op = sbrg.ham.operators[max_residual_index]
		max_residual_coeff = sbrg.ham.coefficients[max_residual_index]

		Delta_ops, Delta_coeffs, Sigma_ops, Sigma_coeffs = sbrg.split_up_operators(max_residual_op)

		sbrg.schrieffer_wolff_rotate(max_residual_op, max_residual_coeff)

		Delta_ops, Delta_coeffs, Sigma_ops, Sigma_coeffs = sbrg.split_up_operators(max_residual_op)

		sbrg.N+=1
		sbrg.update_res_eff_hams()
	
	#next, collect the operators and coefficients
	operators = []
	coefficients = []

	#now collect all the terms and coefficients
	for op, coeff in zip(sbrg.ham.operators, sbrg.ham.coefficients):
		if coeff!=0:
			operators.append(op)
			coefficients.append(coeff)


	return operators, coefficients


def test_application(terms, coeffs, growth_rate, only_clifford = False):
	"""
	
	"""	
	#check that the inputs are appropriate
	assert len(terms) == len(coeffs)
	
	length = len(terms[0])
	for term in terms:
		assert len(term) == length

	#construct the hamiltonian
	Js = np.zeros(length)
	hs = np.zeros(length)
	Ks = np.zeros(length)

	obj = TFIM_ham_1d(Js, hs, Ks, periodic = False)

	for i, term in enumerate(terms):
		coeff = coeffs[i]

		obj.add_operator(term, coeff)

	#now add it to the bifurcation rg operators
	sbrg = BifurcationRG(obj, growth_rate)

	#update it
	sbrg.update_res_eff_hams()

	#apply sbrg step by step
	max_residual_index = sbrg.ham.max_residual_index()

	sbrg.clifford_rotate(max_residual_index, sbrg.N)

	if not only_clifford:
		max_residual_op = sbrg.ham.operators[max_residual_index]
		max_residual_coeff = sbrg.ham.coefficients[max_residual_index]

		Delta_ops, Delta_coeffs, Sigma_ops, Sigma_coeffs = sbrg.split_up_operators(max_residual_op)

		sbrg.schrieffer_wolff_rotate(max_residual_op, max_residual_coeff)

		Delta_ops, Delta_coeffs, Sigma_ops, Sigma_coeffs = sbrg.split_up_operators(max_residual_op)

		sbrg.N+=1
		sbrg.update_res_eff_hams()
	
	#next, collect the operators and coefficients
	operators = []
	coefficients = []

	#now collect all the terms and coefficients
	for op, coeff in zip(sbrg.ham.operators, sbrg.ham.coefficients):
		if coeff!=0:
			operators.append(op)
			coefficients.append(coeff)


	return operators, coefficients





def test_full_application(terms, coeffs, growth_rate, only_clifford = False):
	"""
	
	"""	
	#check that the inputs are appropriate
	assert len(terms) == len(coeffs)
	
	length = len(terms[0])
	for term in terms:
		assert len(term) == length

	#construct the hamiltonian
	Js = np.zeros(length)
	hs = np.zeros(length)
	Ks = np.zeros(length)

	obj = TFIM_ham_1d(Js, hs, Ks, periodic = False)

	for i, term in enumerate(terms):
		coeff = coeffs[i]

		obj.add_operator(term, coeff)

	#now add it to the bifurcation rg operators
	sbrg = BifurcationRG(obj, growth_rate)

	#update it
	sbrg.update_res_eff_hams()

	#apply all sbrg steps
	sbrg.diagonalize()
	
	#next, collect the operators and coefficients
	operators = []
	coefficients = []

	#now collect all the terms and coefficients
	for op, coeff in zip(sbrg.ham.operators, sbrg.ham.coefficients):
		if coeff!=0:
			operators.append(op)
			coefficients.append(coeff)


	return operators, coefficients


"""
terms = [(1, 0, 0, 2, 0), (0, 1, 0, 3, 0), (3, 3, 0, 0, 0), (1, 1, 0, 2, 0), (2, 3, 0, 0, 0), (1, 1, 1, 2, 0), (2, 1, 0, 0, 1)]

coeffs = [10**2.5, 10**2, 10**1.5, 10**1, 10**.5, 1, .1]

ops, coeffs = test_full_application(terms, coeffs, growth_rate=100, only_clifford=False)

for i, op in enumerate(ops):
	print(op, coeffs[i])
"""