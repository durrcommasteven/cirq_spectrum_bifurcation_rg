"""
Here we implement tests of Spectrum Bifurcation Tools

We test each method contained in BifurcationRG and TFIM_ham_1d
confirming that each does precisely what its description entails
"""
import cirq

from cirq_bifurcation_rg import *
import random
from functools import reduce
from time import time

"""
Define the tests for BifurcationRG
"""

integer_to_op_map = {
    0: cirq.I,
    1: cirq.X,
    2: cirq.Y,
    3: cirq.Z
}

def test_is_leading_term_of_correct_form():
    """
    is_leading_term_of_correct_form(self, op_index, target_position) 
    """
    
    # Create a Hamiltonian with 4 terms
    ops = [
        (0,0,1,1),
        (1,1,2,2),
        (0,0,3,1),
        (1,3,2,1)
    ]

    coefficients = np.random.rand(4)

    hamiltonian = cirq.PauliSum()

    sbrg = BifurcationRG(hamiltonian)

    pauli_terms = [
        np.prod([integer_to_op_map[integer] for integer in op]) for i, op in enumerate(ops)
    ]

    term_indices, target_position_list = [0, 0, 1, 2, 2, 3], [2, 3, 0, 2, 3, 2]
    correct_bool_list = [True, False, True, True, True, False]

    for term_index, correct_bool, target_position in enumerate(term_indices, correct_bool_list, target_position_list):

        term = pauli_terms[term_index]
        is_correct_type = sbrg.is_leading_term_of_correct_form(term, target_position)
        assert is_correct_type == correct_bool, "Check the leading_term_of_correct_form function"


#test_executeSwap()
def test_clifford_from_type():
    """
    clifford_from_type takes an operator index, the type of rotation and the target positon, and generates a dictionary R which contains the rotations
    required to bring the largest operating term to the form [000]3[000]
    Note that clifford_from_type does not modify the Hamiltonian in any way
    """
    
    ## we will do the testing by checking the function outputs one-by-one on multiple examples
    ## create a dummy Hamiltonian which will contain all the operators to be tested
    
    
    ################## Type 1 testing ##################
    ## op --- target_pos --- R_exp
    ### Without swap : op = (0,3,0,0,1,3,3,1,0)  --- 4 --- R_exp['RC41'] = (0,3,0,0,2,3,3,1,0) and R_exp['RC41_coef'] = -1
    ### With swap: op = (3,0,0,3,0,2,1,3,0) -- 4 -- R_exp['SWAP'] = [4,6], R_exp['RC41'] = (3,0,0,3,2,2,0,3,0) and R_exp['RC41_coef'] = -1
    
    
    ops = [(0,3,0,0,1,3,3,1,0), (3,0,0,3,0,2,1,3,0)]    
    coefs = np.random.rand(2)
    hamiltonian = gen_Hamil(ops, coefs)
    sbrg = BifurcationRG(hamiltonian)
    
    # type 1 without swap test
    target_position = 4
    op_index = 0
    rot_type = 1
    R_exp = {'RC41': (0,3,0,0,2,3,3,1,0), 'RC41_coef' : -1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    
    # type 1 with swap test
    target_position = 4
    op_index = 1
    rot_type = 1
    R_exp = {'SWAP':[4,6],'RC41': (3,0,0,3,2,2,0,3,0), 'RC41_coef' : -1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    ######################################################
    
    
    ################## Type 2 testing ##################
    ## op --- target_pos --- R_exp
    ### Without swap : op = (3,0,3,0,2,3,0,3,2)  --- 4 --- R_exp['RC41'] = (3,0,3,0,1,3,0,3,2) and R_exp['RC41_coef'] = 1
    ### With swap: op = (3,0,3,3,3,0,3,2,0) -- 2 -- R_exp['SWAP'] = [2,7], R_exp['RC41'] = (3,0,1,3,3,0,3,3,0) and R_exp['RC41_coef'] = 1
    
    
    ops = [(3,0,3,0,2,3,0,3,2), (3,0,3,3,3,0,3,2,0)]    
    coefs = np.random.rand(2)
    hamiltonian = gen_Hamil(ops, coefs)
    sbrg = BifurcationRG(hamiltonian)
    
    # type 2 without swap test
    target_position = 4
    op_index = 0
    rot_type = 2
    R_exp = {'RC41': (3,0,3,0,1,3,0,3,2), 'RC41_coef' : 1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)

    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    
    # type 2 with swap test
    target_position = 2
    op_index = 1
    rot_type = 2
    R_exp = {'SWAP':[2,7],'RC41': (3,0,1,3,3,0,3,3,0), 'RC41_coef' : 1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    ######################################################
    
    
    
    
    ################## Type 2 testing ##################
    ## op --- target_pos --- R_exp
    ### With swap: op = (0,3,0,0,3,3,0,0,3) --- 3 --- R_exp['SWAP'] = [3,4] ---  R_exp['RC41'] = (0,3,0,2,0,3,0,0,3) and R_exp['RC41_coef'] = 1
    ###                                                     R_exp['RC42'] = (0,0,0,2,0,0,0,0,0)  --- R_exp['RC42_coef'] = -1
    ### no swap: op = (3,0,0,3,0,0,0,3,3) --- 0 ---   R_exp['RC41'] = (2,0,0,3,0,0,0,3,3) and R_exp['RC41_coef'] = 1
    ###                                                     R_exp['RC42'] = (2,0,0,0,0,0,0,0,0) --- R_exp['RC42_coef'] = -1
    ### no swap: op = (3,0,0,3,0,0,0,3,3) --- 8 ---   R_exp['RC41'] = (3,0,0,3,0,0,0,3,2) and R_exp['RC41_coef'] = 1
    ###                                                     R_exp['RC42'] = (0,0,0,0,0,0,0,0,2) --- R_exp['RC42_coef'] = -1
    
    ops = [(0,3,0,0,3,3,0,0,3), (3,0,0,3,0,0,0,3,3)]    
    coefs = np.random.rand(2)
    hamiltonian = gen_Hamil(ops, coefs)
    sbrg = BifurcationRG(hamiltonian)
    
    # test1
    target_position = 3
    op_index = 0
    rot_type = 3
    R_exp = {'SWAP':[3,4], 'RC41': (0,3,0,2,0,3,0,0,3), 'RC41_coef' : 1 , 'RC42': (0,0,0,2,0,0,0,0,0), 'RC42_coef' : -1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    
    
    # test1
    target_position = 0
    op_index = 1
    rot_type = 3
    R_exp = {'RC41': (2,0,0,3,0,0,0,3,3), 'RC41_coef' : 1 , 'RC42': (2,0,0,0,0,0,0,0,0), 'RC42_coef' : -1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    
    
    # test1
    target_position = 8
    op_index = 1
    rot_type = 3
    R_exp = {'RC41': (3,0,0,3,0,0,0,3,2), 'RC41_coef' : 1 , 'RC42': (0,0,0,0,0,0,0,0,2), 'RC42_coef' : -1}
    R_func = sbrg.clifford_from_type(rot_type, op_index, target_position)
    assert R_exp == R_func, "Incorrect Clifford rotation obtained"
    ######################################################
    
    print("test_clifford_from_type passed without problems")
    
    
# test_clifford_from_type()

def test_conjugate_with_clifford():
    """
    Test the apply_clifford function
    apply_clifford takes in a Clifford rotation dictionary R and rotates all the terms in the hamiltonian accordingly.

    Returns
    -------
    None.

    """    
    
    #create a test Hamiltonian
    ops = [(3,3,3,1,0,2),
           (3,3,0,0,0,0),
           (3,0,1,2,1,1),
           (3,0,0,0,0,0),
           (0,3,0,0,0,0),
           (0,0,0,3,3,0),
           (0,3,0,1,0,2),
           (3,0,2,2,1,2)]

    coefs = [0.5, 1000, 2, 3000, 4000, 5, 20, 7]

    resop_inds = [0,2,5,6,7]
    effop_inds = [1,3,4]

    pauli_terms = [
        np.prod([integer_to_op_map[integer] for integer in op]) for i, op in enumerate(ops)
    ]

    residual_ops = np.sum([
        coefs[i] * pauli_terms[i] for i in resop_inds
    ])

    effective_ops = np.sum([
        coefs[i] * pauli_terms[i] for i in effop_inds
    ])

    hamiltonian = np.sum(
        coefs[i] * pauli_terms[i] for i in range(len(ops))
    )

    largest_term_ind = 6
    
    sbrg = BifurcationRG(hamiltonian)
    sbrg.N = 2 # implies that target position = 2
    target_position = sbrg.N 
    # Have manually determined that R should be:
    #R = {'SWAP' : [2,3], 'RC41' : (0,3,2,0,0,2), 'RC41_coef' : -1}
    correct_clifford = [
        -1 * sbrg.c4_rotation( -1 * cirq.Z(cirq.LineQubit(1)) * cirq.Z(cirq.LineQubit(2)) * cirq.Z(cirq.LineQubit(5))),
        cirq.SWAP(cirq.LineQubit(2), cirq.LineQubit(3))]

    clifford = sbrg.get_clifford_operations(
        leading_term,
        target_position,
    )

    # do the rotation
    sbrg.apply_clifford(R)
    
    #Expected operators
    exp_operators = [(3, 3, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0),
                     (3, 0, 2, 2, 1, 2), (3, 0, 3, 3, 0, 0), (3, 3, 0, 1, 1, 3), 
                     (0, 3, 1, 0, 3, 2), (0, 0, 3, 0, 0, 0)]
    exp_coefs = [1000, 3000, 4000, 7, (0.5-0j), (2-0j), (-5+0j), (20-0j)]
    
    assert set(exp_operators) == set(sbrg.ham.operators)
    assert set(exp_coefs) == set(sbrg.ham.coefficients)
    
    exp_resops = {(3, 0, 2, 2, 1, 2): 3, (3, 0, 3, 3, 0, 0): 4, (3, 3, 0, 1, 1, 3): 5, (0, 3, 1, 0, 3, 2): 6, (0, 0, 3, 0, 0, 0): 7}
    exp_effops = {(3, 3, 0, 0, 0, 0): 0, (3, 0, 0, 0, 0, 0): 1, (0, 3, 0, 0, 0, 0): 2}
    
    assert set(exp_resops.keys()) == set(sbrg.ham.residual_ops.keys()) and set(exp_effops.keys()) == set(sbrg.ham.effective_ops.keys())

    #assert exp_resops == sbrg.hamiltonian.residual_ops and exp_effops == sbrg.hamiltonian.effective_ops
    
    
    
    ## Explicitly printing all the outputs
    # for i in range(len(sbrg.hamiltonian.operators)):
    #     print(sbrg.hamiltonian.operators[i], sbrg.hamiltonian.coefficients[i])
    
    # print(sbrg.hamiltonian.operator_dict)
    # print("Residual operators:")
    # print(sbrg.hamiltonian.residual_ops)
    # print("effective ops:")
    # print(sbrg.hamiltonian.effective_ops)
    
    
# test_apply_clifford()
    
    
    
#%% 

from Master_Spectrum_Bifurcation_Tools import *
import random
from functools import reduce
from time import time



# ops = [(0,3,2,1,2,1)]
# op_index = 0
# target_position = 2
# coefs = [1]
# hamiltonian = gen_Hamil(ops, coefs)
# sbrg = BifurcationRG(hamiltonian)
# rot_type = sbrg.identify_clif_rot_type(op_index, target_position)
# print(rot_type)
# R = sbrg.clifford_from_type(rot_type, op_index, target_position)
# print(R)
    
def test_clifford_rotate():
    #create a test Hamiltonian
    ops = [(3,3,3,1,0,2),
           (3,3,0,0,0,0),
           (3,0,1,2,1,1),
           (3,0,0,0,0,0),
           (0,3,0,0,0,0),
           (0,0,0,3,3,0),
           (0,3,0,1,0,2),
           (3,0,2,2,1,2)]
    coefs = [0.5, 1000, 2, 3000, 4000, 5, 20, 7]
    resop_inds = [0,2,5,6,7]
    effop_inds = [1,3,4]
    res_ops_init = dict()
    eff_ops_init = dict()
    for i in range(len(ops)):
        if i in resop_inds:
            res_ops_init[ops[i]] = i
        else:
            eff_ops_init[ops[i]] = i
                
    hamiltonian = gen_Hamil(ops, coefs)
    hamiltonian.residual_ops = res_ops_init.copy()
    hamiltonian.effective_ops = eff_ops_init.copy()
    
    
    sbrg = BifurcationRG(hamiltonian)
    sbrg.N = 2 # implies that target position = 2
    op_index = 6
    target_position = sbrg.N 
    sbrg.clifford_rotate(op_index, target_position)
    
    
    
    #Expected operators
    exp_operators = [(3, 3, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0), (0, 3, 0, 0, 0, 0),
                     (3, 0, 2, 2, 1, 2), (3, 0, 3, 3, 0, 0), (3, 3, 0, 1, 1, 3), 
                     (0, 3, 1, 0, 3, 2), (0, 0, 3, 0, 0, 0)]
    exp_coefs = [1000, 3000, 4000, 7, (0.5-0j), (2-0j), (-5+0j), (20-0j)]
    
    assert set(exp_operators) == set(sbrg.ham.operators)
    assert set(exp_coefs) == set(sbrg.ham.coefficients)
    
    exp_resops = {(3, 0, 2, 2, 1, 2): 3, (3, 0, 3, 3, 0, 0): 4, (3, 3, 0, 1, 1, 3): 5, (0, 3, 1, 0, 3, 2): 6, (0, 0, 3, 0, 0, 0): 7}
    exp_effops = {(3, 3, 0, 0, 0, 0): 0, (3, 0, 0, 0, 0, 0): 1, (0, 3, 0, 0, 0, 0): 2}
    
    #assert exp_resops == sbrg.hamiltonian.residual_ops and exp_effops == sbrg.hamiltonian.effective_ops
    
    
    assert set(exp_resops.keys()) == set(sbrg.ham.residual_ops.keys()) and set(exp_effops.keys()) == set(sbrg.ham.effective_ops.keys())

     # Explicitly printing all the outputs
    # for i in range(len(sbrg.hamiltonian.operators)):
    #     print(sbrg.hamiltonian.operators[i], sbrg.hamiltonian.coefficients[i])
    # print(sbrg.hamiltonian.operator_dict)
    # print("Residual operators:")
    # print(sbrg.hamiltonian.residual_ops)
    # print("effective ops:")
    # print(sbrg.hamiltonian.effective_ops)


# test_clifford_rotate()
 

#%%

def test_is_effective():
    """
    this function tests whether a term is 
    ready to be added to the residual hamiltonian
    """

    """
    first create an object to test with
    """
    size = 10

    Js = np.random.rand(size)
    hs = np.random.rand(size)
    Ks = np.random.rand(size)

    ham = TFIM_ham_1d(Js, hs, Ks)

    test_object = BifurcationRG(ham)

    reps = 15

    for i in range(size):

        test_object.N = i

        eff_ops = [[random.randint(0, 1)*3 for _ in range(i)]+[0]*(size-i) for k in range(10)]
        res_ops = [[random.randint(1, 2) for _ in range(size)] for k in range(10)]

        for op in eff_ops:
            assert test_object.is_effective(op), "effective op not being recognized"

        for op in res_ops:
            assert not test_object.is_effective(op), "residual op not being recognized"


def test_split_up_operators():
    """
    in order to test this, we'll custom-make the hamiltonian with some test-operators

    note all that this function uses is self.hamiltonian.operators and self.hamiltonian.coefficients

    So our modifications can be restricted to those lists
    """
    """
    first create an object to test with
    """
    Js = np.random.rand(10)
    hs = np.random.rand(10)
    Ks = np.random.rand(10)

    ham = TFIM_ham_1d(Js, hs, Ks)

    test_object = BifurcationRG(ham)

    """
    now we'll make some operators to choose from 
    """
    ops = []
    #some weird cases
    ops.append([0]*10)
    ops.append([1]*10)
    ops.append([2]*10)
    ops.append([3]*10)
    ops.append([1]+[0]*9)
    ops.append([0]*9 + [1])
    ops.append([2]+[0]*9)
    ops.append([0]*9 + [2])
    ops.append([3]+[0]*9)
    ops.append([0]*9 + [3])

    #now lets add some random ones
    random_num = 10

    for _ in range(random_num):
        rand_op = [random.randint(0, 3) for _ in range(10)]
        ops.append(rand_op)

    #now lets add some random diagonalized ones
    random_num = 15

    for _ in range(random_num):
        rand_op = [random.randint(0, 1)*3 for _ in range(10)]
        ops.append(rand_op)

    #make all ops tuples
    ops = [tuple(op) for op in ops]

    """
    Now we will create subsets of these operators and coefficients, and 
    find the ones that commute and anticommute with a given op (comprising all zeros and one 3)

    we will confirm that the two sets of operators found explicitly agrees with the sets found using our 
    program 

    we will run this test reps time
    on randomly chosen subsets of the operators of size n
    """
    reps = 10
    N = 15

    for _ in range(reps):

        #first we create H0
        nonzero_op = random.randint(0, 9)
        H0_op = [0 if i!=nonzero_op else 3 for i in range(10)]

        #now lets make the H0 matrix
        mat_list_H0 = [pauli(p) for p in H0_op]
        fullmat_H0 = tensorproduct(mat_list_H0)

        #choose the ops
        selected_indices = np.random.choice(range(len(ops)), size = N, replace=False)
        selected_ops = [ops[i] for i in selected_indices]
        #now make some coefficients
        #take the coefficients
        #~1 in 3 will be complex
        selected_coeffs = [np.random.choice([0,1j], p = [.67, .33]) + 2*(np.random.rand()-.5) for op in selected_ops]

        #now find which of these commute

        commuting_ops = []
        commuting_coeffs = []
        anticommuting_ops = []
        anticommuting_coeffs = []

        for i, op in enumerate(selected_ops):
            mat_list_ops = [pauli(p) for p in op]
            fullmat_ops = tensorproduct(mat_list_ops)

            prod1 = fullmat_ops@fullmat_H0 
            prod2 = fullmat_H0@fullmat_ops

            if np.array_equal(prod1, prod2):
                #then the two commute
                commute = 1
            elif np.array_equal(prod1, -prod2):
                #then the two anticommute
                commute = -1
            else:
                #there should be no other case for tensor 
                #products of pure pauli operators
                assert False, "Operators neither commute nor anticommute"

            if commute == 1:
                commuting_ops.append(tuple(op))
                commuting_coeffs.append(selected_coeffs[i])
            else:
                anticommuting_ops.append(tuple(op))
                anticommuting_coeffs.append(selected_coeffs[i])

        #now we'll sort these so we can compare to the case executed by the test object
        #note we sort by the op list 
        zipped_commute = zip(commuting_ops, commuting_coeffs)
        commuting_ops, commuting_coeffs = zip(*sorted(zipped_commute))

        zipped_anticommute = zip(anticommuting_ops, anticommuting_coeffs)
        anticommuting_ops, anticommuting_coeffs = zip(*sorted(zipped_anticommute))

        """
        Now we'll calculate what should be the same lists with the 
        function, self.split_up_operators(op)
        """
        #modify which ops and coeffs are present in the hamiltonian's lists
        test_object.ham.operators = selected_ops
        test_object.ham.coefficients = selected_coeffs

        test_output = test_object.split_up_operators(H0_op)

        test_commuting_ops, test_commuting_coeffs, test_anticommuting_ops, test_anticommuting_coeffs = test_output

        #now we make sure things are sorted so we can compare to the explicitly computed case
        #they should be sorted in the same way as before
        zipped_test_commute = zip(test_commuting_ops, test_commuting_coeffs)
        test_commuting_ops, test_commuting_coeffs = zip(*sorted(zipped_test_commute))

        zipped_test_anticommute = zip(test_anticommuting_ops, test_anticommuting_coeffs)
        test_anticommuting_ops, test_anticommuting_coeffs = zip(*sorted(zipped_test_anticommute))

        #now lets assert that these match
        
        assert test_commuting_ops == commuting_ops, "the commuting ops dont match"

        assert test_anticommuting_ops == anticommuting_ops, "the anticommuting ops dont match"

        assert test_commuting_coeffs == commuting_coeffs, "the commuting coeffs dont match"

        assert test_anticommuting_coeffs == anticommuting_coeffs, "the anticommuting coeffs dont match"

    """
    all clear
    """



def test_schrieffer_wolff_rotate():
    """
    Here we'll set up test hams and enact schrieffer wolff by explicitly

    we'll compare the result to the result found by doing the function in 
    SpectrumBifurcation
    """

    size = 8
    reps = 15

    for _ in range(reps):
        #this gives us the number of new terms to keep
        branching_ratio = 1+2*np.random.rand()

        #this is the first non-identity matrix
        n = random.randint(2, size-2)

        h0_op = tuple(0 if i!=n else 3 for i in range(size))

        assert set(h0_op)!={0}, "h0_op is the identity"

        #the coeff should be large, and either positive or negative
        h0_coeff = (np.random.choice([-1, 1]))*(np.random.rand()+10)

        other_ops = []
        other_coeffs = []

        #now lets make some other operators
        #note there could be zero other operators here
        num_other = random.randint(0, 10)

        for j in range(num_other):

            op = tuple(random.randint(0, 1)*3 if i<n else random.randint(0, 3) for i in range(size))
            coeff = np.random.rand()*2 -1

            other_ops.append(op)
            other_coeffs.append(coeff)

        for op in other_ops:
            diag_part = op[:n]
            assert set(diag_part).issubset({0, 3})

            remaining_part = op[n:]
            assert set(diag_part).issubset({0, 1, 2, 3})




        #we're done making the test hamiltonian
        #next we'll run the function on SpectrumBifurcation to get the explicit result

        #now lets create an object and add these in 
        zero_coeffs = [0]*size
        ham = TFIM_ham_1d(zero_coeffs, zero_coeffs, zero_coeffs)
        test_object = BifurcationRG(ham)
        #set the branching ratio here
        test_object.branching_ratio = branching_ratio

        #add in all our operators
        for i, op in enumerate(other_ops):
            test_object.ham.add_operator(op, other_coeffs[i])

        test_object.ham.add_operator(h0_op, h0_coeff)

        #now apply schrieffer wolff
        test_object.schrieffer_wolff_rotate(h0_op, h0_coeff)

        #now get the hamiltonian that produced
        test_resulting_ops = test_object.ham.operators 
        test_resulting_coeffs = test_object.ham.coefficients

        #now turn this into a full matrix
        test_mat = 0*tensorproduct([pauli(0) for i in range(size)])

        for i, op in enumerate(test_resulting_ops):
            mat_list = [pauli(p) for p in op]
            op_mat = tensorproduct(mat_list)
            op_coeff = test_resulting_coeffs[i]

            test_mat = np.add(test_mat, op_coeff*op_mat, casting ='unsafe')

        #Now we compare to the expected result of
        #h0 + Delta + (h0.Sigma.Sigma)/(2 h0^2)
        #after taking into account the branching process

        #get sigmas
        Sigma_ops = []
        Sigma_coeffs = []

        #Deltas
        Delta_ops = []
        Delta_coeffs = []

        #the index that h0 has a 3
        h0_index = h0_op.index(3)

        for i, op in enumerate(other_ops):

            if op[h0_index] in (1, 2):
                #then it anticommutes
                Sigma_ops.append(op)
                Sigma_coeffs.append(other_coeffs[i])
            else:
                #then it commutes
                Delta_ops.append(op)
                Delta_coeffs.append(other_coeffs[i])

        #now we have the sigmas. It is possible for there to be no such sigmas
        #in this case, I'll go through the procedure regardless, setting its value to zero

        if not Sigma_ops:
            Sigma_ops = [tuple(0 for _ in range(size))]
            Sigma_coeffs = [0.0]

        #I'm going to assume that multiply terms works, as there exists a test for it above
        #and use it here so we have a result decomposed into a tensor product of pauli matrices
        ops_1, coeffs_1 = test_object.multiply_terms([h0_op], [h0_coeff], Sigma_ops, Sigma_coeffs)
        ops_2, coeffs_2 = test_object.multiply_terms(ops_1, coeffs_1, Sigma_ops, Sigma_coeffs)
        coeffs_2 = [c/(2*(h0_coeff**2)) for c in coeffs_2]

        #now we throw away terms if they are too small
        #if there are n sigma terms, we keep the largest in magnitude
        #test_object.growth_rate*n terms

        New_terms = list(zip(ops_2, coeffs_2))

        #all coeffs should be real here

        imaginary_part_1 = np.imag(test_object.ham.coefficients)

        if coeffs_2:
            #note coeffs_2 could simply not be present here
            imaginary_part_2 = np.imag(coeffs_2)
        else:
            imaginary_part_2 = 0

        #assert that these are very small

        assert np.max(np.abs(imaginary_part_1))<10**-10, "test object's coefficients are complex"
        assert np.max(np.abs(imaginary_part_2))<10**-10, "contructed coefficients are complex"


        if New_terms:

            New_terms.sort(key = lambda x: abs(x[-1]), reverse = True)

            num_to_keep = int(round(test_object.growth_rate*len(New_terms)))

            New_terms = New_terms[:num_to_keep]

            ops_3, coeffs_3 = zip(*New_terms)

        else:
            ops_3, coeffs_3  = [[], []]

        #finally, turn this into an explicit matrix
        #h0 + Delta + (h0.Sigma.Sigma)/(2 h0^2)
        #add h0
        explicit_mat = h0_coeff*tensorproduct([pauli(i) for i in h0_op])

        #Delta + (h0.Sigma.Sigma)/(2 h0^2)
        #add deltas
        for i, op in enumerate(Delta_ops):
            explicit_mat = np.add(explicit_mat, Delta_coeffs[i]*tensorproduct([pauli(p) for p in op]), casting = 'unsafe')
        
        #(h0.Sigma.Sigma)/(2 h0^2)
        #add new terms
        for i, op in enumerate(ops_3):
            explicit_mat = np.add(explicit_mat, coeffs_3[i]*tensorproduct([pauli(p) for p in op]), casting = 'unsafe')

        if not np.array_equal(explicit_mat, test_mat):
            diff = np.max(np.abs(explicit_mat- test_mat))
            
        #now we compare the resulting matrices to make sure the two match
        #check that resulting_mat==answer

        total_diff = np.sum(np.abs(test_mat-explicit_mat))
        total_mean = np.mean(np.abs(test_mat)+np.abs(explicit_mat))

        assert total_diff/total_mean < 10**-10, "error with Wolff schrieffer"


def test_update_res_eff_hams():
    """
    Here, we'll repeatedly update the terms in the hamiltonian
    and make sure they match what we expect

    Each time we run this, we'll shift test_object.n up by one

    we'll do this whole procedure reps times
    """

    reps = 15
    size = 10

    #start off the system
    zero_coeffs = [0]*size
    ham = TFIM_ham_1d(zero_coeffs, zero_coeffs, zero_coeffs)
    test_object = BifurcationRG(ham)
    test_object.update_res_eff_hams()

    init_res_dict = test_object.ham.residual_ops
    init_eff_dict = test_object.ham.effective_ops

    #load up the hamiltonian with operators which are effective up to self.n. At self.n+1
    #and on, they are residual. Then iterate through self.n, and make sure that we see what we expect to see
    #well also add in some completely effective ops

    variable_eff_op_num = 10
    always_eff_op_num = 1 #the identity
    always_res_op_num = 10

    #record the ops we added along with their coefficients
    custom_ops_dict = dict()

    effective_at_N = [0]*size

    for i in range(size):

        custom_dict_size_before = len(custom_ops_dict)

        for _ in range(variable_eff_op_num):
            #add an op which is effective at most at 
            #self.n = i+1. For n above that, it is residual

            op = [random.randint(0, 1)*3 for _ in range(i)]+[3]+[0 for _ in range(size - i - 1)]
            coeff = np.random.rand()+1

            assert len(op)==size

            test_object.ham.add_operator(tuple(op), coeff)

            if tuple(op) in custom_ops_dict:
                custom_ops_dict[tuple(op)] += coeff
            else:
                custom_ops_dict[tuple(op)] = coeff

        #how many ops are added here
        effective_at_N[i] = len(custom_ops_dict) - custom_dict_size_before

    for _ in range(always_res_op_num):
        #add an ops which are residual always

        op = [random.randint(1, 2) for _ in range(size)]
        coeff = np.random.rand()+1

        assert len(op)==size

        test_object.ham.add_operator(tuple(op), coeff)
        
        if tuple(op) in custom_ops_dict:
            custom_ops_dict[tuple(op)] += coeff
        else:
            custom_ops_dict[tuple(op)] = coeff

    #lets also add in the identity. This should always be effective
    identity_op = [0 for _ in range(size)]
    identity_coeff = 1

    test_object.ham.add_operator(tuple(identity_op), identity_coeff)

    if tuple(identity_op) in custom_ops_dict:
        custom_ops_dict[tuple(identity_op)] += identity_coeff
    else:
        custom_ops_dict[tuple(identity_op)] = identity_coeff

    """
    now we'll work our way up n and assert that 
    update_res_eff_hams reflects the correct changes
    """

    while test_object.N<=size:

        #first we'll update the residual and effective ops
        test_object.update_res_eff_hams()

        #now we can make assertions about what we expect to see

        #at first, we expect all the terms we added to be effective
        #then after each shift up of n, we expect res_op_num fewer to be residual
        res_dict = test_object.ham.residual_ops
        eff_dict = test_object.ham.effective_ops

        #remove the initial operators present
        for k in init_res_dict.keys():
            if k in eff_dict:
                #if it is in the dictionary, has anything been added to it
                #if not, the corresponding coefficient should be zero
                #otherwise, this is an added in value, dont disturb it
                if test_object.ham.coefficients[eff_dict[k]]==0:
                    eff_dict.pop(k)

            if k in res_dict:
                #if it is in the dictionary, has anything been added to it
                #if not, the corresponding coefficient should be zero
                #otherwise, this is an added in value, dont disturb it
                if test_object.ham.coefficients[res_dict[k]]==0:
                    res_dict.pop(k)
        
        for k in init_eff_dict.keys():
            if k in eff_dict:
                #if it is in the dictionary, has anything been added to it
                #if not, the corresponding coefficient should be zero
                #otherwise, this is an added in value, dont disturb it
                if test_object.ham.coefficients[eff_dict[k]]==0:
                    eff_dict.pop(k)

            if k in res_dict:
                #if it is in the dictionary, has anything been added to it
                #if not, the corresponding coefficient should be zero
                #otherwise, this is an added in value, dont disturb it
                if test_object.ham.coefficients[res_dict[k]]==0:
                    res_dict.pop(k)


        #how many operators do we expect to be effective
        #we expect variable_eff_op_num*self.n to be recognized as effective
        expected_effective_num = sum(effective_at_N[:test_object.N]) + always_eff_op_num

        #how many operators do we expect to be residual
        #variable_eff_op_num*(size - self.n) + always_res_op_num
        expected_residual_num = sum(effective_at_N[test_object.N:]) + always_res_op_num


        assert len(res_dict)==expected_residual_num, "residual num is wrong"

        assert len(eff_dict)==expected_effective_num, "effective num is wrong"

        """
        next, ensure that all coefficients are accurate
        """
        for op, index in res_dict.items():
            ham_coefficient = test_object.ham.coefficients[index]

            test_coefficient = custom_ops_dict[op]

            assert ham_coefficient == test_coefficient, "coefficients do not match"

        test_object.N+=1


def test_apply_diagonalization():
    """
    here we'll test the apply_diagonalization tool in a few ways
    """

    #1 make sure it diagonalizes the hamiltonian it was intended to 

    """
    First I go step by step, ensuring that the diagonalization procedure is recorded 
    and applied as intended
    """
    Js, hs, Ks = [np.random.rand(10)*2 -1 for _ in range(3)]
    SB = BifurcationRG(TFIM_ham_1d(Js, hs, Ks), growth_rate=2)

    while SB.N<SB.length:
        SB.apply_spectrum_bifurcation()
        ham = SB.ham

        #check that at this step, apply_diagonalization matches

        diag_tfim = SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only=False)

        #check that these are equal
        for op, i in ham.operator_dict.items():
            #hamiltonian coefficient

            coeff_1 = ham.coefficients[i]
            
            if coeff_1!=0:
                j = diag_tfim.operator_dict[op]

                coeff_2 = diag_tfim.coefficients[j]

                if coeff_1 != coeff_2:
                    assert False, "coefficients do not match"
                """
                SOMETHING IS WRONG 
                """

                #assert coeff_1==coeff_2, "unequal coefficients"

                del diag_tfim.operator_dict[op]

        #now there should only be zero ops left over in diag_tfim

        for op, i in diag_tfim.operator_dict.items():
            j = diag_tfim.operator_dict[op]

            coeff = diag_tfim.coefficients[j]

            assert coeff==0, "coefficients don't match"




    """
    now confirm that this works on 5 generic hamiltonians
    """
    reps = 5

    for rep in range(reps):

        length = rep+10

        Js, hs, Ks = [np.random.rand(length)*2 -1 for _ in range(3)]

        SB = BifurcationRG(TFIM_ham_1d(Js, hs, Ks), growth_rate=2)

        SB.diagonalize()

        #here's the diagonalized hamiltonian
        ham = SB.ham

        #now apply to the hamiltonian again
        diag_tfim = SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks))

        #check that these are equal
        for op, i in ham.operator_dict.items():
            #hamiltonian coefficient

            coeff_1 = ham.coefficients[i]
            
            if coeff_1!=0:
                j = diag_tfim.operator_dict[op]

                coeff_2 = diag_tfim.coefficients[j]

                #print(coeff_1, coeff_2)
                """
                SOMETHING IS WRONG 
                """

                assert coeff_1==coeff_2, "unequal coefficients"

                del diag_tfim.operator_dict[op]

        #now there should only be zero ops left over in diag_tfim

        for op, i in diag_tfim.operator_dict.items():
            j = diag_tfim.operator_dict[op]

            coeff = diag_tfim.coefficients[j]

            assert coeff==0, "coefficients don't match"

    """
    now test out this, enacting clifford_only = True
    """
    reps = 5

    length = 10

    for rep in range(reps):

        Js, hs, Ks = [np.random.rand(length)*2 -1 for _ in range(3)]

        tfim = TFIM_ham_1d(Js, hs, Ks)

        SB = BifurcationRG(tfim, growth_rate=2)

        SB.diagonalize()

        #here's the diagonalized hamiltonian
        ham = SB.ham

        #now apply to the hamiltonian again
        diag_tfim = SB.apply_diagonalization(TFIM_ham_1d(Js, hs, Ks), clifford_only = True)
        original_tfim = TFIM_ham_1d(Js, hs, Ks)

        #check that the nonzero terms in the hamiltonian remains the same
        initial_nonzero_coeffs = sorted([abs(x) for x in original_tfim.coefficients if x!=0])
        final_nonzero_coeffs = sorted([abs(x) for x in diag_tfim.coefficients if x!=0])

        for i, v in enumerate(initial_nonzero_coeffs):

            diff = abs(v-final_nonzero_coeffs[i])

            #print(diff)

            assert diff<10**-8, "clifford only issue"


    """
    finally, check that under the clifford circuit, the identity maps to the identity
    """

    reps = 5

    length = 10

    for rep in range(reps):

        Js, hs, Ks = [np.random.rand(length)*2 -1 for _ in range(3)]

        tfim = TFIM_ham_1d(Js, hs, Ks)

        SB = BifurcationRG(tfim, growth_rate=2)

        SB.diagonalize()

        #make an identity hamiltonian
        identity_ham = TFIM_ham_1d(Js, hs, Ks)

        identity_ham.operators = []
        identity_ham.coefficients = []
        identity_ham.operator_dict = dict()

        ident = tuple(0 for _ in range(length))
        identity_ham.add_operator(ident, 1)

        #now apply to the hamiltonian again
        diag_identity = SB.apply_diagonalization(identity_ham, clifford_only = True)

        #check that the nonzero terms in the hamiltonian remains the same

        nonzero_ident_coeffs = [x for x in diag_identity.coefficients if x!=0]

        assert len(nonzero_ident_coeffs)==1, "identity hamiltonian has wrong size"
        assert nonzero_ident_coeffs[0]==1, "identity hamiltonian has wrong coeff"

        for op, i in diag_identity.operator_dict.items():
            if op!=ident:
                assert diag_identity.coefficients[i] == 0, "nonidentity present"

            else:
                assert diag_identity.coefficients[i] == 1, "identity has incorrect coefficient"

        assert ident in diag_identity.operator_dict, "no identity present in diag_identity"
        identity_index = diag_identity.operator_dict[ident]

        assert diag_identity.coefficients[identity_index] == 1, "diag_identity has no identity element"

    print("apply_diagonalization passes tests \n")
            




"""
Define the tests for TFIM_ham_1d

define these after we've finished up with the rg tests
"""


































"""
Run the tests for BifurcationRG
"""


test_algebra()
print("algebra passes tests \n")

test_multiply_ops()
print("multiply_ops passes tests \n")

test_multiply_basic_terms()
print("multiply_basic_terms passes tests \n")

test_multiply_terms()
print("multiply_terms passes tests \n")

test_sum_terms()
print("sum_terms passes tests \n")

test_clifford_rotate()
print("clifford_rotate passes tests \n")

test_is_leading_term_of_correct_form()
print("algebra passes tests \n")

test_identify_clif_rot_type()
print("is_leading_term_of_correct_form passes tests \n")

test_clifford_from_type()
print("clifford_from_type passes tests \n")

test_executeSwap()
print("executeSwap passes tests \n")

test_apply_clifford()
print("apply_clifford passes tests \n")

test_multiply_op_list_for_clifford()
print("multiply_op_list_for_clifford passes tests \n")

test_commute_or_anti()
print("commute_or_anti passes tests \n")

test_is_effective()
print("is_effective passes tests \n")

#test_split_up_operators()
#print("split_up_operators passes tests \n")

#test_replace_terms()
#print("replace_terms passes tests \n")

test_schrieffer_wolff_rotate()
print("schrieffer_wolff_rotate passes tests \n")

test_update_res_eff_hams()
print("update_res_eff_hams passes tests \n")

"""
Run the tests for TFIM_ham_1d

fill these in below later
"""






















