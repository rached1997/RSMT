"""
Copyright 2020 Ryan (Mohammad) Solgi
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IP NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IP AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IP CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IP THE
SOFTWARE.
"""

###############################################################################
###############################################################################
###############################################################################
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
import numpy as np
import sys
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
from RSMT.Utils.vector_decoder import VectorDecoder
from RSMT.optimisation_algorithms.psnr import psnr_constraint

def _obj_wrapper_to_generate_trf_patches(func, original_patch, indices, whole_data, x):
    return func(x, original_patch, indices, whole_data)


def generate_transformed_patches(trf_vector, original_patches, indices, whole_data):
    vector_decoder = VectorDecoder(original_patches, indices, whole_data, trf_vector)
    vector_decoder.apply_transformations()
    patches = vector_decoder.patches
    return patches


def compute_adverserial_patches(kwargs, transformed_patches_over_gen):
    if 'transformed_outputs' in kwargs['tracker'].temp_data.keys():
        transformed_outputs = kwargs['tracker'].temp_data['transformed_outputs']
    else:
        transformed_inputs = transformed_patches_over_gen.reshape((-1, transformed_patches_over_gen.shape[2],
                                                                   transformed_patches_over_gen.shape[3],
                                                                   transformed_patches_over_gen.shape[4], 1))
        transformed_outputs = kwargs['main_model'].predict(transformed_inputs)

    adverserial_indices = np.argmax(transformed_outputs, axis=1) != np.argmax(kwargs["target"])

    return adverserial_indices, transformed_outputs


class geneticalgorithm():
    """  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations
    """

    #############################################################
    def __init__(self, function, dimension, args=(), kwargs={}, variable_type='bool', variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=10, algorithm_parameters={'max_num_iteration': None, 'population_size': 100,
                                                            'mutation_probability': 0.1, 'elit_ratio': 0.01,
                                                            'crossover_probability': 0.5, 'parents_portion': 0.3,
                                                            'crossover_type': 'uniform',
                                                            'max_iteration_without_improv': None, 'maxfixgen': 6},
                 convergence_curve=True, progress_bar=True):

        """
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        """
        self.__name__ = geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)), "function must be callable"

        self.f = function
        #############################################################
        # dimension

        self.dim = int(dimension)

        #############################################################
        # args

        self.args = args
        self.kwargs = kwargs

        self.generate_patches = partial(_obj_wrapper_to_generate_trf_patches, generate_transformed_patches,
                                        kwargs['original_patches'], kwargs['indices'], kwargs['whole_data'])
        #############################################################
        # input variable type

        assert (variable_type == 'bool' or variable_type == 'int' or variable_type == 'real'), \
            "\n variable_type must be 'bool', 'int', or 'real'"
        #############################################################
        # input variables' type (MIXED)     

        if variable_type_mixed is None:

            if variable_type == 'real':
                self.var_type = np.array([['real']] * self.dim)
            else:
                self.var_type = np.array([['int']] * self.dim)

        else:
            assert (type(variable_type_mixed).__module__ == 'numpy'), \
                "\n variable_type must be numpy array"

            assert (len(variable_type_mixed) == self.dim), \
                "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i == 'real' or i == 'int'), \
                    "\n variable_type_mixed is either 'int' or 'real' " + \
                    "ex:['int','real','real']" + \
                    "\n for 'boolean' use 'int' and specify boundary as [0,1]"

            self.var_type = variable_type_mixed
        #############################################################
        # input variables' boundaries 

        if variable_type != 'bool' or type(variable_type_mixed).__module__ == 'numpy':

            assert (type(variable_boundaries).__module__ == 'numpy'), \
                "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries) == self.dim), \
                "\n variable_boundaries must have a length equal dimension"

            for i in variable_boundaries:
                assert (len(i) == 2), \
                    "\n boundary for each variable must be a tuple of length two."

                assert (i[0] <= i[1]), \
                    "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        ############################################################# 
        # Timeout
        self.funtimeout = float(function_timeout)
        ############################################################# 
        # convergence_curve
        if convergence_curve:
            self.convergence_curve = True
        else:
            self.convergence_curve = False
        ############################################################# 
        # progress_bar
        if progress_bar:
            self.progress_bar = True
        else:
            self.progress_bar = False
        ############################################################# 
        ############################################################# 
        # input algorithm's parameters

        self.param = algorithm_parameters

        self.pop_s = int(self.param['population_size'])

        assert (1 >= self.param['parents_portion'] >= 0), \
            "parents_portion must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (1 >= self.prob_mut >= 0), \
            "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert (1 >= self.prob_cross >= 0), \
            "mutation_probability must be in range [0,1]"

        assert (1 >= self.param['elit_ratio'] >= 0), \
            "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (self.par_s >= self.num_elit), \
            "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration'] is None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or self.c_type == 'two_point'), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] is None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

        self.maxfixgen = self.param['maxfixgen']

        #############################################################

    def run(self):

        #############################################################
        # Initial Population

        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')

        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)
        mutations = np.zeros((self.pop_s, self.dim + 1))
        var = np.zeros((self.pop_s, self.dim))

        # vector_encoder = VectorEncoder(patch.shape, patch_index)
        # var = vector_encoder.construct_random_tr_vector()
        # solo[:-1] = var.copy()

        for i in self.integers[0]:
            var[:, i] = np.random.randint(self.var_bound[i][0], self.var_bound[i][1] + 1, size=self.pop_s)
            mutations[:, i] = var[:, i].copy()
        for i in self.reals[0]:
            var[:, i] = self.var_bound[i][0] + np.random.random(size=self.pop_s) * (
                        self.var_bound[i][1] - self.var_bound[i][0])
            mutations[:, i] = var[:, i].copy()

        with ThreadPoolExecutor(max_workers=5) as exe:
            result = exe.map(self.generate_patches, var)

        transformed_patches = np.array(list(result))
        self.kwargs["psnr"] = psnr_constraint(transformed_patches, **self.kwargs)
        obj = self.evaluate(transformed_patches)

        adverserial_indices, original_softmax = compute_adverserial_patches(self.kwargs, transformed_patches)
        #  fx, x, iteration, indices, psnrs, adverserial_indices, original_softmax
        trf_data = self.kwargs['tracker'].format_trf_data(self.kwargs['tracker'].temp_data['fx'], var, 0,
                                                          self.kwargs['indices'],
                                                          self.kwargs['tracker'].temp_data['all_psnr'],
                                                          adverserial_indices, original_softmax)
        self.kwargs['tracker'].append_trf_data(trf_data)

        mutations[:, self.dim] = obj
        pop = mutations.copy()

        #############################################################

        #############################################################
        # Report
        self.report = []
        self.test_obj = obj[-1]
        self.best_variable = var[-1].copy()
        self.best_function = obj[-1]
        ##############################################################   

        t = 1
        counter = 0
        fix_gen = self.maxfixgen

        while t <= self.iterate:
            self.kwargs['tracker'].format_ratio_data(self.kwargs['tracker'].temp_data['all_psnr'],
                                                     self.kwargs["psnr"] >= 0, self.best_function, adverserial_indices)

            if self.progress_bar == True:
                self.progress(t, self.iterate, status="GA is running...")
            #############################################################
            # Sort
            pop = pop[pop[:, self.dim].argsort()]
            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
            else:
                counter += 1
            #############################################################
            # Report

            self.report.append(pop[0, self.dim])

            ##############################################################         
            # Normalizing objective function

            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1

            #############################################################        
            # Calculate probability

            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            #############################################################        
            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            par[:self.num_elit] = pop[:self.num_elit].copy()

            index = np.searchsorted(cumprob, np.random.random(size=self.par_s - self.num_elit))
            par[self.num_elit:self.par_s] = pop[index].copy()

            par_count = 0
            while par_count == 0:
                ef_par_list = np.random.choice([True, False], (self.par_s,), p=[self.prob_cross, 1 - self.prob_cross])
                if len(np.where(ef_par_list == True)[0]) >= 1:
                    par_count += 1

            ef_par = par[ef_par_list].copy()

            #############################################################  
            # New generation
            mutations = np.zeros((int(np.round((self.pop_s - self.par_s) / 2)), self.dim + 1))
            indices = np.arange(self.par_s, self.pop_s, 2)
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

            pop[:self.par_s] = par.copy()

            r1 = np.random.randint(0, par_count, size=int(np.round((self.pop_s - self.par_s) / 2)))
            r2 = np.random.randint(0, par_count, size=int(np.round((self.pop_s - self.par_s) / 2)))
            pvar1 = ef_par[r1, : self.dim].copy()
            pvar2 = ef_par[r2, : self.dim].copy()

            ch = self.cross(pvar1, pvar2, self.c_type)
            ch1 = ch[0].copy()
            ch2 = ch[1].copy()

            ch1 = self.mut(ch1)
            ch2 = self.mutmidle(ch2, pvar1, pvar2)
            mutations[:, : self.dim] = ch1.copy()
            with ThreadPoolExecutor(max_workers=5) as exe:
                result = exe.map(self.generate_patches, ch1)

            transformed_patches = np.array(list(result))
            self.kwargs["psnr"] = psnr_constraint(transformed_patches, **self.kwargs)
            obj = self.evaluate(transformed_patches)

            adverserial_indices, original_softmax = compute_adverserial_patches(self.kwargs, transformed_patches)
            trf_data = self.kwargs['tracker'].format_trf_data(self.kwargs['tracker'].temp_data['fx'], ch1, t,
                                                              self.kwargs['indices'],
                                                              self.kwargs['tracker'].temp_data['all_psnr'],
                                                              adverserial_indices, original_softmax)
            self.kwargs['tracker'].append_trf_data(trf_data)

            mutations[:, self.dim] = obj
            pop[indices] = mutations.copy()
            mutations[:, : self.dim] = ch2.copy()
            with ThreadPoolExecutor(max_workers=5) as exe:
                result = exe.map(self.generate_patches, ch2)

            transformed_patches = np.array(list(result))
            self.kwargs["psnr"] = psnr_constraint(transformed_patches, **self.kwargs)
            obj = self.evaluate(transformed_patches)

            adverserial_indices, original_softmax = compute_adverserial_patches(self.kwargs, transformed_patches)
            trf_data = self.kwargs['tracker'].format_trf_data(self.kwargs['tracker'].temp_data['fx'], ch2, t,
                                                              self.kwargs['indices'],
                                                              self.kwargs['tracker'].temp_data['all_psnr'],
                                                              adverserial_indices, original_softmax)
            self.kwargs['tracker'].append_trf_data(trf_data)

            mutations[:, self.dim] = obj
            pop[indices + 1] = mutations.copy()

            generated_adverserial = np.sum(adverserial_indices)
            if generated_adverserial == 0:
                fix_gen -= 1
            else:
                fix_gen = self.maxfixgen
            #############################################################
            t += 1
            if (counter > self.mniwi) or (fix_gen == 0):
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    if self.progress_bar == True:
                        self.progress(t, self.iterate, status="GA is running...")

                    t += 1
                    self.stop_mniwi = True

        #############################################################
        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function': self.best_function}
        if self.progress_bar == True:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % self.best_variable)
        sys.stdout.write('\n\n Objective function:\n %s\n' % self.best_function)
        sys.stdout.flush()
        re = np.array(self.report)
        if self.convergence_curve == True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

        if fix_gen == 0:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without adverserial was met!')
        elif self.stop_mniwi == True:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')

        self.kwargs['tracker'].final_trf_write()
        self.kwargs['tracker'].format_ratio_data(self.kwargs['tracker'].temp_data['all_psnr'], self.kwargs["psnr"] >= 0,
                                                 self.best_function, adverserial_indices)
        self.kwargs['tracker'].add_vector_ratio(self.kwargs['indices'])
        return pop

    ##############################################################################
    ##############################################################################
    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim, size=x.shape[0])
            for i, el in enumerate(ran):
                ofs1[i, :el] = y[i, :el].copy()
                ofs2[i, :el] = x[i, :el].copy()

        if c_type == 'two_point':

            ran1 = np.random.randint(0, self.dim, size=x.shape[0])
            ran2 = np.random.randint(ran1, self.dim)
            for i in range(len(ran1)):
                ofs1[i, ran1[i]:ran2[i]] = y[i, ran1[i]:ran2[i]].copy()
                ofs2[i, ran1[i]:ran2[i]] = x[i, ran1[i]:ran2[i]].copy()

        if c_type == 'uniform':
            ran = np.random.choice([True, False], (x.shape[0], self.dim), p=[0.5, 0.5])
            ofs1[ran] = y[ran].copy()
            ofs2[ran] = x[ran].copy()

        return np.array([ofs1, ofs2])

    ###############################################################################

    def mut(self, x):
        mask = np.random.choice([True, False], (x.shape[0], self.dim,), p=[self.prob_mut, 1 - self.prob_mut])
        for i in range(self.dim):
            indices = np.where(mask[:, i] == True)[0]
            if i in self.integers[0]:
                x[indices, i] = np.random.randint(self.var_bound[i][0], self.var_bound[i][1] + 1, size=len(indices))
            else:
                x[indices, i] = self.var_bound[i][0] + np.random.random(size=len(indices)) * (
                            self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ###############################################################################
    def mutmidle(self, x, p1, p2):
        mask = np.random.choice([True, False], (x.shape[0], self.dim,), p=[self.prob_mut, 1 - self.prob_mut])
        for i in range(self.dim):
            indices1 = np.where((mask[:, i] == True) * (p1[:, i] < p2[:, i]))
            indices2 = np.where((mask[:, i] == True) * (p1[:, i] > p2[:, i]))
            indices3 = np.where((mask[:, i] == True) * (p1[:, i] == p2[:, i]))

            if (indices1 == []) and (indices2 == []) and (indices3 == []):
                if i in self.integers[0]:
                    x[indices1[0], i] = np.random.randint(p1[indices1[0], i], p2[indices1[0], i], size=len(indices1[0]))
                    x[indices2[0], i] = np.random.randint(p2[indices2[0], i], p1[indices2[0], i], size=len(indices2[0]))
                    x[indices3[0], i] = np.random.randint(self.var_bound[i][0], self.var_bound[i][1] + 1)
                else:
                    x[indices1, i] = p1[indices1, i] + np.random.random(size=len(indices1)) * (
                            p2[indices1, i] - p1[indices1, i])
                    x[indices2, i] = p2[indices2, i] + np.random.random(size=len(indices2)) * (
                            p1[indices2, i] - p2[indices2, i])
                    x[indices3, i] = self.var_bound[i][0] + np.random.random(size=len(indices3)) * (
                            self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ###############################################################################
    def evaluate(self, X):
        # import multiprocessing
        # mp_pool = multiprocessing.Pool(2)
        #
        # fx = np.array(mp_pool.map(np.sum, X))

        # fx = np.zeros((X.shape[0],))
        # for i in range(X.shape[0],):
        #     fx[i] = self.f(X[i], **self.kwargs)
        fx = self.f(X, **self.kwargs)
        return fx

    ###############################################################################
    # def sim(self, X):
    #     self.temp = X.copy()
    #     obj = None
    #     try:
    #         obj = func_timeout(self.funtimeout, self.evaluate)
    #     except FunctionTimedOut:
    #         print("given function is not applicable")
    #     assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
    #                           "func_timeout: the given function does not provide any output"
    #     return obj

    ###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
    ###############################################################################
###############################################################################
