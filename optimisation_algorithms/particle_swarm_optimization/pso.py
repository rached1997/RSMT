from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
import numpy as np

from RSMT.Utils.vector_decoder import VectorDecoder
from RSMT.Utils.vector_encoder import VectorEncoder


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


def _obj_wrapper_to_generate_trf_patches(func, original_patch, indices, whole_data, x):
    return func(x, original_patch, indices, whole_data)


def generate_transformed_patches(trf_vector, original_patches, indices, whole_data):
    vector_decoder = VectorDecoder(original_patches, indices, whole_data, trf_vector)
    vector_decoder.apply_transformations()
    patches = vector_decoder.patches
    return patches


def _obj_wrapper(func, args, kwargs, x):
    return func(x, *args, **kwargs)


def _is_feasible_wrapper(func, x):
    # return np.all(func(x) >= 0)

    # return func(x) >= 0
    return func(x)


def _cons_none_wrapper(x):
    return np.array([0])


def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
    return np.array([y(x, *args, **kwargs) for y in ieqcons])


def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
    return np.array(f_ieqcons(x, *args, **kwargs))


class PSO:
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========
    func : function
        The function to be minimized
    lb : array
        The lower bounds of the design variable(s)
    ub : array
        The upper bounds of the design variable(s)

    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    processes : int
        The number of processes to use to evaluate objective function and
        constraints (default: 1)
    particle_output : boolean
        Whether to include the best per-particle position and the objective
        values at those.

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``
    p : array
        The best known position per particle
    pf: arrray
        The objective values at each position in p

    """

    def __init__(self, func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, swarmsize=100, omega=1, phip=0.5,
                 phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8, debug=False, processes=1, particle_output=False,
                 maxfixgen=6):

        assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        # modifications #########################################
        assert np.all(ub >= lb), 'All upper-bound values must be greater than lower-bound values'

        self.vhigh = np.abs(ub - lb)
        self.vlow = - self.vhigh
        self.lb = lb
        self.ub = ub

        # modifications #########################################
        # whole_data, patch, model_name, dataset_name, quant_meth, target, main_model, interpreter, evaluator, layer_names, patch_index = args

        # Initialize objective function
        self.obj = partial(_obj_wrapper, func, args, kwargs)

        self.generate_patches = partial(_obj_wrapper_to_generate_trf_patches, generate_transformed_patches,
                                        kwargs['original_patches'], kwargs['indices'], kwargs['whole_data'])

        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)

        self.is_feasible = partial(_is_feasible_wrapper, cons)

        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            self.mp_pool = multiprocessing.Pool(processes)

        # Initialize the particle swarm ############################################
        self.S = swarmsize
        self.D = len(lb)  # the number of dimensions each particle has

        # Initialize the PSO params ############################################
        self.kwargs = kwargs

        # Initialize the PSO params ############################################
        self.processes = processes
        self.debug = debug
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.maxfixgen = maxfixgen
        self.particle_output = particle_output

    def run(self):
        # modifications #########################################
        # x = np.random.rand(self.S, self.D)  # particle positions

        vector_encoder = VectorEncoder(self.kwargs['original_patches'][0].shape, self.kwargs['indices'][0])
        x = np.zeros((self.S, self.D))  # particle positions
        for i in range(self.S):
            x[i] = vector_encoder.construct_random_tr_vector()

        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(self.S)  # current particle function values
        fs = np.zeros(self.S, dtype=bool)  # feasibility of each particle
        fp = np.ones(self.S) * np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value

        # Initialize the particle's position
        # x = lb + x * (ub - lb)

        # Calculate objective and constraints for each particle
        # for i in range(self.S):
        # fs[i] = self.is_feasible(x[i, :])
        # if fs[i] == True:
        # fx[i] = self.obj(x[i, :])

        with ThreadPoolExecutor(max_workers=5) as exe:
            result = exe.map(self.generate_patches, x)
        # result = None
        # for i in range(len(x)):
        #     patch = generate_transformed_patches(x[i], self.kwargs['original_patches'], self.kwargs['indices'],
        #                                          self.kwargs['whole_data'])
        #     patch = patch.reshape((1,patch.shape[0],patch.shape[1],patch.shape[2],patch.shape[3]))
        #     if result is None:
        #         result=patch
        #     else:
        #         result = np.append(result, patch, axis=0)

        transformed_patches = np.array(list(result))
        fs = self.is_feasible(transformed_patches) >= 0
        self.kwargs["psnr"] = fs
        fx = self.obj(transformed_patches)

        psnrs = self.kwargs['tracker'].temp_data['all_psnr']
        all_fx = self.kwargs['tracker'].temp_data['fx']
        adverserial_indices, original_softmax = compute_adverserial_patches(self.kwargs, transformed_patches)
        trf_data = self.kwargs['tracker'].format_trf_data(all_fx, x, 0, self.kwargs['indices'], psnrs,
                                                          adverserial_indices, original_softmax)
        self.kwargs['tracker'].append_trf_data(trf_data)

        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]

        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()

        # Initialize the particle's velocity
        v = self.vlow + np.random.rand(self.S, self.D) * (self.vhigh - self.vlow)

        # Iterate until termination criterion met ##################################
        it = 1
        fix_gen = self.maxfixgen
        while it <= self.maxiter:
            self.kwargs['tracker'].format_ratio_data(psnrs, fs, fg, adverserial_indices)
            rp = np.random.uniform(size=(self.S, self.D))
            rg = np.random.uniform(size=(self.S, self.D))

            # Update the particles velocities
            v = self.omega * v + self.phip * rp * (p - x) + self.phig * rg * (g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < self.lb
            masku = x > self.ub
            x = x * (~np.logical_or(maskl, masku)) + self.lb * maskl + self.ub * masku

            # Update objectives and constraints

            # for i in range(self.S):
            # fs[i] = self.is_feasible(x[i, :])
            # if fs[i] == True:
            # fx[i] = self.obj(x[i, :])
            with ThreadPoolExecutor(max_workers=5) as exe:
                result = exe.map(self.generate_patches, x)

            transformed_patches = np.array(list(result))
            fs = self.is_feasible(transformed_patches) >= 0
            self.kwargs["psnr"] = fs

            fx = self.obj(transformed_patches)

            psnrs = self.kwargs['tracker'].temp_data['all_psnr']
            all_fx = self.kwargs['tracker'].temp_data['fx']
            adverserial_indices, original_softmax = compute_adverserial_patches(self.kwargs, transformed_patches)
            trf_data = self.kwargs['tracker'].format_trf_data(all_fx, x, it, self.kwargs['indices'], psnrs,
                                                              adverserial_indices, original_softmax)
            self.kwargs['tracker'].append_trf_data(trf_data)

            generated_adverserial = np.sum(adverserial_indices)

            if generated_adverserial == 0:
                fix_gen -= 1
            else:
                fix_gen = self.maxfixgen

            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]

            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            ##################################
            if fp[i_min] < fg:
                if self.debug:
                    print('New best for swarm at iteration {:}: {:} {:}'.format(it, p[i_min, :], fp[i_min]))

                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min) ** 2))

                if np.abs(fg - fp[i_min]) <= self.minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'.format(self.minfunc))

                    self.kwargs['tracker'].final_trf_write()

                    self.kwargs['tracker'].format_ratio_data(psnrs, fs, fg, adverserial_indices)
                    self.kwargs['tracker'].add_vector_ratio(self.kwargs['indices'])

                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= self.minstep:
                    print('Stopping search: Swarm best position change less than {:}'.format(self.minstep))

                    self.kwargs['tracker'].final_trf_write()

                    self.kwargs['tracker'].format_ratio_data(psnrs, fs, fg, adverserial_indices)
                    self.kwargs['tracker'].add_vector_ratio(self.kwargs['indices'])

                    if self.particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]

            if self.debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1

            if fix_gen == 0:
                print('Stopping search: no adverserial found after {:} consecutive generations'.format(self.maxfixgen))

                self.kwargs['tracker'].final_trf_write()

                self.kwargs['tracker'].format_ratio_data(psnrs, fs, fg, adverserial_indices)
                self.kwargs['tracker'].add_vector_ratio(self.kwargs['indices'])

                if self.particle_output:
                    return g, fg, p, fp
                else:
                    return g, fg

        print('Stopping search: maximum iterations reached --> {:}'.format(self.maxiter))

        vector_decoder = VectorDecoder(self.kwargs['original_patches'], self.kwargs['indices'],
                                       self.kwargs['whole_data'])
        vector_decoder.set_trf_vector(g)
        vector_decoder.apply_transformations()
        transformed_patches = vector_decoder.patches
        transformed_patches = transformed_patches.reshape(1, transformed_patches.shape[0], transformed_patches.shape[1],
                                                          transformed_patches.shape[2], transformed_patches.shape[3])

        if not self.is_feasible(transformed_patches) >= 0:
            print("However, the optimization couldn't find a feasible design. Sorry")

        self.kwargs['tracker'].final_trf_write()

        self.kwargs['tracker'].format_ratio_data(psnrs, fs, fg, adverserial_indices)
        self.kwargs['tracker'].add_vector_ratio(self.kwargs['indices'])

        if self.particle_output:
            return g, fg, p, fp
        else:
            return g, fg
