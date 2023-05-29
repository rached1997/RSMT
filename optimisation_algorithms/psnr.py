import sys
import numpy as np
import sewar

"""  this file calculates the PSNR of the distorted data 
"""

def check_psnr(psnr):
    epsilon = (sys.float_info.epsilon)
    if psnr == np.inf:
        return 1 / epsilon
    if psnr == - np.inf:
        return - 10000
    return psnr

# constraint function for PSO
def psnr_constraint(transformed_patches_over_gen, **kwargs):
    transformed_patches = transformed_patches_over_gen.reshape((-1, transformed_patches_over_gen.shape[2],
                                                                transformed_patches_over_gen.shape[3],
                                                                transformed_patches_over_gen.shape[4]))
    original_patches = np.expand_dims(kwargs['original_patches'], axis=0)
    original_patches = np.repeat(original_patches, len(transformed_patches_over_gen), axis=0)
    original_patches = original_patches.reshape((-1, transformed_patches_over_gen.shape[2],
                                                 transformed_patches_over_gen.shape[3],
                                                 transformed_patches_over_gen.shape[4]))

    all_psnr = psnr(original_patches, transformed_patches) - 20
    kwargs['tracker'].temp_data['all_psnr'] = all_psnr + 20
    all_psnr = all_psnr.reshape((len(transformed_patches_over_gen), -1))
    fs = np.median(all_psnr, axis=1)

    return fs


def psnr(originals, transformeds):
    epsilon = sys.float_info.epsilon
    mse = np.mean(np.square(originals - transformeds), axis=(1, 2, 3))

    # MSE is zero means no noise is present in the signal .Therefore PSNR have no importance.
    psnr = 20 * np.log10(np.max(originals, axis=(1, 2, 3)) / np.sqrt(mse))
    indices = np.where(psnr == np.inf)
    psnr[indices] = 1 / epsilon
    indices = np.where((psnr == - np.inf) | (psnr == np.nan))
    psnr[indices] = -10000
    return psnr


def PSNR_metric(original, transformed):
    max = np.max(original)
    psnr_value = sewar.psnr(original, transformed, MAX=max)
    return check_psnr(psnr_value)
