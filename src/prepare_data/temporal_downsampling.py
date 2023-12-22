import numpy as np
from scipy.integrate import trapz
from matplotlib import pyplot as plt

def cartesian_downsampling(hr, sampling_factor, offset = 0):
    assert len(hr.shape) == 4, "Input should be 4D"
    assert sampling_factor >= 1, "Sampling factor should be >= 1"

    return hr[offset::sampling_factor, : , :, :]



#TODO : maybe use convolution instead of averaging to increase speed
def temporal_averaging(hr, radius ):
    """
    Average the temporal dimension with a given radius
    Ideally, radius should be odd, so that the averaging is symmetric.
    This can be used for temporal downsampling as this is simple average smooting
    """
    assert len(hr.shape) == 4, "Input should be 4D"
    assert radius >= 1, "Radius should be >= 1"

    T = hr.shape[0]
    hr_avg = np.zeros_like(hr)

    # loop through all the frames and save the average in hr_avg
    for t in range(T):
        for i in range(t -radius//2, t+radius//2+1):
            
            # use periodical boundary conditions, i.e. after last frame take first frame again and vice verse
            if i >= T :
                i = i%(T)
            
            # sum up all the 3D data 
            hr_avg += np.asarray(hr_avg[i])
    
    # divide by number of frames to take the average
    hr_avg  /= radius

    return hr_avg


#TODO : maybe use convolution instead of averaging to increase speed
def temporal_averaging(hr, radius ):
    """
    Average the temporal dimension with a given radius
    Ideally, radius should be odd, so that the averaging is symmetric.
    This can be used for temporal downsampling as a simple average smooting function
    """
    assert len(hr.shape) == 4, "Input should be 4D"
    assert radius >= 1, "Radius should be >= 1"

    T = hr.shape[0]
    hr_avg = np.zeros_like(hr)

    # loop through all the frames and save the average in hr_avg
    for t in range(T):
        for i in range(t -radius//2, t+radius//2+1):
            
            # use periodical boundary conditions, i.e. after last frame take first frame again and vice verse
            if i >= T :
                i = i%(T)
            
            # sum up all the 3D data 
            hr_avg += np.asarray(hr_avg[i])
    
    # divide by number of frames to take the average
    hr_avg  /= radius

    return hr_avg


def temporal_smoothing_box_function_toeger(hr,t_range, sigma,):
    """
    This is the implementation of the smoothing box function for temporal averaging
    This implementation is based on the paper:  "Blood flow imaging by optimal matching of computational fluid dynamics to 4D-flow data" by Töger et al. 2020
    t_range: range of the heart cycle, going from 0 to (often) 1000 ms (1s), which is then divided into the number of frames
    Note that temporal boundaries are handled periodically.
    Also, differnt to suggested in the paper, not the area under the curve is normalized to 1, but the sum of the discrete weights are normalized to 1. 
    """
    assert len(hr.shape) == 4, "Input should be 4D"

    start_t = t_range[0]
    end_t = t_range[-1]
    len_t = start_t - end_t
    hr_avg = np.zeros_like(hr)
    dt = t_range[1] - t_range[0] # temporal resolution

    # extend range to handle periodic boundary conditions
    # extend the boundaries by a quarter fo total time to cover temporal cycle
    adjusted_start = (start_t - (len_t / 4)) + np.abs(start_t - (len_t / 4)) % dt
    extended_boundaries_left = np.arange(adjusted_start, start_t, dt)
    extended_boundaries_right = np.arange(end_t+dt, end_t+(len_t/4), dt)
    t_range_extended = np.append(np.append(extended_boundaries_left, t_range), (extended_boundaries_right))

    def smoothed_box_fct(t, t0, w, sigma):
            """
            Smoothed box function. With alpha = 1 this is not nomalized to 1
            """
            non_normalized = (1/(1+np.exp(-(t-(t0-w/2))/sigma)) - 1/(1+np.exp(-(t-(t0+w/2))/sigma)))
            alpha = 1
            # alpha = 1/integral_trapez(non_normalized, t)
            return alpha * non_normalized

    def integral_trapez(fct, t):
        """
        Calculate the integral of the smoothed box function with trapey formula   
        """
        return trapz(fct, t)

    
    # loop through all the frames and return the smoothed result hr_avg
    for i, t0 in enumerate(t_range):
        weighting =  smoothed_box_fct(t_range_extended, t0, dt, sigma)
        # plt.plot(t_range, smoothed_box_fct(t_range, t0, dt, sigma))

        # normalize the weighting
        # note: this is not included in the paper 
        weighting /= np.sum(weighting)

        # add the weighting to the periodic boundaries
        periodic_weighting = np.zeros_like(t_range)
        periodic_weighting = weighting[len(extended_boundaries_left):len(extended_boundaries_left)+len(t_range)] # middle
        periodic_weighting[:len(extended_boundaries_right)] += weighting[-len(extended_boundaries_right):] 
        periodic_weighting[-len(extended_boundaries_left):] += weighting[:len(extended_boundaries_left)]

        # weight input by the periodic weighting
        hr_avg[i, :, :, :] = np.sum(hr*periodic_weighting[:, None, None, None], axis = 0)

    #plt.show()
    return hr_avg