### function for analyzing Uri's Sapap3 data
import os
import numpy as np
import matplotlib.pyplot as plt
import pyabf
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.cm as cm


def peaks_finder(trace_filtred,thresh_min,thresh_max,thresh_prominence,fs,plot_it,protocol):    
    fs = 50000
    if protocol == "iv_curve":
        peaks_signal = trace_filtred[497*50:(497*50)+25000] 
    
    elif protocol == "ramp_protocol":
        peaks_signal = trace_filtred[563*50:(563*50)+50000]# this is for the ramp protocol # This is for the IV curve protocol# 497 ms from recording start till stim on and 50 samples per ms] 

    # Event window parameters
    event_no = 0  # Event viewer: 0 is the first event
    pretrigger_window = (2 * fs)/1000  # Pre-event time window in ms
    posttrigger_window = (4 * fs)/1000  # Post-event time window in ms
    std_trace = np.std(peaks_signal)
    # Set parameters of the Find peaks function

    thresh_min_width = 0.1 * (fs/1000)
    
    # Find peaks function
    peaks, peaks_dict = find_peaks(peaks_signal, 
            height=(thresh_min, thresh_max),  # Min and max thresholds to detect peaks.
            threshold=None,  # Min and max vertical distance to neighboring samples.
            distance=None,  # Min horizontal distance between peaks.
            prominence=thresh_prominence,  # Vertical distance between the peak and lowest contour line.
            width=thresh_min_width,  # Min required width (in bins). E.g. For 10Khz, 10 bins = 1 ms.
            wlen=None,  # Window length to calculate prominence.
            rel_height=0.5,  # Relative height at which the peak width is measured.
            plateau_size=None)
    
    # Create table with results
    table = pd.DataFrame(columns = ['event', 'peak_index', 
                                    'peak_time_s',
                                    'event_window_start', 'event_window_end',
                                    'peak_amp', 'width_ms', 
                                    'inst_freq', 'isi_s', 
                                    'area', 'decay_tau_log', 
                                    'decay_tau_exp', 'rise_tau_exp','rise_half_amp_ms','decay_half_amp_ms'])
    
    table.event = np.arange(1, len(peaks) + 1)
    table.peak_index = peaks
    table.peak_time_s = peaks / fs  # Divided by fs to get s
    table.event_window_start = peaks_dict['left_ips'] - pretrigger_window
    table.event_window_end = peaks_dict['right_ips'] + posttrigger_window
    table.peak_amp = peaks_dict['peak_heights']  # height parameter is needed
    table.width_ms = peaks_dict['widths']/(fs/1000) # Width (ms) at half-height
    
    # Additional parameters (remember to add the columns to the dataframe)
    table.rise_half_amp_ms = (peaks - peaks_dict['left_ips'])/(fs/1000) 
    table.decay_half_amp_ms = (peaks_dict['right_ips'] - peaks)/(fs/1000)
    
    # Calculations based on the parameters above
    table.inst_freq = np.insert((1 / (np.array(table.peak_index[1:]) -
                                    np.array(table.peak_index[:-1])) * fs), 
                                0, np.nan) 
    
    table.isi_s = np.diff(peaks, axis=0, prepend=peaks[0]) / fs
    
    for i, event in table.iterrows():
        
        # Event area as absolute value (abs)
        individual_event = peaks_signal[int(event.event_window_start) : int(event.event_window_end)]
        table.loc[i, 'area'] = abs(round(individual_event.sum()/(fs/1000), 2))  # pA x ms
        
        # Decay tau from logistic regression
        decay_tau = abs(peaks_signal[int(event.peak_index) : int(event.event_window_end)])
        decay_tau_log = np.log(decay_tau)
        decay_width = int(len(decay_tau))
        decay_width_array = list(range(0, decay_width))
        slope, _ = np.polyfit(decay_width_array, decay_tau_log, 1)
        tau = -1 / slope
        table.loc[i, 'decay_tau_log'] = tau/(fs/1000) 
    
        # Decay tau from monoexponential fitting
        decay_tau = peaks_signal[int(event.peak_index) : int(event.event_window_end)]
        decay_width = int(len(decay_tau))
        decay_width_array = list(range(0, decay_width))
        a_initial = 200
        b_initial = 0.1
        # popt: optimal values for the parameters, pcov: estimated covariance of popt
        popt, pcov = curve_fit(lambda t, a, b: a * np.exp(b * t), 
                            decay_width_array, decay_tau, 
                            p0=(a_initial, b_initial), 
                            maxfev=2000)  # maxfev: number of iterations
        a = popt[0]  
        b = popt[1]      
        table.loc[i, 'decay_tau_exp'] = abs((1/b)/(fs/1000))
        
        # Rise tau from monoexponential fitting
        rise_tau = peaks_signal[int(event.event_window_start):int(event.peak_index)]
        rise_width = int(len(rise_tau))
        rise_width_array = list(range(0, rise_width))
        a_initial = 200
        b_initial = 0.1
        # popt: optimal values for the parameters, pcov: estimated covariance of popt
        popt, pcov = curve_fit(lambda t, a, b: a * np.exp(b * t), 
                            rise_width_array, rise_tau, 
                            p0=(a_initial, b_initial), 
                            maxfev=2000)  # maxfev: number of iterations
        a = popt[0]  
        b = popt[1]      
        table.loc[i, 'rise_tau_exp'] = abs((1/b)/(fs/1000))
    
    if plot_it == "plot_it":
        fig = plt.figure(figsize=(18,4))
        gridspec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 1])

        # Plot 1: Detected events in the trace
        ax1 = fig.add_subplot(gridspec[0])  # gridspec specifies the ratio between plots
        ax1.set_title("Events detection")   
        ax1.plot(peaks_signal)
        ax1.plot(peaks, peaks_signal[peaks], "r.")
        for i, txt in enumerate(table.event):
            ax1.annotate(table.event[i], (peaks[i], peaks_signal[peaks][i]))
        ax1.set_xlabel("Time bin")
        ax1.set_ylabel("Current (pA)")
        # ax1.axes.set_xlim(4000, 10000)  # OptionaL: Zoom in the trace

        # Plot 2: Event viewer
        ax2 = fig.add_subplot(gridspec[1]) 
        ax2.set_title("Event viewer")
        ax2.plot(peaks_signal, "gray")
        ax2.plot(peaks, peaks_signal[peaks], "r.")
        ax2.set_xlabel("Time bin")
        ax2.set_ylabel("Current (pA)")
        # Event time window
        ax2.set_xlim(table.event_window_start[event_no], table.event_window_end[event_no]) 
        # Labeling the event
        line, = ax2.plot(peaks, peaks_signal[peaks], "r.") 
        line.set_label(table.event[event_no]) 
        ax2.legend()
                
        # Show graph and table

    table.round(3)
    plt.show()
    return table # round: display of decimal numbers in the table

