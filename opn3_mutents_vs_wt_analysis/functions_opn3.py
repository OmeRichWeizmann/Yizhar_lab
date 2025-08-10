import os
import pyabf
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def finding_stim_channels_and_plot(folder_path):
    plt.style.use("seaborn-v0_8-muted")
    plt.figure()
    signals_list = []
    folder_list = []
    all_groups_recovery_points = []
    for folder in os.listdir(folder_path):
        inner_folder_path = os.path.join(folder_path, folder)
        average_singal_smoothed = []
        group_recovery_points = []
        for file in os.listdir(inner_folder_path):
            abf_file_path = os.path.join(inner_folder_path, file)
            abf = pyabf.ABF(abf_file_path)  
            abf.setSweep(0, channel=0, baseline=[0.2, 2.5])
            smoothed = gaussian_filter1d(abf.sweepY, sigma=1)
            average_singal_smoothed.append(smoothed)
            recovery_points = [np.mean(abf.sweepY[20000:25000]),np.mean(abf.sweepY[130000:135000]),np.mean(abf.sweepY[240000:245000])]
            group_recovery_points.append(recovery_points)
            group_recovery_points_all_traces = np.copy(group_recovery_points)
        group_recovery_points_sem = np.std(group_recovery_points, axis=0) / np.sqrt(len(group_recovery_points))
        group_recovery_points = np.mean(group_recovery_points, axis=0)
        average_singal_smoothed = np.mean(average_singal_smoothed, axis=0)
        signals_list.append(average_singal_smoothed)
        folder_list.append(folder)
        all_groups_recovery_points.append({'folder':folder,'mean_recovery_points':group_recovery_points,'sem_recovery_points':group_recovery_points_sem,'all_traces':group_recovery_points_all_traces})
    stim_timings = finding_stims_channels(abf)
        
    for stim_channel, intervals in stim_timings.items():
        match stim_channel:
            case "UV (V)":
                color = "purple"
            case "Teal (V)":
                color = "#11AD86"
            case "Blue (V)":
                color = "blue"
            case "G_Y (V)":
                color = "green"
            case "Red (V)":
                color = "red"
            case _:
                color = "gray"
    
        for onset, offset in intervals:
            plt.axvspan(onset, offset, alpha=0.3, facecolor=color)
    for signal,folder in zip(signals_list,folder_list):
        match folder:
            case 'wt':
                color = '#2E3440'
            case 'WT':
                color = '#2E3440'
            case 'A251T':
                color = '#58D68D'
            case 'F243Y':
                color = '#0173B2'
            case 'm111':
                color = '#A569BD'
            case 'e168D-green':
                color = '#D98880'
            case 'S173F':
                color = '#8B4513'
            
        plt.plot(signal,color = color, label = folder, alpha=0.8)
        plt.legend(loc="upper center",frameon=False,bbox_to_anchor=(0.4, 0.9)) 
        plt.ylim(-50,300)
        plt.xlim(0,270000)
        plt.legend(loc='upper right')
        plt.ylabel("Average Current (pA)")
        plt.xlabel("Time (s)")
        plt.xticks([0, 50000, 100000, 150000, 200000, 250000],
                ['0', '5', '10', '15', '20', '25'])
        #plt.grid(True, linestyle='--', alpha=0.5)
        #plt.savefig(f'Z:\Omer\Ph.D\Project- Opn3_mutents_Moran_Suraj\Plots\{folder_path[-4:]}.svg',format = 'svg')
    return all_groups_recovery_points

def finding_stims_channels(abf):
    stim_channels = []
    stim_timings = {}

    for channel in abf.channelList:
        abf.setSweep(0, channel)
        channel_name = abf.sweepLabelY
        if channel_name in ["Clamp Current (pA)", "Membrane Potential (mV)", "light_sens (V)", "cam_expo (V)", "cam_trig (V)"]:
            continue
        
        # Detect stimulation onset and offset
        stim_signal = abf.sweepY
        threshold = 4  # Assuming <4 indicates stimulation
        stim_active = stim_signal < threshold
        stim_diff = np.diff(stim_active.astype(int))
        
        onsets = np.where(stim_diff == 1)[0] + 1  # +1 to correct for shift from diff
        offsets = np.where(stim_diff == -1)[0] + 1
        
        if len(onsets) != len(offsets):
            print(f"Warning: Mismatched onsets and offsets in channel {channel_name}")
        
        stim_channels.append(channel_name)
        stim_timings[channel_name] = list(zip(onsets, offsets))

    return stim_timings


def plot_recovery_comparison(data, group_name,save_path):
    labels = ['Dark', 'Light', 'Recovery']
    x = np.arange(len(labels))

    # Colors: muted and friendly
    colors = {
        'wt': '#2E3440',        # Safe blue - visible to all
        'mutant': '#DE8F05',    # Orange - high contrast alternative to red
        'bg': '#FAFAFA',        # Very light gray background
        'grid': '#E5E5E5',      # Light gray grid
        'text': '#2E3440',      # Dark blue-gray text
    }

    # Extract WT and mutants
    wt = next((d for d in data if d['folder'].lower() == 'wt'), None)
    mutants = [d for d in data if d['folder'].lower() != 'wt']
    if not wt or not mutants:
        print(f"Incomplete data for {group_name}")
        return

    n = len(mutants)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 10), facecolor=colors['bg'])
    fig.suptitle(f"{group_name} – Mutants vs WT", fontsize=18, fontweight='regular',
                 color=colors['text'], y=0.95)

    axes = np.array(axes).flatten()

    for i, mutant in enumerate(mutants):
        ax = axes[i]
        ax.set_facecolor(colors['bg'])



        match mutant['folder']:
            case 'wt':
                color = '#2E3440'
            case 'WT':
                color = '#2E3440'
            case 'A251T':
                color = '#58D68D'
            case 'F243Y':
                color = '#0173B2'
            case 'm111':
                color = '#A569BD'
            case 'e168D-green':
                color =  '#D98880'
            case 'S173F':
                color = '#8B4513'
        
        
        # WT
        # ax.errorbar(x, wt['mean_recovery_points'], yerr=wt['sem_recovery_points'],
        #             color = '#2E3440', marker='o', linestyle='-', linewidth=2,
        #             capsize=4, label='WT', zorder=2)
        
        # # Mutant
        # ax.errorbar(x, mutant['mean_recovery_points'], yerr=mutant['sem_recovery_points'],
        #             color=color, marker='s', linestyle='-', linewidth=2,
        #             capsize=4, label=mutant['folder'], zorder=3)

                #WT
        ax.plot(x, wt['mean_recovery_points'],
                    color = '#2E3440', marker='o', linestyle='-', linewidth=3,
                    label='WT', zorder=2)
        for trace in wt['all_traces']:
            ax.plot(x, trace, color='#2E3440', linestyle='-', alpha=0.2, zorder=1)
        # Mutant
        ax.plot(x, mutant['mean_recovery_points'], color=color, marker='s', linestyle='-', linewidth=3,
                 label=mutant['folder'], zorder=3)
        for trace in mutant['all_traces']:
            ax.plot(x, trace, color=color, alpha=0.2, zorder=1)

        # Titles and labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, color=colors['text'])
        #ax.set_yticks(np.linspace(-30, 260, 5))
        ax.set_ylim(-20, 300)
        ax.set_ylabel('GIRK current (pA)', fontsize=16, color=colors['text'])

        # Add vertical green lines
        # ax.axvline(x=.136, color='green', linewidth=8, alpha=0.1, zorder=1)
        # ax.axvline(x=1.318, color='blue', linewidth=40, alpha=0.1, zorder=1)
        
        # Clean styling
        #ax.grid(True, axis='y', linestyle='--', linewidth=0.6, color=colors['grid'], alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(colors['grid'])
        ax.spines['bottom'].set_color(colors['grid'])
        ax.tick_params(axis='both', colors=colors['text'], labelsize=24)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')
        ax.legend(frameon=False, fontsize=20, loc='upper left')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(f"{save_path}/{group_name}_recovery_comparison.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.show()