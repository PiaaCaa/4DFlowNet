import param
import numpy as np 
import pandas as pd
import panel as pn
import h5py
from model import *
import scipy


import matplotlib.pyplot as plt

pn.extension('vega', 'plotly', defer_load=True, template='fast')

XLABEL = 'GDP per capita (2000 dollars)'
YLABEL = 'Life expectancy (years)'
YLIM = (20, 90)
ACCENT = "#00A170"

PERIOD = 1000 # milliseconds

# pn.state.template.param.update(
#     site_url="https://panel.holoviz.org",
#     title="Hans Rosling's Gapminder",
#     header_background=ACCENT,
#     accent_base_color=ACCENT,
#     favicon="static/extensions/panel/images/favicon.ico",
#     theme_toggle=False
# )

#------------------Set Dataset------------------------------------------------------
model_name = 'Temporal4DFlowNet_20240124-1757'
hr_file = 'data/CARDIAC/M4_2mm_step2_invivoP02_magn_temporalsmoothing_toeger_periodic_HRfct.h5'
lr_file = 'data/CARDIAC/M4_2mm_step2_invivoP02_magn_temporalsmoothing_toeger_periodic_LRfct_noise.h5'
pred_dir = 'results'
pred_file = f'{pred_dir}/{model_name}/Testset_result_model4_2mm_step2_1757_temporal.h5'


#------------------Data functions------------------------------------------------------

def load_vel_h5( filename, vel_colnames = []):
    data = {}
    with h5py.File(filename, mode = 'r' ) as h5:
        for vel in vel_colnames:
            data[vel] = np.asarray(h5[vel])
    return data

@pn.cache
def get_dataset():
    url =  'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'
    return load_vel_h5(url, vel_colnames=['u', 'v', 'w'])



dataset = get_dataset()

FRAMES = list(np.arange(dataset['u'].shape[0]))

def play_through_frames():
    if frame.value == FRAMES[-1]:
        frame.value = FRAMES[0]
        return

    index = FRAMES.index(frame.value)
    frame.value = FRAMES[index+1]    

frame = pn.widgets.DiscreteSlider(
    value=FRAMES[-1], options=FRAMES, name="Frame", width=280
)
spatial_slicer = pn.widgets.DiscreteSlider(value = 10, options = list(range(0, 24)), name = "Spatial Slicer", width = 280)

show_legend = pn.widgets.Checkbox(value=True, name="Show Legend")

periodic_callback = pn.state.add_periodic_callback(play_through_frames, start=False, period=PERIOD)
player = pn.widgets.Checkbox.from_param(periodic_callback.param.running, name="Autoplay")

widgets = pn.Column(frame,spatial_slicer, player, show_legend, margin=(0,15))

desc = """## 🎓 Info

This is panel to give a good first overview on model results
"""

settings = pn.Column(
    "## ⚙️ Settings", widgets, desc,
    sizing_mode='stretch_width'
).servable(area='sidebar')

#------------------Calculation functions (i.e. model)------------------------------------------------------
eval_model = Evaluate_NNModel(model_name,hr_file, lr_file,pred_dir, pred_file,)

#create random indices
idx_x, idx_y, idx_z = eval_model.create_random_indices(eval_model.c_mask[0], p=0.1)
eval_model.save_random_indices_core(idx_x, idx_y, idx_z)
idx_x, idx_y, idx_z = eval_model.create_random_indices(eval_model.b_mask[0], p=0.1)
eval_model.save_random_indices_boundary(idx_x, idx_y, idx_z)

RE = eval_model.calculate_relative_error_normalized()
# create random indices

#--------------------Plotting functions-----------------------------------------


def get_title(frame):
    return f" Displaying image at frame, {frame}"


@pn.cache
def qualitative_plot(frame=4, spatial_idx=10,  show_legend=True):
    """This function creates a qualitative view of the data"""
    idx = eval_model.get_slice_idx(frame, spatial_idx,  axis = "x")
    
    u_hr = eval_model.hr_data['u'][idx]
    v_hr = eval_model.hr_data['v'][idx]
    w_hr = eval_model.hr_data['w'][idx]

    u_lr = eval_model.lr_data['u'][idx]
    v_lr = eval_model.lr_data['v'][idx]
    w_lr = eval_model.lr_data['w'][idx]

    u_pred = eval_model.pred_data['u'][idx]
    v_pred = eval_model.pred_data['v'][idx]
    w_pred = eval_model.pred_data['w'][idx]

    # TODO: lr only every second slice

    u_diff = np.abs(u_hr - u_pred)
    v_diff = np.abs(v_hr - v_pred)
    w_diff = np.abs(w_hr - w_pred)

    title = get_title(frame)

    plot, axs = plt.subplots(4, 3, figsize=(15, 10), facecolor=(0, 0, 0, 0))

    variables = [ u_lr, v_lr, w_lr, u_hr, v_hr, w_hr, u_pred, v_pred, w_pred, u_diff, v_diff, w_diff]

    labels = ['LR u', 'LR v', 'LR w', 'HR u', ' HR v', 'HR w', '4DFlowNet u', '4DFlowNet v', '4DFLowNet w', 'abs diff u','abs diff v','abs diff w']

    for ax, var, label in zip(axs.flatten(), variables, labels):
        ax.imshow(var, vmin = eval_model.min_v[label[-1]], vmax = eval_model.max_v[label[-1]])
        ax.set_title(label) 
        ax.set_axis_off() 
    
    plt.colorbar(axs[-1, 0].imshow(u_pred, vmin = eval_model.min_v["u"], vmax = eval_model.max_v["u"]), ax=axs[-1, 0], orientation='horizontal', fraction=.1)
    plt.colorbar(axs[-1, 1].imshow(v_pred, vmin = eval_model.min_v["v"], vmax = eval_model.max_v["v"]), ax=axs[-1, 1], orientation='horizontal', fraction=.1)
    plt.colorbar(axs[-1, 2].imshow(w_pred, vmin = eval_model.min_v["w"], vmax = eval_model.max_v["w"]), ax=axs[-1, 2], orientation='horizontal', fraction=.1)
    plt.suptitle(title)
    plt.tight_layout()
    plt.close(plot)

    return plot

@pn.cache
def correlation_plot(frame_idx):
    color_b = (135/255, 0, 82/255)

    # get indices of core and boundary
    idx_core = np.where(eval_model.c_mask[frame_idx] == 1)
    idx_bounds = np.where(eval_model.b_mask[frame_idx] == 1)

    # get random indices for core and boundary to plot a subset of the points
    # core (subtract bounds from mask such that mask only contains core points)
    x_idx, y_idx, z_idx = eval_model.rdn_idxx_c, eval_model.rdn_idxy_c, eval_model.rdn_idxz_c
    # boundary 
    x_idx_b, y_idx_b, z_idx_b = eval_model.rdn_idxx_b, eval_model.rdn_idxy_b, eval_model.rdn_idxz_b
    
    # Get velocity values in all directions
    # HR
    hr_u = np.asarray(eval_model.hr_data['u'][frame_idx])
    hr_u_core = hr_u[x_idx, y_idx, z_idx]
    hr_u_bounds = hr_u[x_idx_b, y_idx_b, z_idx_b]
    hr_v = np.asarray(eval_model.hr_data['v'][frame_idx])
    hr_v_core = hr_v[x_idx, y_idx, z_idx]
    hr_v_bounds = hr_v[x_idx_b, y_idx_b, z_idx_b]
    hr_w = np.asarray(eval_model.hr_data['w'][frame_idx])
    hr_w_core = hr_w[x_idx, y_idx, z_idx]
    hr_w_bounds = hr_w[x_idx_b, y_idx_b, z_idx_b]

    # SR 
    sr_u = np.asarray(eval_model.pred_data['u'][frame_idx])
    sr_u_vals = sr_u[x_idx, y_idx, z_idx]
    sr_u_bounds = sr_u[x_idx_b, y_idx_b, z_idx_b]
    sr_v = np.asarray(eval_model.pred_data['v'][frame_idx])
    sr_v_vals = sr_v[x_idx, y_idx, z_idx]
    sr_v_bounds = sr_v[x_idx_b, y_idx_b, z_idx_b]
    sr_w = np.asarray(eval_model.pred_data['w'][frame_idx])
    sr_w_vals = sr_w[x_idx, y_idx, z_idx]
    sr_w_bounds = sr_w[x_idx_b, y_idx_b, z_idx_b]

    def plot_regression_points(hr_vals, sr_vals, hr_vals_bounds, sr_vals_bounds,all_hr, all_sr, all_hr_bounds, all_sr_bounds, direction = 'u'):
        dimension = 2 #TODO
        N = 100
        # make sure that the range is the same for all plots and make square range
        x_range = np.linspace(-abs_max, abs_max, N)
        
        corr_line, text = get_corr_line_and_r2(all_hr, all_sr, x_range)
        corr_line_bounds, text_bounds = get_corr_line_and_r2(all_hr_bounds, all_sr_bounds, x_range)

        # plot linear correlation line and parms
        plt.gca().text(0.05, 0.95, text,transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.gca().text(0.05, 0.85, text_bounds,transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', color=color_b)
        plt.plot(x_range, x_range, color= 'grey', label = 'diagonal line')
        plt.plot(x_range, corr_line_bounds, '--', color = color_b)
        plt.plot(x_range, corr_line, 'k--')
        plt.scatter(hr_vals, sr_vals, s=0.3, c=["black"], label = 'core voxels')
        plt.scatter(hr_vals_bounds, sr_vals_bounds, s=0.3, c=[color_b], label = 'boundary voxels')
        
        plt.title(direction)
        plt.xlabel("V HR (m/s)")
        plt.ylabel("V prediction (m/s)")
        plt.legend(loc = 'lower right')
        plt.ylim(-abs_max, abs_max)
        plt.xlim(-abs_max, abs_max)

    def get_corr_line_and_r2(hr_vals, sr_vals, x_range):
        '''
        Returns correlation line and text for plot
        '''
        z = np.polyfit(hr_vals, sr_vals, 1)
        corr_line = np.poly1d(z)(x_range)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(hr_vals, sr_vals)
        text = f"$y={z[0]:0.4f}\;x{z[1]:+0.4f}$\n$R^2 = {r_value**2:0.4f}$"
        
        return corr_line, text

    
    print("Plotting correlation lines...")

    min_vals = np.min([np.min(sr_u_vals), np.min(sr_v_vals), np.min(sr_w_vals)])
    max_vals = np.max([np.max(sr_u_vals), np.max(sr_v_vals), np.max(sr_w_vals)])
    abs_max = np.max([np.abs(min_vals), np.abs(max_vals)])
    print('min/max/abs max', min_vals, max_vals, abs_max)

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plot_regression_points(hr_u_core, sr_u_vals, hr_u_bounds, sr_u_bounds,hr_u[idx_core], sr_u[idx_core], hr_u[idx_bounds], sr_u[idx_bounds],direction=r'$V_x$')
    plt.subplot(1, 3, 2)
    plot_regression_points(hr_v_core, sr_v_vals, hr_v_bounds, sr_v_bounds,hr_v[idx_core], sr_v[idx_core], hr_v[idx_bounds], sr_v[idx_bounds],direction=r'$V_y$')
    plt.subplot(1, 3, 3)
    plot_regression_points(hr_w_core, sr_w_vals, hr_w_bounds, sr_w_bounds,hr_w[idx_core], sr_w[idx_core], hr_w[idx_bounds], sr_w[idx_bounds], direction=r'$V_z$')
    plt.tight_layout()
        # if save_as is not None: plt.savefig(f"{save_as}_LRXYZ_subplots.svg")
    
    plt.close(fig)
    return fig


def plot_relative_error(eval_model, RE, frame):
    plot = plt.figure(figsize=(7, 5))
    plt.plot(range(eval_model.N_frames), RE, label = '4DFlowNet')
    plt.scatter([frame], RE[frame])
    plt.title('Relative Error')
    return plot


#--------------------------binding and layout-----------------------


qualitative_view    = pn.bind(qualitative_plot,    frame=frame, show_legend=show_legend, spatial_idx = spatial_slicer)
correlation_view    = pn.bind(correlation_plot, frame_idx = frame)
error_view          = pn.bind(plot_relative_error, frame = frame, eval_model = eval_model, RE = RE)

plots = pn.layout.GridBox(
    pn.pane.Matplotlib(qualitative_view, format='png', sizing_mode='scale_both', tight=True, margin=1),
    pn.pane.Matplotlib(correlation_view, format = 'svg', sizing_mode='scale_both', tight=True, margin=1),
    pn.pane.Matplotlib(error_view, format = 'svg', tight=True),
    pn.pane.Matplotlib(error_view, format = 'svg', tight=True),
    nrows = 2,
    sizing_mode="stretch_both", 
).servable()

# settings.show()
# plots.show()

layout = pn.template.FastListTemplate(
    title= f'Results Overview of {model_name}',
    sidebar=settings, 
    main=plots,
    header_background = '#870052', 
    accent_base_color = '#870052',
).servable()


layout.show()