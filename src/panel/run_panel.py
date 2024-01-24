import panel as pn
import param


from controller import *


model_name = '20240115-1535'
hr_file = 'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'
lr_file = 'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'
pred_dir = 'results/predictions'
pred_file = 'data/CARDIAC/M1_2mm_step2_static_dynamic.h5'



controller = PanelController(model_name=model_name, hr_file=hr_file, lr_file=lr_file, pred_dir=pred_dir, pred_file=pred_file)
controller.run()


# to start enter to commandline
    # panel serve src/panel/run_panel.py --autoreload --port 5000 --show