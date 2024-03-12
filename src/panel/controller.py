'''
Build Model - Visual - Controll struture:
model includes all calulations
visual all graphs, number indicators, layout, etc.
control the communication between visual and model, including taking care of updates, etc.

it should also run without visual thorugh an API


all constants should come from a database

controll file:
'''


import panel as pn
import param


from model import *
from view import *

class PanelController(param.Parameterized):

    
    view = param.ClassSelector(class_=PanelDashboard)
    layout = pn.template.FastListTemplate()

    def __init__(self, model_name,hr_file, lr_file,pred_dir, pred_file, **params):
        super().__init__(**params)
        self.model = Evaluate_NNModel(model_name,hr_file, lr_file,pred_dir, pred_file,)
        self.model.__post_init__()
        self.view = PanelDashboard(self.model.lr_data, self.model.hr_data, self.model.pred_data)
        
        
    
    
    def run(self):
        self.layout = self.view.get_layout()
        self.layout.servable()
        print("-------------------------------------------------------------------------------------")
        print("ruuuuuning")
        # self.layout.show()


    @param.depends(
            "view.psidebar.slicing_x",
            "view.psidebar.slicing_y",
            "view.psidebar.axis",
            "view.psidebar.t_frame",
            watch=True
    )    
    def recalculate(self):
        print("ES PASSIERT WAS!!!")

        #readout of widgets, e.g. slicing, axis, etc.
        slicing_x = self.view.psidebar.slicing_x
        slicing_y = self.view.psidebar.slicing_y
        axis = self.view.psidebar.axis
        t_frame = self.view.psidebar.t_frame

        print(slicing_x, slicing_y, axis)

        # TODO 
        # recaluclate model and return updated data
        self.model.update_calculation(t_frame, slicing_x,  axis)

        # update view,esp. update plots
        self.view.pplots.update_plots(self.model.lr_slice, self.model.hr_slice, self.model.pred_slice)
        
        # also check matplotlib responsive

        self.view.update_layout() # TODO maybe delete this later
        # self.view.layout.servable()
        # self.view.layout.show()
        print('Finished updating')