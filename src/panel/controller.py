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

    def __init__(self, **params):
        super().__init__(**params)
        self.model = Model()
        self.view = PanelDashboard()
    
    def run(self):
        self.layout = self.view.get_layout()
        self.layout.servable()
        print("ruuuuuning")