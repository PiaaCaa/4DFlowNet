'''
Build Model - Visual - Controll struture:
model includes all calulations
visual all graphs, number indicators, layout, etc.
control the communication between visual and model, including taking care of updates, etc.
'''

import param
import panel as pn
import matplotlib.pyplot as plt
import numpy as np

class PanelSidebar(param.Parameterized):
    slider      = param.Integer(label='First slider', default=10, bounds=(0, 24))
    def __init__(self, **params):
        super().__init__(**params)

    def get_sidebar(self):
        return pn.Param(self.param, parameters=['slider'], widgets={'slider': pn.widgets.DiscreteSlider})


class PanelPlots(param.Parameterized):
    plot = pn.pane.Markdown("## Plot")

    # Create your plot here
    fig, ax = plt.subplots()
    t = np.arange(20)
    y = np.sin(t)
    ax.plot(t, y)  # Replace x_values and y_values with your data

    plot2 = pn.pane.Matplotlib(fig)

    def __init__(self, **params):
        super().__init__(**params)
    
    def get_plot(self):
        return pn.Row(self.plot, self.plot2)

class PanelDashboard(param.Parameterized):

    
    # Constants
    MAIN_COLOR = "#0099cc"
    SECOND_COLOR = "#FC4F30"

    def __init__(self, **params):
        super().__init__(**params)

        self.psidebar = PanelSidebar()
        self.pplots = PanelPlots()

        self.layout = pn.template.FastListTemplate(
            title='Results Overview',
            sidebar=self.psidebar,  # alternativ psidebar.get_sidebar() oder psidebar.sidebar wenn nicht alle objekte von self.psidebar angezeigt werden sollen
            main=self.pplots.get_plot(),
            accent_base_color=self.MAIN_COLOR,
            header_background=self.MAIN_COLOR
        )

        self.layout.main.sizing_mode = 'scale_both'
        self.layout.sidebar.sizing_mode = 'scale_both'
        self.layout.sizing_mode = 'scale_both'

    def get_layout(self):
        return self.layout


