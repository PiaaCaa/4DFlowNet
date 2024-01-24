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
# pn.extension('ipywidgets')


class PanelSidebar(param.Parameterized):
    """ This class covers the sidebar containing the widgets"""


    slicing_x      = param.Integer(label='Slicing x', default=10, bounds=(0, 24)) # update to correct bounds
    slicing_y      = param.Integer(label='Slicing y', default=10, bounds=(0, 24)) # update to correct bounds
    t_frame       = param.Integer(label='Time frame', default=10, bounds=(0, 50)) # update to correct bounds
    axis           = param.ObjectSelector(default='x', objects=['x', 'y', 'z'])

    def __init__(self, **params):
        super().__init__(**params)

    def get_sidebar(self):
        return pn.Column(self.param.slicing_x, self.param.t_frame, self.param.axis)


class PanelPlots(param.Parameterized):
    """ This class covers the overall window containing plots"""
    plot1 = pn.pane.Markdown("## Plot")
    
    # Create your plot here
    fig, axs = plt.subplots(2, 2)
    for ax in axs.flat:
        ax.plot(np.random.rand(10))

    plot2 = pn.pane.Markdown("## Plot2")

    plot3 = pn.pane.Matplotlib(fig)
    
    im = axs[0,0].imshow(np.zeros((10,10)))
    # fig, axs = plt.subplots()
    # axs.imshow(lr_data[0,0,:,:])

    # plot4 = pn.pane.Matplotlib(fig)
    
    

    def __init__(self,lr_data, hr_data, pred_data, **params):
        super().__init__(**params)
        #make more memory efficent? either in model or in control? 
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.pred_data = pred_data
        self.im = self.set_imshow_plot(lr_data['u'][10,:, :, 15], hr_data['u'][10,:, :, 15], pred_data['u'][10,:, :, 15])

        
    
    def get_plot(self):
        return pn.Row(self.plot1, self.plot2,self.plot3, self.plot4)

    def set_imshow_plot(self, lr_slice, hr_slice, pred_slice):
        print("------LR slice shape: ", lr_slice.shape)
        fig, axs = plt.subplots()
        im_update = axs.imshow(lr_slice)
        self.im.set_data(lr_slice)
        fig.canvas.draw_idle()
        axs.set_title('LR Data')
        axs.set_axis_off()
        self.plot4 = pn.pane.Matplotlib(fig, sizing_mode="stretch_width") #,
        plt.close(fig)
         
        self.plot4.param.trigger('object')
        return im_update
        
    
    def update_plots(self, lr_data, hr_data, pred_data):
        self.set_imshow_plot(lr_data, hr_data, pred_data)
        return


class PanelDashboard(param.Parameterized):

    
    # Constants
    MAIN_COLOR = "#0099cc"
    SECOND_COLOR = "#FC4F30"

    def __init__(self,lr_data, hr_data, pred_data, **params):
        super().__init__(**params)

        # 
        self.psidebar = PanelSidebar()
        self.pplots = PanelPlots(lr_data, hr_data, pred_data)



        self.layout = pn.template.FastListTemplate(
            title='Results Overview',
            sidebar=self.psidebar.get_sidebar(),  # alternativ psidebar.get_sidebar() oder psidebar.sidebar wenn nicht alle objekte von self.psidebar angezeigt werden sollen
            main=self.pplots.get_plot(),
            accent_base_color=self.MAIN_COLOR,
            header_background=self.MAIN_COLOR
        )

        self.layout.main.sizing_mode = 'scale_both'
        self.layout.sidebar.sizing_mode = 'scale_both'
        self.layout.sizing_mode = 'scale_both'

    def get_layout(self):
        return self.layout
    
    def update_layout(self):
        self.layout = pn.template.FastListTemplate(
            title='Results Overview',
            sidebar=self.psidebar.get_sidebar(),  # alternativ psidebar.get_sidebar() oder psidebar.sidebar wenn nicht alle objekte von self.psidebar angezeigt werden sollen
            main=self.pplots.get_plot(),
            accent_base_color=self.MAIN_COLOR,
            header_background=self.MAIN_COLOR
        )
        return




