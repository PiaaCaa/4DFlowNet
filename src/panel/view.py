'''
Build Model - Visual - Controll struture:
model includes all calulations
visual all graphs, number indicators, layout, etc.
control the communication between visual and model, including taking care of updates, etc.
'''
import matplotlib
matplotlib.use('WebAgg', force = True)
matplotlib.use('Agg', force=True)
print(matplotlib.get_backend())
import param
import panel as pn
import matplotlib.pyplot as plt
import numpy as np
pn.extension()
# pn.extension('ipywidgets')


class PanelSidebar(param.Parameterized):
    """ This class covers the sidebar containing the widgets"""


    slicing_x      = param.Integer(label='Slicing x', default=10, bounds=(0, 24)) # update to correct bounds
    slicing_y      = param.Integer(label='Slicing y', default=10, bounds=(0, 24)) # update to correct bounds
    t_frame        = param.Integer(label='Time frame', default=10, bounds=(0, 50)) # update to correct bounds
    axis           = param.ObjectSelector(default='x', objects=['x', 'y', 'z'])

    def __init__(self, **params):
        super().__init__(**params)

    def get_sidebar(self):
        return pn.Column(self.param.slicing_x, self.param.t_frame, self.param.axis)


class PanelPlots(param.Parameterized):
    """ This class covers the overall window containing plots"""
    
    # Create your plot here
    plot1 = pn.pane.Markdown("## Plot1")
    plot2 = pn.pane.Markdown("## Plot2")
    plot3 = param.ClassSelector(class_=pn.pane.Matplotlib)
    plot4 = param.ClassSelector(class_=pn.pane.Matplotlib)
    plot5 = param.ClassSelector(class_=pn.pane.Matplotlib)
    shape_widget = param.ClassSelector(class_ = pn.indicators.Number)


    

    def __init__(self,lr_data, hr_data, pred_data, **params):
        params["shape_widget"]= pn.indicators.Number()
        params["plot3"] = pn.pane.Matplotlib()
        params["plot4"]= pn.pane.Matplotlib()
        params["plot5"] = pn.pane.Matplotlib()
        super().__init__(**params)

        #make more memory efficent? either in model or in control? 
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.pred_data = pred_data


        image_plt = self.set_imshow_plot(lr_data['u'][10,:, :, 15], hr_data['u'][10,:, :, 15], pred_data['u'][10,:, :, 15], axis = "x")
        fig_vol = self.create_voltage_figure(1.0)
        random_plt = self.random_plot()

        self.plot3 = pn.pane.Matplotlib(random_plt,  tight=True)
        self.plot4 = pn.pane.Matplotlib(image_plt,  tight=True)
        self.plot5 = pn.pane.Matplotlib(fig_vol,  tight=True)
        self.shape_widget = pn.indicators.Number(name = "shape:", value = 10)

        self.plot3.param.trigger('object')
        self.plot4.param.trigger('object')
        self.plot5.param.trigger('object')

        temp  = pn.bind(self.set_imshow_plot, lr_slice = lr_data['u'][10,:, :, 15], hr_slice = lr_data['u'][10,:, :, 15], pred_slice = lr_data['u'][10,:, :, 15], note = '',axis = "x",  watch = True)
        self.plot4 = pn.pane.Matplotlib(temp, sizing_mode="stretch_width") 
        x = 1.0
        temp2 = pn.bind(self.create_voltage_figure, x  =x, watch=True)
        self.plot5 = pn.pane.Matplotlib(temp2) 
    
    def get_plot(self):
        plots = pn.layout.GridBox(
        self.plot4,self.shape_widget,self.plot5, 
        ncols=2,
        sizing_mode="stretch_both", 
        ).servable()
        return plots
        # return pn.Row(self.plot1, self.plot2,self.plot3, self.plot4, self.plot5, self.shape_widget)

    def random_plot(self):
        fig = plt.figure()
        rnd = np.random.randint(10, size=(10))
        plt.plot(rnd)
        return fig
    
    def create_voltage_figure(self, x = 1.0, figsize=(4,3)):
        x = float(x)
        t = np.arange(0.0, x, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(t, s)

        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
                title='Voltage')
        ax.grid()

        plt.close(fig) # CLOSE THE FIGURE!
        
        return fig

    @pn.cache
    def set_imshow_plot(self, lr_slice, hr_slice, pred_slice,axis,  note = ''):
        print("------LR slice shape: ", lr_slice.shape)
        print('Update of imshow is running: ', note)
        # self.plot4.param.trigger('object')

        x_shape = lr_slice.shape[0]
        self.shape_widget.value = x_shape
        fig, axs = plt.subplots()
        axs.imshow(lr_slice)
        # self.im.set_data(lr_slice)
        # fig.canvas.draw_idle()
        axs.set_title(f'LR Data {note}')
        axs.set_axis_off()
        plt.close(fig)
         
        im_update = axs.imshow(lr_slice)
        axs.set_title(f'LR Data {note}')
        # self.im = im_update
        return fig
    
    def set_mean_velocity_plot(self, lr_slice, hr_slice, pred_slice, note = ''):
        print("------LR slice shape: ", lr_slice.shape)
        print('Update is running: ', note)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.random.rand(10))
        return fig
         
    def update_plots(self, lr_data, hr_data, pred_data):
        print("Plots are getting updated..")
        self.set_imshow_plot(lr_data, hr_data, pred_data, note = 'updated')
        # temp  = self.set_imshow_plot(lr_data, hr_data, pred_data, note = 'updated')
        x = 3.0
        # self.plot3 =self.set_mean_velocity_plot(lr_data, hr_data, pred_data, note = 'updated')
        # self.plot5 = self.create_voltage_figure(x = x)
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
            title='Results Overview --',
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
        print("Update layout function")
        self.layout = pn.template.FastListTemplate(
            title='Results Overview updated',
            sidebar=self.psidebar.get_sidebar(),  # alternativ psidebar.get_sidebar() oder psidebar.sidebar wenn nicht alle objekte von self.psidebar angezeigt werden sollen
            main=self.pplots.get_plot(),
            accent_base_color=self.MAIN_COLOR,
            header_background=self.MAIN_COLOR
        )

        # self.pplots.get_plot()
        return 




