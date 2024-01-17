import panel as pn
import param


from controller import *


controller = PanelController()
controller.run()


# to start enter to commandline
    # panel serve src/panel/run_panel.py --autoreload --port 5000 --show