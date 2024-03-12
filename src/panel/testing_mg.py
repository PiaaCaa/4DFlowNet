import param
import numpy as np 
import pandas as pd
import panel as pn


import matplotlib.pyplot as plt

pn.extension('vega', 'plotly', defer_load=True, template='fast')

XLABEL = 'GDP per capita (2000 dollars)'
YLABEL = 'Life expectancy (years)'
YLIM = (20, 90)
ACCENT = "#00A170"

PERIOD = 1000 # milliseconds

pn.state.template.param.update(
    site_url="https://panel.holoviz.org",
    title="Hans Rosling's Gapminder",
    header_background=ACCENT,
    accent_base_color=ACCENT,
    favicon="static/extensions/panel/images/favicon.ico",
    theme_toggle=False
)
@pn.cache
def get_dataset():
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv'
    return pd.read_csv(url)

dataset = get_dataset()

YEARS = [int(year) for year in dataset.year.unique()]

def play():
    if year.value == YEARS[-1]:
        year.value = YEARS[0]
        return

    index = YEARS.index(year.value)
    year.value = YEARS[index+1]    

year = pn.widgets.DiscreteSlider(
    value=YEARS[-1], options=YEARS, name="Year", width=280
)
show_legend = pn.widgets.Checkbox(value=True, name="Show Legend")

periodic_callback = pn.state.add_periodic_callback(play, start=False, period=PERIOD)
player = pn.widgets.Checkbox.from_param(periodic_callback.param.running, name="Autoplay")

widgets = pn.Column(year, player, show_legend, margin=(0,15))

desc = """## 🎓 Info

The [Panel](http://panel.holoviz.org) library from [HoloViz](http://holoviz.org)
lets you make widget-controlled apps and dashboards from a wide variety of 
plotting libraries and data types. Here you can try out four different plotting libraries
controlled by a couple of widgets, for Hans Rosling's 
[gapminder](https://demo.bokeh.org/gapminder) example.

Source: [pyviz-topics - gapminder](https://github.com/pyviz-topics/examples/blob/master/gapminders/gapminders.ipynb)
"""

settings = pn.Column(
    "## ⚙️ Settings", widgets, desc,
    sizing_mode='stretch_width'
).servable(area='sidebar')

@pn.cache
def get_data(year):
    df = dataset[(dataset.year==year) & (dataset.gdpPercap < 10000)].copy()
    df['size'] = np.sqrt(df['pop']*2.666051223553066e-05)
    df['size_hvplot'] = df['size']*6
    return df

def get_title(library, year):
    return f"{library}: Life expectancy vs. GDP, {year}"

def get_xlim(data):
    return (data['gdpPercap'].min()-100,data['gdpPercap'].max()+1000)

@pn.cache
def mpl_view(year=1952, show_legend=True):
    data = get_data(year)
    title = get_title("Matplotlib", year)
    xlim = get_xlim(data)

    plot = plt.figure(figsize=(10, 6), facecolor=(0, 0, 0, 0))
    ax = plot.add_subplot(111)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.set_ylim(YLIM)
    ax.set_xlim(xlim)

    for continent, df in data.groupby('continent'):
        ax.scatter(df.gdpPercap, y=df.lifeExp, s=df['size']*5,
                   edgecolor='black', label=continent)

    if show_legend:
        ax.legend(loc=4)

    plt.close(plot)
    return plot




mpl_view    = pn.bind(mpl_view,    year=year, show_legend=show_legend)

plots = pn.layout.GridBox(
    pn.pane.Matplotlib(mpl_view, format='png', sizing_mode='scale_both', tight=True, margin=10),
    ncols=2,
    sizing_mode="stretch_both", 
).servable()

# settings.show()
# plots.show()

layout = pn.template.FastListTemplate(
    title='Results Overview updated',
    sidebar=settings,  # alternativ psidebar.get_sidebar() oder psidebar.sidebar wenn nicht alle objekte von self.psidebar angezeigt werden sollen
    main=plots,
).servable()
layout.show()