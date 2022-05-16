import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loss_plot(losses:[], save_loc=None, show=True, yrange=(0,2)):
    FILENAME = 'loss.png'
    df = pd.DataFrame(dict(epochs=range(len(losses)), avg_loss=losses))
    g = sns.relplot(x="epochs", y="avg_loss", kind="line", data=df)
    g.figure.autofmt_xdate()
    # options
    if yrange: plt.ylim(yrange)
    if show: plt.show()
    if save_loc:
        g.figure.savefig(save_loc + FILENAME) 
    return save_loc + FILENAME