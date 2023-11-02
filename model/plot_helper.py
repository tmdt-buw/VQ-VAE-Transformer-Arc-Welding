import wandb
import numpy as np
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import threading
import time 


class MyWandbPlotterPlotly:

    def __init__(self, wandb_run: None) -> None:
        super().__init__()
        self.table = wandb.Table(columns=["i", "plotly_figure"])
        self.wandb_run = wandb_run

    def plot_from_batch_to_np(self, x, y, title: str):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        i = np.random.randint(len(x))
        self.plot_recon(x[i], y[i], title=title)


    def plot_recon(self, x, pred, title: str = "Plot of original and reconstructed data"):

        path_to_plotly_html = "./model_notebooks/plots/plotly_figure.html"
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Target", "Prediction"])

        for dim, dim_name in enumerate(['voltage', 'current']):
            x_t = x[:, dim]
            pred_t = pred[:, dim]
            fig.add_trace(
                go.Scatter(x=np.arange(len(x_t)), y=x_t, mode='lines', name=dim_name, legendgroup='group1', showlegend=True,
                           line={'color': 'blue'} if dim == 0 else {'color': 'red'}),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=np.arange(len(pred_t)), y=pred_t, mode='lines', name=dim_name, legendgroup='group1',
                           showlegend=False, line={'color': 'blue'} if dim == 0 else {'color': 'red'}),
                row=1, col=2
            )


        fig.update_layout(height=400, width=1200, title_text="Reconstruction")

        # Write Plotly figure to HTML
        fig.write_html(path_to_plotly_html, auto_play = False)

        # Add Plotly figure as HTML file into Table
        self.table.add_data(0, wandb.Html(path_to_plotly_html))

        self.wandb_run.log({title: self.table})


def plot_recon(logger, x: torch.Tensor, y: torch.Tensor, title: str, plot_wandb: bool = True):
    x = x.cpu().detach().numpy().reshape(-1, x.shape[-1])
    y = y.cpu().detach().numpy().reshape(-1, y.shape[-1])
    if x.shape[1] == 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 12))
        ax1.plot(x)
        ax1.set_title("Target")

        ax2.plot(y, label="current")
        ax2.set_title("Prediction")
        fig.suptitle("Left is the target and on the right the prediction")
        if plot_wandb:
            logger.log({title: fig})
        else:
            plt.show()
        
    elif x.shape[1] == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 12))
        ax1.plot(x[:, 1], label="current")
        ax1.plot(x[:, 0], color="red", label="voltage")
        ax1.set_title("Target")
        ax2.plot(y[:, 1], label="current")
        ax2.plot(y[:, 0], color="red", label="voltage")
        ax2.set_title("Prediction")
        fig.suptitle("Left is the target and on the right the prediction")

        if plot_wandb:
            logger.log({title: fig})
        else:
            plt.show()
    else:
        print("plotting not implemented for this shape")
    plt.close()

def threaded_plot_func(x: torch.Tensor, y: torch.Tensor, title: str):
    plot_thread = threading.Thread(target=plot_recon, args=(wandb, x, y, title))
    plot_thread.start()
    plot_thread.join(timeout=0.1) 



class MyWandbPlotterMatplt:

    def __init__(self, wandb_run: None) -> None:
        super().__init__()
        self.wandb_run = wandb_run

    def plot_from_batch_to_np(self, x, y, title: str):
         
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        i = np.random.randint(len(x))
        self.plot_recon(x[i], y[i], title)


    def plot_recon(self, x, pred, title: str = "Training Plot"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 12))
        ax1.plot(x[:, 1], label="current")
        ax1.plot(x[:, 0], color="red", label="voltage")
        ax1.set_title("Target")
        ax2.plot(pred[:, 1], label="current")
        ax2.plot(pred[:, 0], color="red", label="voltage")
        ax2.set_title("Prediction")
        fig.suptitle("Left is the target and on the right the prediction")
        self.wandb_run.log({title: fig})