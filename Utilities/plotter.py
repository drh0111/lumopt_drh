import matplotlib.pyplot as plt
from matplotlib.animation import FileMovieWriter

class Snapshots(FileMovieWriter):
    """
    This class is used to grab the image of every optimization figure and save it as a movie frame
    """
    support_formats = ['png', 'jpeg', 'pdf']

    def __init__(self, *args, extra_args = None, **kwargs) -> None:
        super().__init__(*args, extra_args = (), **kwargs) # stop None from being passed 

    def setup(self, fig, outfile, dpi, frame_prefix = None) -> None:
        """
        Parameters
        ----------
        FIG: Figure object, the figure that we grab the frame from
        OUTFILE: String, the name of output file
        """
        super().setup(fig, outfile, dpi = dpi, frame_prefix = frame_prefix)
        self.fname_format_str = '%s%%d.%s'
        self.temp_prefix, self.frame_format = self.outfile.split('.')

    def grab_frame(self, **savefig_kwargs) -> None:
        return super().grab_frame(**savefig_kwargs)

    def finish(self) -> None:
        return super().finish()
    

class Plotter():
    """
    This class is used to orchestrate the generation of plots during the optimization
    """

    def __init__(self, movie = True, plot_hist = True) -> None:
        """
        ---INPUTS---
        MOVIE: Boolean class. Whether record the evolution of parameters, field and geometries 
            as a movie
        PLOT_HIST: Boolean class. Whether plot the history of the parameters and gradients. 
            Should be set False for large (> 100) numbers of parameters
        """
        self.movie = movie
        self.plot_hist = plot_hist

        if plot_hist:
            self.fig, self.axs = plt.subplots(nrows = 2, ncols = 3, figsize = (12, 7)) 
        else:
            self.fig, self.axs = plt.subplots(nrows = 2, ncols = 2, figsize = (12, 10))
        self.fig.show()

        if movie:
            metadata = dict(title = 'optimization', artist = 'lumopt', comment = 'continuous adjoint method')
            self.writer = Snapshots(fps = 2, metadata = metadata)

    def update(self, optimization):
        """
        This function is used to update the figure and the movie in course of the optimization
        """
        if self.plot_hist:
            optimization.optimizer.plot(fom_ax = self.axs[0, 0], params_ax = self.axs[0, 2], gradients_ax = self.axs[1, 2])
        else:
            optimization.optimizer.plot(fom_ax = self.axs[0, 0])

        if hasattr(optimization, 'optimizations'):
            for i, opt in enumerate(optimization.optimizations):
                if hasattr(opt, 'gradient_fields'):
                    if not opt.geometry.plot(self.axs[1, 0]):
                        opt.gradient_fields.plot_eps(self.axs[1, 0])
                    opt.gradient_fields.plot(self.fig, self.axs[1, 1], self.axs[0, 1])
                print('Plots updated with optimization {} iteration {} result'.format(i, optimization.optimizer.iter - 1))
        else:
            if hasattr(optimization, 'gradient_fields'):
                if not optimization.geometry.plot(self.axs[1, 0]):
                    optimization.gradient_fields.plot_eps(self.axs[1, 0])
                optimization.gradient_fields.plot(self.fig, self.axs[1, 1], self.axs[0, 1])
            print('Plots updated with iteration {} result'.format(optimization.optimizer.iter - 1))

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.movie:
            self.writer.grab_frame()
            print('Saved frame')
