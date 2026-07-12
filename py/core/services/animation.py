import matplotlib.animation as animation
import matplotlib.pyplot as plt
from random import randint as rd
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Animation:
    """
    Animation class for real-time plotting of velocity, gyroscope, and position data.
    Attributes:
        vl (dict): Dictionary containing velocity data, keyed by timestamp.
        gl (dict): Dictionary containing gyroscope data, keyed by timestamp.
        pl (dict): Dictionary containing position data, keyed by timestamp.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        axs (numpy.ndarray): Array of matplotlib axes for subplots.
        lines (dict): Dictionary of matplotlib Line2D objects for each data component.
        window_size (int): Time window (in seconds) for the x-axis scroll.
    Methods:
        __init__(vl: dict, gl: dict, pl: dict):
            Initializes the Animation object, sets up the figure, axes, and line objects.
        update(frame):
            Updates the plot with new data for each frame. Adjusts x and y limits dynamically
            based on the latest data within the window_size.
        main():
            Starts the animation using matplotlib's FuncAnimation and displays the plot.
    """
    def __init__(self, vl:dict, gl:dict, pl:dict):
        self.vl = vl  # dict
        self.gl = gl
        self.pl = pl
        self.fig, self.axs = plt.subplots(3, 1)
        self.lines = {
            'vxl': self.axs[0].plot([], [], 'r-')[0],
            'vyl': self.axs[0].plot([], [], 'g-')[0],
            'vzl': self.axs[0].plot([], [], 'b-')[0],
            'gxl': self.axs[1].plot([], [], 'r-')[0],
            'gyl': self.axs[1].plot([], [], 'g-')[0],
            'gzl': self.axs[1].plot([], [], 'b-')[0],
            'pxl': self.axs[2].plot([], [], 'r-')[0],
            'pyl': self.axs[2].plot([], [], 'g-')[0],
            'pzl': self.axs[2].plot([], [], 'b-')[0],
        }

        for ax in self.axs:
            ax.set_xlim(0, 10)

        self.window_size = 10  # 초 단위로 스크롤

    def update(self, frame):
        if not self.vl or not self.gl or not self.pl:
            return self.lines.values()

        # 타임스탬프 기준 정렬
        sorted_times = sorted(self.vl.keys())
        if not sorted_times:
            return self.lines.values()

        start_time = sorted_times[-1] - self.window_size
        times = [t for t in sorted_times if t >= start_time]

        def extract(key):
            return [self.vl[t][key] for t in times], \
                   [self.gl[t][key] for t in times], \
                   [self.pl[t][key] for t in times]

        vx, gx, px = extract(0)
        vy, gy, py = extract(1)
        vz, gz, pz = extract(2)

        self.axs[0].set_xlim(times[0], times[-1])
        self.axs[1].set_xlim(times[0], times[-1])
        self.axs[2].set_xlim(times[0], times[-1])

        # y축 자동 조정
        self.axs[0].set_ylim(min(vx+vy+vz), max(vx+vy+vz))
        self.axs[1].set_ylim(min(gx+gy+gz), max(gx+gy+gz))
        self.axs[2].set_ylim(min(px+py+pz), max(px+py+pz))

        self.lines['vxl'].set_data(times, vx)
        self.lines['vyl'].set_data(times, vy)
        self.lines['vzl'].set_data(times, vz)
        self.lines['gxl'].set_data(times, gx)
        self.lines['gyl'].set_data(times, gy)
        self.lines['gzl'].set_data(times, gz)
        self.lines['pxl'].set_data(times, px)
        self.lines['pyl'].set_data(times, py)
        self.lines['pzl'].set_data(times, pz)

        return self.lines.values()

    def main(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
