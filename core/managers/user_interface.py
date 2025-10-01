import matplotlib.pyplot as plt
from matplotlib.table import Table


class UserInterface:
    """
    A class for displaying a 3x3 grid-based user interface using matplotlib, where each cell represents a status indicator.
    Attributes:
        nrows (int): Number of rows in the grid (default: 3).
        ncols (int): Number of columns in the grid (default: 3).
        width (float): Width of each cell, calculated if not provided.
        height (float): Height of each cell, calculated if not provided.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The matplotlib axes object.
    Methods:
        __init__(nrows=3, ncols=3, cell_width=None, cell_height=None):
            Initializes the UserInterface with the specified grid size and cell dimensions.
        get_cell_color(value):
            Returns a color code based on the boolean or string value for cell background.
        update(l):
            Updates the grid display with new values from the provided dictionary `l`.
    """
    
    def __init__(self, nrows=3, ncols=3, cell_width=None, cell_height=None):
        self.nrows = nrows
        self.ncols = ncols
        self.width = cell_width if cell_width else 1.0 / ncols
        self.height = cell_height if cell_height else 1.0 / nrows
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 3)
        self.ax.set_title("UI")

    def get_cell_color(self, value):
        if isinstance(value, bool):
            return '#a8e6a3' if value else '#f4a3a3'  # 초록/빨강
        elif isinstance(value, str) and value.lower() == 'true':
            return '#a8e6a3'
        elif isinstance(value,str) and value.upper() == 'false':
            return '#f4a3a3'
        return '#FFFFFF'

    def update(self, l):
        self.ax.set_axis_off()
        self.ax.clear()

        table = Table(self.ax, bbox=[0, 0, 1, 1])
        
        cell_texts = [
            ["On Floor", "Hitted", "Thrower"],
            ["OutLined", "L In", "R In"],
            ["Running", "L Out", "R Out"]
        ]
        
        # 셀 배치 설계
        cells = [
            [l[cell_texts[0][0]], l[cell_texts[0][1]], l[cell_texts[0][2]]],
            [l[cell_texts[1][0]], l[cell_texts[1][1]], l[cell_texts[1][2]]],
            [l[cell_texts[2][0]], l[cell_texts[2][1]], l[cell_texts[2][2]]]
        ]

        for i in range(self.nrows):
            for j in range(self.ncols):
                value = cells[i][j]
                color = self.get_cell_color(value)
                text = cell_texts[i][j]
                table.add_cell(i, j, self.width, self.height,
                               text=text, loc='center', facecolor=color)
 

        # 테두리 두껍게
        for cell in table.get_celld().values():
            cell.set_linewidth(2)
            cell.set_fontsize(14)

        self.ax.add_table(table)

        # 크기 및 표시
        self.fig.set_size_inches(8, 3)
        plt.pause(0.01)