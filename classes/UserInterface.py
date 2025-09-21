import matplotlib.pyplot as plt
from matplotlib.table import Table


class UserInterface:
    """
    심판에게 보여줄 UI를 구성하는 클래스

    Functions : get_cell_color, plot_table
    initial variables : nrow(default 3), ncols(5), cell_width(None), cell_height(None)

    1. get_cell_color
    properties : value(any)
    UI 셀들의 값들에 색을 넣는 함수

    2. plot_table
    properties : l
    UI를 보여주는 함수
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