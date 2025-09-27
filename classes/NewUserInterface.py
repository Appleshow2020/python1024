import matplotlib.pyplot as plt
from matplotlib.table import Table

class UserInterface:
    """
    UserInterface provides a simple graphical table-based UI for displaying the status of various boolean indicators.
    Attributes:
        nrows (int): Number of rows in the table (default: 3).
        ncols (int): Number of columns in the table (default: 3).
        width (float): Width of each cell as a fraction of the table width.
        height (float): Height of each cell as a fraction of the table height.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        ax (matplotlib.axes.Axes): The axes object for drawing the table.
    Methods:
        __init__(nrows=3, ncols=3):
            Initializes the UserInterface, sets up the figure and axes, and configures the table layout.
        get_cell_color(value):
            Returns a color code based on the boolean value ('True' for green, 'False' for red, otherwise white).
        update(l):
            Updates the table display with the current status values.
            Args:
                l (list): List of status values (as strings), expected to be at least 9 elements long.
    """

    def __init__(self, nrows=3, ncols=3):
        self.nrows = nrows
        self.ncols = ncols
        self.width = 1.0 / ncols
        self.height = 1.0 / nrows
        
        # 1. 대화형 모드 활성화 및 창을 미리 생성
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 3)
        self.ax.set_title("Referee UI")

    def get_cell_color(self, value):
        if isinstance(value, str) and value.lower() == 'true':
            return '#a8e6a3' # 초록
        elif isinstance(value, str) and value.lower() == 'false':
            return '#f4a3a3' # 빨강
        return '#FFFFFF' # 기본 흰색

    # 2. plot_table -> update로 이름 변경 및 로직 수정
    def update(self, l):
        # 기존 내용을 깨끗이 지웁니다.
        self.ax.clear()
        self.ax.set_axis_off()

        table = Table(self.ax, bbox=[0, 0, 1, 1])

        # Positioned_labels 배열에 맞춰 셀 텍스트를 지정합니다.
        # (이 부분은 메인 코드의 Positioned_labels 순서와 일치해야 합니다)
        cell_texts = [
            ["On Floor", "Hitted", "Thrower"],
            ["OutLined", "L In", "R In"],
            ["Running", "L Out", "R Out"]
        ]
        
        # bool 값을 문자열로 변환하여 l 리스트를 준비해야 합니다.
        # 예: l = [str(val) for val in Positioned_status]
        cells_data = [
            [l[0], l[1], l[2]],
            [l[3], l[4], l[5]],
            [l[8], l[6], l[7]]
        ]

        for i in range(self.nrows):
            for j in range(self.ncols):
                value = cells_data[i][j]
                text = cell_texts[i][j]
                color = self.get_cell_color(value)
                table.add_cell(i, j, self.width, self.height,
                               text=text, loc='center', facecolor=color)

        for cell in table.get_celld().values():
            cell.set_linewidth(2)
            cell.set_fontsize(14) # 폰트 크기 조절

        self.ax.add_table(table)
        
        # 3. fig.canvas.draw()와 plt.pause()로 논블로킹 업데이트
        self.fig.canvas.draw()
        plt.pause(0.01) # 0.01초 동안 GUI를 업데이트하고 즉시 리턴