import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


DATA_COLOR = "darkblue"
LINE_COLOR = "red"
SQ_COLOR_PAD = "orange"
SQ_COLOR_MAR = "darkorange"
FIG_SIZE = (8, 5)


def read_csv(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        columns = {h: [] for h in headers}
        for row in reader:
            for h, val in zip(headers, row):
                columns[h].append(float(val))
    return headers, columns

def print_stats(columns):
    print(f"\n{'Столбец':<20}{'N':>6}{'min':>12}{'max':>12}{'mean':>12}")
    print("-" * 62)
    for name, data in columns.items():
        arr = np.array(data)
        print(f"{name:<20}{len(arr):>6}{arr.min():>12.2f}{arr.max():>12.2f}{arr.mean():>12.2f}")

def choose_columns(headers):
    if len(headers) == 2:
        return headers[0], headers[1]

    print("Доступные столбцы:")
    for i, h in enumerate(headers):
        print(f"{i}: {h}")

    while True:
        try:
            xi = int(input("Выберите столбец X: "))
            yi = int(input("Выберите столбец Y: "))
        except ValueError:
            print("Нужно ввести целое число.")
            continue
        n = len(headers)
        if xi == yi or not (0 <= xi < n) or not (0 <= yi < n):
            print(f"Номера должны быть разными и в диапазоне от 0 до {n - 1}.")
            continue
        return headers[xi], headers[yi]

def least_squares(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xx = (x * x).sum()
    sum_xy = (x * y).sum()
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    b = (sum_y - a * sum_x) / n
    return a, b

def make_axes(x_name, y_name, title):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax

def draw_data(ax, x, y, label="Исходные данные"):
    ax.scatter(x, y, color=DATA_COLOR, label=label)

def draw_line(ax, x_line, y_line, label):
    ax.plot(x_line, y_line, color=LINE_COLOR, label=label)

def draw_squares(ax, x, y, residuals, k):
    for xi, yi, ri in zip(x, y, residuals):
        side_y = abs(ri)
        side_x = side_y * k
        y_bottom = min(yi, yi - ri)
        rectangle = plt.Rectangle(
            (xi - side_x / 2, y_bottom),
            side_x, side_y,
            facecolor=SQ_COLOR_PAD, edgecolor=SQ_COLOR_MAR,
            alpha=0.5, hatch="///"
        )
        ax.add_patch(rectangle)

def get_filename():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return input("Введите имя файла (Enter - lab1_DS.csv): ") or "lab1_DS.csv"

def main(filename):
    headers, columns = read_csv(filename)
    print_stats(columns)

    x_name, y_name = choose_columns(headers)
    x = np.array(columns[x_name])
    y = np.array(columns[y_name])

    a, b = least_squares(x, y)
    eq = f"y = {a:.2f} * x + {b:.2f}"
    print("\nУравнение регрессии: " + eq)

    y_pred = a * x + b
    residuals = y - y_pred
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = a * x_line + b

    fig1, ax1 = make_axes(x_name, y_name, "Исходные данные")
    draw_data(ax1, x, y)
    ax1.legend()

    fig2, ax2 = make_axes(x_name, y_name, "Линейная регрессия (МНК)")
    draw_data(ax2, x, y)
    draw_line(ax2, x_line, y_line, label=eq)
    ax2.legend()

    fig3, ax3 = make_axes(x_name, y_name, "Квадраты ошибок")
    k = (x.max() - x.min()) / (y.max() - y.min())
    draw_squares(ax3, x, y, residuals, k)
    draw_data(ax3, x, y)
    draw_line(ax3, x_line, y_line, label=eq)
    ax3.plot([], [], color=SQ_COLOR_MAR, alpha=0.5, linewidth=8, label="Квадраты ошибок")
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = get_filename()
    try:
        main(filename)
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        sys.exit(1)