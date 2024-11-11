import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Генерація даних для побудови поверхні
x1_vals = np.linspace(-3, 3, 100)  # Значення для x1
x2_vals = np.linspace(-3, 3, 100)  # Значення для x2
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Обчислення значень цільової функції для кожної пари (x1, x2)
Z = (X1 ** 2 + X2 ** 2 - 1) ** 2 + (X1 ** 2 - X2) ** 2

# Створення графіку
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Побудова поверхні
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')

# Налаштування підписів
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Поверхня цільової функції')

# Показати графік
plt.show()

# Ініціалізація змінної для підрахунку КОЦФ
count_f_evaluations = 0


# Цільова функція
def f(x1, x2):
    global count_f_evaluations
    count_f_evaluations += 1
    return (x1 ** 2 + x2 ** 2 - 1) ** 2 + (x1 ** 2 - x2) ** 2


# Збереження даних для таблиці
data_table = {
    "iteration": [],
    "coordinate_descent_method": [],
    "A0_x": [],
    "A0_y": [],
    "gradient_descent_method": [],
    "A1_x": [],
    "A1_y": [],
    "function_evaluations": []
}


# Метод координатного спуску
def coordinate_descent(f, x0, num_iterations=100, learning_rate=0.01, tolerance=1e-6):
    global count_f_evaluations
    count_f_evaluations = 0  # Обнулити для підрахунку в даному методі
    x = np.array(x0, dtype=float)

    # Процес ітерацій
    for i in range(num_iterations):
        prev_x = np.copy(x)

        # Крок за x1
        x[0] -= learning_rate * 4 * x[0] * (x[0] ** 2 + x[1] ** 2 - 1)
        # Крок за x2
        x[1] -= learning_rate * 4 * x[1] * (x[0] ** 2 + x[1] ** 2 - 1)

        # Обчислення значення функції для підрахунку
        f(x[0], x[1])

        # Перевірка зупинки за критерієм невеликої зміни значення функції
        if np.linalg.norm(x - prev_x) < tolerance:
            break

        # Запис даних
        data_table["iteration"].append(i)
        data_table["coordinate_descent_method"].append("coordinate_descent")
        data_table["A0_x"].append(x[0])
        data_table["A0_y"].append(x[1])
        data_table["function_evaluations"].append(count_f_evaluations)

    return x, count_f_evaluations


# Виконання координатного спуску для стартових точок
A0 = [-2, 2]
A1 = [2, -2]
result_A0 = coordinate_descent(f, A0)  # Передаємо точку як список
result_A1 = coordinate_descent(f, A1)  # Передаємо точку як список

print(f"Мінімум з A0 (кординаторний спуск): {result_A0}")
print(f"Мінімум з A1 (кординаторний спуск): {result_A1}")


# Метод найшвидшого спуску
def gradient_descent(f, x0, num_iterations=100, learning_rate=0.01):
    x = np.array(x0, dtype=float)

    # Функція для обчислення градієнта
    def gradient(x):
        h = 1e-5
        df_dx1 = (f(x[0] + h, x[1]) - f(x[0] - h, x[1])) / (2 * h)
        df_dx2 = (f(x[0], x[1] + h) - f(x[0], x[1] - h)) / (2 * h)
        return np.array([df_dx1, df_dx2])

    # Процес ітерацій
    for i in range(num_iterations):
        grad = gradient(x)
        x -= learning_rate * grad

        # Запис даних
        data_table["gradient_descent_method"].append("gradient_descent")
        data_table["A1_x"].append(x[0])
        data_table["A1_y"].append(x[1])

    return x, count_f_evaluations


# Виконання найшвидшого спуску для стартових точок
result_A0_grad = gradient_descent(f, A0)
result_A1_grad = gradient_descent(f, A1)

print(f"Мінімум з A0 (найшвидший спуск): {result_A0_grad}")
print(f"Мінімум з A1 (найшвидший спуск): {result_A1_grad}")

data = pd.DataFrame(data_table)
data.to_csv(
    "data.csv",
    index=False,
    encoding="utf-8"
)