import pandas as pd
import math


def check_pow():
    df = pd.read_csv("pow_results.csv")
    correct = 0
    for i in range(100):
        arg1 = df['Arg1'][i]
        arg2 = df['Arg2'][i]
        result = df['Result'][i]

        truth = arg1 ** arg2

        result_rounded = round(result + 0.01, 1)
        truth_rounded = round(truth + 0.01, 1)


        if abs(result_rounded - truth_rounded) < 1e-6:  # Точность сравнения
            correct += 1
            # print(f"Правильно: pow({arg1}, {arg2}) = {truth_rounded:.6f}")
        else:
            print(f"Ошибка: pow({arg1}, {arg2}) = {result_rounded:.6f}, ожидалось {truth_rounded:.6f}")

    print(f"Correct: {correct}/100")

def check_sin():
    df = pd.read_csv("sin_results.csv")
    correct = 0
    for i in range(100):
        arg1 = df['Arg1'][i]
        result = df['Result'][i]

        truth = math.sin(arg1)

        result_rounded = round(result + 0.01, 1)
        truth_rounded = round(truth + 0.01, 1)


        if abs(result_rounded - truth_rounded) < 1e-6:  # Точность сравнения
            correct += 1
            # print(f"Правильно: pow({arg1}, {arg2}) = {truth_rounded:.6f}")
        else:
            print(f"Ошибка: sin({arg1}) = {result_rounded:.6f}, ожидалось {truth_rounded:.6f}")

    print(f"Correct: {correct}/100")

def check_sqrt():
    df = pd.read_csv("sqrt_results.csv")
    correct = 0
    for i in range(100):
        arg1 = df['Arg1'][i]
        result = df['Result'][i]

        truth = math.sqrt(arg1)

        result_rounded = round(result + 0.01, 1)
        truth_rounded = round(truth + 0.01, 1)


        if abs(result_rounded - truth_rounded) < 1e-6:  # Точность сравнения
            correct += 1
            # print(f"Правильно: pow({arg1}, {arg2}) = {truth_rounded:.6f}")
        else:
            print(f"Ошибка: sin({arg1}) = {result_rounded:.6f}, ожидалось {truth_rounded:.6f}")

    print(f"Correct: {correct}/100")


check_pow()
check_sin()
check_sqrt()