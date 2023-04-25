import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt

def segment_month(d, index):
    global F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    n = d[index].size
    print("n=", n)
    F0 = np.zeros((n))
    i = 0
    j = 0
    l = 0
    while d["Месяц"][i] == "Январь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F1 = np.zeros((l))
    for l1 in range(0, l):
        F1[l1] = F0[l1]
    print("F1=", F1)
    j = 0
    l = 0
    while d["Месяц"][i] == "Февраль":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F2 = np.zeros((l))
    for l1 in range(0, l):
        F2[l1] = F0[l1]
    print("F2=", F2)
    j = 0
    l = 0
    while d["Месяц"][i] == "Март":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F3 = np.zeros((l))
    for l1 in range(0, l):
        F3[l1] = F0[l1]
    print("F3=", F3)

    j = 0
    l = 0
    while d["Месяц"][i] == "Апрель":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F4 = np.zeros(l)
    for l1 in range(0, l):
        F4[l1] = F0[l1]
    print("F4=", F4)

    j = 0
    l = 0
    while d["Месяц"][i] == "Май":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F5 = np.zeros((l))
    for l1 in range(0, l):
        F5[l1] = F0[l1]
    print("F5=", F5)

    j = 0
    l = 0
    while d["Месяц"][i] == "Июнь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F6 = np.zeros((l))
    for l1 in range(0, l):
        F6[l1] = F0[l1]
    print("F6=", F6)

    j = 0
    l = 0
    while d["Месяц"][i] == "Июль":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F7 = np.zeros((l))
    for l1 in range(0, l):
        F7[l1] = F0[l1]
    print("F7=", F7)

    j = 0
    l = 0
    while d["Месяц"][i] == "Август":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F8 = np.zeros((l))
    for l1 in range(0, l):
        F8[l1] = F0[l1]
    print("F8=", F8)

    j = 0
    l = 0
    while d["Месяц"][i] == "Сентябрь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F9 = np.zeros((l))
    for l1 in range(0, l):
        F9[l1] = F0[l1]
    print("F9=", F9)

    j = 0
    l = 0
    while d["Месяц"][i] == "Октябрь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F10 = np.zeros((l))
    for l1 in range(0, l):
        F10[l1] = F0[l1]
    print("F10=", F10)

    j = 0
    l = 0
    while d["Месяц"][i] == "Декабрь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F12 = np.zeros((l))
    for l1 in range(0, l):
        F12[l1] = F0[l1]
    print("F12=", F12)
    j = 0
    l = 0
    while d["Месяц"][i] == "Ноябрь":
        F0[j] = d[index][i]
        i = i + 1
        j = j + 1
        l = l + 1
    F11 = np.zeros((l))
    for l1 in range(0, l):
        F11[l1] = F0[l1]
    print("F11=", F11)
    return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12


def Sum_segment_month(d, index, F_in):
    global F_segment_month
    n = d[index].size
    F0 = np.zeros((n))
    F_segment_month = np.zeros((12))

    i = 0
    j = 0
    l = 0
    while d["Месяц"][i] == "Январь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[0] = F_segment_month[0] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Февраль":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[1] = F_segment_month[1] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Март":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[2] = F_segment_month[2] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Апрель":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[3] = F_segment_month[3] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Май":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[4] = F_segment_month[4] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Июнь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[5] = F_segment_month[5] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Июль":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[6] = F_segment_month[6] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Август":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[7] = F_segment_month[7] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Сентябрь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[8] = F_segment_month[8] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Октябрь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[9] = F_segment_month[9] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Декабрь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[11] = F_segment_month[11] + F0[l1]

    j = 0
    l = 0
    while d["Месяц"][i] == "Ноябрь":
        F0[j] = F_in[i]
        i = i + 1
        j = j + 1
        l = l + 1
    for l1 in range(0, l):
        F_segment_month[10] = F_segment_month[10] + F0[l1]
    print("F_segment_month=", F_segment_month)
    return F_segment_month


def sale(d):
    global F_sale
    n = d["КолВо реализации"].size
    F_sale = np.zeros((n))
    for i in range(0, n):
        F_sale[i] = d["КолВо реализации"][i] * d["Цена реализации"][i]
    print("F_sale=", F_sale)
    return F_sale


def profit(d):
    global F_profit
    n = d["КолВо реализации"].size
    F_profit = np.zeros((n))
    for i in range(0, n):
        F_profit[i] = d["КолВо реализации"][i] * (
            d["Цена реализации"][i] - d["Себестоимость единицы"][i]
        )
    print("F_profit=", F_profit)
    return F_profit

d = pd.read_excel("data\\Pr12.xls", parse_dates=["Дата"])
dd = pd.read_excel("data\\Pr12.xls", parse_dates=["Дата"], index_col="Дата")
print("d=", d)
print("dd=", dd)
print("---------------- Цена реализации -------------------")
print(d["Цена реализации"])
print(type(float(d["Цена реализации"][0])))

index = "Цена реализации"
segment_month(d, index)
plt.title(index)
d[index].plot()
plt.show()
plt.title(index)
dd[index].plot()
plt.show()
plt.title("Month 1-6")
plt.plot(F1)
plt.plot(F2)
plt.plot(F3)
plt.plot(F4)
plt.plot(F5)
plt.plot(F6)
plt.show()

sale(d)  # за рік
Sum_segment_month(d, index, F_sale)  # по місяцях
plt.title("Sale")
plt.plot(F_sale)  # візуалізація за рік
plt.show()
s = pd.Series(F_segment_month)
plt.title("Sale")
s.plot(kind="bar")  # візуалізація по місяцях
plt.show()
F_segment_month_sale = F_segment_month

profit(d)  # за рік
Sum_segment_month(d, index, F_profit)  # по місяцях
plt.title("Profit")
plt.plot(F_profit)  # візуалізація за рік
plt.show()
s = pd.Series(F_segment_month)
plt.title("Profit")
s.plot(kind="bar")  # візуалізація по місяцях
plt.show()
F_segment_month_profit = F_segment_month

plt.title("Sale + Profit")
plt.plot(F_sale)
plt.plot(F_profit)
plt.show()
s1 = pd.Series(F_segment_month_profit)
s2 = pd.Series(F_segment_month_sale)
plt.title("Sale + Profit")
s2.plot(kind="bar", color="b")
s1.plot(kind="bar", color="g")
plt.show()

def Stst_A(S, Y_coord, title):
    iter = Y_coord.size
    S0 = np.zeros(iter)
    for i in range(iter):
        S0[i] = abs(S[i] - 0)
    mS = np.mean(S0)
    dS = np.var(S0)
    scvS = mt.sqrt(dS)
    print("----- статистичні характеристики  -----")
    print("математичне сподівання ВВ=", mS)
    print("дисперсія ВВ =", dS)
    print("СКВ ВВ=", scvS)
    plt.title(title)
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return Stst_A

def MNK(Y_coord):
    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout

def MNK_extrapolation(Y_coord, koef):

    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)

    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    j = koef
    for i in range(0, koef):
        Yout[i, 0] = (
            C[0, 0]
            + C[1, 0] * j
            + (C[2, 0] * j * j)
            + (C[3, 0] * j * j * j)
            + (C[4, 0] * j * j * j * j)
        )
        j = j + 1
    return Yout

Yout0 = MNK(F_sale)
print("------------ вхідна вибірка  ----------")
Stst_A(F_sale, Yout0, "вхідна вибірка за рік")
print("-------------- МНК оцінка  ------------")
Stst_A(Yout0, Yout0, "МНК оцінка за рік")
plt.title("MNK_sale")
plt.plot(F_sale)
plt.plot(Yout0)
plt.show()

Yout1 = MNK(F_segment_month_sale)
print("------------ вхідна вибірка  ----------")
Stst_A(F_segment_month_sale, Yout1, "вхідна вибірка за місяцями")
print("-------------- МНК оцінка  ------------")
Stst_A(Yout1, Yout1, "МНК оцінка за місяцями")
plt.title("MNK_segment_month_sale")
plt.plot(F_segment_month_sale)
plt.plot(Yout1)
plt.show()

prognoz = int(0.5 * (F_sale.size))
Graf_Yout0 = np.zeros(prognoz)
Graf_Yout0_1 = np.zeros(6)
Yout0 = MNK_extrapolation(F_sale, prognoz)
Graf_Yout0_1[0] = Yout0[1, 0]
Graf_Yout0_1[1] = Yout0[(int(prognoz / 6)), 0]
Graf_Yout0_1[2] = Yout0[(int((2 * prognoz / 6))), 0]
Graf_Yout0_1[3] = Yout0[(int((3 * prognoz / 6))), 0]
Graf_Yout0_1[4] = Yout0[(int((4 * prognoz / 6))), 0]
Graf_Yout0_1[5] = Yout0[(int((5 * prognoz / 6))), 0]
print("------------ MNK_extrapolation_sale ---------------")
for i in range(0, 6):
    print("Yout0[", i, ",0]=", Graf_Yout0_1[i])
plt.title("MNK_extrapolation_sale")
for i in range(0, prognoz):
    Graf_Yout0[i] = Yout0[i, 0]
plt.plot(Graf_Yout0)
plt.show()

prognoz = 6
Graf_Yout1 = np.zeros(prognoz)
Yout1 = MNK_extrapolation(F_segment_month_sale, prognoz)
print("------ MNK_extrapolation_segment_month_sale --------")
for i in range(prognoz):
    print("Yout1[", i, ",0]=", Yout1[i, 0])
plt.title("MNK_extrapolation_segment_month_sale")
for i in range(0, prognoz):
    Graf_Yout1[i] = Yout1[i, 0]
plt.plot(Graf_Yout1)
plt.show()

plt.title("MNK_extrapolation за рік та місяці")
plt.plot(Graf_Yout0_1)
plt.plot(Graf_Yout1)
plt.show()
