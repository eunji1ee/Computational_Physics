import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

bins, count = [], []
with open("hist2.csv", "r") as f:
    for line in f.readlines():
        _b, _c = [float(i) for i in line.split(",")]
        bins.append(_b)
        count.append(_c)
        
bins = np.array(bins)
count = np.array(count)

def particle(x,a,b,c):
    return a * np.exp(-((x -b)**2)/(2*c**2))

def model(x,a_0,a_1,a_2,a_3,a_4,a_5):
    p_a = particle(x,a_0,a_1,a_2)
    p_b = particle(x,a_3,a_4,a_5)
    return p_a + p_b

initial_guess = [100, 0, 1, 100, 2, 1]  # 초기 추정값
popt, pcov = curve_fit(model, bins, count, p0=initial_guess)

# 피팅 결과 출력
print("Fitted parameters:", popt)

# 피팅 결과 그래프
x_fit = np.linspace(min(bins), max(bins), 100)
y_fit = model(x_fit, *popt)
plt.figure(figsize=(10, 6))
plt.hist(x_fit, bins=len(bins), weights=particle(x_fit, *popt[:3]), alpha=0.5, label='Particle A', color='red')
plt.hist(x_fit, bins=len(bins), weights=particle(x_fit, *popt[3:]), alpha=0.5, label='Particle B', color='blue')
plt.title('Energy spectrum of particles A, B (using Gaussian distributions)')
plt.legend()
plt.grid()
plt.show()

# 각 가우시안 적분
integral_a = trapezoid(particle(x_fit, *popt[:3]), x_fit) 
integral_b = trapezoid(particle(x_fit, *popt[3:]), x_fit)

# 결과 출력
print("Integral of Particle A:", integral_a)
print("Integral of Particle B:", integral_b)

# 생성비 계산
creation_ratio = integral_a / integral_b if integral_b != 0 else np.inf
print("Creation Ratio (A/B):", creation_ratio)
