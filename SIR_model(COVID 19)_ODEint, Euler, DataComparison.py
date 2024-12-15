##odeint

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#SIR 모델 정의
def sir_model(y, t, beta, gamma):
    S, I, R = y

    dSdt = -beta * S * I
    
    dIdt = beta * S * I - gamma * I
    
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]

#초기값 설정
I0 = 31

N = 2436488
S0 = (N-I0)/N    #미감염자(Susceptible)
I0 = 34/N        #감염자(Infected)
R0 = 12/N        #회복자(Recovered)

y0 = [S0, I0, R0] 

beta = 0.471     #감염률
gamma = 1/14     #회복율
r0 = 6.6         #감염재생산지수(R0)

t = np.linspace(0, 160, 160)  

#미분방정식 풀고 결과 저장
solution = odeint(sir_model, y0, t, args=(beta, gamma)) #chat gpt에게 odeint함수 형식에 대해서만 묻고 제 프로젝트 목적에 맞게 수정하였습니다.
S, I, R = solution.T 

#일별 변화 출력 
for day, s, i, r in zip(t, S, I, R): 
   print(f"Day {int(day)}: S = {s*N:.4f}, I = {i*N:.4f}, R = {r*N:.4f}") #chat gpt에게 각 데이터 값을 출력할 수 있는 코드를 부탁하였습니다.

#결과 시각화
plt.figure(figsize=(10, 6)) 
plt.plot(t, S, label="S (Susceptible)", color="blue") 
plt.plot(t, I, label="I (Infected)", color="red") 
plt.plot(t, R, label="R (Recovered)", color="green")
plt.title("Prediction of the number of confirmed COVID-19 cases (using the SIR model)") 
plt.xlabel("Day")
plt.ylabel("S / I / R")
plt.legend() 
plt.grid(True)
plt.show()



##euler

import numpy as np
import matplotlib.pyplot as plt

#SIR 모델 정의
def sir_model(y, t, beta, gamma):
    S, I, R = y

    dSdt = -beta * S * I
    
    dIdt = beta * S * I - gamma * I
    
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]

#초기값 설정
I0 = 31

N = 2436488
S0 = (N-I0)/N    #미감염자(Susceptible)
I0 = 34/N        #감염자(Infected)
R0 = 12/N        #회복자(Recovered)

beta = 0.471     #감염률
gamma = 1/14     #회복율
r0 = 6.6         #감염재생산지수(R0)\

y0 = [S0, I0, R0]
t = np.linspace(0, 160, 160) 

#오일러 방식을 이용
def euler_method(sir_model, y0, t, beta, gamma):
    
    S = [S0] 
    I = [I0] 
    R = [R0] 
    dt = t[1] - t[0]

    for i in range(1, len(t)): #처음에 t의 형태로 짠 코드를 실행하였는데 오류가 발생하여 chat gpt에게 물어보고 len(t)의 형태로 수정하였습니다.

        dS, dI, dR = sir_model([S[i-1], I[i-1], R[i-1]], t[i-1], beta, gamma)
        
        S.append(S[i-1] + dS * dt)
        I.append(I[i-1] + dI * dt)
        R.append(R[i-1] + dR * dt)

    return np.array(S), np.array(I), np.array(R)
    
S, I, R = euler_method(sir_model, y0, t, beta, gamma)

#일별 변화 출력 
for day, s, i, r in zip(t, S, I, R):
    print(f"Day {int(day)}: S = {s * N:.4f}, I = {i * N:.4f}, R = {r * N:.4f}") #chat gpt에게 각 데이터 값을 출력할 수 있는 코드를 부탁하였습니다.(위 odeint에서와 동일함)

#결과 시각화
plt.figure(figsize=(10, 6)) 
plt.plot(t, S, label="S (Susceptible)", color="blue")
plt.plot(t, I, label="I (Infected)", color="red")
plt.plot(t, R, label="R (Recovered)", color="green")
plt.title("Prediction of the number of confirmed COVID-19 cases (using the SIR model)") 
plt.xlabel("Day")
plt.ylabel("S / I / R")
plt.legend() 
plt.grid(True)
plt.show()




##data comparison

import matplotlib.pyplot as plt

#I1 = Real Data / I2 = odeint코드의 결과 / I3 = euler코드의 결과 / t = Day
t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
I1 = [34, 16, 74, 190, 209, 206, 129, 252, 447, 909]  
I2 = [34.0000, 50.8651, 76.0506, 113.7195, 170.0317, 254.1600, 379.9463, 567.9572, 848.9680, 1268.8745]   
I3 = [34.0000, 47.6707, 66.8379, 93.7114, 131.3895, 184.2154, 258.2781, 362.1129, 507.6834, 711.7569]  

#결과 시각화
plt.figure(figsize=(8, 6))
plt.plot(t, I1, label='real data', marker='o', linestyle='-', color='b')  
plt.plot(t, I2, label='odeint', marker='s', linestyle='--', color='r')  
plt.plot(t, I3, label='euler', marker='^', linestyle='-.', color='g') 
plt.title('Comparison of Data')
plt.xlabel('Day (t)')
plt.ylabel('I (Infected)')
plt.legend()
plt.grid(True)
plt.show()



#최종학점: A
#이유: 저는 약 5년 전 들었던 소프트웨어 관련 필수교양을 겨우 따라가며 코딩이 적성에 맞지 않다고 생각했고, 
#      올해 전산물리학 수업을 처음 수강하던 시기엔 그마저도 까먹어서 문구를 print하는 방법조차 가물가물했습니다.
#      그렇지만 이번 기말 프로젝트는 SIR모델을 파이썬으로 직접 구현해보는 데에 그치지 않고 실제 데이터와 비교해보기도 하고
#      파이썬으로 미분방정식을 풀 때 어떤 함수가 더 적절한지에 대해 깊이 고민해보았다는 점에서 스스로 성장했다고 느꼈습니다.
#      그렇지만 SIR모델에 방역정책 및 기타 다양한 변수들을 반영하지 못한 게 아쉬움으로 남습니다.
#      사실 기말 발표에서 보여드린 것 이외에도 코로나19 바이러스의 확산에 따라 변하는 R0값을 변화시키는 모델(방역정책 반영 목적)을 구현하려고도 해보았고
#      잠복기에 관한 변수를 추가로 설정하여 SEIR 코드도 짜보았으나, 
#      일별 R0값의 정확한 수치에 대한 정보를 얻는 데에 일반인 신분으로서는 한계가 있어 R0가 방역정책에 따라 어떻게 변화했는지 알 수 없었고
#      잠복기 E를 반영한 모델이 오히려 (제가 발표했던 SIR모델의 결과보다) 실제 데이터와 더 동떨어진 값을 도출하는 문제를 해결하지 못한 것이 아쉽습니다.
