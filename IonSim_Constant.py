import numpy as np

n_e =2.5*10**8  #electron destiny of Dlayer
n_e1 =5*10**11  #electron destiny of F Layer
nu_c = 10**(7) #HZ########
eposilon_0 = 8.85*10**(-12) 
e = 1.6*10**(-19) 
m = 9.10956*10**(-31)
R_E=6378 #km
delta_H_D = 30#Km
H_D  =75 #Km
C = 2.99792*10**(5)
nu_p_D=np.sqrt(e**2*n_e/(eposilon_0*m))/(2*np.pi) #electron plasma frequency of D layer
nu_p_F=np.sqrt(e**2*n_e1/(eposilon_0*m))/(2*np.pi) #electron plasma frequency of F layer
L=np.zeros(50)
yita_D=np.zeros(50)
T_gama = 2.72548 # CMB temperature
T_s = 2#spin temperature
z =  20#redshift
x_HI = 0.51 # mean neutral hydrogen fraction
T_e = 800   #typical D-layer electron temperature of Te = 800 K for mid-latitude ionosphere
T_eba = 1500 #或许可以直接假设不随时间变化？
n_e =5*10**8  #electron destiny
TEC_D = 20#10**(0.5)
h_m = 250
d = 15