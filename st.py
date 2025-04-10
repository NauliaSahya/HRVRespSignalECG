import numpy as np
import math
import streamlit as st
import pandas as pd
import csv
import altair as alt
import ipywidgets as widgets
from IPython.display import display


st.set_page_config(layout="wide")

fs=125

def Absolute(y):
    y_ab = np.zeros(len(y))
    for n in range (len(y_ab)):
      y_ab[n] = abs(y[n])
    return y_ab

def MAV(orde, y):
    #forward
    y_mavf = np.zeros(len(y))
    for i in range (len(y)):
      for j in range (orde):
        if i-j >= 0:
          y_mavf[i] += y[i-j]
      y_mavf[i] /= orde
    #backward
    y_mav = np.zeros(len(y_mavf))
    for i in range (len(y_mavf)):
      for j in range (orde):
        if i+j < len(y_mavf):
          y_mav[i] += y_mavf[i+j]
      y_mav[i] /= orde
    return y_mav

def dirac(x):
            if (x ==0):
                dirac_delta = 1
            else:
                dirac_delta = 0 
            return dirac_delta

def read_data(filepath, type="ecg"):
    data = []
    a = 5 if type=="ecg" else 1
    try:
        with open(filepath) as file:
            lines = csv.reader(file, delimiter='\t')
            next(lines)
            next(lines)
            for column in lines:
                data.append(float(column[a]))
    except Exception as e:
        st.error(f"Error membaca file: {e}")
    return np.array(data)

def hg_list():
    h, g, n_list = [], [], []
    for n in range(-2, 2):
        n_list.append(n)
        h.append(1/8 * (dirac(n-1) + 3*dirac(n) + 3*dirac(n+1) + dirac(n+2)))
        g.append(-2 * (dirac(n) - dirac(n+1)))
    return h, g, n_list

def hwgw_freq(h, g):
    Hw, Gw, i_list = np.zeros(20000), np.zeros(20000), []
    for i in range(fs+1):
        i_list.append(i)
        reH, imH, reG, imG = 0, 0, 0, 0
        for k in range(-2, 2):
            reH += h[k+2] * np.cos(k*2*np.pi*i/fs)
            imH -= h[k+2] * np.sin(k*2*np.pi*i/fs)
            reG += g[k+2] * np.cos(k*2*np.pi*i/fs)
            imG -= g[k+2] * np.sin(k*2*np.pi*i/fs)
        Hw[i], Gw[i] = np.sqrt(reH**2 + imH**2), np.sqrt(reG**2 + imG**2)
    return Hw, Gw, i_list

def mallat(ecg, h, g):
    w2fm, s2fm = np.zeros((8, len(ecg))), np.zeros((8, len(ecg)))
    for n in range(len(ecg)):
        for j in range(1, 9):
            for k in range(-2, 2):
                try:
                    w2fm[j-1, n] += g[k+2] * ecg[round(n - np.power(2, j-1) * k)]
                    s2fm[j-1, n] += h[k+2] * ecg[round(n - np.power(2, j-1) * k)]
                except:
                    pass
    return w2fm, s2fm

def fr_filbank(Gw, Hw, fs):
    Q = np.zeros((9, round(fs/2)+1))
    i_list = list(range(0, round(fs/2)+1))
    for i in i_list:
        Q[1][i] = Gw[i]
        Q[2][i] = Gw[2*i]*Hw[i]
        Q[3][i] = Gw[4*i]*Hw[2*i]*Hw[i]
        Q[4][i] = Gw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[5][i] = Gw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[6][i] = Gw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[7][i] = Gw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
        Q[8][i] = Gw[128*i]*Hw[64*i]*Hw[32*i]*Hw[16*i]*Hw[8*i]*Hw[4*i]*Hw[2*i]*Hw[i]
    return Q

def compute_qj(j):
    qj = np.zeros(100000)  # Initialize filter for level j
    k_list = []
    a = -(round(2**j) + round(2**(j-1))-2)
    b = -(1-round(2**(j-1))) + 1
        
    for k in range(a, b):
        k_list.append(k)
        if j == 1:
            qj[k + abs(a)] = -2 * (dirac(k) - dirac(k + 1))
        elif j == 2:
            qj[k + abs(a)] = -1/4 * (dirac(k-1) + 3*dirac(k) + 2*dirac(k+1) - 2*dirac(k+2)
                                     - 3*dirac(k+3) - dirac(k+4))
        elif j == 3:
            qj[k + abs(a)] = -1/32 * (dirac(k-3) + 3*dirac(k-2) + 6*dirac(k-1) + 10*dirac(k)
                                    + 11*dirac(k+1) + 9*dirac(k+2) + 4*dirac(k+3) - 4*dirac(k+4)
                                    - 9*dirac(k+5) - 11*dirac(k+6) - 10*dirac(k+7) - 6*dirac(k+8)
                                    - 3*dirac(k+9) - dirac(k+10))
        elif j == 4:
            qj[k + abs(a)] = -1/256 * (dirac(k-7) + 3*dirac(k-6) + 6*dirac(k-5) + 10*dirac(k-4)
                                    + 15*dirac(k-3) + 21*dirac(k-2) + 28*dirac(k-1) + 36*dirac(k)
                                    + 41*dirac(k+1) + 43*dirac(k+2) + 42*dirac(k+3) + 38*dirac(k+4)
                                    + 31*dirac(k+5) + 21*dirac(k+6) + 8*dirac(k+7) - 8*dirac(k+8)
                                    - 21*dirac(k+9) - 31*dirac(k+10) - 38*dirac(k+11) - 42*dirac(k+12)
                                    - 43*dirac(k+13) - 41*dirac(k+14) - 36*dirac(k+15) - 28*dirac(k+16)
                                    - 21*dirac(k+17) - 15*dirac(k+18) - 10*dirac(k+19) - 6*dirac(k+20)
                                    - 3*dirac(k+21) - dirac(k+22))
        elif j == 5:
            qj[k + abs(a)] = -1/512 * (dirac(k-15) + 3*dirac(k-14) + 6*dirac(k-13) + 10*dirac(k-12)
                                    + 15*dirac(k-11) + 21*dirac(k-10) + 28*dirac(k-9) + 36*dirac(k-8)
                                    + 45*dirac(k-7) + 55*dirac(k-6) + 66*dirac(k-5) + 78*dirac(k-4)
                                    + 91*dirac(k-3) + 105*dirac(k-2) + 120*dirac(k-1) + 136*dirac(k)
                                    + 149*dirac(k+1) + 159*dirac(k+2) + 166*dirac(k+3) + 170*dirac(k+4)
                                    + 171*dirac(k+5) + 169*dirac(k+6) + 164*dirac(k+7) + 156*dirac(k+8)
                                    + 145*dirac(k+9) + 131*dirac(k+10) + 114*dirac(k+11) + 94*dirac(k+12)
                                    + 71*dirac(k+13) + 45*dirac(k+14) + 16*dirac(k+15) - 16*dirac(k+16)
                                    - 45*dirac(k+17) - 71*dirac(k+18) - 94*dirac(k+19) - 114*dirac(k+20)
                                    - 131*dirac(k+21) - 145*dirac(k+22) - 156*dirac(k+23) - 164*dirac(k+24)
                                    - 169*dirac(k+25) - 171*dirac(k+26) - 170*dirac(k+27) - 166*dirac(k+28)
                                    - 159*dirac(k+29) - 149*dirac(k+30) - 136*dirac(k+31) - 120*dirac(k+32)
                                    - 105*dirac(k+33) - 91*dirac(k+34) - 78*dirac(k+35) - 66*dirac(k+36)
                                    - 55*dirac(k+37) - 45*dirac(k+38) - 36*dirac(k+39) - 28*dirac(k+40)
                                    - 21*dirac(k+41) - 15*dirac(k+42) - 10*dirac(k+43) - 6*dirac(k+44)
                                    - 3*dirac(k+45) - dirac(k+46))
        elif j==6:
            qj[k + abs(a)] = -1 / 16384 * (dirac(k - 31) + 3 * dirac(k - 30) + 6 * dirac(k - 29) + 10 * dirac(k - 28) +
                        15 * dirac(k - 27) + 21 * dirac(k - 26) + 28 * dirac(k - 25) + 36 * dirac(k - 24) +
                        45 * dirac(k - 23) + 55 * dirac(k - 22) + 66 * dirac(k - 21) + 78 * dirac(k - 20) +
                        91 * dirac(k - 19) + 105 * dirac(k - 18) + 120 * dirac(k - 17) + 136 * dirac(k - 16) +
                        153 * dirac(k - 15) + 171 * dirac(k - 14) + 190 * dirac(k - 13) + 210 * dirac(k - 12) +
                        231 * dirac(k - 11) + 253 * dirac(k - 10) + 276 * dirac(k - 9) + 300 * dirac(k - 8) +
                        325 * dirac(k - 7) + 351 * dirac(k - 6) + 378 * dirac(k - 5) + 406 * dirac(k - 4) +
                        435 * dirac(k - 3) + 465 * dirac(k - 2) + 496 * dirac(k - 1) + 528 * dirac(k) +
                        557 * dirac(k + 1) + 583 * dirac(k + 2) + 606 * dirac(k + 3) + 626 * dirac(k + 4) +
                        643 * dirac(k + 5) + 657 * dirac(k + 6) + 668 * dirac(k + 7) + 676 * dirac(k + 8) +
                        681 * dirac(k + 9) + 683 * dirac(k + 10) + 682 * dirac(k + 11) + 678 * dirac(k + 12) +
                        671 * dirac(k + 13) + 661 * dirac(k + 14) + 648 * dirac(k + 15) + 632 * dirac(k + 16) +
                        613 * dirac(k + 17) + 591 * dirac(k + 18) + 566 * dirac(k + 19) + 538 * dirac(k + 20) +
                        507 * dirac(k + 21) + 473 * dirac(k + 22) + 436 * dirac(k + 23) + 396 * dirac(k + 24) +
                        353 * dirac(k + 25) + 307 * dirac(k + 26) + 258 * dirac(k + 27) + 206 * dirac(k + 28) +
                        151 * dirac(k + 29) + 93 * dirac(k + 30) + 32 * dirac(k + 31) - 32 * dirac(k + 32) -
                        93 * dirac(k + 33) - 151 * dirac(k + 34) - 206 * dirac(k + 35) - 258 * dirac(k + 36) -
                        307 * dirac(k + 37) - 353 * dirac(k + 38) - 396 * dirac(k + 39) - 436 * dirac(k + 40) -
                        473 * dirac(k + 41) - 507 * dirac(k + 42) - 538 * dirac(k + 43) - 566 * dirac(k + 44) -
                        591 * dirac(k + 45) - 613 * dirac(k + 46) - 632 * dirac(k + 47) - 648 * dirac(k + 48) -
                        661 * dirac(k + 49) - 671 * dirac(k + 50) - 678 * dirac(k + 51) - 682 * dirac(k + 52) -
                        683 * dirac(k + 53) - 681 * dirac(k + 54) - 676 * dirac(k + 55) - 668 * dirac(k + 56) -
                        657 * dirac(k + 57) - 643 * dirac(k + 58) - 626 * dirac(k + 59) - 606 * dirac(k + 60) -
                        583 * dirac(k + 61) - 557 * dirac(k + 62) - 528 * dirac(k + 63) - 496 * dirac(k + 64) -
                        465 * dirac(k + 65) - 435 * dirac(k + 66) - 406 * dirac(k + 67) - 378 * dirac(k + 68) -
                        351 * dirac(k + 69) - 325 * dirac(k + 70) - 300 * dirac(k + 71) - 276 * dirac(k + 72) -
                        253 * dirac(k + 73) - 231 * dirac(k + 74) - 210 * dirac(k + 75) - 190 * dirac(k + 76) -
                        171 * dirac(k + 77) - 153 * dirac(k + 78) - 136 * dirac(k + 79) - 120 * dirac(k + 80) -
                        105 * dirac(k + 81) - 91 * dirac(k + 82) - 78 * dirac(k + 83) - 66 * dirac(k + 84) -
                        55 * dirac(k + 85) - 45 * dirac(k + 86) - 36 * dirac(k + 87) - 28 * dirac(k + 88) -
                        21 * dirac(k + 89) - 15 * dirac(k + 90) - 10 * dirac(k + 91) - 6 * dirac(k + 92) -
                        3 * dirac(k + 93) - dirac(k + 94))
        elif j==7:
            qj[k + abs(a)] = -1 / 131072 * (
                        dirac(k-63) + 3*dirac(k-62) + 6*dirac(k-61) + 10*dirac(k-60) + 15*dirac(k-59) +
                        21*dirac(k-58) + 28*dirac(k-57) + 36*dirac(k-56) + 45*dirac(k-55) +
                        55*dirac(k-54) + 66*dirac(k-53) + 78*dirac(k-52) + 91*dirac(k-51) +
                        105*dirac(k-50) + 120*dirac(k-49) + 136*dirac(k-48) + 153*dirac(k-47) +
                        171*dirac(k-46) + 190*dirac(k-45) + 210*dirac(k-44) + 231*dirac(k-43) +
                        253*dirac(k-42) + 276*dirac(k-41) + 300*dirac(k-40) + 325*dirac(k-39) +
                        351*dirac(k-38) + 378*dirac(k-37) + 406*dirac(k-36) + 435*dirac(k-35) +
                        465*dirac(k-34) + 496*dirac(k-33) + 528*dirac(k-32) + 561*dirac(k-31) +
                        595*dirac(k-30) + 630*dirac(k-29) + 666*dirac(k-28) + 703*dirac(k-27) +
                        741*dirac(k-26) + 780*dirac(k-25) + 820*dirac(k-24) + 861*dirac(k-23) +
                        903*dirac(k-22) + 946*dirac(k-21) + 990*dirac(k-20) + 1035*dirac(k-19) +
                        1081*dirac(k-18) + 1128*dirac(k-17) + 1176*dirac(k-16) + 1225*dirac(k-15) +
                        1275*dirac(k-14) + 1326*dirac(k-13) + 1378*dirac(k-12) + 1431*dirac(k-11) +
                        1485*dirac(k-10) + 1540*dirac(k-9) + 1596*dirac(k-8) + 1653*dirac(k-7) +
                        1711*dirac(k-6) + 1770*dirac(k-5) + 1830*dirac(k-4) + 1891*dirac(k-3) +
                        1953*dirac(k-2) + 2016*dirac(k-1) + 2080*dirac(k) + 2141*dirac(k+1) +
                        2199*dirac(k+2) + 2254*dirac(k+3) + 2306*dirac(k+4) + 2355*dirac(k+5) +
                        2401*dirac(k+6) + 2444*dirac(k+7) + 2484*dirac(k+8) + 2521*dirac(k+9) +
                        2555*dirac(k+10) + 2586*dirac(k+11) + 2614*dirac(k+12) + 2639*dirac(k+13) +
                        2661*dirac(k+14) + 2680*dirac(k+15) + 2696*dirac(k+16) + 2709*dirac(k+17) +
                        2719*dirac(k+18) + 2726*dirac(k+19) + 2730*dirac(k+20) + 2731*dirac(k+21) +
                        2729*dirac(k+22) + 2724*dirac(k+23) + 2716*dirac(k+24) + 2705*dirac(k+25) +
                        2691*dirac(k+26) + 2674*dirac(k+27) + 2654*dirac(k+28) + 2631*dirac(k+29) +
                        2605*dirac(k+30) + 2576*dirac(k+31) + 2544*dirac(k+32) + 2509*dirac(k+33) +
                        2471*dirac(k+34) + 2430*dirac(k+35) + 2386*dirac(k+36) + 2339*dirac(k+37) +
                        2289*dirac(k+38) + 2236*dirac(k+39) + 2180*dirac(k+40) + 2121*dirac(k+41) +
                        2059*dirac(k+42) + 1994*dirac(k+43) + 1926*dirac(k+44) + 1855*dirac(k+45) +
                        1781*dirac(k+46) + 1704*dirac(k+47) + 1624*dirac(k+48) + 1541*dirac(k+49) +
                        1455*dirac(k+50) + 1366*dirac(k+51) + 1274*dirac(k+52) + 1179*dirac(k+53) +
                        1081*dirac(k+54) + 980*dirac(k+55) + 876*dirac(k+56) + 769*dirac(k+57) +
                        659*dirac(k+58) + 546*dirac(k+59) + 430*dirac(k+60) + 311*dirac(k+61) +
                        189*dirac(k+62) + 64*dirac(k+63) - 64*dirac(k+64) - 189*dirac(k+65) -
                        311*dirac(k+66) - 430*dirac(k+67) - 546*dirac(k+68) - 659*dirac(k+69) -
                        769*dirac(k+70) - 876*dirac(k+71) - 980*dirac(k+72) - 1081*dirac(k+73) -
                        1179*dirac(k+74) - 1274*dirac(k+75) - 1366*dirac(k+76) - 1455*dirac(k+77) -
                        1541*dirac(k+78) - 1624*dirac(k+79) - 1704*dirac(k+80) - 1781*dirac(k+81) -
                        1855*dirac(k+82) - 1926*dirac(k+83) - 1994*dirac(k+84) - 2059*dirac(k+85) -
                        2121*dirac(k+86) - 2180*dirac(k+87) - 2236*dirac(k+88) - 2289*dirac(k+89) -
                        2339*dirac(k+90) - 2386*dirac(k+91) - 2430*dirac(k+92) - 2471*dirac(k+93) -
                        2509*dirac(k+94) - 2544*dirac(k+95) - 2576*dirac(k+96) - 2605*dirac(k+97) -
                        2631*dirac(k+98) - 2654*dirac(k+99) - 2674*dirac(k+100) - 2691*dirac(k+101) -
                        2705*dirac(k+102) - 2716*dirac(k+103) - 2724*dirac(k+104) - 2729*dirac(k+105) -
                        2731*dirac(k+106) - 2730*dirac(k+107) - 2726*dirac(k+108) - 2719*dirac(k+109) -
                        2709*dirac(k+110) - 2696*dirac(k+111) - 2680*dirac(k+112) - 2661*dirac(k+113) -
                        2639*dirac(k+114) - 2614*dirac(k+115) - 2586*dirac(k+116) - 2555*dirac(k+117) -
                        2521*dirac(k+118) - 2484*dirac(k+119) - 2444*dirac(k+120) - 2401*dirac(k+121) -
                        2355*dirac(k+122) - 2306*dirac(k+123) - 2254*dirac(k+124) - 2199*dirac(k+125) -
                        2141*dirac(k+126) - 2080*dirac(k+127) - 2016*dirac(k+128) - 1953*dirac(k+129) -
                        1891*dirac(k+130) - 1830*dirac(k+131) - 1770*dirac(k+132) - 1711*dirac(k+133) -
                        1653*dirac(k+134) - 1596*dirac(k+135) - 1540*dirac(k+136) - 1485*dirac(k+137) -
                        1431*dirac(k+138) - 1378*dirac(k+139) - 1326*dirac(k+140) - 1275*dirac(k+141) -
                        1225*dirac(k+142) - 1176*dirac(k+143) - 1128*dirac(k+144) - 1081*dirac(k+145) -
                        1035*dirac(k+146) - 990*dirac(k+147) - 946*dirac(k+148) - 903*dirac(k+149) -
                        861*dirac(k+150) - 820*dirac(k+151) - 780*dirac(k+152) - 741*dirac(k+153) -
                        703*dirac(k+154) - 666*dirac(k+155) - 630*dirac(k+156) - 595*dirac(k+157) -
                        561*dirac(k+158) - 528*dirac(k+159) - 496*dirac(k+160) - 465*dirac(k+161) -
                        435*dirac(k+162) - 406*dirac(k+163) - 378*dirac(k+164) - 351*dirac(k+165) -
                        325*dirac(k+166) - 300*dirac(k+167) - 276*dirac(k+168) - 253*dirac(k+169) -
                        231*dirac(k+170) - 210*dirac(k+171) - 190*dirac(k+172) - 171*dirac(k+173) -
                        153*dirac(k+174) - 136*dirac(k+175) - 120*dirac(k+176) - 105*dirac(k+177) -
                        91*dirac(k+178) - 78*dirac(k+179) - 66*dirac(k+180) - 55*dirac(k+181) -
                        45*dirac(k+182) - 36*dirac(k+183) - 28*dirac(k+184) - 21*dirac(k+185) -
                        15*dirac(k+186) - 10*dirac(k+187) - 6*dirac(k+188) - 3*dirac(k+189) -
                        dirac(k+190))
        elif j==8:
            sub1 = (dirac(k-127) + 3*dirac(k-126) + 6*dirac(k-125) + 10*dirac(k-124) + 15*dirac(k-123) +
                21*dirac(k-122) + 28*dirac(k-121) + 36*dirac(k-120) + 45*dirac(k-119) + 55*dirac(k-118) +
                66*dirac(k-117) + 78*dirac(k-116) + 91*dirac(k-115) + 105*dirac(k-114) + 120*dirac(k-113) +
                136*dirac(k-112) + 153*dirac(k-111) + 171*dirac(k-110) + 190*dirac(k-109) + 210*dirac(k-108) +
                231*dirac(k-107) + 253*dirac(k-106) + 276*dirac(k-105) + 300*dirac(k-104) + 325*dirac(k-103) +
                351*dirac(k-102) + 378*dirac(k-101) + 406*dirac(k-100) + 435*dirac(k-99) + 465*dirac(k-98) +
                496*dirac(k-97) + 528*dirac(k-96) + 561*dirac(k-95) + 595*dirac(k-94) + 630*dirac(k-93) +
                666*dirac(k-92) + 703*dirac(k-91) + 741*dirac(k-90) + 780*dirac(k-89) + 820*dirac(k-88) +
                861*dirac(k-87) + 903*dirac(k-86) + 946*dirac(k-85) + 990*dirac(k-84) + 1035*dirac(k-83) +
                1081*dirac(k-82) + 1128*dirac(k-81) + 1176*dirac(k-80) + 1225*dirac(k-79) + 1275*dirac(k-78) +
                1326*dirac(k-77) + 1378*dirac(k-76) + 1431*dirac(k-75) + 1485*dirac(k-74) + 1540*dirac(k-73) +
                1596*dirac(k-72) + 1653*dirac(k-71) + 1711*dirac(k-70) + 1770*dirac(k-69) + 1830*dirac(k-68) +
                1891*dirac(k-67) + 1953*dirac(k-66) + 2016*dirac(k-65) + 2080*dirac(k-64) + 2145*dirac(k-63) +
                2211*dirac(k-62) + 2278*dirac(k-61) + 2346*dirac(k-60) + 2415*dirac(k-59) + 2485*dirac(k-58) +
                2556*dirac(k-57) + 2628*dirac(k-56) + 2701*dirac(k-55) + 2775*dirac(k-54) + 2850*dirac(k-53) +
                2926*dirac(k-52) + 3003*dirac(k-51) + 3081*dirac(k-50) + 3160*dirac(k-49) + 3240*dirac(k-48) +
                3321*dirac(k-47) + 3403*dirac(k-46) + 3486*dirac(k-45) + 3570*dirac(k-44) + 3655*dirac(k-43) +
                3741*dirac(k-42) + 3828*dirac(k-41) + 3916*dirac(k-40) + 4005*dirac(k-39) + 4095*dirac(k-38) +
                4186*dirac(k-37) + 4278*dirac(k-36) + 4371*dirac(k-35) + 4465*dirac(k-34) + 4560*dirac(k-33) +
                4656*dirac(k-32) + 4753*dirac(k-31) + 4851*dirac(k-30) + 4950*dirac(k-29) + 5050*dirac(k-28) +
                5151*dirac(k-27) + 5253*dirac(k-26) + 5356*dirac(k-25) + 5460*dirac(k-24) + 5565*dirac(k-23) +
                5671*dirac(k-22) + 5778*dirac(k-21) + 5886*dirac(k-20) + 5995*dirac(k-19) + 6105*dirac(k-18) +
                6216*dirac(k-17) + 6328*dirac(k-16) + 6441*dirac(k-15) + 6555*dirac(k-14) + 6670*dirac(k-13) +
                6786*dirac(k-12) + 6903*dirac(k-11) + 7021*dirac(k-10) + 7140*dirac(k-9) + 7260*dirac(k-8) +
                7381*dirac(k-7) + 7503*dirac(k-6) + 7626*dirac(k-5) + 7750*dirac(k-4) + 7875*dirac(k-3) +
                8001*dirac(k-2) + 8128*dirac(k-1) + 8256*dirac(k) + 8381*dirac(k+1) + 8503*dirac(k+2) +
                8622*dirac(k+3) + 8738*dirac(k+4) + 8851*dirac(k+5) + 8961*dirac(k+6) + 9068*dirac(k+7) +
                9172*dirac(k+8) + 9273*dirac(k+9) + 9371*dirac(k+10) + 9466*dirac(k+11) + 9558*dirac(k+12) +
                9647*dirac(k+13) + 9733*dirac(k+14) + 9816*dirac(k+15) + 9896*dirac(k+16) + 9973*dirac(k+17) +
                10047*dirac(k+18) + 10118*dirac(k+19) + 10186*dirac(k+20) + 10251*dirac(k+21) + 10313*dirac(k+22) +
                10372*dirac(k+23) + 10428*dirac(k+24) + 10481*dirac(k+25) + 10531*dirac(k+26) + 10578*dirac(k+27) +
                10622*dirac(k+28) + 10663*dirac(k+29) + 10701*dirac(k+30) + 10736*dirac(k+31) + 10768*dirac(k+32) +
                10797*dirac(k+33) + 10823*dirac(k+34) + 10846*dirac(k+35) + 10866*dirac(k+36) + 10883*dirac(k+37) +
                10897*dirac(k+38) + 10908*dirac(k+39) + 10916*dirac(k+40) + 10921*dirac(k+41) + 10923*dirac(k+42) +
                10922*dirac(k+43) + 10918*dirac(k+44) + 10911*dirac(k+45) + 10901*dirac(k+46) + 10888*dirac(k+47) +
                10872*dirac(k+48) + 10853*dirac(k+49) + 10831*dirac(k+50) + 10806*dirac(k+51) + 10778*dirac(k+52) +
                10747*dirac(k+53) + 10713*dirac(k+54) + 10676*dirac(k+55) + 10636*dirac(k+56) + 10593*dirac(k+57) +
                10547*dirac(k+58) + 10498*dirac(k+59) + 10446*dirac(k+60) + 10391*dirac(k+61) + 10333*dirac(k+62) +
                10272*dirac(k+63) + 10208*dirac(k+64) + 10141*dirac(k+65) + 10071*dirac(k+66) + 9998*dirac(k+67) +
                9922*dirac(k+68) + 9843*dirac(k+69) + 9761*dirac(k+70) + 9676*dirac(k+71) + 9588*dirac(k+72) +
                9497*dirac(k+73) + 9403*dirac(k+74) + 9306*dirac(k+75) + 9206*dirac(k+76) + 9103*dirac(k+77) +
                8997*dirac(k+78) + 8888*dirac(k+79) + 8776*dirac(k+80) + 8661*dirac(k+81) + 8543*dirac(k+82) +
                8422*dirac(k+83) + 8298*dirac(k+84) + 8171*dirac(k+85) + 8041*dirac(k+86) + 7908*dirac(k+87) +
                7772*dirac(k+88) + 7633*dirac(k+89) + 7491*dirac(k+90) + 7346*dirac(k+91) + 7198*dirac(k+92) +
                7047*dirac(k+93) + 6893*dirac(k+94) + 6736*dirac(k+95) + 6576*dirac(k+96) + 6413*dirac(k+97) +
                6247*dirac(k+98) + 6078*dirac(k+99) + 5906*dirac(k+100) + 5731*dirac(k+101) + 5553*dirac(k+102) +
                5372*dirac(k+103) + 5188*dirac(k+104) + 5001*dirac(k+105) + 4811*dirac(k+106) + 4618*dirac(k+107) +
                4422*dirac(k+108) + 4223*dirac(k+109) + 4021*dirac(k+110) + 3816*dirac(k+111) + 3608*dirac(k+112) +
                3397*dirac(k+113) + 3183*dirac(k+114) + 2966*dirac(k+115) + 2746*dirac(k+116) + 2523*dirac(k+117) +
                2297*dirac(k+118) + 2068*dirac(k+119) + 1836*dirac(k+120) + 1601*dirac(k+121) + 1363*dirac(k+122) +
                1122*dirac(k+123) + 878*dirac(k+124) + 631*dirac(k+125) + 381*dirac(k+126) + 128*dirac(k+127) )
            sub2= (-128*dirac(k+128) - 381*dirac(k+129) - 631*dirac(k+130) - 878*dirac(k+131) - 1122*dirac(k+132) -
                1363*dirac(k+133) - 1601*dirac(k+134) - 1836*dirac(k+135) - 2068*dirac(k+136) - 2297*dirac(k+137) -
                2523*dirac(k+138) - 2746*dirac(k+139) - 2966*dirac(k+140) - 3183*dirac(k+141) - 3397*dirac(k+142) -
                3608*dirac(k+143) - 3816*dirac(k+144) - 4021*dirac(k+145) - 4223*dirac(k+146) - 4422*dirac(k+147) -
                4618*dirac(k+148) - 4811*dirac(k+149) - 5001*dirac(k+150) - 5188*dirac(k+151) - 5372*dirac(k+152) -
                5553*dirac(k+153) - 5731*dirac(k+154) - 5906*dirac(k+155) - 6078*dirac(k+156) - 6247*dirac(k+157) -
                6413*dirac(k+158) - 6576*dirac(k+159) - 6736*dirac(k+160) - 6893*dirac(k+161) - 7047*dirac(k+162) -
                7198*dirac(k+163) - 7346*dirac(k+164) - 7491*dirac(k+165) - 7633*dirac(k+166) - 7772*dirac(k+167) -
                7908*dirac(k+168) - 8041*dirac(k+169) - 8171*dirac(k+170) - 8298*dirac(k+171) - 8422*dirac(k+172) -
                8543*dirac(k+173) - 8661*dirac(k+174) - 8776*dirac(k+175) - 8888*dirac(k+176) - 8997*dirac(k+177) -
                9103*dirac(k+178) - 9206*dirac(k+179) - 9306*dirac(k+180) - 9403*dirac(k+181) - 9497*dirac(k+182) -
                9588*dirac(k+183) - 9676*dirac(k+184) - 9761*dirac(k+185) - 9843*dirac(k+186) - 9922*dirac(k+187) -
                9998*dirac(k+188) - 10071*dirac(k+189) - 10141*dirac(k+190) - 10208*dirac(k+191) - 10272*dirac(k+192) -
                10333*dirac(k+193) - 10391*dirac(k+194) - 10446*dirac(k+195) - 10498*dirac(k+196) - 10547*dirac(k+197) -
                10593*dirac(k+198) - 10636*dirac(k+199) - 10676*dirac(k+200) - 10713*dirac(k+201) - 10747*dirac(k+202) -
                10778*dirac(k+203) - 10806*dirac(k+204) - 10831*dirac(k+205) - 10853*dirac(k+206) - 10872*dirac(k+207) -
                10888*dirac(k+208) - 10901*dirac(k+209) - 10911*dirac(k+210) - 10918*dirac(k+211) - 10922*dirac(k+212) -
                10923*dirac(k+213) - 10921*dirac(k+214) - 10916*dirac(k+215) - 10908*dirac(k+216) - 10897*dirac(k+217) -
                10883*dirac(k+218) - 10866*dirac(k+219) - 10846*dirac(k+220) - 10823*dirac(k+221) - 10797*dirac(k+222) -
                10768*dirac(k+223) - 10736*dirac(k+224) - 10701*dirac(k+225) - 10663*dirac(k+226) - 10622*dirac(k+227) -
                10578*dirac(k+228) - 10531*dirac(k+229) - 10481*dirac(k+230) - 10428*dirac(k+231) - 10372*dirac(k+232) -
                10313*dirac(k+233) - 10251*dirac(k+234) - 10186*dirac(k+235) - 10118*dirac(k+236) - 10047*dirac(k+237) -
                9973*dirac(k+238) - 9896*dirac(k+239) - 9816*dirac(k+240) - 9733*dirac(k+241) - 9647*dirac(k+242) -
                9558*dirac(k+243) - 9466*dirac(k+244) - 9371*dirac(k+245) - 9273*dirac(k+246) - 9172*dirac(k+247) -
                9068*dirac(k+248) - 8961*dirac(k+249) - 8851*dirac(k+250) - 8738*dirac(k+251) - 8622*dirac(k+252) -
                8503*dirac(k+253) - 8381*dirac(k+254) - 8256*dirac(k+255) - 8128*dirac(k+256) - 8001*dirac(k+257) -
                7875*dirac(k+258) - 7750*dirac(k+259) - 7626*dirac(k+260) - 7503*dirac(k+261) - 7381*dirac(k+262) -
                7260*dirac(k+263) - 7140*dirac(k+264) - 7021*dirac(k+265) - 6903*dirac(k+266) - 6786*dirac(k+267) -
                6670*dirac(k+268) - 6555*dirac(k+269) - 6441*dirac(k+270) - 6328*dirac(k+271) - 6216*dirac(k+272) -
                6105*dirac(k+273) - 5995*dirac(k+274) - 5886*dirac(k+275) - 5778*dirac(k+276) - 5671*dirac(k+277) -
                5565*dirac(k+278) - 5460*dirac(k+279) - 5356*dirac(k+280) - 5253*dirac(k+281) - 5151*dirac(k+282) -
                5050*dirac(k+283) - 4950*dirac(k+284) - 4851*dirac(k+285) - 4753*dirac(k+286) - 4656*dirac(k+287) -
                4560*dirac(k+288) - 4465*dirac(k+289) - 4371*dirac(k+290) - 4278*dirac(k+291) - 4186*dirac(k+292) -
                4095*dirac(k+293) - 4005*dirac(k+294) - 3916*dirac(k+295) - 3828*dirac(k+296) - 3741*dirac(k+297) -
                3655*dirac(k+298) - 3570*dirac(k+299) - 3486*dirac(k+300) - 3403*dirac(k+301) - 3321*dirac(k+302) -
                3240*dirac(k+303) - 3160*dirac(k+304) - 3081*dirac(k+305) - 3003*dirac(k+306) - 2926*dirac(k+307) -
                2850*dirac(k+308) - 2775*dirac(k+309) - 2701*dirac(k+310) - 2628*dirac(k+311) - 2556*dirac(k+312) -
                2485*dirac(k+313) - 2415*dirac(k+314) - 2346*dirac(k+315) - 2278*dirac(k+316) - 2211*dirac(k+317) -
                2145*dirac(k+318) - 2080*dirac(k+319) - 2016*dirac(k+320) - 1953*dirac(k+321) - 1891*dirac(k+322) -
                1830*dirac(k+323) - 1770*dirac(k+324) - 1711*dirac(k+325) - 1653*dirac(k+326) - 1596*dirac(k+327) -
                1540*dirac(k+328) - 1485*dirac(k+329) - 1431*dirac(k+330) - 1378*dirac(k+331) - 1326*dirac(k+332) -
                1275*dirac(k+333) - 1225*dirac(k+334) - 1176*dirac(k+335) - 1128*dirac(k+336) - 1081*dirac(k+337) -
                1035*dirac(k+338) - 990*dirac(k+339) - 946*dirac(k+340) - 903*dirac(k+341) - 861*dirac(k+342) -
                820*dirac(k+343) - 780*dirac(k+344) - 741*dirac(k+345) - 703*dirac(k+346) - 666*dirac(k+347) -
                630*dirac(k+348) - 595*dirac(k+349) - 561*dirac(k+350) - 528*dirac(k+351) - 496*dirac(k+352) -
                465*dirac(k+353) - 435*dirac(k+354) - 406*dirac(k+355) - 378*dirac(k+356) - 351*dirac(k+357) -
                325*dirac(k+358) - 300*dirac(k+359) - 276*dirac(k+360) - 253*dirac(k+361) - 231*dirac(k+362) -
                210*dirac(k+363) - 190*dirac(k+364) - 171*dirac(k+365) - 153*dirac(k+366) - 136*dirac(k+367) -
                120*dirac(k+368) - 105*dirac(k+369) - 91*dirac(k+370) - 78*dirac(k+371) - 66*dirac(k+372) -
                55*dirac(k+373) - 45*dirac(k+374) - 36*dirac(k+375) - 28*dirac(k+376) - 21*dirac(k+377) -
                15*dirac(k+378) - 10*dirac(k+379) - 6*dirac(k+380) - 3*dirac(k+381) - dirac(k+382))
            qj[k + abs(a)] = -1 / 1048576 * (sub1+sub2)


        delays = round(2**(j-1)) - 1
    
    return qj, delays, a, b, k_list

def filbank_ecg(ecg, qj_dict, delays):
    w2fb = np.zeros((9, len(ecg) + delays[8]))  # Matrik untuk hasil filter untuk setiap level
    
    for n in range(len(ecg)):
        for j in range(1, 9):
            w2fb[j][n + delays[j]] = 0  # Inisialisasi sinyal terfilter untuk level j
            a = -(round(2**j) + round(2**(j-1)) - 2)
            b = -(1 - round(2**(j-1)))
            
            # Menghitung hasil filter untuk level j
            for k in range(a, b + 1):
                w2fb[j][n + delays[j]] += qj_dict[j][k + abs(a)] * ecg[n - (k + abs(a))]
    
    return w2fb

def plot_grid(data1, title=None, data2=None, data3=None, data4=None, label1="Signal", label2="Thresholded", label3="Absolute", label4="MAV", jenis="ecg", kolom="2", timev=None):
    # Lakukan pemeriksaan jika data2, data3, atau data4 ada
    min_len = len(data1)
    yser = list(data1)
    
    for data, label in zip([data2, data3, data4], [label2, label3, label4]):
        if data is not None and len(data) > 0:
            min_len = min(min_len, len(data))
            yser.extend(data[:min_len])
    
    signal_types = [label1] * len(data1) + \
                   [label for data, label in zip([data2, data3, data4], [label2, label3, label4]) if data is not None and len(data) > 0 for _ in range(min_len)]

    time = np.arange(min_len) / fs
    freq = np.linspace(0, fs/2, min_len)
    if jenis == "ecg"or jenis=="ecg2":
        height = 200 if jenis=="ecg" else 300
        width = 1000 if jenis=="ecg" else 450
    
        df = pd.DataFrame({
            "Time (s)": np.tile(time, len(yser) // min_len),
            "Amplitude": yser, #np.concatenate([data1, data2]) if data2 is not None else data1,
            "Signal Type": signal_types
        })
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("Time (s)", title="Time (s)", axis=alt.Axis(grid=True)),
                y=alt.Y("Amplitude", title="Amplitude", axis=alt.Axis(grid=True), scale=alt.Scale(domain=[min(yser), max(yser)])),
                color="Signal Type:N"
            )
            .properties(width=width, height = height)
        )
    elif jenis == "freq":
        df = pd.DataFrame({
            "Freq (Hz)": np.tile(freq, len(yser) // min_len),
            "Magnitude": yser
        })
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("Freq (Hz)", title="Freq (Hz)", axis=alt.Axis(grid=True)),
                y=alt.Y("Magnitude", title="Magnitude", axis=alt.Axis(grid=True), scale=alt.Scale(domain=[min(yser), max(yser)]))
            )
            .properties(width=1000, height = 500)
        )
    
    elif jenis == "hrv":
        
        df = pd.DataFrame({
            "Sequence (s)": timev,
            "HR (bpm)": data1,
        })
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("Sequence (s)", title="Sequence (s)", axis=alt.Axis(grid=True)),
                y=alt.Y("HR (bpm)", title="HR (bpm)", axis=alt.Axis(grid=True,tickMinStep=0.01), scale=alt.Scale(domain=[min(data1)/2, 1.5*max(data1)]))
            )
            .properties(width=1000, height = 300)
        )
    
    if kolom == "1":    
        st.altair_chart(chart, use_container_width=True)

    elif kolom == "2":
        col1, col2 = st.columns([1, 8])
        with col1:
            st.markdown(f"### {title}")
        with col2:
            st.altair_chart(chart, use_container_width=True) 


def plot_bar_chart(df, title, x_label, y_label, width=500):
    chart = (
        alt.Chart(df)
        .mark_bar(color="blue")
        .encode(
            x=alt.X("n:O", title=x_label),
            y=alt.Y(df.columns[1], title=y_label)
        )
        .properties(width=500, height=300, title=title)
    )
    return chart

def thres(dat, threshold, delay):
    Out = np.zeros(len(dat) + delay[8])  # Buat array kosong (1D)
    absdat = abs(dat)
    dat = MAV(5,absdat)
    for n in range(len(dat)):
        for j in range(1, 9):
            if dat[n] > threshold:
                Out[n] = 1.5  # T3-T1
            else:
                Out[n] = 0

    return absdat, dat, Out   

def detect_rpeak(rpeak):
    RR = []
    for i in range(len(rpeak) - 1):
        if rpeak[i] == 1 and rpeak[i + 1] == 0:  # Transisi dari 1 ke 0
            RR.append(i / fs)  
    return RR

def compute_rr_intervals(RR):
    RR_intervals = []
    for i in range(len(RR) - 1):
        RR_intervals.append(RR[i + 1] - RR[i])  # Selisih antar R-peak
    return RR_intervals

def compute_hr(RR_intervals):
    HR = []
    for rr in RR_intervals:
        if rr > 0:
            HR.append(60 / rr)  # Konversi RR ke bpm
        else:
            HR.append(0)  # Jika ada kesalahan
    return HR


def main():
    st.sidebar.title("FP ASN Kelompok 2")
    st.sidebar.title("Non Stationary Signal 📚")
    selected_option = st.sidebar.selectbox("Choose an option", ["Mallat Algorithm Theory", "Filter Bank Theory", "HRV dan Resp Signal"])
    ecg = read_data('samples.txt')
    resp = read_data('samples.txt', type="resp")

    if selected_option == "Mallat Algorithm Theory":
        st.title("Mallat Algorithm")

        h, g, n_list = hg_list()
        Hw, Gw, i_list = hwgw_freq(h, g)

        #show h(n) g(n)
        df_h = pd.DataFrame({"n": n_list, "h(n)": h})
        df_g = pd.DataFrame({"n": n_list, "g(n)": g})
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("h(n)")
            st.altair_chart(plot_bar_chart(df_h, "h(n) Coefficients", "n", "Amplitude"))

        with col2:
            st.subheader("g(n)")
            st.altair_chart(plot_bar_chart(df_g, "g(n) Coefficients", "n", "Amplitude"))
        # Freq response Hw (LPF) & Gw (HPF)       
        i_list = i_list[:round(fs/2)]
        hHw = Hw[:round(fs/2)]
        hGw = Gw[:round(fs/2)]
        st.subheader("H(w) Frequency Response")
        plot_grid(hHw, "H(w) Frequency Response", jenis="freq", kolom="1")
        st.subheader("G(w) Frequency Response")
        plot_grid(hGw, "G(w) Frequency Response", jenis="freq", kolom="1")
        
        #Contoh penerapan di ECG
        st.title("Mallat Algorithm with ECG Data")
        st.subheader("Raw ECG Signal")
        plot_grid(ecg, "Raw ECG", kolom="1", label1="Raw ECG")

        #penerapan mallat
        w2fm, s2fm = mallat(ecg, h, g)

        # Plot hasil Mallat 
        st.subheader("Mallat Transform")
        for i in range(8):
            # Mengatur tampilan w2f dan s2f dalam satu baris
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader(f"w2f{i+1}")
                plot_grid(w2fm[i, :1250], "", jenis="ecg2", kolom="1", label1=f"w2f{i+1}")
                # st.line_chart(w2fm[i, :1250])  # Menampilkan w2f

            with col4:
                st.subheader(f"s2f{i+1}")
                plot_grid(s2fm[i, :1250], "", jenis="ecg2", kolom="1", label1=f"s2f{i+1}")
                # st.line_chart(s2fm[i, :1250])  # Menampilkan s2f


    elif selected_option == "Filter Bank Theory":
        st.title("Filter Bank")
        # Freq Response
        h, g, n_list = hg_list()
        Hw, Gw, i_list = hwgw_freq(h, g)
        Q = fr_filbank(Gw, Hw, fs)
        st.subheader("Frequency Response")
        checkbox_values = []
        cols = st.columns(8)  
        for i in range(1, 9):
            with cols[i-1]:
                checkbox_values.append(st.checkbox(f"Q{i}", value=True))

        if any(checkbox_values):
            selected_Q = []
            labels = []
            for i in range(1, 9):
                if checkbox_values[i-1]:
                    selected_Q.append(Q[i])
                    labels.append(f"Q{i}")

            freq = np.linspace(0, round(fs/2), len(Q[0])) 
            df = pd.DataFrame(np.array(selected_Q).T, columns=labels)
            df.index = freq
            df.index.name = "Frequency (Hz)"

            st.line_chart(df)
        else:
            st.write("No frequency response selected.")
        st.write("**X-Axis:** Frequency (Hz)")
        st.write("**Y-Axis:** Magnitude")

        qj = {}
        delays = {}  
        for j in range(1, 9):
            qj_dict, delay, a, b, k_list = compute_qj(j)
            qj[j] = qj_dict
            delays[j] = delay
            st.subheader(f"Level {j}")
            st.write(f"a = {a}, b = {b} ")
            df = pd.DataFrame({
                "n": k_list,  # Use k_list values for the x-axis
                "Amplitude": qj[j][:len(k_list)]  # Use your existing amplitude values
            })
            st.altair_chart(plot_bar_chart(df, " ", "n", "Amplitude", width=1000))


        T1 = delays[1]
        T2 = delays[2]
        T3 = delays[3]
        T4 = delays[4]
        T5 = delays[5]
        T6 = delays[6]
        T7 = delays[7]
        T8 = delays[8]

        st.write(f"T1 = {T1}, T2 = {T2}, T3 = {T3}, T4 = {T4}, T5 = {T5}, T6 = {T6}, T7 = {T7}, T8 = {T8}")
        
        st.title("Filter Bank with ECG Data")
        plot_grid(ecg, "ECG Raw", label1="ECG Raw")
        w2fb = filbank_ecg(ecg, qj, delays)

        # Plot the processed signals
        st.subheader("Filtered ECG Signal")
        for i in range(1, 9):
            plot_grid(w2fb[i][:1250], f"Scale {i}", label1=f"Scale {i}")
            
    
    elif selected_option == "HRV dan Resp Signal": 
        st.title("Wavelet Plot HRV dan Respiratory Signal")
        # st.subheader("Uploaded .txt File")
        # uploaded_file = st.file_uploader("Upload .txt File", type=["txt"])
        # if uploaded_file is not None:
        #     ecg2 = read_ecg(uploaded_file)
        #     st.write(f"ECG data successfully loaded with {len(ecg2)} data points.")
        
        ecg2 = list(ecg-0.4)  # Konversi numpy array ke list
        for _ in range(2 * fs): #padding
            ecg2.append(0) 
        ecg2 = np.array(ecg2)
        
        qj = {}
        delays = {}  
        for j in range(1, 9):
            qj_dict, delay, a, b, k_list = compute_qj(j)
            qj[j] = qj_dict
            delays[j] = delay

        T1 = delays[1]
        T2 = delays[2]
        T3 = delays[3]
        T4 = delays[4]
        T5 = delays[5]
        T6 = delays[6]
        T7 = delays[7]
        T8 = delays[8]

        st.title("Filter Bank for ECG")
        plot_grid(ecg2, "Raw ECG", label1="Raw ECG")
        w2fb = filbank_ecg(ecg, qj, delays)
        for i in range(1, 9):
            plot_grid(w2fb[i], f"Scale {i}", label1= f"DWT Scale {i}")
        
        st.subheader("Threshold")
        absdat = {}
        mav = {}
        thresout = {}
        plot_grid(ecg2, "Raw ECG")
        checkbox_data2 = st.checkbox("Show Threshold", value=True)
        checkbox_data3 = st.checkbox("Show Absolute", value=True)
        checkbox_data4 = st.checkbox("Show MAV", value=True)
        for i in range(1, 9):
            # Tentukan threshold berdasarkan paper
            if i == 1:  # Skala 2^1 - 2^3
                th = 0.18
            elif i == 2:
                th = 0.13
            elif i == 3:
                th = 0.09
            elif i == 4:
                th = 0.3
            elif i == 5:
                th = 0.7
            elif i == 6:  # Skala 2^4 - 2^6
                th = 0.05 #0.5
            elif i == 7:  # Skala 2^7
                th = 0.025 #0.4
            else:  # Skala 2^8
                th = 0 #0.1
            absdat[i], mav[i], thresout[i] = thres(w2fb[i], th, delays)

            # Plot dua sinyal di satu grafik
            # plot_grid(w2fb[i], f"Thresholded Scale {i}", data2=thresout[i], label1=f"ECG DWT Scale {i}", label2="Threshold")
            plot_data2 = thresout[i] if checkbox_data2 else None
            plot_data3 = absdat[i] if checkbox_data3 else None
            plot_data4 = mav[i] if checkbox_data4 else None

            plot_grid(
                w2fb[i],  # Data pertama
                f"Threshold Scale {i}",  # Judul plot
                data2=plot_data2,  
                data3=plot_data3,  
                data4=plot_data4,
                label1=f"ECG DWT Scale {i}",  # Label pertama
                label2="Threshold",  # Label kedua
                label3="Absolute",  # Label ketiga
                label4="MAV",  # Label keempat
            )
        
        st.subheader("R peak")
        st.write("**R-peak detection using ECG DWT scale 1-3**")
        rpeak = np.zeros_like(thresout[1])  # Sama panjang dengan thresout[1]

        # Loop setiap titik data
        for n in range(len(rpeak)):  
            if (thresout[1][n] == 1.5 and 
                thresout[2][n] == 1.5 and 
                thresout[3][n] == 1.5):
                # thresout[4][n] == 1.5 and 
                # thresout[5][n] == 1.5):
                rpeak[n-3] = 1 #T3-T1
            else:
                rpeak[n-3] = 0

        # Plot hasilnya
        plot_grid(ecg2, "R-Peak Detection", data2=rpeak, label1="Raw ECG", label2="R-Peak")

        RR = detect_rpeak(rpeak)  # Ambil waktu deteksi R-peak
        RR_intervals = compute_rr_intervals(RR)  # Hitung RR Interval
        HR = compute_hr(RR_intervals)  # Hitung HR dalam bpm
        meanRR = sum(RR_intervals)/len(RR_intervals)
        fs_HRV = 1 / meanRR if RR_intervals else 1  # Sampling rate HRV
        sequence_time = np.arange(len(HR)) / fs_HRV  # Waktu sequence (detik)
        st.write("**RR (s):**", RR)
        st.write("**RR Intervals (s):**", RR_intervals)
        st.write("**HR (Bpm):**", HR)
        st.write("**Mean RR Intervals (s):**", meanRR)
        st.write("**fs HRV:**", fs_HRV)
        st.write("**Sequence(s):**", sequence_time)
        st.subheader("Heart Rate Variability (HRV)")
        plot_grid(HR, "HRV Plot", jenis="hrv", kolom="1", timev=sequence_time)

        st.subheader("Respiratory Signal Using DWT Scale 8")
        respi = np.zeros(len(ecg))
        respi2 = np.zeros(len(ecg))
        for i in range (T8,128*10):
            respi[i-T8]=w2fb[8][i]
            respi2[i-T8]= respi[i-T8]*20

        # plot_grid(w2fb[8], "Orde 8", data2=resp)
        plot_grid(respi, "ECG-Derived Respiratory (Scale 8)", label1="DWT Scale 8")
        plot_grid(resp, "Respiratory Signal", label1="Resp Signal")
        st.subheader("Comparing Signal")
        st.write("Note: The amplitude of DWT scale 8 is magnified 20 times.")
        plot_grid(resp, label1="Resp Signal", data2=respi2, label2="DWT Scale 8", kolom="1")
        



if __name__ == "__main__":
    main()
