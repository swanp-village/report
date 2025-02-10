import matplotlib.pyplot as plt
import numpy as np
import h5py
from MRR.transfer_function import simulate_transfer_function
import csv
from generate_figure import generate_figure

def graph_cretate(axis,dataset,nameset): #datasetは配列　axisはx軸　namesetは配列で凡例用の名前　Lとsは実験用なので削除予定　一枚表示or二枚重ねる用の関数です
    if len(dataset) == 1:
        _onegraph_create(axis,dataset[0],nameset[0])
        _onetopgraph_create(axis,dataset[0],nameset[0])
    elif len(dataset) == 2:
        _twograph_create(axis,dataset[0],dataset[1],nameset[0],nameset[1])
        _twotopgraph_create(axis,dataset[0],dataset[1],nameset[0],nameset[1])


def _onegraph_create(axis,data1,name1):     #一つのグラフのみ表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    fig=plt.plot(axis,data1,label=name1)        #プロット
    plt.xlabel("Wavelength (nm)",fontsize=13)       #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-30,0)                                 #y軸表示範囲-60~0dB
    plt.xticks(np.arange(1540,1565,5))              #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()      #グラフ表示
    plt.savefig("M_2.csv")
    plt.close
    with open("M_2.csv",'w',newline='')as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        for i in range(len(axis)):
            writer.writerow([axis[i], data1[i]])
    
def _onetopgraph_create(axis,data1,name1):  #一つのグラフをトップの部分を拡大して表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)        #プロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-6,0)                                 #y軸表示範囲-6~0dB
    plt.xlim(1549,1551)                            #x軸表示範囲1549nm~1551nm
    plt.xticks(np.arange(1549,1551.25,0.5))         #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()                                          #グラフ表示

def _twograph_create(axis,data1,data2,name1,name2):     #二つのグラフを重ねて表示
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)        #一つ目のグラフをプロット
    plt.plot(axis,data2,label=name2,linestyle = "dashed")        #二つ目のグラフをプロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)  #y軸ラベル
    plt.ylim(-40,0)                                 #y軸表示範囲-60~0dB
    plt.xticks(np.arange(1540,1565,5))              #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()              #グラフ表示

def _twotopgraph_create(axis,data1,data2,name1,name2):
    plt.rcParams["xtick.direction"]="in"    #目盛り内向き
    plt.rcParams["ytick.direction"]="in"    #目盛り内向き
    plt.plot(axis,data1,label=name1)       #一つ目のグラフをプロット
    plt.plot(axis,data2,label=name2,linestyle = "dashed")       #二つ目のグラフをプロット
    plt.xlabel("Wavelength (nm)",fontsize = 13)     #x軸ラベル
    plt.ylabel("Transmittance (dB)",fontsize = 13)     #y軸ラベル
    plt.ylim(-6,0)                                 #y軸表示範囲-6~0dB
    plt.xlim(1549,1551)                            #x軸表示範囲1549nm~1551nm
    plt.xticks(np.arange(1549,1551.25,0.5))         #x軸目盛りの表示位置を設定
    plt.minorticks_on()                             #補助目盛オン
    plt.legend(bbox_to_anchor=(1,1),loc="upper right")  #凡例を右上に表示　locだけでなく位置を変えたいならanchorも変える必要ありなため、調べてください
    plt.show()              #グラフ表示



K1=np.array([
        0.3804785998770388,
        0.04311135245013454,
        0.027324823681336888,
        0.040581224489887505,
        0.06934857655822832,
        0.07267724646180711,
        0.41197244177117703
    ])
L1=np.array([
        5.495454545454545e-05,
        5.495454545454545e-05,
        5.495454545454545e-05,
        8.243181818181816e-05,
        8.243181818181816e-05,
        8.243181818181816e-05
    ])


data1=simulate_transfer_function(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})

axis = np.arange(1540e-9,1560e-9,0.01e-9)
xaxis=np.arange(1540,1560.01,0.01)
data_1=data1.simulate(axis)
_onegraph_create(axis,data1,"M=1")
    

