#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt #, mpld3
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import time
from dateutil import parser
from datetime import datetime
import pandas as pd
import numpy as np
# import datetime
import math
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import config as cfg
#import seaborn
from matplotlib.font_manager import FontProperties
#from PointBrowser import PointBrowser
import config as cfg
from matplotlib import style
import sys
import subprocess
import NN_config as cfg

style.use('seaborn-paper')


def getMinMax(df_):
    global dmaxVal, dminVal
    minimums = []
    minimums.append(min( df_['accuracy'] )) 
    minimums.append(min( df_['loss'] )) 
    maximums = []
    maximums.append(max( df_['accuracy'] ))
    maximums.append(max( df_['loss'] ))
    return max(maximums), min(minimums)


# init everything first
fig = plt.figure(figsize=(7, 4))
fig.fontsize = 9
latest = 400 #1440 #int(cfg.latest)

states_path = cfg.checkpoint # 'R:/temp/' 
sw = True
f=0
## ---------- wait for data file to fill ------------
while (f == 0):
    try:
        df = pd.read_csv(states_path + cfg.log_file, sep=",", header=0, encoding="utf8", low_memory=False) #, parse_dates=True)
        if (len(df.batch) > 3): 
            f = 1
        else:
            print ("waiting for data...", end="\r")          
    except:     
        pass
    time.sleep(2)
    
lastEpisode = int(df['batch'].max()) #int(df['epoch'].iloc[-1])
lastRun = int(df['batch'].max())
dmaxVal, dminVal = getMinMax(df)
# ax1.xaxis_date()
print ("dmaxVal, dminVal:", dmaxVal,",", dminVal)
zoomX1, zoomX2, zoomY1, zoomY2 = -100, lastRun + 10, dminVal, dmaxVal    # specify the limits
xview = lastEpisode - df['batch'].iloc[-2]
lastEpisode_old = lastEpisode
lastRun_old = lastRun
qlive = []

# Declare and register callbacks
def on_xlims_change(axes):
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, sw, xview
    zoomX1, zoomX2 = ax1.get_xlim()
    #xview = float()
    sw = True
    print ("updated xlims: ", zoomX1, zoomX2)


def on_ylims_change(axes):
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, sw
    zoomY1, zoomY2 = ax1.get_ylim()
    sw = True
    print ("updated ylims: ", zoomY1, zoomY2)


def resetView():
    global ax1, zoomX1, zoomX2, zoomY1, zoomY2, plt, sw, lastEpisode, dmaxVal, dminVal, df, states_path
    df = pd.read_csv(states_path + cfg.log_file, sep=",", header=0, encoding="utf8", low_memory=False) # , parse_dates=True)
    dmaxVal, dminVal = getMinMax(df)
    zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastRun + 10, dminVal, dmaxVal  # specify the limits
    sw = False
    plt.xlim(zoomX1, zoomX2)
    plt.ylim(zoomY1, zoomY2)
    plt.show()

def to_datetime_object(date_string, date_f):
    s = datetime.strptime(date_string, date_f)
    return s

def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == ' ':
        resetView()


def animate(i):
    global dminVal, lastRun_old, lastRun, dmaxVal, qlive, ax1, zoomX1, zoomX2, zoomY1, zoomY2, plt, df, sw, xview, lastEpisode_old, states_path, lastEpisode
    print(df.columns)
    # ax1 = fig.add_subplot(1,1,1)
    # ax1.xaxis_date()
    #plt.savefig('test.pdf')
    # ax1.set_xlim(xlim)
    # ax1.set_ylim(ylim)
    df = pd.read_csv(states_path + cfg.log_file, sep=",", header=0, encoding="utf8", low_memory=False) #, parse_dates=True)

    lastEpisode = int(df['batch'].max()) #int(df['epoch'].iloc[-1])
    
    lastRun = int(df['batch'].max())
    dmaxVal, dminVal = getMinMax(df)
    zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastRun + 10, dminVal, dmaxVal  # specify the limits

    xview = int(lastEpisode - df['batch'].iloc[-2])   
    print (xview)   

    latest = len(df.batch) # to shrink the diagramm (not used)
    #reduce to latest factor
    if len(df.batch) > int(latest):        # Drop 

        ld = len(df.batch) - int(latest)
        df.drop(df.batch[:ld], inplace=True)

    else:                           # do not drop (log < 24h = 1440)
        print('nothing to drop')
        print('\r\n')

    fig.clear()
    ax1 = fig.add_subplot(1,1,1)

    ax1.set_xlabel("batch")    #x-achse set
    ax1.set_ylabel("loss")  #y-achse set
    ax1.yaxis.grid(color='gray', linestyle='dashed') # backgr. grid
    ax1.xaxis.grid(color='gray', linestyle='dashed') # backgr. grid
    #ax1.autoscale_view()
    ax1.set_facecolor((0.8, 0.8, 0.8))   # backgr. color grey
    if not (lastRun_old == lastRun):   
        plt.xlim(zoomX1, zoomX2)
        plt.ylim(zoomY1, zoomY2)
    else:
        plt.xlim(zoomX1, zoomX2)
        plt.ylim(zoomY1, zoomY2)

    ax1.set_title('lstm training state  @trained batch '+ str(int(np.max(df.batch)+2)) )
    for label in ax1.xaxis.get_ticklabels(): label.set_rotation(20)  #x-achse rotate annot.

    

    plt.tight_layout()


    print ("lastEpisode: ", lastEpisode)

    if (sw==False):        #  no zoomt or movement = reset
        if not (lastRun_old == lastRun): 
            dmaxVal, dminVal = getMinMax(df)   
            zoomX1, zoomX2, zoomY1, zoomY2 = -10, lastRun + 10, dminVal, dmaxVal  # specify the limits  
            lastRun_old = lastRun
            #xview = int(df['epoch'].iloc[-1]) - int(df['epoch'].iloc[-2]) 
            plt.xlim(int(zoomX1), int(zoomX2))
            plt.ylim(zoomY1, zoomY2)
            #zoomX1, zoomX2 = ax1.get_xlim()
            #plt.show()
        else:
            dmaxVal, dminVal = getMinMax(df)   
            plt.xlim(zoomX1, zoomX2)
            plt.ylim(dminVal, dmaxVal)

    elif (sw==True):      # with zoom or movement
        if not (lastRun_old == lastRun):              
            lastRun_old = lastRun
            xview = int(df['batch'].iloc[-1]) - int(df['batch'].iloc[-2]) 
            plt.xlim(int(zoomX1), int(zoomX2+xview))
            plt.ylim(zoomY1, zoomY2)
            zoomX1, zoomX2 = ax1.get_xlim()
            #plt.show()
        else:
            plt.xlim(zoomX1, zoomX2)
            plt.ylim(zoomY1, zoomY2)
    
    plot1, = plt.plot(df.index, df['loss'], label="loss.", marker="", picker=3, linewidth=1)
    plot2, = plt.plot(df.index, df['accuracy'], label="acc.", marker="", picker=3, linewidth=1)
    
    #plot6, = plt.plot(cfg.getQ(), label="run", marker="+", picker=5, linewidth=.8)
    #plot7, = plt.plot(df.stateTime, df['cfg.maxVTemp'], label="maxV", marker="x", linestyle='dashed', picker=3)
    #plot8, = plt.plot(df.stateTime, df['fehler'], label="F1", marker="o", picker=5)

    #ax2 = ax1.twinx()
    #ax2.set_ylim([-30,30])

    plt.legend([plot1,plot2],['loss.', 'acc.'], loc="upper left",  bbox_transform=fig.transFigure)
    #fig.savefig('test.pdf')

    #ax2.plot(df.time, df['fehler'], label="F1", marker="o", color='C7', picker=5)
    #ax2.set_ylabel('fehler', color='C7', font="Arial")
    #ax2.tick_params('f', color='C7', font="Arial")
    
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax1.callbacks.connect('ylim_changed', on_ylims_change)
    fig.canvas.mpl_connect('key_press_event', press)

    

ani = animation.FuncAnimation(fig, animate, interval=1000)
#fig.canvas.mpl_connect('button_press_event', onpress)
plt.show()