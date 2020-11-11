#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:27:26 2019

@author: adityamate, killian-34
"""
import numpy as np
from scipy import optimize
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
import glob
import time
from compute_exact_yundi import *
from compute_exact_yundi_infinite import yundi_whittle_exact as yundi_beta 
from adherence_simulation import *

   
def plotBeliefTrajectory(p11, p01, p11active, p01active, read=False, 
                         belief_plots=True, whittle_plots=True, scatter_plots=True, 
                         days=15, scatter_sv=''):
    
    #days=15
    L=days
    
    p_pass01=p01
    p_pass11=p11
    p_act11=p11active
    p_act01=p01active
    
    '''
    BELIEF TRAJECTORIES
    '''
    ba, bna= precomputeBelief(p11, p01, p11active, p01active)
    w0= p01/(p01 + (1.-p11))
    
    x=[i for i in range(L)]
    expectedbeliefA=[ba[i]*ba[0]+(1-ba[i])*bna[0] for i in range(L)]
    expectedbeliefNA=[bna[i]*ba[0]+(1-bna[i])*bna[0] for i in range(L)]
    jumpA=[expectedbeliefA[i]-ba[i] for i in range(L)]
    jumpNA=[expectedbeliefNA[i]-bna[i] for i in range(L)]

    '''
    WHITTLE TRAJECTORIES
    '''
    
    print ("p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
    Tpass= np.array([[1-p_pass01,p_pass01],[1.-p_pass11,p_pass11]])
    Tact=np.array([[1.-p_act01,p_act01],[1.-p_act11,p_act11]])

    #w1, w2, w3, w4, yundi_w1, yundi_w2 =[], [],[],[],[],[]
    if not read:
        w1,w2=whittleIndex([Tpass,Tact], L)
        w3, w4= newWhittle([Tpass,Tact], L=180)
        print ("computing new_whittle")
        w5, w6= newnewWhittle([Tpass,Tact], L=180)
        print("computing yundi")
        yundi_w1= [yundi_whittle_exact([Tpass,Tact], ba[i], 180-i) for i in range(days)]
        yundi_w2= [yundi_whittle_exact([Tpass,Tact], bna[i], 180-i) for i in range(days)]
        
        np.save('whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w1, w2]))
        np.save('whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w3, w4]))
        np.save('whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([yundi_w1, yundi_w2]))
        np.save('whittleValues/newnew_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w3, w4]))

    else:
        
        generation_necessary=False
        if len(glob.glob('./whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:
            w1,w2=np.load(glob.glob('./whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
        else:
            generation_necessary=True
        if len(glob.glob('./whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:            
            w3,w4=np.load(glob.glob('./whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
        else:
            generation_necessary=True
        if len(glob.glob('./whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:
            yundi_w1,yundi_w2=np.load(glob.glob('./whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
        else:
            generation_necessary=True
        if generation_necessary:
            #plotBeliefTrajectory(p11, p01, p11active, p01active, read=False, belief_plots=belief_plots,
                                 #whittle_plots=whittle_plots, scatter_plots=scatter_plots)
            print ("Value absent:", "p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
            return 
        w5, w6= newnewWhittle([Tpass,Tact], L=180)
    


    print ("W1:", w1[:days])
    print("W2:", w2[:days])
    print ("W3:", w3[:days])
    print("W4:", w4[:days])
    print ("W5:", w5[:days])
    print("W6:", w6[:days])
    
                   

    x=[i for i in range(days)]
    print ("This", w1[:len(x)])
    
    '''
    Belief plots
    '''
    plt.plot(x, [w0 for i in range(L)], 'b--', label='stationary')
    plt.plot(x, ba[:days], 'r-', label='starting from A')
    plt.plot(x, bna[:days], 'g-', label='starting from NA')
    
    plt.plot(x, jumpA[:days], 'm-', label='jump from A ')
    plt.plot(x, jumpNA[:days], 'y-',label='jump from NA')
    

    plt.plot(x, expectedbeliefA[:days], 'r--', label='E[next belief] for A')
    plt.plot(x, expectedbeliefNA[:days], 'g--', label='E[next belief] for NA')
    
    plt.xlabel ("Unobserved days")
    plt.ylabel ('probability of A = belief')
    
    plt.title("p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p11, p01, p11active, p01active))
    #plt.legend()
    #plt.savefig('./beliefTrajectories/'+"p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p11, p01, p11active, p01active)+'.png')
    plt.show()

    '''
    Whittle index plots
    '''
    
    ba, bna= precomputeBelief(p_pass11,p_pass01,p_act11,p_act01)
    plt.plot(x[:days], [round(w, 5) for w in w1[:len(x)]], 'r--', label='Old A chain')
    plt.plot(x[:days], [round(w, 5) for w in w2[:len(x)]], 'g--', label='Old NA chain')
    plt.plot(x[:days], [round(w, 5) for w in yundi_w1[:len(x)]], 'r-', label='Yundi A chain')
    plt.plot(x[:days], [round(w, 5) for w in yundi_w2[:len(x)]], 'g-', label='Yundi NA chain')
                    
    plt.xlabel ("Unobserved days")
    plt.ylabel ('Whittle index')
    
    plt.title("old_p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
    plt.legend()
    #plt.savefig('./whittleTrajectories/'+"old_p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01)+'.png')
    plt.show()
    
    
    plt.plot(x, [round(w, 5) for w in w3[:len(x)]], 'r-', label='New A chain')
    plt.plot(x, [round(w, 5) for w in w4[:len(x)]], 'g-', label='New NA chain')
    plt.plot(x, [round(w, 5) for w in w5[:len(x)]], 'ro-', label='NewNew A chain')
    plt.plot(x, [round(w, 5) for w in w6[:len(x)]], 'go-', label='NewNew NA chain')
    plt.xlabel ("Unobserved days")
    plt.ylabel ('Whittle index')
    
    plt.title("new_p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
    plt.legend()
    #plt.savefig('./whittleTrajectories/'+"new_p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01)+'.png')
    plt.show()
    
    '''
    whittle vs belief scatter plots
    '''
    fig, ax=plt.subplots(2,3, figsize=(12,12))
    fig.suptitle("scatterPlot_p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01), fontsize=16)
    
    ax[0][0].scatter(ba[:days], yundi_w1, label='Yundi A chain')
    ax[0][0].scatter(bna[:days], yundi_w2, label='Yundi NA chain')
    
    ax[0][1].scatter(ba[:days], w1[:days], label='Threshold A chain')
    ax[0][1].scatter(bna[:days], w2[:days], label='Threshold NA chain')
    ax[0][2].scatter(ba[:days], w5[:days], label='NewNewThreshold A chain')
    ax[0][2].scatter(bna[:days], w6[:days], label='NewNewThreshold NA chain')
    
    ax[1][0].plot(x, [w0 for i in range(L)], 'b--', label='stationary')
    ax[1][0].plot(x, ba[:days], 'r-', label='starting from A')
    ax[1][0].plot(x, bna[:days], 'g-', label='starting from NA')
    
    ax[1][1].plot(x, jumpA[:days], 'm-', label='jump from A ')
    ax[1][1].plot(x, jumpNA[:days], 'y-',label='jump from NA')
    ax[0][0].set(xlabel='Belief', ylabel='Yundi Index value')


    ax[1][0].plot(x, expectedbeliefA[:days], 'r--', label='E[next belief] for A')
    ax[1][0].plot(x, expectedbeliefNA[:days], 'g--', label='E[next belief] for NA')
    ax[1][0].set(xlabel='Unobserved days', ylabel='Belief')
    
    ax[0][0].legend()
    ax[0][1].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    
    
    plt.savefig('./whittleTrajectories/scatter/'+scatter_sv+'_p11:%s, p01:%s, p_act11:%s, p_act01:%s.png'%(p_pass11, p_pass01, p_act11, p_act01))
    
    plt.figure()
    plt.scatter(ba[:days], yundi_w1, label='Yundi A chain')
    plt.scatter(bna[:days], yundi_w2, label='Yundi NA chain')
    plt.title('Yundi')
    
    plt.figure()
    plt.scatter(ba[:days], w1[:days], label='Threshold A chain')
    plt.scatter(bna[:days], w2[:days], label='Threshold NA chain')
    plt.title('Threshold')
    plt.legend()
    plt.show()
    
    return 

def plotWhittleTrendReverseThreshold(T, savename, color, read=False, 
                         belief_plots=True, whittle_plots=True, scatter_plots=True, 
                         days=15, scatter_sv='',alpha=0.5):
    
    #days=15
    L=180
    
    print("Len T", len(T))
    fig, ax=plt.subplots(5,2, figsize=(6,9))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    
    for ind,T_sub in enumerate(T):

        p11, p01 = T_sub[0][1][1], T_sub[0][0][1]
        p11active, p01active = T_sub[1][1][1], T_sub[1][0][1]


        p_pass01=p01
        p_pass11=p11
        p_act11=p11active
        p_act01=p01active
        
        '''
        BELIEF TRAJECTORIES
        '''
        ba, bna= precomputeBelief(p11, p01, p11active, p01active)
        w0= p01/(p01 + (1.-p11))
        
        x=[i for i in range(L)]
        expectedbeliefA=[ba[i]*ba[0]+(1-ba[i])*bna[0] for i in range(L)]
        expectedbeliefNA=[bna[i]*ba[0]+(1-bna[i])*bna[0] for i in range(L)]
        jumpA=[expectedbeliefA[i]-ba[i] for i in range(L)]
        jumpNA=[expectedbeliefNA[i]-bna[i] for i in range(L)]

        '''
        WHITTLE TRAJECTORIES
        '''
        
        print ("p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
        Tpass= np.array([[1-p_pass01,p_pass01],[1.-p_pass11,p_pass11]])
        Tact=np.array([[1.-p_act01,p_act01],[1.-p_act11,p_act11]])

        #w1, w2, w3, w4, yundi_w1, yundi_w2 =[], [],[],[],[],[]
        if not read:
            w1,w2=whittleIndex([Tpass,Tact], L)
            # w3, w4= newWhittle([Tpass,Tact], L=180)
            # print ("computing new_whittle")
            # w5, w6= newnewWhittle([Tpass,Tact], L=180)
            # print("computing yundi")

            # yundi_w1= [yundi_whittle_exact([Tpass,Tact], ba[i], 180-i) for i in range(days)]
            # yundi_w2= [yundi_whittle_exact([Tpass,Tact], bna[i], 180-i) for i in range(days)]
            
            # np.save('whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w1, w2]))
            # np.save('whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w3, w4]))
            # np.save('whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([yundi_w1, yundi_w2]))
            # np.save('whittleValues/newnew_p11:%s, p01:%s, p_act11:%s, p_act01:%s'%(p_pass11, p_pass01, p_act11, p_act01), np.array([w3, w4]))

        else:
            
            generation_necessary=False
            if len(glob.glob('./whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:
                w1,w2=np.load(glob.glob('./whittleValues/old_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
            else:
                generation_necessary=True
            if len(glob.glob('./whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:            
                w3,w4=np.load(glob.glob('./whittleValues/new_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
            else:
                generation_necessary=True
            if len(glob.glob('./whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01)))>0:
                yundi_w1,yundi_w2=np.load(glob.glob('./whittleValues/yundi_p11:%s, p01:%s, p_act11:%s, p_act01:%s.npy'%(p_pass11, p_pass01, p_act11, p_act01))[0])
            else:
                generation_necessary=True
            if generation_necessary:
                #plotBeliefTrajectory(p11, p01, p11active, p01active, read=False, belief_plots=belief_plots,
                                     #whittle_plots=whittle_plots, scatter_plots=scatter_plots)
                print ("Value absent:", "p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p_pass11, p_pass01, p_act11, p_act01))
                return 
            w5, w6= newnewWhittle([Tpass,Tact], L=180)
        
        '''
        Whittle index plots
        '''



        b_combined = np.concatenate([ba[:-1], bna[:-1]],axis=0)
        w_combined = np.concatenate([w1[:-1], w2[:-1]],axis=0)
        
        sort_ind = np.argsort(b_combined)

        b_sorted = b_combined[sort_ind]
        w_sorted= w_combined[sort_ind]



        # plt.plot(b_sorted, w_sorted, color, alpha=alpha)
        ax[ind%5,ind//5].plot(b_sorted, w_sorted, color=color)
        ax[ind%5,ind//5].set_yticks([w_sorted.min(),w_sorted.max()])
        ax[ind%5,ind//5].set_xticks([b_sorted.min(),b_sorted.max()])
        ax[ind%5,ind//5].set_yticklabels([round(w_sorted.min(),1),round(w_sorted.max(),1)])
        ax[ind%5,ind//5].set_xticklabels([round(b_sorted.min(),1),round(b_sorted.max(),1)])
 
    plt.xlabel("Belief")
    plt.ylabel('Whittle index')
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(savename)
    plt.show()
    
    



def plotBeliefTrajectorySimple(p11, p01, p11active, p01active, L=15):
    
    
    p_pass01=p01
    p_pass11=p11
    p_act11=p11active
    p_act01=p01active
    
    '''
    BELIEF TRAJECTORIES
    '''
    ba, bna= precomputeBelief(p11, p01, p11active, p01active)
    w0= p01/(p01 + (1.-p11))
    
    x=[i for i in range(L)]

    
    '''
    Belief plots
    '''
    plt.plot(x, [w0 for i in range(L)], 'b--', label='stationary')
    plt.plot(x, ba[:L], 'r-', label='starting from A')
    plt.plot(x, bna[:L], 'g-', label='starting from NA')
    

    
    plt.xlabel ("Unobserved days")
    plt.ylabel ('probability of A = belief')
    
    plt.title("p11:%.2f, p01:%.2f, p_act11:%.2f, p_act01:%.2f"%(p11, p01, p11active, p01active))
    #plt.legend()
    #plt.savefig('./beliefTrajectories/'+"p11:%s, p01:%s, p_act11:%s, p_act01:%s"%(p11, p01, p11active, p01active)+'.png')


    
    return plt



def precomputeBelief(p11, p01, p11active, p01active,  L=180):
    #precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1])
    bA=np.zeros(L)
    bNA=np.zeros(L)
    
    bA[-1]=1.
    bNA[-1]=0.
    
    for t in range(L):  
        if t==0:
            bA[t]= bA[t-1]*p11active + (1.- bA[t-1])*p01active
            bNA[t]= bNA[t-1]*p11active + (1.- bNA[t-1])*p01active
        else:            
            bA[t]= bA[t-1]*p11 + (1.- bA[t-1])*p01
            bNA[t]= bNA[t-1]*p11 + (1.- bNA[t-1])*p01
    return bA, bNA    

def fastCavg(x1, x2, c0=None, Tpass=np.identity(2), Tact=np.identity(2)):
    
    x1=int(x1)
    x2=int(x2)
    T=[Tpass, Tact]
    q= ((x1*getB(x2,T[1][0][1],T))/(1-getB(x1,T[1][1][1],T)) +x2)**-1
    p=q*(getB(x2,T[1][0][1],T)/(1-getB(x1,T[1][1][1],T)))
    if c0:
        cavg=p*(x1- np.sum([getB(i,T[1][1][1],T) for i in range(1, x1+1)]))+q*(x2- np.sum([getB(j,T[1][0][1],T) for j in range(1, x2+1)])) + (p+q)*c0
        return cavg, cavg-(p+q)*c0, (p+q)
    else:
        # return slope, intercept 
        return (p+q), p*(x1- np.sum([getB(i,T[1][1][1],T) for i in range(1, x1+1)]))+q*(x2- np.sum([getB(j,T[1][0][1],T) for j in range(1, x2+1)]))
   
    
def Cavg(x1, x2, c0=None, Tpass=np.identity(2), Tact=np.identity(2), ba=[], bna=[]):
    
    if len(ba)==0 or len(bna)==0:
        ba, bna= precomputeBelief(Tpass[1][1],Tpass[0][1],Tact[1][1],Tact[0][1])
    
    q= ((x1*bna[x2-1])/(1-ba[x1-1]) +x2)**-1
    p=q*(bna[x2-1]/(1-ba[x1-1]))
        
    if c0:
        cavg=p*(x1- np.sum(ba[:x1]))+q*(x2- np.sum(bna[:x2])) + (p+q)*c0
        return cavg, p*(x1- np.sum(ba[:x1]))+q*(x2- np.sum(bna[:x2])), (p+q)
        #return cavg, (p+q), p*(x1- np.sum(ba[:x1]))+q*(x2- np.sum(bna[:x2]))
    else:
        # return slope, intercept 
        return (p+q), p*(x1- np.sum(ba[:x1]))+q*(x2- np.sum(bna[:x2]))
        

def fastgetThresholdC(point1, point2, Tpass, Tact):
    
    # print (point1, point2)
    slope1, const1 = fastCavg(point1[0], point1[1], Tpass=Tpass, Tact=Tact)
    slope2, const2 = fastCavg(point2[0], point2[1], Tpass=Tpass, Tact=Tact)
    c_threshold= (const1-const2)/(slope2-slope1)
    
    if np.isnan(c_threshold):
        print(const1,const2,slope1,slope2)
        print(Tpass)
        print(Tact)
        raise ValueError
    
    return c_threshold
 
def getThresholdC(point1, point2, Tpass, Tact,ba=[], bna=[]):
    
    if len(ba)==0 or len(bna)==0:
        ba, bna= precomputeBelief(Tpass[1][1],Tpass[0][1],Tact[1][1],Tact[0][1])
        
    # print (point1, point2)
    slope1, const1 = Cavg(point1[0], point1[1], Tpass=Tpass, Tact=Tact, ba=ba, bna=bna)
    slope2, const2 = Cavg(point2[0], point2[1], Tpass=Tpass, Tact=Tact, ba=ba, bna=bna)
    c_threshold= (const1-const2)/(slope2-slope1)

  
    return c_threshold
    


def hittingTime(b_start, b_hitting, T, L=180):
    
    p01=T[0][0][1]
    p11=T[0][1][1]
    random_big_number=200
    b_stat= p01/(p01+1-p11)
    
    if b_start< b_hitting:
        return 0
    elif (b_start >= b_hitting) and (b_hitting < b_stat):
        return random_big_number
    else:
        
        term_inside_log= (p01 - b_hitting*(1-p11+p01) )/(p01 - b_start*(1-p11+p01))
    
        return min(L, np.floor((np.log(term_inside_log))/(np.log(p11-p01)))+1 )
    
def getB(k, start_b, T):
    '''
    Returns belief of kth item in chain (k starts from 1)
    Please give first belief of chain as input start_b (This is k=1 belief)
    T is the passive transition matrix    
    '''
    
    p01=T[0][0][1]
    p11=T[0][1][1]
    
    return (p01-((p11-p01)**(k-1))*(p01-(1+p01-p11)*start_b))/(1+p01-p11)
    
    
    
def whittleIndex(T, L=180, ba=[], bna=[], limit_a=None, limit_na=None):
    
    '''
    Pre-compute whittle's index for all possible values of x1 and all values of x2. 
    If limit_a is not None, then compute only until limit_a for A chain.   
    If limit_na is not NOne, then compute only until limit_na for NA chain. 
    
    T should be such that T[0]= T_passive and T[1] is T_active
    w1 stores whittle index for current belief state for the A chain
    w2 stores whittle index for current belief state for the NA chain
    '''
    if len(ba)==0 or len(bna)==0:
        ba, bna= precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1], L=L)
        
    w1=np.zeros(L)
    w2=np.zeros(L)
    
    x1_current=1 
    x2_current=1
    
    if limit_a is None: 
        limit_a=L -1
        
    if limit_na is None:
        limit_na=L-1
        
    while (x1_current<=limit_a or x2_current<=limit_na):
        
        #print (x1_current, x2_current)
        
        if not x1_current<L:        # x1 is full only look for x2(down) whittle index
            c_down= getThresholdC((x1_current, x2_current), (x1_current, x2_current+1), T[0], T[1], ba=ba, bna=bna)
            w2[x2_current-1]=c_down
            x2_current+=1
        
        elif not x2_current<L:        # x2 is full only look for x1(right) whittle index
            c_right= getThresholdC((x1_current, x2_current), (x1_current+1, x2_current), T[0], T[1], ba=ba, bna=bna)
            w1[x1_current-1]=c_right
            x1_current+=1
        
        else:        # neither directions is full, look for both whittle index
            c_down= getThresholdC((x1_current, x2_current), (x1_current, x2_current+1), T[0], T[1], ba=ba, bna=bna)
            c_right= getThresholdC((x1_current, x2_current), (x1_current+1, x2_current), T[0], T[1], ba=ba, bna=bna)
            
            if c_down<c_right:
                w2[x2_current-1]=c_down
                x2_current+=1
            else:
                w1[x1_current-1]=c_right
                x1_current+=1
                
    return w1, w2
                

def newWhittle(T, L=180, ba=[], bna=[], limit_a=None, limit_na=None):
    
    if len(ba)==0 or len(bna)==0:
        ba, bna= precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1])
        
    w1=np.zeros(L)
    w2=np.zeros(L)
    
    x1_current=1 
    x2_current=1
    
    p01= T[0][0][1]
    p10= T[0][1][0]
    b_stationary = p01/(p01+p10)
    
    if limit_a is None: 
        limit_a=L 
        
    if limit_na is None:
        limit_na=L
    
    while (x1_current<limit_a or x2_current<limit_na):
        
        print ("new:", x1_current, x2_current)

        if not x1_current<L:
            
            slope, constant= Cavg(x1_current, 1, Tpass=T[0], Tact=T[1], ba=ba, bna=bna) 
            c_down = (1-b_stationary- constant)/slope
            
            for i in range(L):
                w2[i]= c_down
            
            x2_current=L       
            
            
        else:
            c_up = getThresholdC((x1_current,1), (x1_current+1,1), T[0], T[1], ba=ba, bna=bna)
            
            slope, constant= Cavg(x1_current, 1, Tpass=T[0], Tact=T[1], ba=ba, bna=bna) 
            c_down = (1-b_stationary- constant)/slope
            
            if c_down > c_up:
                w1[x1_current-1]=c_up
                x1_current+=1
            
            else:
            
                for i in range(L):        
                    w2[i]= c_down
                x2_current=0
                
                for i in range(x1_current-1, L):    
                    w1[i]=c_down
                x1_current=L
                    
            
    return w1, w2  
 
    
def newnewWhittle(T, L=180, ba=[], bna=[], limit_a=None, limit_na=None, t= 180, finite=False,
                  version=-1):
    
    '''
    t is the finite time horizon
    '''
    if limit_a is None: 
        limit_a=L -1
        
    if limit_na is None:
        limit_na=L-1
        
        
    if len(ba)==0 or len(bna)==0:
        ba, bna= precomputeBelief(T[0][1][1],T[0][0][1],T[1][1][1],T[1][0][1])
        
    w1=np.zeros(L)
    w2=np.zeros(L)
    
    b_stat =  T[0][0][1]/(T[0][0][1]+T[0][1][0])
    

    slope, intercept= (Cavg(1, L, c0=0, Tpass=T[0], Tact=T[1], ba=ba, bna=bna))
    for i in range(len(w1)):  
        w1[i]= (ba[i]*(1-b_stat- intercept))/(1+ba[i]*slope)
    
    for i in range(len(w2)-1):
        w2[i]= getThresholdC((1,i+1), (1,i+2), T[0], T[1],ba=ba, bna=bna)

    if version==-1 and limit_a>0:
        
        prev_m=w2[-2]
        prev_b=bna[-2]
        #A=L*b_stat-sum(bna)
        A=0
        #A=sum()-sum(bna)
        
        
        for i in range(L-1,-1,-1):
            
            #A= sum(ba[i:])+i*b_stat - sum(bna)
            
            delta_b= ba[i]-prev_b
            #dA_by_db=prev_b/delta_b

            delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) 
            #delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) - prev_b*(1-ba[0])/(1+prev_b-ba[0])
            #delta_m=(((prev_m+A)/prev_b)-dA_by_db)*((1-ba[0])/(1+prev_b-ba[0]))
            
            w1[i] = delta_m+prev_m
            
            if i<limit_a-1:
                break
            
            prev_m= w1[i]
            prev_b=ba[i]   
            
    if version==0:
        
        prev_m=w2[-2]
        prev_b=bna[-2]
        A=L*b_stat-sum(bna)
        
        #A=sum()-sum(bna)
        
        
        for i in range(L-1,-1,-1):
            
            #A= sum(ba[i:])+i*b_stat - sum(bna)
            
            delta_b= ba[i]-prev_b
            #dA_by_db=prev_b/delta_b

            delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) 
            #delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) - prev_b*(1-ba[0])/(1+prev_b-ba[0])
            #delta_m=(((prev_m+A)/prev_b)-dA_by_db)*((1-ba[0])/(1+prev_b-ba[0]))
            
            w1[i] = delta_m+prev_m
            
            
            prev_m= w1[i]
            prev_b=ba[i]    
        
    if version==1:
        
        prev_m=w2[-2]
        prev_b=bna[-2]
        A=L*b_stat-sum(bna)
        
        #A=sum()-sum(bna)
        
        
        for i in range(L-1,-1,-1):
            
            A= sum(ba[i:])+i*b_stat - sum(bna)
            
            delta_b= ba[i]-prev_b
            #dA_by_db=prev_b/delta_b

            delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) 
            #delta_m=((prev_m+A)*(1-ba[0])*delta_b)/(prev_b*(1+prev_b-ba[0])) - prev_b*(1-ba[0])/(1+prev_b-ba[0])
            #delta_m=(((prev_m+A)/prev_b)-dA_by_db)*((1-ba[0])/(1+prev_b-ba[0]))
            
            w1[i] = delta_m+prev_m
            
            
            prev_m= w1[i]
            prev_b=ba[i]
            
    if version==2:
        
        prev_m=w2[-2]
        prev_b=bna[-2]
        
        for i in range(L-1,-1,-1):
            
            db= ba[i]-prev_b
            dm= db/(1-ba[0]) - ba[i]+ b_stat
            w1[i]=prev_m+dm
            
            prev_b=ba[i]
            prev_m=w1[i]
    

    if version==3:
        
        prev_m=w2[-2]
        prev_b=bna[-2]
        
        for i in range(L-1,-1,-1):
            
            db= ba[i]-prev_b
            dm= ((1-prev_m)*db)/(1-ba[0]) - ba[i]+ b_stat
            w1[i]=prev_m+dm
            
            prev_b=ba[i]
            prev_m=w1[i]
            
    if version==4:
        
        for i in range(1, len(w1)-1):
            w1[i]= getThresholdC((i,1), (i+1,1), T[0], T[1],ba=ba, bna=bna)
            
    if version==5:
        
        for i in range(1, len(w1)-1):
            w1[i]= getThresholdC((i,L), (i+1,L), T[0], T[1],ba=ba, bna=bna)
                    
            

    if finite: 
        
        for i in range(len(w1)-1):
            
            t=L-i
            '''
            v_calling= v_calling_constant + v_calling_slope*c0 
            '''
            v_calling_slope= ba[i]* (sum([((ba[0])**(j-1))*(j)*(1-ba[0]) for j in range(t)]))
            v_calling_constant=( (1-bna[i])*(t-sum(bna[:t])) + ba[i]* (sum([ ( (t-j-sum(bna[:t-j])) + ((ba[0])**(j-1))*((1-ba[0])**2)*j ) for j in range(1, t+1)])))

            v_notcalling = sum ([1-ba[j] for j in range(i, i+t)]) 
            
            
            w1[i]= (v_notcalling - v_calling_constant)/v_calling_slope
        
        
            w2[i]= getThresholdC((1,i+1), (1,i+2), T[0], T[1],ba=ba, bna=bna)
        
    return w1, w2  
    

def fastWhittle(T, x1=None, x2=None):
    
    
          
  if x2 is None and x1 is None:
      print("Specify which whittle index is required")
      return 
  elif x2 is None:
      x1_current=x1
      b=getB(x1_current, T[1][1][1], T)
      x2_current= 1+hittingTime(T[1][0][1], b, T)
      return fastgetThresholdC((x1_current, x2_current), (x1_current+1, x2_current), T[0], T[1])
  else:
      x2_current=x2
      b=getB(x2_current, T[1][0][1], T)
      x1_current= 1+hittingTime(T[1][1][1], b, T)
      return fastgetThresholdC((x1_current, x2_current), (x1_current, x2_current+1), T[0], T[1])  


def isBadPatient(T, whittle_test=True):
    
    p_act01=T[1][0][1]
    
    p01=T[0][0][1]
    p10=T[0][1][0]
    
    if whittle_test:
        '''
        It is bad patient is whittle index is decreasing as a function of time.
        '''
        if fastWhittle(T, x1=2)- fastWhittle(T, x1=1)>=0:
            return False
        else:
            return True
    
    
    b_stationary= p01/(p01+p10)
    simple_test=False
    
    if p_act01<b_stationary:
        simple_test=True
    else:
        simple_test= False
    
    L=3
    ba, bna= precomputeBelief(1-p10, p01, T[1][1][1], p_act01, L=5)
    #w0= p01/(p01 + (1.-p11))
    
    
    expectedbeliefA=[ba[i]*ba[0]+(1-ba[i])*bna[0] for i in range(L)]
    expectedbeliefNA=[bna[i]*ba[0]+(1-bna[i])*bna[0] for i in range(L)]
    jumpA=[expectedbeliefA[i]-ba[i] for i in range(L)]
    jumpNA=[expectedbeliefNA[i]-bna[i] for i in range(L)]
    
    complex_test= not ((jumpNA[1]>jumpNA[0]) and (jumpNA[2]>jumpNA[1]) and (jumpA[1]>jumpA[0]) )
    
    if not simple_test==complex_test:
        pass
        #print ("NOT TRUE", simple_test, complex_test)
    
    #return complex_test
    print(simple_test)
    return simple_test

def getAdherenceComparison():
    
    N=50
    L=180
    k=10
    
    N_TRIALS=5
    
    adherence0=[]
    adherence1=[]
    
    first_seedbase=np.random.randint(0, high=100000)
    first_seedbase=4
    
    for i in range(N_TRIALS):
        
        print (i)
        seedbase = first_seedbase + i
        np.random.seed(seed=seedbase)
        
        T = None
        T = generateTmatrix(N)
        
        ############################ Whittle
        np.random.seed(seed=seedbase)
        adherence0.append(np.mean(np.sum(simulateAdherence(N,L,T,k,policy_option=5, new_whittle=False), axis=1)))
        
        ############################ New Whittle
        np.random.seed(seed=seedbase)
        adherence1.append(np.mean(np.sum(simulateAdherence(N,L,T,k,policy_option=5, new_whittle=True), axis=1)))
        
        
    print ("Old whittle:", np.mean(adherence0))
                    
    print ("New whittle:",np.mean(adherence1))
                
    #end=time.time()
    #print ("Time taken: ", end-start)
    
    '''
    barPlot(labels, values, errors, ylabel='Average Adherence out of %s days'%L,
            title='%s patients, %s calls per day' % (N, k), 
            filename='img/results_'+savestring+'_N%s_k%s_trials%s_data%s_s%s.png'%(N,k,N_TRIALS, args.data, first_seedbase), root=args.file_root)
    '''
    return 
    

if __name__=="__main__":
    
    days=15
    L=180
    #b1, b2= precomputeBelief(0.9, 0.1)
    '''
    p_pass01s=[0.1]#[0.30, 0.5, 0.8]
    p_pass11s=[0.8]#[0.3, 0.6, 0.9]
    
    p_act01s=[0.15]#[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    p_act11s=[0.95]#[0.95, 0.55, 0.25]
    
    for p_act11 in p_act11s:
        for p_pass01 in p_pass01s:
            for p_pass11 in p_pass11s:
                for p_act01 in p_act01s:
                    
                    
                    #plotBeliefTrajectory(p_pass11, p_pass01, p_act11, p_act01, read=True,
                                         #belief_plots=False, whittle_plots=False, days=15, 
                                         #scatter_sv='')
                        
    
    '''
    threshopt=True
    while threshopt:
        T=generateTmatrixNIBandIB(1,thresh_opt_frac=0,quick_check=False)[0]
        if not isThresholdOptimal(T, 0.5, quick_check=False):
            threshopt=False
    ba, bna= precomputeBelief(T[0][1][1], T[0][0][1], T[1][1][1],T[1][0][1],  L=180)
    s1=time.time()
    print ("Computing Threshold Whittle. Chains: A, NA")

    wa, wna= whittleIndex(T, L=180, ba=[], bna=[], limit_a=None, limit_na=None)
    
    
    
    threshopt=True
    while threshopt:
        T2=generateTmatrixNIBandIB(1,thresh_opt_frac=0,quick_check=False)[0]
        if not isThresholdOptimal(T2, 0.5, quick_check=False):
            threshopt=False
    ba, bna= precomputeBelief(T2[0][1][1], T2[0][0][1], T2[1][1][1],T2[1][0][1],  L=180)
    s1=time.time()
    print ("Computing Threshold Whittle. Chains: A, NA")

    wa2, wna2= whittleIndex(T2, L=180, ba=[], bna=[], limit_a=None, limit_na=None)
    
    
    
    
    threshopt=True
    while threshopt:
        T3=generateTmatrixNIBandIB(1,thresh_opt_frac=0,quick_check=False)[0]
        if not isThresholdOptimal(T3, 0.5, quick_check=False):
            threshopt=False
    ba, bna= precomputeBelief(T3[0][1][1], T3[0][0][1], T3[1][1][1],T2[1][0][1],  L=180)
    s1=time.time()
    print ("Computing Threshold Whittle. Chains: A, NA")

    wa3, wna3= whittleIndex(T3, L=180, ba=[], bna=[], limit_a=None, limit_na=None)
    
    
    
    #wy=yundi_whittle_exact(T, ba[], 140)
    days=5
    #wy_a=[yundi_whittle_exact(T, ba[i], 180-i) for i in range(days)]
    #wy_na=[yundi_whittle_exact(T, bna[i], 180-i) for i in range(days)]
    beta1=0.8
    beta2=0.85
    beta3=0.9
    beta4=0.95
    print ("Computing Yundi. Chain: A, beta1")
    #wy_a_beta1=[yundi_beta(T, ba[i], beta1, solver='normal') for i in range(days)]
    print ("Computing Yundi. Chain: A, beta2")
    #wy_a_beta2=[yundi_beta(T, ba[i], beta2, solver='normal') for i in range(days)]
    print ("Computing Yundi. Chain: A, beta3")
    #wy_a_beta3=[yundi_beta(T, ba[i], beta3, solver='normal') for i in range(days)]
    print ("Computing Yundi. Chain: A, beta4")
    #wy_a_beta4=[yundi_beta(T, ba[i], beta4, solver='normal') for i in range(days)]
    
    #print ("Computing Yundi. Chain: NA, beta1")
    #wy_na_beta1=[yundi_beta(T, bna[i], beta1, solver='normal') for i in range(days)]
    #print ("Computing Yundi. Chain: A, beta2")
    #wy_a_beta2=[yundi_beta(T, ba[i], beta2, solver='normal') for i in range(days)]
    #print ("Computing Yundi. Chain: NA, beta2")
    #wy_na_beta2=[yundi_beta(T, bna[i], beta2, solver='normal') for i in range(days)]

    x=[i for i in range(days)]
    fig, ax=plt.subplots(3,1, figsize=(4,6))
    ax[0].plot(x, wa[:days], '--', label='Th. whittle1')
    ax[1].plot(x, wa2[:days], '--', label='Th. whittle2')
    ax[2].plot(x, wa3[:days], '--', label='Th. whittle3')
    #plt.plot(x, wa[:days], '--', label='Th. whittle')
    #plt.plot(x, wy_a_beta4, '-', label='Yundi beta=%s'%(beta4))
    #plt.plot(x, wy_a_beta3, '-', label='Yundi beta=%s'%(beta3))
    #plt.plot(x, wy_a_beta2, '-', label='Yundi beta=%s'%(beta2))
    #plt.plot(x, wy_a_beta1, '-', label='Yundi beta=%s'%(beta1))

    
    #plt.plot(x, wy_na_beta1, 'g-', label='Yundi NA chain, beta=0.9')
    #plt.plot(x, wy_a_beta2, 'r^-', label='Yundi A chain, beta=0.99999')
    #plt.plot(x, wy_na_beta2, 'g^-', label='Yundi NA chain, beta=0.9999')
    plt.xlabel("Days since called")
    plt.ylabel("Whittle Index")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    plt.show()

    
    
    '''
    e1=time.time()
    s2=time.time()
    fastw=fastWhittle(T, x1=5)
    e2=time.time()
    print ("FastW: ",fastw, e2-s2)
    print ("SlowW: ", w[0][4], e1-s1)
    print ("W_a: ",w[0][:10])
    print("W_na: ",w[1][:10])
    
    print ("Transition matrix:",T[0][1][1], T[0][0][1], T[1][1][1],T[1][0][1])
    print ("A chain beliefs: ", ba[:10])
    print ("NA chain beliefs: ", bna[:10])
    #w1, w2= whittleIndex(T[0], L=180, ba=[], bna=[], limit_a=0, limit_na=4) 
    
    #print ("W1:", w1[:10] )
    #print ("W2:", w2[:10] )
    
    
    '''
    '''
    def equations(p):
        x, y = p
        return x*x, x*y
        return (x+y**2-4, math.exp(x) + x*y - 3)

    x, y =  fsolve(equations, (1, 1))
    print (x, y)
    for i in range(10):
        #print (Cavgold(i, 10, 0, T=T))
        print (Cavg(i, 100,1, T=T))
    '''
    
    #getAdherenceComparison()
    
    
    
    
    
    
    