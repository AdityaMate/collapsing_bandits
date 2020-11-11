#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:56:31 2019

@author: adityamate
"""
import matplotlib
# matplotlib.use('pdf')

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib import rcParams
import brute_force_optimal_policy_check
from brute_force_optimal_policy_check import brute_force_check_threshold_policy_type

#Ensure type 1 fonts are used
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode']=True

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', weight='bold')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# from whittle import precomputeBelief

def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)

def isThresholdOptimal(T, beta, quick_check=True):
    T=np.array(T)
    p01a = T[1, 0, 1]
    p11a = T[1, 1, 1]
    p01 =  T[0, 0, 1]
    p11 =  T[0, 1, 1]
    if quick_check:
        return (p11-p01)*(1+beta*(p11a-p01a))*(1-beta)>=(p11a-p01a)
    else:
        reverse=brute_force_check_threshold_policy_type(T, check_forward=False, check_backward=True, L=180, num_ms=500, beta=beta)
        #forward=brute_force_check_threshold_policy_typebrute_force_check_threshold_policy_type(T, check_forward=True, check_backward=False, L=180, num_ms=500, beta=0.5)
        return (not reverse)
def isReverseThresholdOptimal(T, beta, quick_check=True):
    T=np.array(T)
    p01a = T[1, 0, 1]
    p11a = T[1, 1, 1]
    p01 =  T[0, 0, 1]
    p11 =  T[0, 1, 1]
    if quick_check:
        return (p11-p01)*(1+(beta*(p11a-p01a))/(1-beta))<=(p11a-p01a)
    else:
        #reverse=brute_force_check_threshold_policy_typebrute_force_check_threshold_policy_type(T, check_forward=False, check_backward=True, L=180, num_ms=500, beta=0.5)
        forward=brute_force_check_threshold_policy_type(T, check_forward=True, check_backward=False, L=180, num_ms=500, beta=beta)
        return (not forward)

def how_many_NIB(T):
    count_nib = 0
    for t in T:
        p01a = t[1, 0, 1]
        p01 =  t[0, 0, 1]
        p10 =  t[0, 1, 0]
        if p01a >= p01 / (p01+p10):
            count_nib+=1
    return float(count_nib)/T.shape[0]



def verify_T_matrix(T):

    valid = True
    # print(T[0, 0, 1], T[0, 1, 1])
    valid &= T[0, 0, 1] <= T[0, 1, 1] # non-oscillate condition
    # print(valid)
    valid &= T[1, 0, 1] <= T[1, 1, 1] # must be true for active as well
    # print(valid)
    valid &= T[0, 1, 1] <= T[1, 1, 1] # action has positive "maintenance" value
    # print(valid)
    valid &= T[0, 0, 1] <= T[1, 0, 1] # action has non-negative "influence" value
    # print(valid)
    return valid


def returnKGreatestIndices(arr,k):
    
    arr=np.array(arr)
    ans=arr.argsort()[-k:][::-1]
    return ans

def getIB(ad_p=None, ad_0=None, ad_oracle=None, 
          ad_p_files=None,ad_0_files=None,ad_oracle_files=None ):
    
    
    if ad_p is None:
        ad_p= [np.mean(np.sum(np.load(f), axis=1)) for f in ad_p_files]
        ad_0= [np.mean(np.sum(np.load(f), axis=1)) for f in ad_0_files]
        ad_oracle= [np.mean(np.sum(np.load(f), axis=1)) for f in ad_oracle_files]
        
    
    ib=[]
    for i in range(len(ad_p)):
        ib.append(100*(ad_p[i]-ad_0[i])/(ad_oracle[i%len(ad_oracle)]-ad_0[i]))
        
    ib=np.mean(np.array(ib))
    error=np.std(np.array(ib))
    
    return ib, error
    

    
    
    

def runTimePlots(save=False):
    '''
    Generates runtime comparison and performance comparison plots for varying values of N
    
    '''
    
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et al.',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'Oracle',
           15: 'roundRobin', 16:'new_whittle',  17: 'fast_whittle', 
           18: 'buggyWhittle'}

    # policies=[0,1,2,3,5,10,15]
    
    #policies=[14, 10,16, 5,17, 3, 15, 2, 0]
    policies = [14, 10, 5, 17, 3, 15, 2, 0]
    policies=[8,5,10,3, 2, 0]
    policies=[0,2,3,5,10,8]
    #Ns=[20, 40, 100, 150, 200, 250]
    Ns=[20,40,100,200, 300]
    Ns_new=[500,1000,2000]
    learning_modes = [0,1][:1]#, 2]
    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    lr_strings=['None','TS','EG']
    plot_dict={}
    badf=0.2
    badf_new=0.4
    data='real'
    sv='1220_fixed'
    folder=sv+'_lr0'
    
    sv_new='0524'
    folder_new='0524'
    ratio=0
    K_PERCENT = 10

    learning_mode_line_styles = ['-',':','-.']
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(policies) > len(policy_colors):
        raise ValueError("Error: More policies than available colors. Add more colors")

    for lr in learning_modes:
        #folder=sv+'_simulated_lr'+str(lr)
        plot_dict[lr] = {}
        ibs={}
        ibs[14]=[]
        errors={}
        errors[14]=[]
        for p in policies:
            print("Reading policy...: ", p)
            runtimes=[]
            adherences=[]
            runtimes_err=[]
            adherences_err=[]
            ibs[p]=[]
            
            errors[p]=[]
            for n in Ns:
            
                k=int(n/K_PERCENT)
                ratio=round(k/n, 2)
                
                runtime_files=glob.glob('../logs/runtime/%s/*_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s.npy'%(folder, sv, n, k, p, data, badf, lr))
                adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s.npy'%(folder,sv, n, k, p,data, badf, lr))
                adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s.npy'%(folder,sv, n, k, 0,data, badf, lr))
                adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s.npy'%(folder,sv, n, k, 8,data, badf, lr))
                print ("seeds for N=",n,":", len(runtime_files), len(adherence_files))
                runtimes.append((np.mean([np.load(f) for f in runtime_files])))
                runtimes_err.append((np.std([np.load(f) for f in runtime_files]))/np.sqrt(len(runtime_files)))
                adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
                adherences_err.append(np.std([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files])/np.sqrt(len(adherence_files)))
                if p==8:
                    p=14
                ibs[p].append(getIB(ad_p_files=adherence_files,ad_0_files=adherence_files_0,ad_oracle_files=adherence_files_14)[0])
                errors[p].append(getIB(ad_p_files=adherence_files,ad_0_files=adherence_files_0,ad_oracle_files=adherence_files_14)[1])
            for n in Ns_new:  
                if p==8:
                    p=14
                k=int(n/K_PERCENT)
                ratio=round(k/n, 2)
                runtime_files=glob.glob('../logs/runtime/%s/*_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder_new, sv_new, n, k, p, data, badf_new, lr))
                adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder_new,sv_new, n, k, p,data, badf_new, lr))
                adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder_new,sv_new, n, k, 0,data, badf_new, lr))
                adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder_new,sv_new, n, k, 14,data, badf_new, lr))
                print ("seeds for N=",n,":", len(runtime_files), len(adherence_files))
                runtimes.append((np.mean([np.load(f) for f in runtime_files])))
                runtimes_err.append((np.std([np.load(f) for f in runtime_files]))/np.sqrt(len(runtime_files)))
                adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
                adherences_err.append(np.std([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files])/np.sqrt(len(adherence_files)))
                
                #ibs[p].append(getIB(ad_p_files=adherence_files,ad_0_files=adherence_files_0,ad_oracle_files=adherence_files_14)[0])
                #errors[p].append(getIB(ad_p_files=adherence_files,ad_0_files=adherence_files_0,ad_oracle_files=adherence_files_14)[1])
                
                if p==14:
                    p=8
            
            plot_dict[lr][p]={ 'runtime': runtimes,
                            'adherence':adherences,
                            'runtime_err':runtimes_err,
                            'adherence_err':adherences_err
                        }
    
    '''
    Temporarily switched off
    fig, ax=plt.subplots(2,2, figsize=(16,10))
    fig.suptitle('Performance/runtime plot vs N; k=%sN, bad_f=%s'%(ratio, badf), fontsize=16)
    learning_mode_line_styles = ['-',':','-.']
    '''
    fig2=plt.figure()
    fig, ax = plt.subplots(1,2, figsize=(12,3.5))
    symbols=['o-','s-','^-']
    #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    for ind_lr, lr in enumerate(learning_modes):
        for ind_p, p in enumerate([5,10]):
            print ("Policy:", p, "runtime is ", plot_dict[lr][p]['runtime'])
            #print ("MODE:", lr)
            #print (len(plot_dict[lr][p]['runtime']))
            #print (Ns, plot_dict[lr][p]['runtime'] )
            '''
            Temporarily seitched off
            ax[0][lr].plot(Ns, plot_dict[lr][p]['runtime'], 'o', color=policy_colors[ind_p], linestyle=learning_mode_line_styles[ind_lr])
            #ax[1].plot(Ns, [plot_dict[p]['adherence'][i]-plot_dict[2]['adherence'][i] for i in range(len(Ns))], 'o-', label=pname[p])
            ax[1][lr].plot(Ns, [plot_dict[lr][p]['adherence'][i] for i in range(len(Ns))], 'o', color=policy_colors[ind_p], linestyle=learning_mode_line_styles[ind_lr])
            '''
            if p==10:
                print (len(plot_dict[lr][p]['runtime'][:-2]), len(plot_dict[lr][p]['runtime_err'][:-2]))
                print ((plot_dict[lr][p]['runtime_err'][:-2]))
                print ((plot_dict[lr][p]['runtime'][:-2]))
                ax[0].errorbar((Ns+Ns_new)[:-2], plot_dict[lr][p]['runtime'][:-2], yerr=plot_dict[lr][p]['runtime_err'][:-2],  fmt=symbols[ind_p], color=policy_colors[ind_p], linestyle=learning_mode_line_styles[ind_lr], label=pname[p])            
            else:    
                ax[0].errorbar(Ns+Ns_new, plot_dict[lr][p]['runtime'], yerr=plot_dict[lr][p]['runtime_err'] , fmt=symbols[ind_p], color='g', linestyle=learning_mode_line_styles[ind_lr], label=pname[p])
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        #values=[100*(plot_dict[lr][p]['adherence'][5]-plot_dict[lr][0]['adherence'][5])/(plot_dict[lr][8]['adherence'][5]-plot_dict[lr][0]['adherence'][5]) for p in policies[1:]]
        bottom=0
        
        #print ("VAL:", values)
        print (width)
        print (x)
        Ns=[100,200,300,500]
        x = np.arange(len(Ns))
        pname[5]='Threshold \nWhittle'
        for pidx, p in enumerate((policies[1:])):
            
            values=[100*(plot_dict[lr][p]['adherence'][n]-plot_dict[lr][0]['adherence'][n])/(plot_dict[lr][8]['adherence'][n]-plot_dict[lr][0]['adherence'][n])  for n in range(2,6)]
            errors0=plot_dict[lr][0]['adherence_err']
            errorsp=plot_dict[lr][p]['adherence_err']
            errorso=plot_dict[lr][8]['adherence_err']
            
            errors=[((errorsp[n]+errors0[n])/(plot_dict[lr][p]['adherence'][n]-plot_dict[lr][0]['adherence'][n]) + (errorso[n]+errors0[n])/(plot_dict[lr][8]['adherence'][n]-plot_dict[lr][0]['adherence'][n]))*values[n-2] for n in range(2,6)]
            #print (len(values), len(ibs[p]))
            rects1 = ax[1].bar(x+pidx*width, values, width, yerr=errors, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            #rects1 = ax[1].bar(x+pidx*width, ibs[p][2:6], width, yerr=errors[p][2:6],bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            print ("Policy:", p, " intervention benefit is: ", values)
        
        
        
        ax[1].set_ylabel(" Intervention benefit (\%)", fontsize=20)
        ax[0].set_ylabel("Runtime in seconds", fontsize=20)
        ax[0].set_xlabel(" Number of patients", fontsize=20)
        ax[1].set_xlabel(" Number of patients", fontsize=20)
        #ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax[1].set_title("Intervention Benefit comparison", fontsize=14)
        ax[1].set_xticks([i+2*width for i in [0,1,2,3]])
        ax[1].set_xticklabels(["N="+str(n) for n in Ns+[500]], rotation=0)
        #ax[1].set_xticklabels(labels, rotation=30)
        ax[1].set_yticks([0,50,100])
        #ax[1].legend(fontsize=14)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=16)
        handles, labels = ax[1].get_legend_handles_labels()
        ax[1].legend(reversed(handles), reversed(labels),loc='lower center', bbox_to_anchor=(1.35, 0.04), ncol=1, fontsize=18)
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax[1].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        #autolabel(rects1)       
        plt.tight_layout()
    """
    ax[0][0].set(xlabel='Number of patients',  ylabel='log of runtime in sec')
    ax[0][1].set(xlabel='Number of patients', ylabel='log of runtime in sec')
    
    ax[1][0].set(xlabel='NO LEARNING',  ylabel='Adherence- NO L ')#, ylim=[149,161])
    ax[1][1].set(xlabel='THOMSPON SAMPLING',  ylabel='Adherence- TS')#, ylim=[149,161])
    #plt.xlabel('Number of patients')
    #plt.ylabel('')
    legend_elements = []
    for ind_p, p in enumerate(policies):
        legend_elements.append(Line2D([0], [0], color=policy_colors[ind_p], label='%s'%pname[p]))

    for ind_lr, lr in enumerate(learning_modes):
        legend_elements.append(Line2D([0], [0], color='k', alpha=0.5, linestyle=learning_mode_line_styles[ind_lr], label='%s'%lr_strings[ind_lr]))

    plt.legend(handles=legend_elements)
    """
    if save:
        sv='linear_scale_full'
        plt.savefig('./img/runtime/vsN/performance_runtime'+sv+'_vsN_k_%s.png'%(ratio))
    plt.show()

def kPlots():    
    '''
    performance-vs-k      
    '''
    pname={0: 'no calls',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et. al',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle',
           15: 'roundRobin', 16:'new_whittle', 17: 'fast_whittle', 
           18: 'buggyWhittle'}

    #policies=[0,1,2,3,5,10,15]
    policies=[10, 5, 3, 2, 0,14,1]
    
    #Kps=[5, 10,20, 30, 40, 50,60,70,80]
    Kps=[5, 10,15, 20, 30, 40,50,60,70,80][:3]
    plot_dict={}
    badf=0.4
    data='real'
    sv='1220_fixed_new'
    additional_sv='percentagePlot'
    ratio=0
    N=200
    lr=0
    #folder=sv+'_real_lr'+str(lr)
    folder = sv
    
    
    lrnames=['no_learn', 'TS', 'EG']
    plot_dict={}
    percentages={}
    for p in policies:
        print ("POLICY:", p)
        adherences=[]
        for kp in Kps:
            #seeds=range(5,10)
            
            #k=int(n/20)
            #ratio=round(k/n, 2)
            k=int((kp*N)/100)
            #print ("K", k)
            adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*lr%s*.npy'%(folder, sv, N, k, p,data, badf, lr))
            adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*lr%s*.npy'%(folder, sv, N, k, 0,data, badf, lr))
            adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_N%s_k%s_L180_policy%s_data%s_badf%s_s*lr%s*.npy'%(folder, sv, N, k, 14,data, badf, lr))
            print ("Files for kp=%s :"%(kp),len(adherence_files))
            #runtimes.append(np.log(np.mean([np.load(f) for f in runtime_files])))
            adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
        
        plot_dict[p]=adherences
    
    
    for p in policies:
        
        percentages[p]= [100*(plot_dict[p][i]-plot_dict[0][i])/(plot_dict[14][i]-plot_dict[0][i]) for i in range (len(plot_dict[p]))]
    
    """
    Temporary code below:
    """
    '''
    policies=[2,3,5,10,8]
    k_to_plot=Kps[:3]
    
    fig, ax=plt.subplots(len(k_to_plot),1, figsize=(5,8))
    
    for idx, k in enumerate(k_to_plot):
        x = np.arange(len(policies))  # the label locations
        width = 0.85  # the width of the bars
        bottom=0
        labels=[pname[p] for p in policies]
        rects1 = ax[idx].bar(x, [percentages[p][idx] for p in policies], width, bottom=bottom, label=' k = '+str(k)+"% of N")
        if idx==0:
            #ax[idx].set_ylabel("% of intervention benefit acheived")
            ax[idx].set_title("Intervention benefit comparison")   
            ax[idx].set_xticks(x)
            ax[idx].set_xticklabels(labels, rotation=15)
            ax[idx].legend()
        if idx==1:
            ax[idx].set_ylabel("% of intervention benefit acheived")
            #ax[idx].set_title("Intervention benefit comparison")   
            ax[idx].set_xticks(x)
            ax[idx].set_xticklabels(labels, rotation=15)
            ax[idx].legend()
        if idx==2:
            #ax[idx].set_ylabel("% of intervention benefit acheived")
            #ax[idx].set_title("Intervention benefit comparison")   
            ax[idx].set_xticks(x)
            ax[idx].set_xticklabels(labels, rotation=15)
            ax[idx].legend()
            
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax[idx].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 1),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        autolabel(rects1)       
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig('./img/vsK/newKplot.png')
    '''
    
    
    
    """
    GROUPED PLOT Temporary code below:
    """
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 0)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    policies=[2,3,5,10,14]
    k_to_plot=Kps[:3]
    
    fig, ax=plt.subplots(1,1, figsize=(5,3))
    x = np.arange(len(k_to_plot))
    width = 0.18
    bottom=0
    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    for pidx, p in enumerate(policies):
        
        rects1 = ax.bar(x+pidx*width, [percentages[p][idx] for idx in range(len(k_to_plot))] 
        , width, bottom=bottom, label=pname[p], color= colors[pidx], edgecolor='black')
        #autolabel(rects1) 
    labels=['k='+str(item)+'\%N' for item in k_to_plot]    
    ax.set_xticks(x+width*2)
    ax.set_yticks([0,50,100])
    ax.set_xticklabels(labels, rotation=0, fontsize=18)

    ax.set_ylabel("Intervention benefit (\%)", fontsize=20)
    ax.set_xlabel("Resource level (k)", fontsize=20)
    #ax.set_title("Intervention benefit comparison")   
    #ax.legend()
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=16)            
    
          
    plt.tight_layout()
    #plt.savefig('./img/vsK/newKplot.pdf', bbox_inches="tight")
    
    
    
    '''
    for idx, k in enumerate(k_to_plot):
        
        fig, ax=plt.subplots(1, figsize=(5,3))
        
        x = np.arange(len(policies))  # the label locations
        width = 0.85  # the width of the bars
        bottom=0
        labels=[pname[p] for p in policies]
        rects1 = ax.bar(x, [percentages[p][idx] for p in policies], width, bottom=bottom, label='Resources (k) = '+str(k)+"% of N")
        
        if idx==0:
            #ax[idx].set_ylabel("Intervention benefit acheived in %")
            ax.set_title("Intervention benefit comparison")   
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.legend()
        if idx==1:
            #ax.set_ylabel("Intervention benefit acheived as a % of total possible")
            #ax[idx].set_title("Intervention benefit comparison")   
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.legend()
        if idx==2:
            #ax[idx].set_ylabel("Intervention benefit acheived in %")
            #ax[idx].set_title("Intervention benefit comparison")   
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15)
            ax.legend()
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 1),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        
        autolabel(rects1)       
        plt.tight_layout()
        plt.tight_layout()
        
        plt.savefig('./img/vsK/newKplot_'+str(k)+'.png')
        plt.show()
    '''
        
        
    
    
    plt.figure(figsize=(7,4))
    for p in policies[:-2]:
        plt.plot(Kps, percentages[p], 'o-',label =pname[p] )
    plt.legend()
    plt.xlabel("Percentage of available resources: k/N")
    plt.ylabel ("adherence improvement percentage")
    #plt.title('performance_'+sv+'_vsKfrac_N%s_k_badf_%s_lr%s.png'%(N, badf, lrnames[lr]))
    #plt.title('Adherence improvement achieved as a percentage of total possible for N_100')
    #plt.savefig('./img/vsK/performance_'+sv+additional_sv+'_vsKfrac_N%s_k_badf_%s_lr%s.png'%(N, badf, lr))
    plt.show()
    
def threshOptFracPlots(folder='thresoptExps', sv='thresoptExps_threshopt_frac', save=False,
                       fig1=None,ax1=None, return_early=False):
    '''
    Generates runtime comparison and performance comparison plots for varying values of fraction of threshold optimal patients
    
    '''
    
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et al.',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle',
           15: 'roundRobin', 16:'new_whittle',  17: 'fast_whittle', 
           18: 'buggyWhittle'}

    policies=[0,2,3,5,10,14]
    fracs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data='real'
    #sv='thresoptExps_threshopt_frac'
    #folder='thresoptExps'



    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    plot_dict={}
    badf=0.4
    ratio=0
    K_PERCENT = 10
    k=20
    lr=0
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(policies) > len(policy_colors):
        raise ValueError("Error: More policies than available colors. Add more colors")

    for p in policies:
        print("Reading policy...: ", p)
        runtimes=[]
        adherences=[]
        runtimes_err=[]
        adherences_err=[]
        errors_a=[]
        for frac in fracs:
        
            
            ratio=round(10, 2)
            
            runtime_files=glob.glob('../logs/runtime/%s/*_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder, sv, frac, k, p, data, badf, lr))
            adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, p,data, badf, lr))
            adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 0,data, badf, lr))
            adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 14,data, badf, lr))
            print ("seeds for frac=",frac,":", len(runtime_files), len(adherence_files))
            
            
            runtimes.append(np.log10(np.mean([np.load(f) for f in runtime_files])))
            adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
            #runtimes_err.append(np.log10(np.mean([np.load(f) for f in runtime_files])))
            adherences_err.append(np.std([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files])/np.sqrt(len(adherence_files)))
            #errors_a.append(np.std([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
        
        plot_dict[p]={'runtime': runtimes,
                         'adherence':adherences,
                         'adherence_err':adherences_err
                    }
    """   
    Plotting code below:
    """
    plot_both_graphs= True
    if plot_both_graphs:
        fig2=plt.figure()
        fig, ax = plt.subplots(1,2, figsize=(9,3))
        if fig1 is None or ax1 is None:
            fig1, ax1 = plt.subplots(1,1, figsize=(5,3))
        symbols=['o-','s-','^-', '.-']
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        for ind_p, p in enumerate([10,5,3,0]):
            print ("Policy:", p, "runtime is ", plot_dict[p]['runtime'])
            
            ax[0].plot(fracs, plot_dict[p]['adherence'], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
        
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        values=[100*(plot_dict[p]['adherence'][3]-plot_dict[0]['adherence'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        
            
        
        
        #errors_a=[100*(plot_dict[p]['errors_a'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        bottom=0
        
        print ("VAL:", values)
        print ("RAW adherence:",plot_dict[p]['adherence'])
        #print ("Errors:",plot_dict[p]['errors_a'])
        print (x)
        fracs_to_show=[0,2,6,10]
        x = np.arange(len(fracs_to_show))
        for pidx, p in enumerate((policies[1:])):
            
            values=[100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            
            
            errors0=plot_dict[0]['adherence_err']
            errorsp=plot_dict[p]['adherence_err']
            errorso=plot_dict[14]['adherence_err']
                
            errors=[((errorsp[fr]+errors0[fr])/(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr]) + (errorso[fr]+errors0[fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]))*values[i] for i, fr in enumerate(fracs_to_show)]
        
            
            
            
            #errors_a=[100*(plot_dict[p]['errors_a'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            rects1 = ax[1].bar(x+pidx*width, values, width, yerr=errors, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            #rects2 = ax1.bar(x+pidx*width, values, width, yerr=errors_a, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            rects1 = ax1.bar(x+pidx*width, values, width, yerr=errors, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            
        
        ax1.set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax1.set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        if return_early:
            return
        
        ax[1].set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        ax[0].set_ylabel(" Adherence out of 180", fontsize=16)
        ax[0].set_xlabel(" Fraction of threshold opt patients", fontsize=16)
        ax[1].set_xlabel(" Fraction of threshold optimal patients", fontsize=16)           #Uncomment: 
        #ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax.set_title("Intervention Benefit comparison", fontsize=14)                #Uncomment: 
        ax[1].set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax[1].set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        #ax[1].set_xticklabels(labels, rotation=30)
        ax[1].set_yticks([0,50,100])                                                   #Uncomment: 
        #ax[1].legend(fontsize=14)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
    else:
        
        
        fig, ax = plt.figure()
        for ind_p, p in enumerate([10,5,0]):
            print ("Policy:", p, "runtime is ", plot_dict[p]['runtime'])
            
            #ax[0].plot(fracs, plot_dict[p]['adherence'], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
        
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        values=[100*(plot_dict[p]['adherence'][3]-plot_dict[0]['adherence'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        bottom=0
        
        print ("VAL:", values)
        print (width)
        print (x)
        fracs_to_show=[2,4,6,8,10]
        x = np.arange(len(fracs_to_show))
        for pidx, p in enumerate((policies[1:])):
            
            values=[100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            #rects1 = ax[1].bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            rects1 = ax.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
        
        
        
        
        ax.set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        #Uncomment: ax[0].set_ylabel(" Adherence out of 180", fontsize=16)
        #Uncomment: ax[0].set_xlabel(" Fraction of threshold opt patients", fontsize=16)
        ax.set_xlabel(" Fraction of threshold optimal patients", fontsize=16)           #Uncomment: 
        #Uncomment: ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax.set_title("Intervention Benefit comparison", fontsize=14)                #Uncomment: 
        ax.set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax.set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        #ax[1].set_xticklabels(labels, rotation=30)
        ax.set_yticks([0,50,100])                                                   #Uncomment: 
        #ax[1].legend(fontsize=14)
        #Uncomment: ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4, fontsize=12)
        
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 0)
            ax[1].annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    #autolabel(rects1)       
    plt.tight_layout()
    #save=True
    if save:
        plt.savefig('./img/threshOptFrac/patient_random_brute_force_N%s_k_20.png'%(200))
    plt.show()



    
def groupedBarPlot(infile_prefix, ylabel='Average Adherence out of 180 days',
            title='', filename='image.png', root='.'):
    
    import glob
    d={}
    labels=[]
    for fname in glob.glob(infile_prefix+'*'):
        df = pd.read_csv(fname)
        d[fname] = {}
        d[fname]['labels'] = df.columns.values
        labels = df.columns.values
        d[fname]['values'] = df.values[0]
        d[fname]['errors'] = df.values[1]

    print(d)

    fname = os.path.join(root,'test.png')

    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    rects = []
    fig, ax = plt.subplots(figsize=(8,6))
    for i,key in enumerate(d.keys()):
        rects1 = ax.bar(x+i*width, d[key]['values'], width, yerr=d[key]['errors'], label='average adherence'+key[-8:])
        rects.append(rects1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # for r in rects:
    #     autolabel(r)       
    plt.tight_layout() 
    plt.savefig(fname)
    # plt.show()

def barPlot(labels, values, errors, ylabel='Average Adherence out of 180 days',
            title='Adherence simulation for 20 patients/4 calls', filename='image.png', root='.',
            bottom=0):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)       
    plt.tight_layout() 
    plt.savefig(fname)
    plt.show()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def write_action_logs(file_root,N,k,N_TRIALS,data, action_logs ):
    
    for policy_option in action_logs.keys():
        fname = os.path.join(file_root,'action_logs/action_logs_N%s_k%s_trials%s_data%s_policy%s.csv'%(N,k,N_TRIALS, data, policy_option))
        columns = list(map(str, np.arange(N)))
        df = pd.DataFrame(action_logs[policy_option], columns=columns)
        df.to_csv(fname, index=False)


def createAdherenceResultsMatrix(filename, N_trials, num_policies, L):
    
    adherence_array=np.zeros((N_trials, num_policies, L))
    ''' adherence_array is a results array of size: adherence_array[Trial_number][policy_number][Day of the program(0-179)]
    '''
    np.save(filename, adherence_array)
    

def error(T1, T2):
    
    diff=T1-T2
    return np.linalg.norm(np.linalg.norm(np.linalg.norm(np.linalg.norm(diff, ord=2, axis=0), ord=2, axis=0), ord=2, axis=0), ord=2)
    
    

def counterExampleVariableRatio():
    
    #[ Nobody, Random, Myopic, Whittle]
    badfs={0:[166.05, 167.27,167.62,167.71],
           10:[153.56, 155.50, 155.14, 164.35],
           20:[135.47, 139.03, 137.05, 147.85],
           30:[],
           40:[]}
    
    lol=0
    badf_performance={}
    for N in [10, 20, 100]:
        badf_performance[N]={}
        for kp in [5, 10]:
            badf_performance[N][kp]={}
            for badf in [10*i for i in range (10)]:
                badf_performance[N][kp][badf]={}
                for p in [0,2,3,5]:
                    #badf_performance[N][kp][badf][policy]=value
                    k=int((kp*N)/100)
                    seed_value_matrices=glob.glob('/Users/adityamate/Harvard/phd/projects/tb_bandits/adherence_log/*_workshop*N%d_k%d*policy%d*badf%d.0_s*'%(N,k,p,badf))
                    badf_performance[N][kp][badf][p]=np.mean(np.array([np.mean(np.sum(np.load(file), axis=1)) for file in seed_value_matrices]))
                
                    print (lol)
                    lol+=1
                    
                    #print ("Value1:", [np.mean(np.sum(np.load(file), axis=1)) for file in seed_value_matrices])
                    
    
    
    #adherence_TRIALworkshopExps_N20_k1_L180_policy0_datademo_badf10.0_s48_lr0.npy
    print (badf_performance)
    
    fig, ax =plt.subplots(1,2, figsize=(9,4))
    
    
    x= [10*i for i in range(10)]
    
    y1_N20_k10_p0= [0 for i in range(10)]
    y1_N20_k10_p2= [100*(badf_performance[20][10][10*i][2]-badf_performance[20][10][10*i][0])/(badf_performance[20][10][10*i][5]-badf_performance[20][10][10*i][0]) for i in range(9,-1,-1)]
    y1_N20_k10_p3= [100*(badf_performance[20][10][10*i][3]-badf_performance[20][10][10*i][0])/(badf_performance[20][10][10*i][5]-badf_performance[20][10][10*i][0]) for i in range(9,-1,-1)]
    y1_N20_k10_p5= [100*(badf_performance[20][10][10*i][5]-badf_performance[20][10][10*i][0])/(badf_performance[20][10][10*i][5]-badf_performance[20][10][10*i][0]) for i in range(9,-1,-1)]

    y2_N100_k5_p0= [100*(badf_performance[100][5][10*i][0]-badf_performance[100][5][10*i][0])/(badf_performance[100][5][10*i][5]-badf_performance[100][5][10*i][0]) for i in range(9,-1,-1)]
    y2_N100_k5_p2= [100*(badf_performance[100][5][10*i][2]-badf_performance[100][5][10*i][0])/(badf_performance[100][5][10*i][5]-badf_performance[100][5][10*i][0]) for i in range(9,-1,-1)]
    y2_N100_k5_p3= [100*(badf_performance[100][5][10*i][3]-badf_performance[100][5][10*i][0])/(badf_performance[100][5][10*i][5]-badf_performance[100][5][10*i][0]) for i in range(9,-1,-1)]
    y2_N100_k5_p5= [100*(badf_performance[100][5][10*i][5]-badf_performance[100][5][10*i][0])/(badf_performance[100][5][10*i][5]-badf_performance[100][5][10*i][0]) for i in range(9,-1,-1)]
    
    ax[0].plot(x, y1_N20_k10_p0, label='No calls')
    ax[0].plot(x, y1_N20_k10_p2, 'o-', label='Random')
    ax[0].plot(x, y1_N20_k10_p3, 's-',label='Myopic')
    ax[0].plot(x, y1_N20_k10_p5, '^-',label='Whittle')
    
    ax[1].plot(x, y2_N100_k5_p0, label='No calls')
    ax[1].plot(x, y2_N100_k5_p2, 'o-',label='Random')
    ax[1].plot(x, y2_N100_k5_p3, 's-',label='Myopic')
    ax[1].plot(x, y2_N100_k5_p5, '^-',label='Whittle')
    
    #ax[0].set(xlabel=r'$\delta_2$',  ylabel='Intervention benefit (\%)' )
    ax[0].set(xlabel=r'Fraction of self-correcting patients', ylabel=r'Intervention benefit (\%)')
    ax[1].set(xlabel=r'Fraction of self-correcting patients')
    ax[0].set_title(r'$N=20, k=2$')
    ax[1].set_title(r'$N=100, k=5$')

    plt.legend(loc='lower center', bbox_to_anchor=(-0.1, 1.07), ncol=4)
    #plt.tight_layout()
    plt.savefig('./img/selfCorrecting.pdf')
    
    
    plt.show()
    
    return

def perturbPlots():
    
    x4=[0.01,0.05,0.09,0.13,0.17,0.25,0.4]
    yr4=[4.28,4.56,4.89,5.29,5.65,6.66,8.86]
    ym4=[9.09,9.36,9.51,10.12,11.13,12.76,16.19]
    yw4=[13.7,14.09,14.78,15.34,16.23,18.19,21.49]
    
    x2=[0.01,	0.05,	0.09,	0.13,	0.17,	0.25,	0.4]
    yr2=[8.07,8.07,8.08,8.08,8.09,8.11,8.12]
    ym2=[14.1,14.2,14.19,14.27,14.25,14.41,14.73]
    yw2=[19.74,19.96,20.02,20.2,20.05,20.08,20.32]
    
    
    x3=[0.01,0.05,0.09,0.13,0.17,0.25,0.4]
    yr3=[1.37,1.96,2.36,2.77,3.19,4.18,5.82]
    ym3=[ 5.07,6.2,6.6,6.22,7.52,7.1,8.37]
    yw3=[7.63,9.28,10.22,11,11.76,13.44,15.58]
    
    
    x1=[0.01,0.03,0.05,0.07,0.09,0.13,0.17,0.2,0.25,0.3,0.4]
    yr1=[1.12,1.23,1.37,1.52,1.66,1.88,2.04,2.18,2.41,2.58,2.98]
    ym1=[2.62,2.31,3.63,4.41,5,5.87,6.33,6.59,7.08,7.54,8.37]
    yw1=[5.81,6.26,6.73,7.41,7.75,8.51,8.95,9.2,9.56,9.87,10.41]
    
    fig, ax =plt.subplots(1,4, figsize=(16,3.5))
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    
    percentyr1=[(100*yr1[i])/yw1[i] for i in range(len(yr1))]
    percentym1=[(100*ym1[i])/yw1[i] for i in range(len(yr1))]
    percentyw1=[(100*yw1[i])/yw1[i] for i in range(len(yr1))]
    
    percentyr2=[(100*yr2[i])/yw2[i] for i in range(len(yr2))]
    percentym2=[(100*ym2[i])/yw2[i] for i in range(len(yr2))]
    percentyw2=[(100*yw2[i])/yw2[i] for i in range(len(yr2))]

    percentyr3=[(100*yr3[i])/yw3[i] for i in range(len(yr3))]
    percentym3=[(100*ym3[i])/yw3[i] for i in range(len(yr3))]
    percentyw3=[(100*yw3[i])/yw3[i] for i in range(len(yr3))]

    percentyr4=[(100*yr4[i])/yw4[i] for i in range(len(yr4))]
    percentym4=[(100*ym4[i])/yw4[i] for i in range(len(yr4))]
    percentyw4=[(100*yw4[i])/yw4[i] for i in range(len(yr4))]

    
    #fig.suptitle('Intervention benefit as % of Threshold Whittle for varying perturbation', fontsize=14)
    #fig4=plt.figure(figsize=(12,12))

    ax[3].plot(x4, percentyr4, '^-',label='Random', color=policy_colors[4])
    ax[3].plot(x4, percentym4, '^-',label='Myopic', color=policy_colors[3])
    ax[3].plot(x4, percentyw4, '^-',label='Whittle', color=policy_colors[2])
    ax[3].set(xlabel=r' Perturbation, $\delta_4$')
    
    
    
    ax[2].plot(x3, percentyr3, '-',label='Random', color=policy_colors[4])
    ax[2].plot(x3, percentym3, '.-',label='Myopic', color=policy_colors[3])
    ax[2].plot(x3, percentyw3, 's-',label='Threshold Whittle',color=policy_colors[2])
    ax[2].set(xlabel=r' Perturbation, $\delta_3$')


    ax[1].plot(x2, percentyr2, '^-',label='Random', color=policy_colors[4])
    ax[1].plot(x2, percentym2, '^-',label='Myopic', color=policy_colors[3])
    ax[1].plot(x2, percentyw2, '^-',label='Whittle', color=policy_colors[2])
    ax[1].set(xlabel=r' Perturbation, $\delta_2$')
    
    
    
    ax[0].plot(x1, percentyr1, '-',label='Random', color=policy_colors[4])
    ax[0].plot(x1, percentym1, '.-',label='Myopic', color=policy_colors[3])
    ax[0].plot(x1, percentyw1, 's-',label='Threshold Whittle',color=policy_colors[2])
    ax[0].set(xlabel=r' Perturbation, $\delta_1$', ylabel='Intervention benefit (\%)' )

    '''
    ax[0].plot(x2, percentyr2, 'o-',label='Random')
    ax[0].plot(x2, percentym2, '^-',label='Myopic')
    ax[0].plot(x2, percentyw2, 's-',label='Threshold Whittle')
    ax[0].set(xlabel=r'$\delta_2$',  ylabel='Intervention benefit (\%)' )
    '''
    
    """
    ax[1][1].plot(x1, percentyr1, '^-',label='Random')
    ax[1][1].plot(x1, percentym1, '^-',label='Myopic')
    ax[1][1].plot(x1, percentyw1, '^-',label='Whittle')
    ax[1][1].set(xlabel='delta_1')
    """
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(reversed(handles), reversed(labels),loc='lower center', bbox_to_anchor=(0, 1.01), ncol=3)
    #plt.tight_layout()
    #plt.savefig('./img/perturbation.png')
    
    
    plt.show()
    '''
    plt.xlabel("delta 4")
    plt.ylabel("Increase in adherence: (Policy - Nobody)")
    plt.title("Effect of perturbating parameter 4" )
    plt.legend()
    plt.show()
    '''
    
    '''
    fig3=plt.figure()
    plt.plot(x3, yr3, '',label='Random')
    plt.plot(x3, ym3, label='Myopic')
    plt.plot(x3, yw3, label='Whittle')
    plt.xlabel("delta 3")
    plt.ylabel("Increase in adherence: (Policy - Nobody)")
    plt.title("Effect of perturbating parameter 3" )
    plt.legend()
    plt.show()
    
    
    fig2=plt.figure()
    plt.plot(x2, yr2, '',label='Random')
    plt.plot(x2, ym2, label='Myopic')
    plt.plot(x2, yw2, label='Whittle')
    plt.xlabel("delta 2")
    plt.ylabel("Increase in adherence: (Policy - Nobody)")
    plt.title("Effect of perturbating parameter 2" )
    plt.legend()
    plt.show()
    
    fig1=plt.figure()
    plt.plot(x1, yr1, '',label='Random')
    plt.plot(x1, ym1, label='Myopic')
    plt.plot(x1, yw1, label='Whittle')
    plt.xlabel("delta 1")
    plt.ylabel("Increase in adherence: (Policy - Nobody)")
    plt.title("Effect of perturbating parameter 1" )
    plt.legend()
    plt.show()
    
    '''


def selfCorrectingPlots(folder='', sv='', save=False, fig0=None, ax0=None, K_PERCENT=None, return_early=False):
    '''
    Generates  performance comparison plots for varying values of fraction of self correcting patients
    
    '''
    
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et al.',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle',
           15: 'roundRobin', 16:'new_whittle',  17: 'fast_whittle', 
           18: 'buggyWhittle'}

    policies=[0,2,3,5,10,14]
    policies_to_show=[14,10,5,3,2]
    fracs_to_show=[2,4,6,8,10]
    
    fracs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data='demo'
    if sv=='':
        sv='selfcorrecting'
    if folder=='':
        folder='selfcorrecting'



    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    plot_dict={}
    badf=0.4
    ratio=0
    if K_PERCENT is None:
        K_PERCENT = 10
    N=200
    k=int(N*K_PERCENT/100.)
    lr=0
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(policies) > len(policy_colors):
        raise ValueError("Error: More policies than available colors. Add more colors")

    for p in policies:
        print("Reading policy...: ", p)
        runtimes=[]
        adherences=[]
        for frac in fracs:
        
            runtime_files=glob.glob('../logs/runtime/%s/*_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder, sv, frac, k, p, data, badf, lr))
            adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_frac_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, p,data, badf, lr))
            adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_frac_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 0,data, badf, lr))
            adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_frac_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 14,data, badf, lr))
            print ("seeds for frac=",frac,":", len(runtime_files), len(adherence_files))
            
            runtimes.append(np.log10(np.mean([np.load(f) for f in runtime_files])))
            adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
            
        
        plot_dict[p]={'runtime': runtimes,
                         'adherence':adherences
                    }
    """   
    Plotting code below:
    """
    plot_both_graphs=True
    if plot_both_graphs:
        if fig0 is None or ax0 is None:
            fig0, ax0=plt.subplots(1,1, figsize=(4,3))

        fig1, ax1=plt.subplots(1,1, figsize=(4,3))
        fig, ax = plt.subplots(1,2, figsize=(9,4))
        symbols=['o-','s-','^-', '.-','-']
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        for ind_p, p in enumerate(policies_to_show):
            #print ("Policy:", p, "Adherence is ", plot_dict[p]['adherence'])
            
            ax[0].plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax[0].plot(fracs, (plot_dict[p]['adherence'][fr], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            ax0.plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax0.plot(fracs, (plot_dict[p]['adherence'][fr], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        values=[100*(plot_dict[p]['adherence'][3]-plot_dict[0]['adherence'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        bottom=0
        #print ("VAL:", values)
        #print (width)
        #print (x)
        
        x = np.arange(len(fracs_to_show))
        for pidx, p in enumerate((policies[1:])):
            
            values=[100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            rects1 = ax[1].bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            #rects1 = ax.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            rects11 = ax1.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            
        if return_early:
            return
        
        
        ax[1].set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        ax1.set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        #ax[0].set_ylabel(" Adherence out of 180", fontsize=16)
        ax[0].set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax0.set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax[0].set_xlabel(" Fraction of self-correcting agents", fontsize=16)
        ax0.set_xlabel(" Fraction of self-correcting agents", fontsize=16)
        ax[1].set_xlabel(" Fraction of self-correcting agents", fontsize=16)           #Uncomment: 
        ax1.set_xlabel(" Fraction of self-correcting agents", fontsize=16)           #Uncomment: 
        #ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax.set_title("Intervention Benefit comparison", fontsize=14)                #Uncomment: 
        ax[1].set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax1.set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax[1].set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        ax1.set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        #ax[1].set_xticklabels(labels, rotation=30)
        ax[1].set_yticks([0,50,100])                                                   #Uncomment: 
        ax1.set_yticks([0,50,100])                                                   #Uncomment: 
        #ax[1].legend(fontsize=14)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)

    def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax[1].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    
    #autolabel(rects1)       
    plt.tight_layout()
    #save=True
    if save:
        plt.savefig('../img/selfcorrecting/selfcorrecting_N%s_kp_%s.png'%(200, K_PERCENT))
    plt.show()



def quickCheckFractionPlots(folder='', sv='', save=False):
    '''
    Generates  performance comparison plots for varying values of fraction of patients satisfyinf quick check for fixed
    number of threshold optimal patients
    
    '''
    
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et al.',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle',
           15: 'roundRobin', 16:'new_whittle',  17: 'fast_whittle', 
           18: 'buggyWhittle'}

    policies=[0,2,3,5,10,14]
    policies_to_show=[14,10,5,3,2]
    fracs_to_show=[2,4,6,8,10]
    
    fracs=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data='uniform'
    if sv=='':
        sv='quick_frac_unif_threshfrac_80'
    if folder=='':
        folder='quickcheckfraction'



    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    plot_dict={}
    badf=0.4
    ratio=0
    K_PERCENT = 10
    N=200
    k=int(N*K_PERCENT/100.)
    lr=0
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(policies) > len(policy_colors):
        raise ValueError("Error: More policies than available colors. Add more colors")

    for p in policies:
        print("Reading policy...: ", p)
        runtimes=[]
        adherences=[]
        for frac in fracs:
        
            runtime_files=glob.glob('../logs/runtime/%s/*_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder, sv, frac, k, p, data, badf, lr))
            adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_quickfrac_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, p,data, badf, lr))
            print ("seeds for frac=",frac,":", len(runtime_files), len(adherence_files))
            
            runtimes.append(np.log10(np.mean([np.load(f) for f in runtime_files])))
            adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
            
        
        plot_dict[p]={'runtime': runtimes,
                         'adherence':adherences
                    }
    """   
    Plotting code below:
    """
    plot_both_graphs=True
    if plot_both_graphs:
        fig0, ax0=plt.subplots(1,1, figsize=(4,3))
        fig1, ax1=plt.subplots(1,1, figsize=(4,3))
        fig, ax = plt.subplots(1,2, figsize=(9,4))
        symbols=['o-','s-','^-', '.-','.-','-']
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        for ind_p, p in enumerate(policies_to_show):
            #print ("Policy:", p, "Adherence is ", plot_dict[p]['adherence'])
            
            ax[0].plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax[0].plot(fracs, (plot_dict[p]['adherence'][fr], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            ax0.plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax0.plot(fracs, (plot_dict[p]['adherence'][fr], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        values=[100*(plot_dict[p]['adherence'][3]-plot_dict[0]['adherence'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        bottom=0
        #print ("VAL:", values)
        #print (width)
        #print (x)
        
        x = np.arange(len(fracs_to_show))
        for pidx, p in enumerate((policies[1:])):
            
            values=[100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            rects1 = ax[1].bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            #rects1 = ax.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            rects11 = ax1.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            
        
        
        
        ax[1].set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        ax1.set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        #ax[0].set_ylabel(" Adherence out of 180", fontsize=16)
        ax[0].set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax0.set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax[0].set_xlabel(" Fraction of agents verifiable using quick check", fontsize=16)
        ax0.set_xlabel(" Fraction of agents verifiable using quick check", fontsize=16)
        ax[1].set_xlabel(" Fraction of agents verifiable using quick check", fontsize=16)           #Uncomment: 
        ax1.set_xlabel(" Fraction of agents verifiable using quick check", fontsize=16)           #Uncomment: 
        #ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax.set_title("Intervention Benefit comparison", fontsize=14)                #Uncomment: 
        ax[1].set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax1.set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax[1].set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        ax1.set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        #ax[1].set_xticklabels(labels, rotation=30)
        ax[1].set_yticks([0,50,100])                                                   #Uncomment: 
        ax1.set_yticks([0,50,100])                                                   #Uncomment: 
        #ax[1].legend(fontsize=14)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)

    def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax[1].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    #autolabel(rects1)       
    plt.tight_layout()
    #save=True
    if save:
        plt.savefig('../img/quickcheckfraction/thresholdoptfrac_80_N%s_kp_%s.png'%(200, K_PERCENT))
    plt.show()


def syntheticRangePlots(folder='', sv='', save=False, 
                        fracs=None, data=None, fig0=None, ax0=None, return_early=False , policies_to_show=[]):
    '''
    Generates  performance comparison plots for varying values of fraction of patients satisfyinf quick check for fixed
    number of threshold optimal patients
    
    '''
    
    pname={0: 'nobody',     1: 'everyday',     2: 'Random',
           3: 'Myopic',     4: 'optimal',      5: 'Threshold Whittle',
           6: '2-day',      7: 'oracl_m',      8: 'Oracle',
           9: 'despot',     10: 'Qian et al.',       11: 'naiveBelief',
           12: 'naiveReal', 13: 'naiveReal2',  14: 'oracle',
           15: 'roundRobin', 16:'new_whittle',  17: 'fast_whittle', 
           18: 'buggyWhittle'}

    policies=[0,2,3,5,10,14][:]
    if policies_to_show==[]:
        policies_to_show=[14,10,5,3,2][:]
    fracs_to_show=[1,3,5,7,8][:]
    
    #fracs=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9][1:]
    if fracs is None:
        fracs=[10,20,30,40,50,60,70,80,90,100]
    if data is None:
        data='entropy'
    #data='simulated'
    if sv=='':
        sv='entropyfraction2'
        #sv='syntheticrange2'
    if folder=='':
        folder='entropyfraction2'
        #folder='syntheticrange2'



    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    plot_dict={}
    badf=0.4
    ratio=0
    K_PERCENT = 10
    N=200
    k=int(N*K_PERCENT/100.)
    lr=0
    policy_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(policies) > len(policy_colors):
        raise ValueError("Error: More policies than available colors. Add more colors")

    for p in policies:
        print("Reading policy...: ", p)
        runtimes=[]
        adherences=[]
        for frac in fracs:
        
            runtime_files=glob.glob('../logs/runtime/%s/*_%s_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder, sv, frac, k, p, data, badf, lr))
            adherence_files=glob.glob('../logs/adherence_log/%s/adherence_%s_baseprob_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, p,data, badf, lr))
            adherence_files_0=glob.glob('../logs/adherence_log/%s/adherence_%s_baseprob_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 0,data, badf, lr))
            adherence_files_14=glob.glob('../logs/adherence_log/%s/adherence_%s_baseprob_%s_N200_k%s_L180_policy%s_data%s_badf%s_s*_lr%s_*.npy'%(folder,sv, frac, k, 14,data, badf, lr))
            print ("seeds for frac=",frac,":", len(runtime_files), len(adherence_files))
            
            runtimes.append(np.log10(np.mean([np.load(f) for f in runtime_files])))
            adherences.append(np.mean([np.mean(np.sum(np.load(f), axis=1)) for f in adherence_files]))
            
        
        plot_dict[p]={'runtime': runtimes,
                         'adherence':adherences
                    }
        plot_dict[14]={'runtime': runtimes,
                         'adherence':adherences
                    }
    """   
    Plotting code below:
    """
    plot_both_graphs=True
    if plot_both_graphs:
        if fig0 is None or ax0 is None:
            fig0, ax0=plt.subplots(1,1, figsize=(4,3))
        fig1, ax1=plt.subplots(1,1, figsize=(4,3))
        fig, ax = plt.subplots(1,2, figsize=(9,4))
        symbols=['o-','s-','^-', '.-','-']
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        for ind_p, p in enumerate(policies_to_show):
            #print ("Policy:", p, "Adherence is ", plot_dict[p]['adherence'])
            
            ax[0].plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax[0].plot(fracs, (plot_dict[p]['adherence'][fr], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            ax0.plot(fracs, [100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr]) for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            #ax0.plot(fracs, [plot_dict[p]['adherence'][fr] for fr in range(len(fracs))], symbols[ind_p], color=policy_colors[ind_p], linestyle='-', label=pname[p])
            
        
        x = np.arange(len(policies[1:]))  # the label locations
        width = 0.15  # the width of the bars
        #rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
        labels=[pname[p] for p in policies[1:]]
        #values=[100*(plot_dict[p]['adherence'][3]-plot_dict[0]['adherence'][3])/(plot_dict[14]['adherence'][3]-plot_dict[0]['adherence'][3]) for p in policies[1:]]
        bottom=0
        #print ("VAL:", values)
        #print (width)
        #print (x)
        
        x = np.arange(len(fracs_to_show))
        for pidx, p in enumerate((policies[1:])):
            print("Policy: ", p, len(plot_dict[0]['adherence']),len(plot_dict[14]['adherence']),len(plot_dict[p]['adherence']), fracs_to_show)
            values=[100*(plot_dict[p]['adherence'][fr]-plot_dict[0]['adherence'][fr])/(plot_dict[14]['adherence'][fr]-plot_dict[0]['adherence'][fr])  for fr in fracs_to_show]
            rects1 = ax[1].bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            #rects1 = ax.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            rects11 = ax1.bar(x+pidx*width, values, width, bottom=bottom, color=colors[pidx],label=pname[p], edgecolor='black')
            
        
        if return_early:
            return
        
        ax[1].set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        ax1.set_ylabel(" Intervention benefit (\%)", fontsize=16 )                    #Uncomment:
        #ax[0].set_ylabel(" Adherence out of 180", fontsize=16)
        ax[0].set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax0.set_ylabel(" Intervention benefit (\%)", fontsize=16)
        ax[0].set_xlabel("p0, p11 (passive), p01, p11 (active)", fontsize=16)
        ax0.set_xlabel(" p0, p11 (passive), p01, p11 (active)", fontsize=16)
        ax[1].set_xlabel(" Base prob range for p01, p11 (passive)", fontsize=16)           #Uncomment: 
        ax1.set_xlabel(" Base prob range for p01, p11 (passive)", fontsize=16)           #Uncomment: 
        #ax[0].set_title("Runtime comparison", fontsize=14)  
        #ax.set_title("Intervention Benefit comparison", fontsize=14)                #Uncomment: 
        ax[1].set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax1.set_xticks([i+2*width for i in range(len(fracs_to_show))])               #Uncomment: 
        ax[1].set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        ax1.set_xticklabels([str(10*fr)+"\%" for fr in fracs_to_show], rotation=0)   #Uncomment: 
        #ax[1].set_xticklabels(labels, rotation=30)
        ax[1].set_yticks([0,50,100])                                                   #Uncomment: 
        ax1.set_yticks([0,50,100])                                                   #Uncomment: 
        #ax[1].legend(fontsize=14)
        ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax0.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax[1].legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=12)

    def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 0)
                ax[1].annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    
    #autolabel(rects1)       
    plt.tight_layout()
    #save=True
    if save:
        plt.savefig('../img/syntheticrange/syntheticrange_N%s_kp_%s.png'%(200, K_PERCENT))
    plt.show()
   
def counterExample():

    values=[56.0,9.0,100.0,100.0]
    labels=['Random', 'Myopic', 'Threshold Whittle','', 'Optimal']
    fig, ax=plt.subplots(1,1, figsize=(4,3))
    x = np.arange(len(values))
    width = 0.8
    bottom=0
    colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']
    
    rect1=ax.bar(x, values, width, label='intervention \n benefit as \%', edgecolor='black' )
        
    ax.set_xticks([0,1,1.6,2,3])
    ax.set_yticks([0,50,100])
    ax.set_xticklabels(labels, rotation=30, fontsize=16)

    ax.set_ylabel("Intervention benefit (\%)", fontsize=18)
    #ax.set_title("Intervention benefit comparison")   
    #ax.legend()
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=1)
    plt.savefig('./img/counterExample.png')
    plt.show()

def combiningFigs():
    
    fig, ax=plt.subplots(1,4, figsize=(18,4))
    
    selfCorrectingPlots(folder='', sv='', save=False, fig0=fig, ax0=ax[0], K_PERCENT=10, return_early=True)
    #selfCorrectingPlots(folder='', sv='', save=False, fig0=fig, ax0=ax[1], K_PERCENT=25, return_early=True)
    syntheticRangePlots(folder='entropyfraction2', sv='entropyfraction2', save=False, 
                        fracs=[10,20,30,40,50,60,70,80,90,100], data='entropy', fig0=fig, ax0=ax[2], return_early=True)
    syntheticRangePlots(folder='syntheticrange2', sv='syntheticrange2', save=False, 
                        fracs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], data='simulated', fig0=fig, ax0=ax[1], return_early=True, policies_to_show=[14,10,5,3])
    threshOptFracPlots(folder='thresoptExps_bf_unif', sv='thresoptExps_bf_unif_threshopt_frac', fig1=fig,ax1=ax[3], return_early=True)
    

    ax[0].set_ylabel(" Intervention benefit (\%)", fontsize=22)
    
    ax[0].set_xlabel(" Self-correcting fraction", fontsize=18)
    ax[2].set_xlabel(" Fraction of bad-state-preferring processes", fontsize=18)
    ax[1].set_xlabel(r'Range of $P_{01}^p, P_{11}^p, P_{01}^a, P_{11}^a$', fontsize=18)
    ax[3].set_xlabel(" Threshold optimal fraction", fontsize=18)
    
    ax[0].set_title('(a)')
    ax[2].set_title('(c)')
    ax[1].set_title('(b)')
    ax[3].set_title('(d)')
    
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(reversed(handles), reversed(labels), loc='lower right', bbox_to_anchor=(2, 1.15), ncol=3, fontsize=16)
    ax[3].legend(loc='lower left', bbox_to_anchor=(-1.02,1.15), ncol=3, fontsize=16)
    #return fig,ax
    plt.tight_layout()
    plt.show(fig)

    
    
if __name__=="__main__":
    
    '''
    l=np.array([10,30,40,30,20,5])
    
    returnKGreatestIndices(l, 7)
    print (l)
    d={1:1, 2:1, 2:3}
    
    
    '''
    
    # labels = ['Call nobody', 'k Random', 'k Myopic', 'Everybody']
    # values = [68.6, 98.39, 111.87, 149.29]
    # barPlot(labels, values)

    # Example command:
    # python3 utils.py results/*N10_k2_trials5_datareal_s28

    #infile_prefix = sys.argv[1]
    #groupedBarPlot(infile_prefix, title='10 patients, k=2, 5 trials')

    
    runTimePlots()
    #threshOptFracPlots(folder='thresoptExps_bf_unif', sv='thresoptExps_bf_unif_threshopt_frac')
    #kPlots()
    #selfCorrectingPlots()
    #quickCheckFractionPlots()
    #syntheticRangePlots()
    #combiningFigs()
    #perturbPlots()
    #counterExample()
    #counterExampleVariableRatio()
    
    