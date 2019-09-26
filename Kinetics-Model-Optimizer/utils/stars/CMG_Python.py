# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:37:06 2019
CMG DAT files generator for different heat loss and air flow rates
@author: yunanli
This file is updated 2019/8/13
2019/8/13: new options for ignition criteria (energy reacted rate)
2019/8/22: collect energy reacted rate data even using criteria 1 (temperature)
2019/8/28: Functions to generate our expected CMG STARS model to match with experimental data.
2019/9/5: add the features to output both ignition and extinction cases, and the rate of net reacted energy profile
2019/9/18: temporary solution to the unexpected error in os.remove function. Jump the error if winerr05 shows, permission denied

Attention:
    1. the K table and initial pressure of the option 1 need to be changed
"""

import numpy as np
import struct

#import makeVectors as mV
import sys
import os


class Component():
    '''
    Component for writing STARS runfile

    Fluid options:
        1 - Morten's setup
        2 - Murat's setup


    Attributes: 
        Option 1:

        Option 2: 

    '''

    def __init__(self, fluid_option=None, **kwargs):
        if fluid_option is None:
            raise Exception('Must enter a valid value for fluid_option.')
        self.__dict__.update(kwargs)


class Reaction():
    '''
    Reaction definition for writing STARS runfile

    '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def print_rxn(self, filename):

        print('** {}'.format(self.NAME), fid=filename)
        attrs_list = ['STOREAC', 'STOPROD', 'RORDER', 'FREQFAC', 'EACT', 'RENTH']
        
        for attr in attrs_list:
            if hasattr(self, attr):
                print_ln = '*{}'.format(attr)
                if isinstance(getattr(self, attr), list):
                    print_ln += ' '.join(str(x for x in getattr(self, attr)))
                else:
                    print_ln += str(getattr(self, attr))
            print(print_ln, fid=filename)
    
    
def delete_dat_file(delete_dat_file_path, dat_file_name):
    '''
    Example:
        delete_dat_file_path = 'C:\\Users\\yunanli\\Desktop\\CMG\\'
    '''
    extensions = ['.dat', '.irf', '.mrf', '.out', '.sr3']
    del_lines = [delete_dat_file_path + dat_file_name + ext for ext in extensions]

    for l in del_lines:
        try:
            os.remove(l)
        except OSError as e:
            print(e)

    
def copy_up_to_string(fid, fileID, i, flag_string):
    '''
    Copies contents of fid up to where it sees 'flag_string'
    '''
    flag = True
    while flag:
        print(fid[i], file = fileID, end="")
        i+=1
        if not fid[i-1].split():
            flag = True
        else:
            if fid[i-1].split()[0] == flag_string:
                flag = False
            else:
                flag = True
    return i


def print_array(fileID, start_text, vec):
    '''
    Prints:
        start_text vec[0] vec[1] ...
    
    '''
    print(start_text, file = fileID, end = "")
    for j in range(vec.shape[0]):
        print('{m}  '.format(m=vec[j]), file=fileID, end="")
    print('\n', file=fileID, end="")


def print_sat(fid, fileID, i, sat):
    '''
    Print saturations
    '''
    line_split = fid[i].split()
    for word in line_split[:-1]:
        print(word, file=fileID, end="")
        print('  ', file=fileID, end="")
    print(str(sat), file = fileID)
    return i+1


def print_thermal_conductivity(fid, fileID, i, thermocond):
    ''' 
    Prints:
            heat loss defined lines. 
    This function is required when changing the heat loss properties
    '''
    print('*HLOSSPROP *OVERBUR 35 ', str(thermocond), file=fileID)
    print('*HLOSSPROP *UNDERBUR 35 ', str(thermocond), file=fileID)
    return i+2
    
    
def print_air_rate(fid, fileID, i, airrate):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('OPERATE  STG  ', str(airrate), file=fileID)
    return i+1


def print_inject_temperature(fid, fileID, i, inject_temp):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('TINJW  ', str(inject_temp), file=fileID)
    return i+1


def print_BHP(fid, fileID, i, BHP):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('OPERATE  MIN  BHP  ', str(BHP), '  CONT REPEAT', file=fileID)
    return i+1


def print_UHTR(fid, fileID, i, UHTR):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('*UHTR   *IJK 1 1 1:12  ', str(UHTR), file=fileID)
    return i+1


def print_HEATR(fid, fileID, i, HEATR):
    print('*HEATR  *IJK 1 1 1:12  ', str(HEATR), file=fileID)
    return i+1
    

def print_HLOSSPROP(fid, fileID, i, HLOSSPROP):
    print_strs = ['-I', '+I', '-J', '+J', '-K', '+K']
    for s in print_strs:
        print('*HLOSSPROP *{} 764.37  '.format(s), str(HLOSSPROP), file=fileID)
    return i+6
            
        
"""
Created on Fri Aug 23 15:45:20 2019

@author: liyunan
Functions to generate our expected CMG STARS model to match with experimental data.

Option = 1: means using the Kristensen's minimal model system
Option = 2: means using the Murat's RTO base case
Option = 3: means using the Bo's RTO base case

"""

def print_heating_rate(fileID, HR, unit_option):
    """
    option = 1: Kristensen's VKC system
    option = 2: Murat's VKC system
    option = 3: Folake's VKC system
    """
    print('** ==========Linear Ramped Heating==========', file=fileID)
    if unit_option == 1:
        for t in range(499):
            if (293.15 + HR * t) > 1023.15:
                print('*TIME ', str((t+1)/60), file = fileID)
                print('*TMPSET *IJK 1 1 1:12 ', str(1023.15), file = fileID)
                print("*INJECTOR 'INJECTOR'", file = fileID)
                print('*TINJW ', str(1023.15), file = fileID)
            else:
                print('*TIME ', str((t+1)/60), file = fileID)
                print('*TMPSET *IJK 1 1 1:12 ', str(293.15 + HR * t), file = fileID)
                print("*INJECTOR 'INJECTOR'", file = fileID)
                print('*TINJW ', str(293.15 + HR * t), file = fileID)
        print('*TIME  500', file = fileID)   
        
    if unit_option == 2:
        for t in range(499):
            if (20 + HR*t) > 750:
                print('*TIME ', str(t+1), file = fileID)
                print('*TMPSET *IJK 2 1 1:11 ', str(750), file = fileID)
                print("*INJECTOR 'INJE'", file = fileID)
                print('*TINJW ', str(750), file = fileID)
            else:
                print('*TIME ', str(t+1), file = fileID)
                print('*TMPSET *IJK 2 1 1:11 ', str(20 + HR*t), file = fileID)
                print("*INJECTOR 'INJE'", file = fileID)
                print('*TINJW ', str(20 + HR*t), file = fileID)
        print('*TIME  500', file = fileID)
        
    
    if unit_option == 3:
        for t in range(249):
            if (20 + HR*2*t) > 750:
                print('*TIME ', str(2*(t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1 ', str(750), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(750), file = fileID)
            else:
                print('*TIME ', str(2*(t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1 ', str(20 + HR*2*t), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(20 + HR*2*t), file = fileID)
        print('*TIME  500', file = fileID)
        
    if unit_option == 4:
        for t in range(499):
            if (20 + HR*t) > 750:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1 ', str(750), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(750), file = fileID)
            else:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1 ', str(20 + HR*t), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(20 + HR*t), file = fileID)
        print('*TIME  500', file = fileID)
        
        
    print('*STOP', file = fileID)


def K_value_table(K_option, fileID):

    print("""
*KV1 0 0 
*KV2 0 0 
*KV3 0 0 
*KV4 0 0 
*KV5 0 0 

              """, file = fileID)


def print_fluid(fileID, fluid_option, Components, Kinetics):
    print('**  ==============  FLUID DEFINITIONS  ======================', file = fileID)

    # Print fluid
    comp_names = Components.keys()

    def print_attrs(attrslist):
        for attr in attrslist:
            print_str='*' + attr
            for comp in comp_names:
                if hasattr(Components[comp], attr):
                    if attr =='NAME':
                        print_str+= '\'' + str(*getattr(Components[comp], attr)) + '\''
                    else:
                        print_str+= str(*getattr(Components[comp], attr))
            print(print_str, fid=fileID)

    if fluid_option == 1:
        print("""
*MODEL 6 5 3   ** Number of noncondensible gases is numy-numx = 2
** Number of solid components is ncomp-numy = 1
              """, file = fileID)
        
        comp_attrs = ['NAME', 'CMM', 'PCRIT', 'TCRIT', 'ACEN', 'AVG', 'BVG', 'AVISC', 'BVISC', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'MOLDEN', 'CP', 'CT1']
        print_attrs(comp_attrs)

        print('*SOLID_DEN ',"\'"+Components["COKE"].NAME+"\'",*Components["WATER"].SOLID_DEN,*Components["HEVY OIL"].SOLID_DEN,*Components["LITE OIL"].SOLID_DEN,*Components["INRT GAS"].SOLID_DEN,*Components["OXYGEN"].SOLID_DEN,*Components["COKE"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["COKE"].NAME+"\'",*Components["WATER"].SOLID_CP,*Components["HEVY OIL"].SOLID_CP,*Components["LITE OIL"].SOLID_CP,*Components["INRT GAS"].SOLID_CP,*Components["OXYGEN"].SOLID_CP,*Components["COKE"].SOLID_CP, file = fileID)

        
    elif fluid_option == 2:
        print("""
*MODEL 9 6 2 
** Number of noncondensible gases is numy-numx = 4
** Number of solid components is ncomp-numy = 4

              """, file = fileID)
        
        comp_attrs1 = ['NAME', 'CMM', 'PCRIT', 'TCRIT', 'AVG', 'BVG', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'CPL1', 'CPL2']
        print_attrs(comp_attrs1)
        
        print("""
*HVAPR 0 0 
*EV 0 0 
              """, file = fileID)
        
        comp_attrs2 = ['MASSDEN', 'CP', 'CT1', 'CT2'] # Add in 'AVISC', 'BVISC' if necessary
        print_attrs(comp_attrs2)

        print('*SOLID_DEN ',"\'"+Components["Coke1"].NAME+"\'",*Components["Coke1"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Coke1"].NAME+"\'",*Components["Coke1"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+Components["Coke2"].NAME+"\'",*Components["Coke2"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Coke2"].NAME+"\'",*Components["Coke2"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+Components["Ci"].NAME+"\'",*Components["Ci"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Ci"].NAME+"\'",*Components["Ci"].SOLID_CP, file = fileID)
        

    #### Print Reactions
    if fluid_option == 2:
        print("""
*VISCTABLE
10      0    100000
80      0      2084
100     0       580
1000    0         1
              """, file = fileID)
        
    print('**Reactions', file = fileID)
    print('**-----------', file = fileID)
    
    # Loop over reactions in list
    for r in Kinetics:
        r.print_rxn(fileID)

    if fluid_option == 2:
        print("*O2PP 'O2'", file = fileID)



def print_IO_control(fileID, IO_option):
    print('** ============== INPUT/OUTPUT CONTROL ======================', file=fileID)
    if IO_option == 1:
        print("""
RESULTS SIMULATOR STARS 201710
*INTERRUPT *STOP
*TITLE1 'STARS Test Bed No. 1'
*TITLE2 'Dry Combustion Tube Experiment'
*INUNIT *SI  *EXCEPT  2 0  ** K instead of degree C
	     *EXCEPT  1 1  ** hr instead of days
*OUTPRN *GRID *ALL
*OUTPRN *WELL *ALL
*WRST 1
*WPRN *GRID 1
*WPRN *ITER 1
OUTSRF WELL MOLE 
OUTSRF WELL COMPONENT ALL
OUTSRF GRID ALL
OUTSRF SPECIAL BLOCKVAR TEMP 1,1,2 
               matbal reaction 'OXYGEN'
               matbal current 'OXYGEN'
               matbal adsorbed 'OXYGEN'
               MOLEFRAC  'PRODUCER' 'OXYGEN' 
OUTSRF WELL MASS COMPONENT ALL
OUTSRF WELL MOLE COMPONENT ALL
OUTSRF WELL DOWNHOLE
              """, file = fileID)

        
    if IO_option == 2:
        print("""
RESULTS SIMULATOR STARS 201710
*TITLE1 'ENSAYO RTO 1.7 C/min - Core @ 1200 psi'
*TITLE2 'ICP-Stanford University'
*INTERRUPT 	*STOP
*INUNIT 	*LAB
*OUTUNIT 	*LAB
*OUTPRN *GRID *ALL
*OUTPRN *WELL *ALL
*WRST 1
*WPRN *GRID 5
*WPRN *ITER 5
*WSRF *GRID *TIME
*WSRF *WELL *TIME
*OUTSRF *WELL *MOLE *COMPONENT *ALL
*OUTSRF *SPECIAL MATBAL CURRENT 'O2'
*OUTSRF *SPECIAL MATBAL CURRENT 'Coke1'
*OUTSRF *SPECIAL MATBAL CURRENT 'CO'
*OUTSRF *SPECIAL MATBAL CURRENT 'CO2'
*OUTSRF *SPECIAL MATBAL CURRENT 'Coke2'
*OUTSRF *SPECIAL MATBAL CURRENT 'OIL'
*OUTSRF *SPECIAL MATBAL CURRENT 'H2O'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'N2'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'O2'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'H2O'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'CO'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'CO2'
*OUTSRF *SPECIAL MATBAL REACTION 'N2'
*OUTSRF *SPECIAL MATBAL REACTION 'O2'
*OUTSRF *SPECIAL MATBAL REACTION 'Coke1'
*OUTSRF *SPECIAL MATBAL REACTION 'CO'
*OUTSRF *SPECIAL MATBAL REACTION 'CO2'
*OUTSRF *SPECIAL MATBAL REACTION 'Coke2'
*OUTSRF *SPECIAL MATBAL REACTION 'OIL'
*OUTSRF *SPECIAL MATBAL REACTION 'H2O'
*OUTSRF *SPECIAL MATBAL REACTION ENERGY
*OUTSRF *SPECIAL AVGVAR TEMP
*OUTSRF GRID TEMP X Y SO SG SW VPOROS FPOROS MOLDENO MOLDENG MOLDENW SOLCONC MOLE
              
              """, file = fileID)
        


def print_grid(fileID, grid_option):
    print('**  ==============  GRID AND RESERVOIR DEFINITION  =================', file = fileID)
    if grid_option == 1:
        print("""
*GRID *CART 1 1 12  ** 12 blocks in the J direction (horizontal)

*DI *CON 0.111869
*DJ *CON 0.111869
*DK *CON 0.0093224

NULL CON  1

*POR *CON 0.4142
*PERMI *CON 12700
*PERMJ  EQUALSI
*PERMK  EQUALSI

PINCHOUTARRAY CON  1

*END-GRID
ROCKTYPE 1

*ROCKCP 2348300  0      **[J/m3*K]
**Unit for thermal conduc. [J/hr*K*m]
*THCONR 6231.6
*THCONW 2242.8
*THCONO 478.8
*THCONG 518.4
*THCONS 6231.6              

              """, file = fileID)
        
    if grid_option == 2:
        print("""
*GRID *RADIAL 2 1 11 *RW 0
*KDIR *Up

*DI   *IVAR 1.0668 0.600456
*DJ   *CON 360
*DK   *ALL 22*1
*NULL *CON 1
*POR  *IJK 1 1 1:11 0.4
           2 1 1:11 0.0

*PERMI *IVAR   30000 0.0 
*PERMJ *CON    30000
*PERMK *CON    30000
PINCHOUTARRAY CON 1
*END-GRID

** SAND
*ROCKTYPE 1
*ROCKCP 2.0375 0.0032
*THCONTAB
**T C thconr thconw thcono thcong thcons
15    0.2528 0.356245009 0.067746578 0.014840486 1.122770711
40    0.2528 0.376899404 0.06682453  0.015961606 1.159069446
65    0.2528 0.392542792 0.065902483 0.017063277 1.194559461
90    0.2528 0.403306893 0.064980436 0.018145781 1.229240756
115   0.2528 0.409323426 0.064058388 0.019209399 1.263113331
140   0.2528 0.410724108 0.063136341 0.020254411 1.296177186
165   0.2528 0.407640659 0.062214293 0.0212811   1.328432321
190   0.2528 0.400204798 0.061292246 0.022289747 1.359878735
215   0.2528 0.388548243 0.060370199 0.023280632 1.39051643
240   0.2528 0.372802713 0.059448151 0.024254037 1.420345405
265   0.2528 0.353099926 0.058526104 0.025210243 1.44936566
290   0.2528 0.329571602 0.057604057 0.026149532 1.477577195
315   0.2528 0.30234946  0.056682009 0.027072185 1.50498001
340   0.2528 0.271565217 0.055759962 0.027978483 1.531574105
365   0.2528 0.237350593 0.054837914 0.028868707 1.557359479
390   0.2528 0.199837307 0.053915867 0.029743138 1.582336134
415   0.2528 0.159157076 0.05299382  0.030602058 1.606504069
440   0.2528 0.115441621 0.052071772 0.031445748 1.629863284
465   0.2528 0.068822660 0.051149725 0.03227449  1.652413779
490   0.2528 0.000000000 0.050227678 0.033088564 1.674155554
515   0.2528 0.000000000 0.04930563  0.033888251 1.695088608
540   0.2528 0.000000000 0.048383583 0.034673834 1.715212943
565   0.2528 0.000000000 0.047461535 0.035445593 1.734528558
590   0.2528 0.000000000 0.046539488 0.036203809 1.753035453
615   0.2528 0.000000000 0.045617441 0.036948764 1.770733628
640   0.2528 0.000000000 0.044695393 0.037680739 1.787623083

**TXT
*ROCKTYPE 2
*ROCKCP 3.3821 0.0013
*THCONTAB
**T C thconr thconw thcono thcong thcons
15  7.907490000 0.356245009 0.069639211 0.014840486 1.122770711
40  8.141165000 0.376899404 0.068691405 0.015961606 1.159069446
65  8.372340000 0.392542792 0.067743598 0.017063277 1.194559461
90  8.601015000 0.403306893 0.066795792 0.018145781 1.229240756
115 8.827190000 0.409323426 0.065847985 0.019209399 1.263113331
140 9.050865000 0.410724108 0.064900178 0.020254411 1.296177186
165 9.272040000 0.407640659 0.063952372 0.021281100 1.328432321
190 9.490715000 0.400204798 0.063004565 0.022289747 1.359878735
215 9.706890000 0.388548243 0.062056759 0.023280632 1.390516430
240 9.920565000 0.372802713 0.061108952 0.024254037 1.420345405
265 10.13174000 0.353099926 0.060161146 0.025210243 1.449365660
290 10.34041500 0.329571602 0.059213339 0.026149532 1.477577195
315 10.54659000 0.302349460 0.058265532 0.027072185 1.504980010
340 10.75026500 0.271565217 0.057317726 0.027978483 1.531574105
365 10.95144000 0.237350593 0.056369919 0.028868707 1.557359479
390 11.15011500 0.199837307 0.055422113 0.029743138 1.582336134
415 11.34629000 0.159157076 0.054474306 0.030602058 1.606504069
440 11.53996500 0.115441621 0.0535265   0.031445748 1.629863284
465 11.73114000 0.068822660 0.052578693 0.032274490 1.652413779
490 11.91981500 0.019431911 0.051630886 0.033088564 1.674155554
515 12.10599000 0.000000000 0.05068308  0.033888251 1.695088608
540 12.28966500 0.000000000 0.049735273 0.034673834 1.715212943
565 12.47084000 0.000000000 0.048787467 0.035445593 1.734528558
590 12.64951500 0.000000000 0.04783966  0.036203809 1.753035453
615 12.82569000 0.000000000 0.046891854 0.036948764 1.770733628
640 12.99936500 0.000000000 0.045944047 0.037680739 1.787623083

**TXT
*ROCKTYPE 3 
*ROCKCP 20 
*THCONR 2E6

*THTYPE *IJK 1 1 1:11 1
	         2 1 1:11 2              

              """, file = fileID)
        

def print_rock_fluid(fileID, rock_fluid_option):
    print('**  ==============  ROCK-FLUID PROPERTIES  ======================', file = fileID)
    if rock_fluid_option == 1:
        print("""
*rockfluid
RPT 1

*swt   **  Water-oil relative permeabilities

**   Sw         Krw       Krow
**  ----      -------    -------
    0.1        0.0        0.9
    0.25       0.004      0.6
    0.44       0.024      0.28
    0.56       0.072      0.144
    0.672      0.168      0.048
    0.752      0.256      0.0

*SLT   **  Liquid-gas relative permeabilities

**   Sl         Krg       Krog
**  ----      -------    -------
    0.21       0.784      0.0
    0.32       0.448      0.01
    0.4        0.288      0.024
    0.472      0.184      0.052
    0.58       0.086      0.152
    0.68       0.024      0.272
    0.832      0.006      0.448
    0.872      0.0        0.9
*sorg 0.2
*sgr 0.12
*sorw 0.25

**  Override critical saturations on table
*swr 0.25
              
              """, file = fileID)
        
    if rock_fluid_option == 2:
        print("""
*ROCKFLUID
*RPT 1 LININTERP WATWET
*swt ** Water-oil relative permeabilities
**  Sw     Krw        Krow
** ----  -------      -------
   0.18  0            1
   0.2   0.000000217  0.862816
   0.22  0.00000347   0.740736
   0.235 0.0000124    0.658348
   0.25  3.25E-05     0.58327
   0.265 7.07E-05     0.515026
   0.28  0.000135     0.453161
   0.3   0.000281     0.379842
   0.35  0.00113      0.236425
   0.4   0.003171     0.139158
   0.45  0.007193     0.076303
   0.5   0.014193     0.038131
   0.55  0.025367     0.016787
   0.6   0.042117     0.00615
   0.625 0.053007     0.003372
   0.65  0.066047     0.001685
   0.7   0.098964     0.000272
   0.75  0.142877     0.000012
   0.775 0.169641     0.000000531
   0.8   0.2          0
   1     0.6          0

*slt ** Liquid-gas relative permeabilities
**  Sl       Krg     Krog
** ----     ------- -------
   0.3      0.718068     0
   0.320968 0.672114 0.000174
   0.362903 0.584799 0.000681
   0.404839 0.503605 0.001952
   0.446774 0.428524 0.004596
   0.530645 0.296665 0.017658
   0.614516 0.189147 0.049986
   0.698387 0.105876 0.116768
   0.740323 0.073298 0.169396
   0.782258 0.046733 0.23901
   0.824194 0.026159 0.329258
   0.866129 0.011546 0.444239
   0.908065 0.002853 0.588519
   0.92     0.001    0.63
   0.93     0.0003   0.67
   0.94     0.00005  0.71
   0.95     0        0.76715
   1        0        1

              """, file = fileID)
        
        

def print_initial_cond(fileID, initial_cond_option):
    print('**  ==============  INITIAL CONDITIONS  ======================', file = fileID)
    if initial_cond_option == 1:
        print("""
*INITIAL
*VERTICAL OFF
**INITREGION 1
*PRES *CON 689.476     ** high initial pressure
*TEMP *CON 293.15
*SW *CON 0.826
*SO *CON 0.174      ** initial gas saturation is 0.168
MFRAC_GAS 'INRT GAS' CON       0.79
MFRAC_GAS 'OXYGEN' CON         0.21
MFRAC_OIL 'HEVY OIL' CON        0.55
MFRAC_OIL 'LITE OIL' CON        0.45      
        
              """, file = fileID)
        
    if initial_cond_option == 2:
        print("""
*INITIAL
*PRES *CON   8273.709
*TEMP *CON   25
*VERTICAL OFF
*INITREGION 1
*SW *IJK 
**WAT_SAT
		 1 1 1:11 0.000
**W_SATEND
         2 1 1:11 0.0

*SO *IJK 
**OIL_SAT
		 1 1 1:11 0.031
**O_SATEND
         2 1 1:11 0.0

*SG *IJK 
**GAS_SAT
	     1 1 1:11 0.969
**G_SATEND
         2 1 1:11 0.0

**Gas in tube is air(79%N2 & 21%O2)
**MFRAC_GAS 'N2' CON       1
*MFRAC_GAS 'O2' *con 0.2094
*MFRAC_GAS 'N2' *con 0.7906      
        
              """, file = fileID)
        
        
def print_ref_cond(fileID, ref_cond_option):
    print('**  ==============  Reference CONDITIONS  ======================', file = fileID)
    if ref_cond_option == 1:
        print("""
*PRSR 101.3529
*TEMR 298.15
*TSURF 293.15
*PSURF 101.3529
              """, file = fileID)
        
    if ref_cond_option == 2:
        print("""
*PRSR 101.325
*TEMR 25
*PSURF 101.325
*TSURF 20  
              """, file = fileID)



def print_numerical(fileID, numerical_option):
    print('**  ==============  NUMERICAL CONTROL  ======================', file = fileID)
    if numerical_option == 1:
        print("""
*NUMERICAL
*RUN
 
              """, file = fileID)

        
    if numerical_option == 2:
        print("""
*NUMERICAL
*MAXSTEPS 100000
*DTMAX    0.1
**DTMIN   0.1
**NCUTS   20
**CONVERGE *TOTRES *TIGHTER
**MATBALTOL 0.0000001      
*RUN    
              """, file = fileID)

        



def print_recurrent(fileID, recurrent_option):
    print('**  ==============  RECURRENT DATA  ======================', file = fileID)
    if recurrent_option == 1:
        print("""
*TIME 0
   *DTWELL .05
WELL 1 'INJECTOR'
                               ** air injection
INJECTOR 'INJECTOR'
INCOMP  GAS  0.0  0.0  0.0  0.79  0.21
TINJW  273.15
**Key_Word_for_Air_Flow_Rate**
OPERATE  STG  0.01
                     ** i  j  k  wi(gas)
** UBA              wi          Status  Connection  
      PERF        WI  'INJECTOR'
** UBA             wi        
    1 1 1        5.54  

**     *WELL 2 'PRODUCER'
WELL 2 'PRODUCER'
                                 **pressure unit is psi
PRODUCER 'PRODUCER'
*OPERATE  BHP  689.476  
**          rad  geofac  wfrac  skin
GEOMETRY  K  1.0  1.0  1.0  0.0
      PERF  TUBE-END  'PRODUCER'
** UBA              ff        
    1 1 12         1.0        
        
              """, file = fileID)
        
    if recurrent_option == 2:
        print("""

*TIME   0
*DTWELL 0.01
*WELL   'INJE'
*WELL   'PROD'
*INJECTOR UNWEIGHT 'INJE'
*INCOMP  GAS  0.  0.  0.7906  0.2094  0.  0.
*TINJW  26.
*OPERATE  MAX  STG  166.67  CONT
*GEOMETRY  K  1.5  1.  1.  0.
*PERF  TUBE-END  'INJE'
1 1 1  1.  OPEN    FLOW-FROM  'SURFACE'
*PRODUCER 'PROD'
*OPERATE  *MIN   *BHP  8273.709
*GEOMETRY  K  1.5  1.  1.  0.
*PERF  TUBE-END  'PROD'
1 1 11  1.  OPEN    FLOW-TO  'SURFACE'      
        
              """, file = fileID)
        


def print_heater(fileID, heater_option):
    print('**  ============== DEFINE HEATERS ======================', file = fileID)
    if heater_option == 1:
        print("""
**Key_Word_for_Proportional_Heat_Transfer_Coefficient**
*UHTR   *IJK 1 1 1:12 3000
*TMPSET *IJK 1 1 1:12 273.15
**Key_Word_for_Constant_Heat_Transfer_Rate**
*HEATR  *IJK 1 1 1:12 30000
*AUTOHEATER *ON 1 1 1:12 
        
              """, file = fileID)
        
    if heater_option == 2:
        print("""
*UHTR *IJK 2 1 1:11 3000
*TMPSET *IJK 2 1 1:11 25
*HEATR *IJK 2 1 1 30000
*AUTOHEATER *ON 2 1 1:11              

              """, file = fileID)

