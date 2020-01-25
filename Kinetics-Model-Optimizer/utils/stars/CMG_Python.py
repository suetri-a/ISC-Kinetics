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
2019/9/26: updated the tim's code regarding a loose Component and Kinetics class. And printing functions associated with them.
Attention:
    1. the K table and initial pressure of the option 1 need to be changed
"""

import numpy as np
import struct

#import makeVectors as mV
import sys
import h5py as hdf
import datetime as dat
import io
import os

def ignition_extinction_envelope(sim_folder, base_file, total_airrate, variable_type, total_variable, exe_path, cd_path, delete_dat_file_path, tempthreshold, option):
    '''
    Input details:
        sim_folder: the name for the folder collecting all the simulation models (e.g.:'VKC/')
        base_file: the name for the base file (.dat file) to change the values for selected variables (e.g.:'base_file.dat')
        Attention: base_file need to be saved in the sim_folder
        exe_path: the path for the CMG exe files, (e.g.: '"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"')
        cd_path: the path until to the folder collecting all dat files to be run (e.g.: 'cd C:\\Users\\yunanli\\Desktop\\CMG\\VKC ')
        delete_dat_file_path: this path helps to delete the extinction files in a special format (e.g.: 'C:\\Users\\yunanli\\Desktop\\CMG\\VKC\\')
        
        airrate_start: the initial point for air flow rate in appropriate format
        airrate_end: the end point for air flow rate in appropriate format
        airrate_step: the step size for the air flow array
        variable_type: the variable type need to be defined in str format
        variable_value_start, 
        variable_value_end, 
        variable_value_step, 
        flag_string: the key information to capture from out files to find the peak temperature (e.g.: flag_string = ['Temperature', '(Fahrenheit)'])
        tempthreshold: the value to  classify ignition extinction cases (440.33 [F] or 500 [K] pay attention to unit system)
        
    '''
    #total_airrate=np.arange(airrate_start, airrate_end, airrate_step)
    #total_variable=np.arange(variable_value_start, variable_value_end, variable_value_step)
    ignition_result = []
    extinction_result = []
    Ereact_rate_result = []
    Ereact_rate_result_extinction = []
    status_record = ['extinction']
    sim_count = 0
    
    for variable in total_variable:
        for airrate in total_airrate:
            # Make sure no rounding happens
            airrate = np.round(airrate, decimals = 8)
            variable = np.round(variable, decimals = 8)
            # Write a new dat file
            dat_file_name = write_dat_file(sim_folder, airrate, variable, variable_type, base_file)
            # Run this file
            run_dat_file(exe_path, cd_path, dat_file_name)
            # Result analysis
            file_name_out = sim_folder + dat_file_name + '.out'
            fid = [line for line in open(file_name_out)]
            fileID = open(file_name_out,'r')
            #resultlist = result_summary(fid, fileID, 0, flag_string)
            #status = ignite(resultlist, tempthreshold, option)
            status_list = ignite(fid, fileID, tempthreshold, option)
            status = status_list[0]
            Ereact_rate = status_list[1]
            fileID.close()
            status_record.append(status)
            sim_count = sim_count + 1
            if status == 'ignition':
                ignition_result.append([airrate, variable])
                Ereact_rate_result.append(Ereact_rate)
            else:
                extinction_result.append([airrate, variable])
                Ereact_rate_result_extinction.append(Ereact_rate)
            if status_record[sim_count-1] == status_record[sim_count]:
                delete_dat_file(delete_dat_file_path, dat_file_name)
            #else: 
                #print('Please check your code, something wrong')
    print('The lower boundary is the ignition case, and the upper boundary is the extinction case')    
    return ignition_result, Ereact_rate_result, extinction_result, Ereact_rate_result_extinction                
                
"""
    dat_file_dir = os.listdir(delete_dat_file_path)

    for item in dat_file_dir:
        if item.endswith(".irf"):
            try:
                os.remove(os.path.join(dir_name, item))
            except OSError as e:
                print(e)
            
        if item.endswith(".mrf"):
            try: 
                os.remove(os.path.join(dir_name, item))
            except OSError as e:
                print(e)
"""




#This function helps to establish the batch file
def write_batch_file(thermocondstart, thermocondend, thermocondstep, airratestart, airrateend, airratestep, path, batch_file_name):
    '''
    This function can help to generate the batch file including all cases.
    Instructinos: find the generated txt file following the path we defined
                  change the property from .txt to .bat
                  drag the .bat file to the terminal to run all cases
    Examples: path='M:\\temp_airflow\\Run2_low'
                
    '''
    
    cd='cd'+' '+path
    CMG='"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"'
    # Construct the name for .dat files
    thermocond=np.arange(thermocondstart,thermocondend,thermocondstep)
    airrate=np.arange(airratestart,airrateend,airratestep)
    
    filename=[]
    for i in thermocond:
        for j in airrate:
            s='{m}_{n}.dat'
            name=s.format(m=np.round(i,decimals=3),n=np.round(j,decimals=4))
            
            #t='"C:\Program Files (x86)\CMG\STARS\2017.10\Win_x64\EXE\st201710.exe"   \-f "'{FILENAME}'" -wd "C:\Users\yunanli\Desktop\CMG\VKC"  -wait'
            #line=t.format(FILENAME=name)
            filename=np.append(filename,name)
            
    batcharray=[]
    with io.open(path + batch_file_name,'a',encoding="utf-8") as f:
        f.write(f'{cd}'+'\n')
        for number in range(len(filename)):
            batchline=CMG+'  '+'-f'+' '+f'"{filename[number]}"'+' -wd '+f'"{path}"'+'  '+'-wait'
            batcharray=np.append(batcharray,batchline)
            f.write(f'{batchline}'+'\n')
            


def write_dat_file(sim_folder, airrate, variable_value, variable_type, base_file):
#def writeSTARSrunfile(sim_folder, freqfac, eact, renth, sat_oil, coeff_P, coeff_R, R_Order, MW_vec, base_file):
    '''
    This function helps to generate all the cases in .dat format for CMG to run.
    Instructions: this function only works for the change of air flow rate and several other variables (05/12/2019)
                  We need to add the keywords indicator to the base case .dat files
                  The keywords indicators are right above the line we want to change
    Inputs:
        
        sim_folder - directory where simulations will be written (sim_folder='Run2_low/')
        thermocond - thermal conductivity in CMG to count for heat loss, 
        airrate - injected air flow rate, 
        
    Variable types:
        0. Air flow rate (airrate)
           CMG key word: **Key_Word_for_Air_Flow_Rate**        
        1. Thermal conductivity (thermocond)
           CMG key word: **Key_Word_for_Thermal_Conductivity**
        2. Bottom hole pressure (BHP)
           CMG key word: **Key_Word_for_Bottom_Hole_Pressure**
        3. Proportional heat transfer coefficient (UHTR). This is a factor in CMG Constant and Convective Heat Transfer Model part.
           CMG key word: **Key_Word_for_Proportional_Heat_Transfer_Coefficient**
        4. Injection temperature (Tinj)
           CMG key word: **Key_Word_for_Injection_Temperature**
        5. Constant heat transfer rate (HEATR)
           CMG key word: **Key_Word_for_Constant_Heat_Transfer_Rate**
        6. Heat loss properties (HLOSSPROP)
           CMG key word: **Key_Word_for_Heatloss_Properties**
        
        HRates - list of heating rates
        num_rxns - number of reactions, integer
        t_init - initial time, scalar
        t_steps - total number of time steps, integer
        temp_max - maximum temperature scalar
        freqfac - frequency factors, NumPy Array
        eact - activation energies, NumPy Array
        renth - enthalpies of reaction, NumPy Array
        sat_oil - oil saturation, scalar
        coeff_P - product coefficients, NumPy Array
        coeff_R - product coefficients, NumPy Array
        R_Order - product coefficients, NumPy Array
        MW_vec - product coefficients, NumPy Array
        base_file - product coefficients, string

    '''

    # Load baseline simulation deck

    # Path to baseline case
    BaseCaseDatPath = sim_folder + base_file

    # Allocate a cell array that we will use to store the baseline
    fid = [line for line in open(BaseCaseDatPath)]
    
    if variable_type == 'thermocond':
        thermocond = variable_value
        # Create new file
        dat_file_name = 'thermocond' + str(thermocond) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'thermocond' + str(thermocond) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0
        # Loading everything before the key word ** Define heat loss (included)
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Thermal_Conductivity**')
        # Change the values for thermal conductivity
        i = print_thermal_conductivity(fid, fileID, i, thermocond)
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)

        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()
        
    elif variable_type == 'BHP':
        BHP = variable_value
        # Create new file
        dat_file_name = 'BHP' + str(BHP) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'BHP' + str(BHP) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0

        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)
        # Loading everything before the key word ** Define heat loss (included)
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Bottom_Hole_Pressure**')
        # Change the values for thermal conductivity
        i = print_BHP(fid, fileID, i, BHP)

        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()
        
    elif variable_type == 'UHTR':
        UHTR = variable_value
        # Create new file
        dat_file_name = 'UHTR' + str(UHTR) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'UHTR' + str(UHTR) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0

        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)
        # Loading everything before the key word ** Define heat loss (included)
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Proportional_Heat_Transfer_Coefficient**')
        # Change the values for thermal conductivity
        i = print_UHTR(fid, fileID, i, UHTR)

        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()
        
    elif variable_type == 'Tinj':
        Tinj = variable_value
        # Create new file
        dat_file_name = 'Tinj' + str(Tinj) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'Tinj' + str(Tinj) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0
        
        # Loading everything before the key word ** Define heat loss (included)
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Injection_Temperature**')
        # Change the values for thermal conductivity
        i = print_inject_temperature(fid, fileID, i, Tinj)
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)


        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()
        
    elif variable_type == 'HEATR':
        HEATR = variable_value
        # Create new file
        dat_file_name = 'HEATR' + str(HEATR) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'HEATR' + str(HEATR) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0
        
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)
        
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Constant_Heat_Transfer_Rate**')
        # Change the values for the injected air flow rate
        i = print_HEATR(fid, fileID, i, HEATR)

        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()
        
        
    elif variable_type == 'HLOSSPROP':
        HLOSSPROP = variable_value
        # Create new file
        dat_file_name = 'HLOSSPROP' + str(HLOSSPROP) + '_' + 'airrate' + str(airrate)
        file_name = sim_folder + 'HLOSSPROP' + str(HLOSSPROP) + '_' + 'airrate' + str(airrate) + '.dat'
        fileID = open(file_name,'w')
        i = 0
        
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Heatloss_Properties**')
        # Change the values for the injected air flow rate
        i = print_HLOSSPROP(fid, fileID, i, HLOSSPROP)
        
        # Loading everything before the Reaction
        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Air_Flow_Rate**')
        # Change the values for the injected air flow rate
        i = print_air_rate(fid, fileID, i, airrate)
        

        i = copy_up_to_string(fid, fileID, i, '**Key_Word_for_Stop**')
    
        fileID.close()        
        
    else:
        print('Invaliad variable tpye in this CMG_python file version')
    # The dat file name here is only the name, without the .dat
    return dat_file_name

def run_dat_file(exe_path, cd_path, dat_file_name):
    '''
    Example for definition of path
    exe_path='"C:\\Program Files (x86)\\CMG\\STARS\\2017.10\\Win_x64\\EXE\\st201710.exe"'
    cd_path='cd C:\\Users\\yunanli\\Desktop\\CMG\\VKC '
    dat_file_name can be returned from writedatfile function
    '''
    os_system_line = cd_path + '&' + exe_path + '  -f ' + f'"{dat_file_name}"' + '.dat'
    os.system(os_system_line)
    
def delete_dat_file(delete_dat_file_path, dat_file_name):
    '''
    Example:
        delete_dat_file_path = 'C:\\Users\\yunanli\\Desktop\\CMG\\'
    '''
    delete_line_1 = delete_dat_file_path + dat_file_name + '.dat'
    delete_line_2 = delete_dat_file_path + dat_file_name + '.irf'
    delete_line_3 = delete_dat_file_path + dat_file_name + '.mrf'
    delete_line_4 = delete_dat_file_path + dat_file_name + '.out'
    delete_line_5 = delete_dat_file_path + dat_file_name + '.sr3'
    try:
        os.remove(delete_line_1)
    except OSError as e:
        print(e)
    try:
        os.remove(delete_line_2)
    except OSError as e:
        print(e)
    try:
        os.remove(delete_line_3)
    except OSError as e:
        print(e)
    try:
        os.remove(delete_line_4)
    except OSError as e:
        print(e)
    try:
        os.remove(delete_line_5)
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
    i +=1
    return i

def print_thermal_conductivity(fid, fileID, i, thermocond):
    ''' 
    Prints:
            heat loss defined lines. 
    This function is required when changing the heat loss properties
    '''
    print('*HLOSSPROP *OVERBUR 35 ', str(thermocond), file=fileID)
    #print(str(thermocond), file=fileID)
    #print('\n', file=fileID, end="")
    print('*HLOSSPROP *UNDERBUR 35 ', str(thermocond), file=fileID)
    #print(str(thermocond), file=fileID)
    #print('\n', file=fileID, end="")
    i=i+2
    return i
    
    
def print_air_rate(fid, fileID, i, airrate):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('OPERATE  STG  ', str(airrate), file=fileID)
    #print(str(airrate), file=fileID)
    #print('  CONT REPEAT', file=fileID)
    #print('\n', file=fileID, end="")
    i=i+1
    return i

def print_inject_temperature(fid, fileID, i, inject_temp):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('TINJW  ', str(inject_temp), file=fileID)
    #print(str(airrate), file=fileID)
    #print('  CONT REPEAT', file=fileID)
    #print('\n', file=fileID, end="")
    i=i+1
    return i

def print_BHP(fid, fileID, i, BHP):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('OPERATE  MIN  BHP  ', str(BHP), '  CONT REPEAT', file=fileID)
    #print(str(airrate), file=fileID)
    #print('  CONT REPEAT', file=fileID)
    #print('\n', file=fileID, end="")
    i=i+1
    return i

def print_UHTR(fid, fileID, i, UHTR):
    '''
    Prints:
        airrate defined lines
    This function is required when changing the air flow rates
    '''
    print('*UHTR   *IJK 1 1 1:12  ', str(UHTR), file=fileID)
    i=i+1
    return i

def print_HEATR(fid, fileID, i, HEATR):
    print('*HEATR  *IJK 1 1 1:12  ', str(HEATR), file=fileID)
    i=i+1
    return i
    
def print_HLOSSPROP(fid, fileID, i, HLOSSPROP):
    print('*HLOSSPROP *-I 764.37  ', str(HLOSSPROP), file=fileID)
    print('*HLOSSPROP *+I 764.37  ', str(HLOSSPROP), file=fileID)
    print('*HLOSSPROP *-J 764.37  ', str(HLOSSPROP), file=fileID)
    print('*HLOSSPROP *+J 764.37  ', str(HLOSSPROP), file=fileID)
    print('*HLOSSPROP *-K 764.37  ', str(HLOSSPROP), file=fileID)
    print('*HLOSSPROP *+K 764.37  ', str(HLOSSPROP), file=fileID)
    i=i+6
    return i
    
def find_keyword_result(fid, fileID, i, option):
    '''
    Find the index of the characteristic flag_string in the .out files
    '''
    if option == 1:
        flag_string = ['Temperature', '(Kelvin)']
        flag = True
        while flag:
            #print(fid[i], file = fileID, end="")
            i+=1
            if not fid[i-1].split():
                flag = True
            else:
                if fid[i-1].split() == flag_string:
                    flag = False
                else:
                    flag = True
            # if there is no flag_string anymore, we need this to avoid out of range error        
            if i == len(fid):
                flag = False
        return i
    
    if option == 4:
        flag_string = ['Temperature', '(Celsius)']
        flag = True
        while flag:
            #print(fid[i], file = fileID, end="")
            i+=1
            if not fid[i-1].split():
                flag = True
            else:
                if fid[i-1].split() == flag_string:
                    flag = False
                else:
                    flag = True
            # if there is no flag_string anymore, we need this to avoid out of range error        
            if i == len(fid):
                flag = False
        return i    
    
    if option == 2:
        flag_string = ['Energy', '(J   )']
        flag = True
        while flag:
            i+=1
            if not fid[i-1].split():
                flag = True
            else:
                if fid[i-1].split()[0:3] == ['The', 'Time', 'is']:
                    flag = False
                else:
                    flag = True
            if i == len(fid):
                flag = False
        return i
            
    if option == 3:
        flag_string = ['Gas', 'Molar', 'Fraction']
        flag = True
        while flag:
            i+=1
            if not fid[i-1].split():
                flag = True
            else:
                if fid[i-1].split() == flag_string:
                    flag = False
                else:
                    flag = True
            if i == len(fid):
                flag = False
        return i

    if option == 5:
        flag_string = ['Global', 'Molar', 'Fraction']
        flag = True
        while flag:
            i+=1
            if not fid[i-1].split():
                flag = True
            else:
                if fid[i-1].split() == flag_string:
                    flag = False
                else:
                    flag = True
            if i == len(fid):
                flag = False
        return i



def result_summary(fid, fileID, i, option):
    '''
    This function helps to summarize all needed resutls in a list.
    In the resultlist, the first string in each sublist represents time.
    The remaining strings are the properties values, such as temperature, etc.
    The length for each sublist may be different.
    '''
    if option == 1:
        resultlist=[]
        #flag_string = ['Temperature', '(Kelvin)']
        flag = True
        while flag:
            i=find_keyword_result(fid, fileID, i, option)
            if i == len(fid):
                break
            time=fid[i-2].split()[2]
            if fid[i+2].split()[0:3] == ['All', 'values', 'are']:
                temp=[fid[i+2].split()[3]]
            else:
                temp=fid[i+3].split()[2:len(fid[i+3])]
        
            resultlist.append([time, temp])
        return resultlist
    
    if option == 2:
        time = [0] #unit is hr
        Ereact = [0] #unit is J
        #flag_string = ['Energy', '(J   )']
        flag = True
        while flag:
            i=find_keyword_result(fid, fileID, i, option)
            if i == len(fid):
                break
            time.append(fid[i-1].split()[3])
            if (fid[i+20].split()[0] == 'Energy')&(fid[i+12].split()[4:6] == ['Net', 'Reactions']):
                Ereact.append(fid[i+20].split()[5])
            else:
                print('Please check the I/O of CMG, this is not matched with your code')
            
        return time, Ereact
 

       

def ignite(fid, fileID, threshold, option):
    '''
    This function helps to determine if a resultlist, such as the temperature profile, ignites or not.
    resultlist is a input list including all the result_summary.
    tempthreshold is a input for ignition determination.
    This function returns the state of a case, ignition or extinction.
    Attention: temperature threshold unit must be the same with result list. e.g.: fahrenheit.
    '''
    if option == 1:
        tempthreshold = threshold
        resultlist = result_summary(fid, fileID, 0, option)
        resultlist_option2 = result_summary(fid, fileID, 0, 2)
        rate_Ereact = []
        rate_Ereact_max = []
        for i in range(len(resultlist_option2[0])-1):
            if (float(resultlist_option2[0][i+1]) - float(resultlist_option2[0][i])) == 0:
                continue
            rate_Ereact.append((float(resultlist_option2[1][i+1]) - float(resultlist_option2[1][i]))/(float(resultlist_option2[0][i+1]) - float(resultlist_option2[0][i])))

        
        status = 'extinction'
        num_sublist=len(resultlist)
        for i in range(num_sublist):
            len_sublist=len(resultlist[i][1])
            if len_sublist == 1:
                if float(resultlist[i][1][0]) >= tempthreshold:
                    status = 'ignition'
                    rate_Ereact_max = (max(rate_Ereact))
                else:
                    rate_Ereact_max = (max(rate_Ereact))
            else:
                for j in range(1,len_sublist-1,1):
                    if float(resultlist[i][1][j]) >= tempthreshold:
                        status = 'ignition'
                        rate_Ereact_max = (max(rate_Ereact))
                    else:
                        rate_Ereact_max = (max(rate_Ereact))
                        
        return status, rate_Ereact_max
                        
    if option == 2:
        
        resultlist = result_summary(fid, fileID, 0, option)
        rate_Ereact = []
        """
        Following code calculates the rate of energy net reacted first.
        Using the differential method: dEreact / dtime
        """
        for i in range(len(resultlist[0])-1):
            if (float(resultlist_option2[0][i+1]) - float(resultlist_option2[0][i])) == 0:
                continue
            rate_Ereact.append((float(resultlist[1][i+1]) - float(resultlist[1][i]))/(float(resultlist[0][i+1]) - float(resultlist[0][i])))
        
        status = 'extinction'
        num_sublist = len(rate_Ereact)
        for i in range(num_sublist):
            if rate_Ereact[i] >= threshold:
                status = 'ignition'
                
        return status
            
            
        
"""
###############################################################################
###############################################################################
Created on Fri Aug 23 15:45:20 2019

@author: liyunan
Functions to generate our expected CMG STARS model to match with experimental data.

Option = 1: means using the Kristensen's minimal model system
Option = 2: means using the Murat's RTO base case
Option = 3: means using the Bo's RTO base case
###############################################################################
###############################################################################
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
        
    if unit_option == 'Bazargan_KC':
        for t in range(499):
            if (25 + HR*t) > 750:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 4 1 1:7 ', str(750), file = fileID)
                print("*INJECTOR 'INJECTOR'", file = fileID)
                print('*TINJW ', str(750), file = fileID)
            else:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 4 1 1:7 ', str(25 + HR*t), file = fileID)
                print("*INJECTOR 'INJECTOR'", file = fileID)
                print('*TINJW ', str(25 + HR*t), file = fileID)
        print('*TIME  500', file = fileID)        
        
    if unit_option == 'nonArr_KC':
        for t in range(499):
            if (25 + HR*t) > 750:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1:3 ', str(750), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(750), file = fileID)
            else:
                print('*TIME ', str((t+1)), file = fileID)
                print('*TMPSET *IJK 1 1 1:3 ', str(25 + HR*t), file = fileID)
                print("*INJECTOR 'MB-INJECTOR'", file = fileID)
                print('*TINJW ', str(25 + HR*t), file = fileID)
        print('*TIME  500', file = fileID)        
    
        
    print('*STOP', file = fileID)


def K_value_table(K_option, fileID):
# if K_option = 1: use the Wilson K relations for the K value table
# if K_option = 2: use Murat model K value table coefficients
    if K_option == 1:
        K_value_file = 'C:\\Users\\yunanli\\OneDrive - Leland Stanford Junior University\\CMG_Python\\K_value_table.txt'
        K_fid = [line for line in open(K_value_file)]
        #K_fileID = open(K_value_file, 'r')
        i = 0
        i = copy_up_to_string(K_fid, fileID, i, '**End_of_K_value_table**')

        
    if K_option == 2:
        print("""
*KV1 0 0 
*KV2 0 0 
*KV3 0 0 
*KV4 0 0 
*KV5 0 0 

              """, file = fileID)
        
        

def Fluid_definition(fluid_option):
    if fluid_option == 1:
        class Component:
            def __init__(self, NAME, CMM, PCRIT, TCRIT, ACEN, AVG, BVG, AVISC, BVISC, CPG1, CPG2, CPG3, CPG4, MOLDEN, CP, CT1, SOLID_DEN, SOLID_CP):
                self.NAME = NAME
                self.CMM = CMM
                self.PCRIT = PCRIT
                self.TCRIT = TCRIT
                self.ACEN = ACEN
                self.AVG = AVG
                self.BVG = BVG
                self.AVISC = AVISC
                self.BVISC = BVISC
                self.CPG1 = CPG1
                self.CPG2 = CPG2
                self.CPG3 = CPG3
                self.CPG4 = CPG4
                self.MOLDEN = MOLDEN
                self.CP = CP
                self.CT1 = CT1
                self.SOLID_DEN = SOLID_DEN
                self.SOLID_CP = SOLID_CP

    
            def getNAME(self):
                return self.NAME
            def getCMM(self):
                return self.CMM
            def getPCRIT(self):
                return self.PCRIT
            def getTCRIT(self):
                return self.TCRIT     
            def getACEN(self):
                return self.ACEN                 
            def getAVG(self):
                return self.AVG                 
            def getBVG(self):
                return self.BVG    
            def getAVISC(self):
                return self.AVISC     
            def getBVISC(self):
                return self.BVISC     
            def getCPG1(self):
                return self.CPG1          
            def getCPG2(self):
                return self.CPG2                 
            def getCPG3(self):
                return self.CPG3                 
            def getCPG4(self):
                return self.CPG4                 
            def getMOLDEN(self):
                return self.MOLDEN                 
            def getCP(self):
                return self.CP                 
            def getCT1(self):
                return self.CT1                 
            def getSOLID_DEN(self):
                return self.SOLID_DEN       
            def getSOLID_CP(self):
                return self.SOLID_CP

            

            def setNAME(self, value):
                self.NAME = value
            def setCMM(self, value):
                self.CMM = value
            def setPCRIT(self, value):
                self.PCRIT = value
            def setTCRIT(self, value):
                self.TCRIT = value   
            def setACEN(self, value):
                self.ACEN = value               
            def setAVG(self, value):
                self.AVG = value              
            def setBVG(self, value):
                self.BVG = value 
            def setAVISC(self, value):
                self.AVISC = value   
            def setBVISC(self, value):
                self.BVISC = value   
            def setCPG1(self, value):
                self.CPG1 = value         
            def setCPG2(self, value):
                self.CPG2 = value               
            def setCPG3(self, value):
                self.CPG3 = value               
            def setCPG4(self, value):
                self.CPG4 = value               
            def setMOLDEN(self, value):
                self.MOLDEN = value               
            def setCP(self, value):
                self.CP = value               
            def setCT1(self, value):
                self.CT1 = value              
            def setSOLID_DEN(self, value):
                self.SOLID_DEN = value      
            def setSOLID_CP(self, value):
                self.SOLID_CP = value

            

        class Kinetics:
            def __init__(self, NAME, REAC, PROD, FREQFAC, EACT, RENTH):
                self.NAME = NAME
                self.REAC = REAC
                self.PROD = PROD
                self.FREQFAC = FREQFAC
                self.EACT = EACT
                self.RENTH = RENTH

            def getNAME(self):
                return self.NAME  
            def getREAC(self):
                return self.REAC
            def getPROD(self):
                return self.PROD
            def getFREQFAC(self):
                return self.FREQFAC
            def getEACT(self):
                return self.EACT
            def getRENTH(self):
               return self.RENTH
            
            def setNAME(self, value):
                self.NAME = value
            def setREAC(self, value):
                self.REAC = value
            def setPROD(self, value):
                self.PROD = value
            def setFREQFAC(self, value):
                self.FREQFAC = value
            def setEACT(self, value):
               self.EACT = value
            def setRENTH(self, value):
                self.RENTH = value
            
        
        Component_map = dict()
        Component_map["WATER"] = Component("WATER",[0.018],[21754.478],[647.4],[0.344],[1.7e-5],[1.116],[7.52e-5],[1384.86],[7.701],[4.595e-4],[2.521e-6],[-0.859e-9],[55520],[4.352331606e-7],[2.16e-4],[],[])
        Component_map["HEVY OIL"] = Component("HEVY OIL",[0.675],[830.865],[887.6],[1.589],[7.503e-6],[1.102],[4.02e-4],[3400.89],[-34.081],[4.137],[-2.279e-3],[4.835e-7],[1464],[7.25388601e-7],[2.68e-4],[],[])
        Component_map["LITE OIL"] = Component("LITE OIL",[0.157],[2107.56],[617.4],[0.449],[3.77e-6],[0.943],[4.02e-4],[3400.89],[-7.913],[0.961],[-5.29e-4],[1.123e-7],[5118],[7.25388601e-7],[5.11e-4],[],[])
        Component_map["INRT GAS"] = Component("INRT GAS",[0.041],[3445.05],[126.5],[0.04],[3.213e-4],[0.702],[],[],[31.150],[-1.357e-2],[2.679e-5],[-1.167e-8],[],[],[],[],[])
        Component_map["OXYGEN"] = Component("OXYGEN",[0.032],[5035.852],[154.8],[0.022],[3.355e-4],[0.721],[],[],[28.106],[-3.68e-6],[1.746e-5],[-1.065e-8],[],[],[],[],[])
        Component_map["COKE"] = Component("COKE",[0.013],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[916.344, 0, 0],[16, 0])
        
        Kinetics_map = dict()
        Kinetics_map["Cracking"] = Kinetics("Cracking",[0, 1, 0, 0, 0, 0],[0, 0, 2.154, 0, 0, 25.96],4.167e5,62802,93000)
        Kinetics_map["HO burning"] = Kinetics("HO burning",[0, 1, 0, 0, 60.55, 0],[28.34, 0, 0, 51.53, 0, 0],4.4394e11,138281,29.1332e6)
        Kinetics_map["LO burning"] = Kinetics("LO burning",[0, 0, 1, 0, 14.06, 0],[6.58, 0, 0, 11.96, 0, 0],4.4394e11,138281,6.7625e6)
        Kinetics_map["Coke burning"] = Kinetics("Coke burning",[0, 0, 0, 0, 1.18, 1],[0.55, 0, 0, 1, 0, 0],6123.8,58615,523400)

        
    if fluid_option ==2:
        
        class Component:
            def __init__(self, NAME, CMM, PCRIT, TCRIT, AVG, BVG, AVISC, BVISC, CPG1, CPG2, CPG3, CPG4, CPL1, CPL2, MASSDEN, CP, CT1, CT2, SOLID_DEN, SOLID_CP):
                self.NAME = NAME
                self.CMM = CMM
                self.PCRIT = PCRIT
                self.TCRIT = TCRIT
                self.AVG = AVG
                self.BVG = BVG
                self.AVISC = AVISC
                self.BVISC = BVISC
                self.CPG1 = CPG1
                self.CPG2 = CPG2
                self.CPG3 = CPG3
                self.CPG4 = CPG4
                self.CPL1 = CPL1
                self.CPL2 = CPL2
                self.MASSDEN = MASSDEN
                self.CP = CP
                self.CT1 = CT1
                self.CT2 = CT2
                self.SOLID_DEN = SOLID_DEN
                self.SOLID_CP = SOLID_CP

    
            def getNAME(self):
                return self.NAME
            def getCMM(self):
                return self.CMM
            def getPCRIT(self):
                return self.PCRIT
            def getTCRIT(self):
                return self.TCRIT                  
            def getAVG(self):
                return self.AVG                 
            def getBVG(self):
                return self.BVG    
            def getAVISC(self):
                return self.AVISC     
            def getBVISC(self):
                return self.BVISC     
            def getCPG1(self):
                return self.CPG1          
            def getCPG2(self):
                return self.CPG2                 
            def getCPG3(self):
                return self.CPG3                 
            def getCPG4(self):
                return self.CPG4
            def getCPL1(self):
                return self.CPL1 
            def getCPL2(self):
                return self.CPL2            
            def getMASSDEN(self):
                return self.MASSDEN                 
            def getCP(self):
                return self.CP                 
            def getCT1(self):
                return self.CT1        
            def getCT2(self):
                return self.CT2   
            def getSOLID_DEN(self):
                return self.SOLID_DEN       
            def getSOLID_CP(self):
                return self.SOLID_CP

            

            def setNAME(self, value):
                self.NAME = value
            def setCMM(self, value):
                self.CMM = value
            def setPCRIT(self, value):
                self.PCRIT = value
            def setTCRIT(self, value):
                self.TCRIT = value              
            def setAVG(self, value):
                self.AVG = value              
            def setBVG(self, value):
                self.BVG = value 
            def setAVISC(self, value):
                self.AVISC = value   
            def setBVISC(self, value):
                self.BVISC = value   
            def setCPG1(self, value):
                self.CPG1 = value         
            def setCPG2(self, value):
                self.CPG2 = value               
            def setCPG3(self, value):
                self.CPG3 = value               
            def setCPG4(self, value):
                self.CPG4 = value       
            def setCPL1(self, value):
                self.CPL1 = value               
            def setCPL2(self, value):
                self.CPL2 = value     
            def setMASSDEN(self, value):
                self.MASSDEN = value               
            def setCP(self, value):
                self.CP = value               
            def setCT1(self, value):
                self.CT1 = value 
            def setCT2(self, value):
                self.CT2 = value     
            def setSOLID_DEN(self, value):
                self.SOLID_DEN = value      
            def setSOLID_CP(self, value):
                self.SOLID_CP = value

            

        class Kinetics:
            def __init__(self, NAME, REAC, PROD, RORDER, FREQFAC, EACT, RENTH):
                self.NAME = NAME
                self.REAC = REAC
                self.PROD = PROD
                self.RORDER = RORDER
                self.FREQFAC = FREQFAC
                self.EACT = EACT
                self.RENTH = RENTH

            def getNAME(self):
                return self.NAME  
            def getREAC(self):
                return self.REAC
            def getPROD(self):
                return self.PROD
            def getRORDER(self):
                return self.RORDER
            def getFREQFAC(self):
                return self.FREQFAC
            def getEACT(self):
                return self.EACT
            def getRENTH(self):
               return self.RENTH
            
            def setNAME(self, value):
                self.NAME = value
            def setREAC(self, value):
                self.REAC = value
            def setPROD(self, value):
                self.PROD = value
            def setRORDER(self, value):
                self.RORDER = value
            def setFREQFAC(self, value):
                self.FREQFAC = value
            def setEACT(self, value):
               self.EACT = value
            def setRENTH(self, value):
                self.RENTH = value
        
        
        #(NAME, CMM, PCRIT, TCRIT, AVG, BVG, AVISC, BVISC, CPG1, CPG2, CPG3, CPG4, CPL1, CPL2, MASSDEN, CP, CT1, CT2, SOLID_DEN, SOLID_CP):
        Component_map = dict()
        Component_map["H2O"] = Component("H2O",[1.8e-2],[22083],[373.8],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0.001],[0],[0],[0],[],[])
        Component_map["OIL"] = Component("OIL",[4.73e-1],[890],[171],[0.0001610891804],[0.7453161006],[1.426417368e-11],[10823.06574],[26.804420692906],[0.005649089963],[0.000095012314],[-0.000000054709],[524.8821790],[1.148635444845],[0.000999798],[7.25e-7],[0.00069242],[0],[],[])
        Component_map["N2"] = Component("N2",[2.8e-2],[3392],[-147],[0.0003500869287],[0.6927470725],[],[],[30.956477056957],[-0.012716023994],[0.000025490143],[-0.000000011065],[0],[0],[],[],[],[],[],[])
        Component_map["O2"] = Component("O2",[3.2e-2],[5033],[-118],[0.000362791571],[0.7120986013],[],[],[28.600167325729],[-0.003497011859],[0.000024399453],[-0.000000014928],[0],[0],[],[],[],[],[],[])
        Component_map["CO2"] = Component("CO2",[4.4e-2],[7377],[31],[0.0001865724378],[0.7754816784],[],[],[19.474325955388],[0.075654731286],[-0.000060750197],[0.000000020109],[0],[0],[],[],[],[],[],[])
        Component_map["CO"] = Component("CO",[2.8e-2],[3496],[-144],[0.0003315014585],[0.7037315714],[],[],[30.990187019402],[-0.01392019971],[0.00003014996],[-0.00000001415],[0],[0],[],[],[],[],[],[])
        Component_map["Coke1"] = Component("Coke1",[1.88e-2],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[0.0014,0,0],[8.3908475,0.0439425])
        Component_map["Coke2"] = Component("Coke2",[1.36e-2],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[0.0014,0,0],[6.96015,0.03645])
        Component_map["Ci"] = Component("Ci",[2.08e-2],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[0.0014,0,0],[7.192155,0.037665])
        
        Kinetics_map = dict()
        Kinetics_map["RXN1"] = Kinetics("RXN1",[0.00E+00, 1.00E+00, 0.00E+00, 9.50E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00],[22.292,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.0E+01,0.00E+00,0.00E+00],[0.00E+00,1.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00],7E5,1E5,3.98E+06)
        Kinetics_map["RXN2"] = Kinetics("RXN2",[0.00E+00,0.00E+00,0.00E+00,1.50E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],[1.3815E+00,0.00E+00,0.00E+00,0.00E+00,0.75E+00,0.3181E+00,0.00E+00,0.00E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],1E6,1.14E5,6.28E5)
        Kinetics_map["RXN3"] = Kinetics("RXN3",[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.3824E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],1.6E6,8E4,0)
        Kinetics_map["RXN4"] = Kinetics("RXN4",[0.00E+00,0.00E+00,0.00E+00,0.650000,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00],[0.239300,0.00E+00,0.00E+00,0.00E+00,0.565200,0.186200,0.00E+00,0.00E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00],4.5E6,1.4E5,2.72E5)
        Kinetics_map["RXN5"] = Kinetics("RXN5",[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.9134615],[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.00E+00,0.00E+00,0.00E+00],5.334E4,5.099E4,0)
        Kinetics_map["RXN6"] = Kinetics("RXN6",[0.00E+00,0.00E+00,0.00E+00,0.900000,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00],[0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,2.00E-01,0.00E+00,0.00E+00,0.00E+00],[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00],1.621E10,100000,1.514E6)
          


    return Component_map, Kinetics_map


   
    


def print_fluid(fileID, fluid_option, Component, Kinetics):
    print('**  ==============  FLUID DEFINITIONS  ======================', file = fileID)
    if fluid_option == 1:
        print("""
*MODEL 6 5 3   ** Number of noncondensible gases is numy-numx = 2
** Number of solid components is ncomp-numy = 1
              """, file = fileID)
        print('*COMPNAME ',"\'"+Component["WATER"].getNAME()+"\'","\'"+Component["HEVY OIL"].getNAME()+"\'","\'"+Component["LITE OIL"].getNAME()+"\'","\'"+Component["INRT GAS"].getNAME()+"\'","\'"+Component["OXYGEN"].getNAME()+"\'","\'"+Component["COKE"].getNAME()+"\'", file = fileID)
        print('*CMM ',*Component["WATER"].getCMM(),*Component["HEVY OIL"].getCMM(),*Component["LITE OIL"].getCMM(),*Component["INRT GAS"].getCMM(),*Component["OXYGEN"].getCMM(),*Component["COKE"].getCMM(), file = fileID)
        print('*PCRIT ',*Component["WATER"].getPCRIT(),*Component["HEVY OIL"].getPCRIT(),*Component["LITE OIL"].getPCRIT(),*Component["INRT GAS"].getPCRIT(),*Component["OXYGEN"].getPCRIT(),*Component["COKE"].getPCRIT(), file = fileID)
        print('*TCRIT ',*Component["WATER"].getTCRIT(),*Component["HEVY OIL"].getTCRIT(),*Component["LITE OIL"].getTCRIT(),*Component["INRT GAS"].getTCRIT(),*Component["OXYGEN"].getTCRIT(),*Component["COKE"].getTCRIT(), file = fileID)
        print('*ACEN ',*Component["WATER"].getACEN(),*Component["HEVY OIL"].getACEN(),*Component["LITE OIL"].getACEN(),*Component["INRT GAS"].getACEN(),*Component["OXYGEN"].getACEN(),*Component["COKE"].getACEN(), file = fileID)
        print('*AVG ',*Component["WATER"].getAVG(),*Component["HEVY OIL"].getAVG(),*Component["LITE OIL"].getAVG(),*Component["INRT GAS"].getAVG(),*Component["OXYGEN"].getAVG(),*Component["COKE"].getAVG(), file = fileID)
        print('*BVG ',*Component["WATER"].getBVG(),*Component["HEVY OIL"].getBVG(),*Component["LITE OIL"].getBVG(),*Component["INRT GAS"].getBVG(),*Component["OXYGEN"].getBVG(),*Component["COKE"].getBVG(), file = fileID)
        print('*AVISC ',*Component["WATER"].getAVISC(),*Component["HEVY OIL"].getAVISC(),*Component["LITE OIL"].getAVISC(),*Component["INRT GAS"].getAVISC(),*Component["OXYGEN"].getAVISC(),*Component["COKE"].getAVISC(), file = fileID)
        print('*BVISC ',*Component["WATER"].getBVISC(),*Component["HEVY OIL"].getBVISC(),*Component["LITE OIL"].getBVISC(),*Component["INRT GAS"].getBVISC(),*Component["OXYGEN"].getBVISC(),*Component["COKE"].getBVISC(), file = fileID)
        print('*CPG1 ',*Component["WATER"].getCPG1(),*Component["HEVY OIL"].getCPG1(),*Component["LITE OIL"].getCPG1(),*Component["INRT GAS"].getCPG1(),*Component["OXYGEN"].getCPG1(),*Component["COKE"].getCPG1(), file = fileID)
        print('*CPG2 ',*Component["WATER"].getCPG2(),*Component["HEVY OIL"].getCPG2(),*Component["LITE OIL"].getCPG2(),*Component["INRT GAS"].getCPG2(),*Component["OXYGEN"].getCPG2(),*Component["COKE"].getCPG2(), file = fileID)
        print('*CPG3 ',*Component["WATER"].getCPG3(),*Component["HEVY OIL"].getCPG3(),*Component["LITE OIL"].getCPG3(),*Component["INRT GAS"].getCPG3(),*Component["OXYGEN"].getCPG3(),*Component["COKE"].getCPG3(), file = fileID)
        print('*CPG4 ',*Component["WATER"].getCPG4(),*Component["HEVY OIL"].getCPG4(),*Component["LITE OIL"].getCPG4(),*Component["INRT GAS"].getCPG4(),*Component["OXYGEN"].getCPG4(),*Component["COKE"].getCPG4(), file = fileID)
        print('*MOLDEN ',*Component["WATER"].getMOLDEN(),*Component["HEVY OIL"].getMOLDEN(),*Component["LITE OIL"].getMOLDEN(),*Component["INRT GAS"].getMOLDEN(),*Component["OXYGEN"].getMOLDEN(),*Component["COKE"].getMOLDEN(), file = fileID)
        print('*CP ',*Component["WATER"].getCP(),*Component["HEVY OIL"].getCP(),*Component["LITE OIL"].getCP(),*Component["INRT GAS"].getCP(),*Component["OXYGEN"].getCP(),*Component["COKE"].getCP(), file = fileID)
        print('*CT1 ',*Component["WATER"].getCT1(),*Component["HEVY OIL"].getCT1(),*Component["LITE OIL"].getCT1(),*Component["INRT GAS"].getCT1(),*Component["OXYGEN"].getCT1(),*Component["COKE"].getCT1(), file = fileID)
        print('*SOLID_DEN ',"\'"+Component["COKE"].getNAME()+"\'",*Component["WATER"].getSOLID_DEN(),*Component["HEVY OIL"].getSOLID_DEN(),*Component["LITE OIL"].getSOLID_DEN(),*Component["INRT GAS"].getSOLID_DEN(),*Component["OXYGEN"].getSOLID_DEN(),*Component["COKE"].getSOLID_DEN(), file = fileID)
        print('*SOLID_CP ',"\'"+Component["COKE"].getNAME()+"\'",*Component["WATER"].getSOLID_CP(),*Component["HEVY OIL"].getSOLID_CP(),*Component["LITE OIL"].getSOLID_CP(),*Component["INRT GAS"].getSOLID_CP(),*Component["OXYGEN"].getSOLID_CP(),*Component["COKE"].getSOLID_CP(), file = fileID)
        
        print('**Reactions', file = fileID)
        print('**-----------', file = fileID)
        
        print('** Cracking:  heavy oil -> light oil + coke (50/50 by mass)', file = fileID)
        print('*STOREAC ',Kinetics["Cracking"].getREAC()[0],'',Kinetics["Cracking"].getREAC()[1],'',Kinetics["Cracking"].getREAC()[2],'',Kinetics["Cracking"].getREAC()[3],'',Kinetics["Cracking"].getREAC()[4],'',Kinetics["Cracking"].getREAC()[5], file = fileID)
        print('*STOPROD ',Kinetics["Cracking"].getPROD()[0],'',Kinetics["Cracking"].getPROD()[1],'',Kinetics["Cracking"].getPROD()[2],'',Kinetics["Cracking"].getPROD()[3],'',Kinetics["Cracking"].getPROD()[4],'',Kinetics["Cracking"].getPROD()[5], file = fileID)
        print('*FREQFAC ',Kinetics["Cracking"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["Cracking"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["Cracking"].getRENTH(), file = fileID)
        
        print('** Light oil burning:  light oil + o2 -> water + inert gas + energy', file = fileID)
        print('*STOREAC ',Kinetics["LO burning"].getREAC()[0],'',Kinetics["LO burning"].getREAC()[1],'',Kinetics["LO burning"].getREAC()[2],'',Kinetics["LO burning"].getREAC()[3],'',Kinetics["LO burning"].getREAC()[4],'',Kinetics["LO burning"].getREAC()[5], file = fileID)
        print('*STOPROD ',Kinetics["LO burning"].getPROD()[0],'',Kinetics["LO burning"].getPROD()[1],'',Kinetics["LO burning"].getPROD()[2],'',Kinetics["LO burning"].getPROD()[3],'',Kinetics["LO burning"].getPROD()[4],'',Kinetics["LO burning"].getPROD()[5], file = fileID)
        print('*FREQFAC ',Kinetics["LO burning"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["LO burning"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["LO burning"].getRENTH(), file = fileID)
        
        print('** Heavy oil burning:  heavy oil + o2 -> water + inert gas + energy', file = fileID)
        print('*STOREAC ',Kinetics["HO burning"].getREAC()[0],'',Kinetics["HO burning"].getREAC()[1],'',Kinetics["HO burning"].getREAC()[2],'',Kinetics["HO burning"].getREAC()[3],'',Kinetics["HO burning"].getREAC()[4],'',Kinetics["HO burning"].getREAC()[5], file = fileID)
        print('*STOPROD ',Kinetics["HO burning"].getPROD()[0],'',Kinetics["HO burning"].getPROD()[1],'',Kinetics["HO burning"].getPROD()[2],'',Kinetics["HO burning"].getPROD()[3],'',Kinetics["HO burning"].getPROD()[4],'',Kinetics["HO burning"].getPROD()[5], file = fileID)
        print('*FREQFAC ',Kinetics["HO burning"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["HO burning"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["HO burning"].getRENTH(), file = fileID)
        
        print('** Coke oil burning:  coke oil + o2 -> water + inert gas + energy', file = fileID)
        print('*STOREAC ',Kinetics["Coke burning"].getREAC()[0],'',Kinetics["Coke burning"].getREAC()[1],'',Kinetics["Coke burning"].getREAC()[2],'',Kinetics["Coke burning"].getREAC()[3],'',Kinetics["Coke burning"].getREAC()[4],'',Kinetics["Coke burning"].getREAC()[5], file = fileID)
        print('*STOPROD ',Kinetics["Coke burning"].getPROD()[0],'',Kinetics["Coke burning"].getPROD()[1],'',Kinetics["Coke burning"].getPROD()[2],'',Kinetics["Coke burning"].getPROD()[3],'',Kinetics["Coke burning"].getPROD()[4],'',Kinetics["Coke burning"].getPROD()[5], file = fileID)
        print('*FREQFAC ',Kinetics["Coke burning"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["Coke burning"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["Coke burning"].getRENTH(), file = fileID)

        
    if fluid_option == 2:
        print("""
*MODEL 9 6 2 
** Number of noncondensible gases is numy-numx = 4
** Number of solid components is ncomp-numy = 4

              """, file = fileID)

        print('*COMPNAME ',"\'"+Component["H2O"].getNAME()+"\'","\'"+Component["OIL"].getNAME()+"\'","\'"+Component["N2"].getNAME()+"\'","\'"+Component["O2"].getNAME()+"\'","\'"+Component["CO2"].getNAME()+"\'","\'"+Component["CO"].getNAME()+"\'","\'"+Component["Coke1"].getNAME()+"\'","\'"+Component["Coke2"].getNAME()+"\'","\'"+Component["Ci"].getNAME()+"\'", file = fileID)
        print('*CMM ',*Component["H2O"].getCMM(),*Component["OIL"].getCMM(),*Component["N2"].getCMM(),*Component["O2"].getCMM(),*Component["CO2"].getCMM(),*Component["CO"].getCMM(),*Component["Coke1"].getCMM(),*Component["Coke2"].getCMM(),*Component["Ci"].getCMM(), file = fileID)
        print('*PCRIT ',*Component["H2O"].getPCRIT(),*Component["OIL"].getPCRIT(),*Component["N2"].getPCRIT(),*Component["O2"].getPCRIT(),*Component["CO2"].getPCRIT(),*Component["CO"].getPCRIT(),*Component["Coke1"].getPCRIT(),*Component["Coke2"].getPCRIT(),*Component["Ci"].getPCRIT(), file = fileID)
        print('*TCRIT ',*Component["H2O"].getTCRIT(),*Component["OIL"].getTCRIT(),*Component["N2"].getTCRIT(),*Component["O2"].getTCRIT(),*Component["CO2"].getTCRIT(),*Component["CO"].getTCRIT(),*Component["Coke1"].getTCRIT(),*Component["Coke2"].getTCRIT(),*Component["Ci"].getTCRIT(), file = fileID)
      
        print('*AVG ',*Component["H2O"].getAVG(),*Component["OIL"].getAVG(),*Component["N2"].getAVG(),*Component["O2"].getAVG(),*Component["CO2"].getAVG(),*Component["CO"].getAVG(),*Component["Coke1"].getAVG(),*Component["Coke2"].getAVG(),*Component["Ci"].getAVG(), file = fileID)
        print('*BVG ',*Component["H2O"].getBVG(),*Component["OIL"].getBVG(),*Component["N2"].getBVG(),*Component["O2"].getBVG(),*Component["CO2"].getBVG(),*Component["CO"].getBVG(),*Component["Coke1"].getBVG(),*Component["Coke2"].getBVG(),*Component["Ci"].getBVG(), file = fileID)
        #print('*AVISC ',*Component["H2O"].getAVISC(),*Component["OIL"].getAVISC(),*Component["N2"].getAVISC(),*Component["O2"].getAVISC(),*Component["CO2"].getAVISC(),*Component["CO"].getAVISC(),*Component["Coke1"].getAVISC(),*Component["Coke2"].getAVISC(),*Component["Ci"].getAVISC(), file = fileID)
        #print('*BVISC ',*Component["H2O"].getBVISC(),*Component["OIL"].getBVISC(),*Component["N2"].getBVISC(),*Component["O2"].getBVISC(),*Component["CO2"].getBVISC(),*Component["CO"].getBVISC(),*Component["Coke1"].getBVISC(),*Component["Coke2"].getBVISC(),*Component["Ci"].getBVISC(), file = fileID)
        print('*CPG1 ',*Component["H2O"].getCPG1(),*Component["OIL"].getCPG1(),*Component["N2"].getCPG1(),*Component["O2"].getCPG1(),*Component["CO2"].getCPG1(),*Component["CO"].getCPG1(),*Component["Coke1"].getCPG1(),*Component["Coke2"].getCPG1(),*Component["Ci"].getCPG1(), file = fileID)
        print('*CPG2 ',*Component["H2O"].getCPG2(),*Component["OIL"].getCPG2(),*Component["N2"].getCPG2(),*Component["O2"].getCPG2(),*Component["CO2"].getCPG2(),*Component["CO"].getCPG2(),*Component["Coke1"].getCPG2(),*Component["Coke2"].getCPG2(),*Component["Ci"].getCPG2(), file = fileID)
        print('*CPG3 ',*Component["H2O"].getCPG3(),*Component["OIL"].getCPG3(),*Component["N2"].getCPG3(),*Component["O2"].getCPG3(),*Component["CO2"].getCPG3(),*Component["CO"].getCPG3(),*Component["Coke1"].getCPG3(),*Component["Coke2"].getCPG3(),*Component["Ci"].getCPG3(), file = fileID)
        print('*CPG4 ',*Component["H2O"].getCPG4(),*Component["OIL"].getCPG4(),*Component["N2"].getCPG4(),*Component["O2"].getCPG4(),*Component["CO2"].getCPG4(),*Component["CO"].getCPG4(),*Component["Coke1"].getCPG4(),*Component["Coke2"].getCPG4(),*Component["Ci"].getCPG4(), file = fileID)
        print('*CPL1 ',*Component["H2O"].getCPL1(),*Component["OIL"].getCPL1(),*Component["N2"].getCPL1(),*Component["O2"].getCPL1(),*Component["CO2"].getCPL1(),*Component["CO"].getCPL1(),*Component["Coke1"].getCPL1(),*Component["Coke2"].getCPL1(),*Component["Ci"].getCPL1(), file = fileID)
        print('*CPL2 ',*Component["H2O"].getCPL2(),*Component["OIL"].getCPL2(),*Component["N2"].getCPL2(),*Component["O2"].getCPL2(),*Component["CO2"].getCPL2(),*Component["CO"].getCPL2(),*Component["Coke1"].getCPL2(),*Component["Coke2"].getCPL2(),*Component["Ci"].getCPL2(), file = fileID)
        
        print("""
*HVAPR 0 0 
*EV 0 0 
              """, file = fileID)
        
        print('*MASSDEN ',*Component["H2O"].getMASSDEN(),*Component["OIL"].getMASSDEN(),*Component["N2"].getMASSDEN(),*Component["O2"].getMASSDEN(),*Component["CO2"].getMASSDEN(),*Component["CO"].getMASSDEN(),*Component["Coke1"].getMASSDEN(),*Component["Coke2"].getMASSDEN(),*Component["Ci"].getMASSDEN(), file = fileID)
        print('*CP ',*Component["H2O"].getCP(),*Component["OIL"].getCP(),*Component["N2"].getCP(),*Component["O2"].getCP(),*Component["CO2"].getCP(),*Component["CO"].getCP(),*Component["Coke1"].getCP(),*Component["Coke2"].getCP(),*Component["Ci"].getCP(), file = fileID)
        print('*CT1 ',*Component["H2O"].getCT1(),*Component["OIL"].getCT1(),*Component["N2"].getCT1(),*Component["O2"].getCT1(),*Component["CO2"].getCT1(),*Component["CO"].getCT1(),*Component["Coke1"].getCT1(),*Component["Coke2"].getCT1(),*Component["Ci"].getCT1(), file = fileID)
        print('*CT2 ',*Component["H2O"].getCT2(),*Component["OIL"].getCT2(),*Component["N2"].getCT2(),*Component["O2"].getCT2(),*Component["CO2"].getCT2(),*Component["CO"].getCT2(),*Component["Coke1"].getCT2(),*Component["Coke2"].getCT2(),*Component["Ci"].getCT2(), file = fileID)
        print('*SOLID_DEN ',"\'"+Component["Coke1"].getNAME()+"\'",*Component["Coke1"].getSOLID_DEN(), file = fileID)
        print('*SOLID_CP ',"\'"+Component["Coke1"].getNAME()+"\'",*Component["Coke1"].getSOLID_CP(), file = fileID)
        print('*SOLID_DEN ',"\'"+Component["Coke2"].getNAME()+"\'",*Component["Coke2"].getSOLID_DEN(), file = fileID)
        print('*SOLID_CP ',"\'"+Component["Coke2"].getNAME()+"\'",*Component["Coke2"].getSOLID_CP(), file = fileID)
        print('*SOLID_DEN ',"\'"+Component["Ci"].getNAME()+"\'",*Component["Ci"].getSOLID_DEN(), file = fileID)
        print('*SOLID_CP ',"\'"+Component["Ci"].getNAME()+"\'",*Component["Ci"].getSOLID_CP(), file = fileID)
        
        print("""
*VISCTABLE
10      0    100000
80      0      2084
100     0       580
1000    0         1
              """, file = fileID)
        
        print('**Reactions', file = fileID)
        print('**-----------', file = fileID)
        
        print('**RXN1: Oil + 6O2 > 19Coke1 + 23.55555H2O', file = fileID)
        print('*STOREAC ',Kinetics["RXN1"].getREAC()[0],'',Kinetics["RXN1"].getREAC()[1],'',Kinetics["RXN1"].getREAC()[2],'',Kinetics["RXN1"].getREAC()[3],'',Kinetics["RXN1"].getREAC()[4],'',Kinetics["RXN1"].getREAC()[5],'',Kinetics["RXN1"].getREAC()[6],'',Kinetics["RXN1"].getREAC()[7],'',Kinetics["RXN1"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN1"].getPROD()[0],'',Kinetics["RXN1"].getPROD()[1],'',Kinetics["RXN1"].getPROD()[2],'',Kinetics["RXN1"].getPROD()[3],'',Kinetics["RXN1"].getPROD()[4],'',Kinetics["RXN1"].getPROD()[5],'',Kinetics["RXN1"].getPROD()[6],'',Kinetics["RXN1"].getPROD()[7],'',Kinetics["RXN1"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN1"].getRORDER()[0],'',Kinetics["RXN1"].getRORDER()[1],'',Kinetics["RXN1"].getRORDER()[2],'',Kinetics["RXN1"].getRORDER()[3],'',Kinetics["RXN1"].getRORDER()[4],'',Kinetics["RXN1"].getRORDER()[5],'',Kinetics["RXN1"].getRORDER()[6],'',Kinetics["RXN1"].getRORDER()[7],'',Kinetics["RXN1"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN1"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN1"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN1"].getRENTH(), file = fileID)
        print("*O2PP 'O2'", file = fileID)
        
        print('**RXN2: **Coke1 + 1.5O2 > CO2 + 0.5CO + 0.5H2O', file = fileID)
        print('*STOREAC ',Kinetics["RXN2"].getREAC()[0],'',Kinetics["RXN2"].getREAC()[1],'',Kinetics["RXN2"].getREAC()[2],'',Kinetics["RXN2"].getREAC()[3],'',Kinetics["RXN2"].getREAC()[4],'',Kinetics["RXN2"].getREAC()[5],'',Kinetics["RXN2"].getREAC()[6],'',Kinetics["RXN2"].getREAC()[7],'',Kinetics["RXN2"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN2"].getPROD()[0],'',Kinetics["RXN2"].getPROD()[1],'',Kinetics["RXN2"].getPROD()[2],'',Kinetics["RXN2"].getPROD()[3],'',Kinetics["RXN2"].getPROD()[4],'',Kinetics["RXN2"].getPROD()[5],'',Kinetics["RXN2"].getPROD()[6],'',Kinetics["RXN2"].getPROD()[7],'',Kinetics["RXN2"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN2"].getRORDER()[0],'',Kinetics["RXN2"].getRORDER()[1],'',Kinetics["RXN2"].getRORDER()[2],'',Kinetics["RXN2"].getRORDER()[3],'',Kinetics["RXN2"].getRORDER()[4],'',Kinetics["RXN2"].getRORDER()[5],'',Kinetics["RXN2"].getRORDER()[6],'',Kinetics["RXN2"].getRORDER()[7],'',Kinetics["RXN2"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN2"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN2"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN2"].getRENTH(), file = fileID)
        print("*O2PP 'O2'", file = fileID)
        
        print('**RXN3: Coke1 > 1.032608Coke2', file = fileID)
        print('*STOREAC ',Kinetics["RXN3"].getREAC()[0],'',Kinetics["RXN3"].getREAC()[1],'',Kinetics["RXN3"].getREAC()[2],'',Kinetics["RXN3"].getREAC()[3],'',Kinetics["RXN3"].getREAC()[4],'',Kinetics["RXN3"].getREAC()[5],'',Kinetics["RXN3"].getREAC()[6],'',Kinetics["RXN3"].getREAC()[7],'',Kinetics["RXN3"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN3"].getPROD()[0],'',Kinetics["RXN3"].getPROD()[1],'',Kinetics["RXN3"].getPROD()[2],'',Kinetics["RXN3"].getPROD()[3],'',Kinetics["RXN3"].getPROD()[4],'',Kinetics["RXN3"].getPROD()[5],'',Kinetics["RXN3"].getPROD()[6],'',Kinetics["RXN3"].getPROD()[7],'',Kinetics["RXN3"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN3"].getRORDER()[0],'',Kinetics["RXN3"].getRORDER()[1],'',Kinetics["RXN3"].getRORDER()[2],'',Kinetics["RXN3"].getRORDER()[3],'',Kinetics["RXN3"].getRORDER()[4],'',Kinetics["RXN3"].getRORDER()[5],'',Kinetics["RXN3"].getRORDER()[6],'',Kinetics["RXN3"].getRORDER()[7],'',Kinetics["RXN3"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN3"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN3"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN3"].getRENTH(), file = fileID)
        print("**O2PP 'O2'", file = fileID)
        
        print('**RXN4: Coke2 + 1.2O2 > 0.4CO + 1CO2', file = fileID)
        print('*STOREAC ',Kinetics["RXN4"].getREAC()[0],'',Kinetics["RXN4"].getREAC()[1],'',Kinetics["RXN4"].getREAC()[2],'',Kinetics["RXN4"].getREAC()[3],'',Kinetics["RXN4"].getREAC()[4],'',Kinetics["RXN4"].getREAC()[5],'',Kinetics["RXN4"].getREAC()[6],'',Kinetics["RXN4"].getREAC()[7],'',Kinetics["RXN4"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN4"].getPROD()[0],'',Kinetics["RXN4"].getPROD()[1],'',Kinetics["RXN4"].getPROD()[2],'',Kinetics["RXN4"].getPROD()[3],'',Kinetics["RXN4"].getPROD()[4],'',Kinetics["RXN4"].getPROD()[5],'',Kinetics["RXN4"].getPROD()[6],'',Kinetics["RXN4"].getPROD()[7],'',Kinetics["RXN4"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN4"].getRORDER()[0],'',Kinetics["RXN4"].getRORDER()[1],'',Kinetics["RXN4"].getRORDER()[2],'',Kinetics["RXN4"].getRORDER()[3],'',Kinetics["RXN4"].getRORDER()[4],'',Kinetics["RXN4"].getRORDER()[5],'',Kinetics["RXN4"].getRORDER()[6],'',Kinetics["RXN4"].getRORDER()[7],'',Kinetics["RXN4"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN4"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN4"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN4"].getRENTH(), file = fileID)
        print("*O2PP 'O2'", file = fileID)
        
        print('**RXN5: Coke1 > 0.9134615 Ci', file = fileID)
        print('*STOREAC ',Kinetics["RXN5"].getREAC()[0],'',Kinetics["RXN5"].getREAC()[1],'',Kinetics["RXN5"].getREAC()[2],'',Kinetics["RXN5"].getREAC()[3],'',Kinetics["RXN5"].getREAC()[4],'',Kinetics["RXN5"].getREAC()[5],'',Kinetics["RXN5"].getREAC()[6],'',Kinetics["RXN5"].getREAC()[7],'',Kinetics["RXN5"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN5"].getPROD()[0],'',Kinetics["RXN5"].getPROD()[1],'',Kinetics["RXN5"].getPROD()[2],'',Kinetics["RXN5"].getPROD()[3],'',Kinetics["RXN5"].getPROD()[4],'',Kinetics["RXN5"].getPROD()[5],'',Kinetics["RXN5"].getPROD()[6],'',Kinetics["RXN5"].getPROD()[7],'',Kinetics["RXN5"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN5"].getRORDER()[0],'',Kinetics["RXN5"].getRORDER()[1],'',Kinetics["RXN5"].getRORDER()[2],'',Kinetics["RXN5"].getRORDER()[3],'',Kinetics["RXN5"].getRORDER()[4],'',Kinetics["RXN5"].getRORDER()[5],'',Kinetics["RXN5"].getRORDER()[6],'',Kinetics["RXN5"].getRORDER()[7],'',Kinetics["RXN5"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN5"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN5"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN5"].getRENTH(), file = fileID)
        print("**O2PP 'O2'", file = fileID)
        
        print('**RXN6: 1Ci + 0.9O2 > 0.2CO + 1CO2', file = fileID)
        print('*STOREAC ',Kinetics["RXN6"].getREAC()[0],'',Kinetics["RXN6"].getREAC()[1],'',Kinetics["RXN6"].getREAC()[2],'',Kinetics["RXN6"].getREAC()[3],'',Kinetics["RXN6"].getREAC()[4],'',Kinetics["RXN6"].getREAC()[5],'',Kinetics["RXN6"].getREAC()[6],'',Kinetics["RXN6"].getREAC()[7],'',Kinetics["RXN6"].getREAC()[8], file = fileID)
        print('*STOPROD ',Kinetics["RXN6"].getPROD()[0],'',Kinetics["RXN6"].getPROD()[1],'',Kinetics["RXN6"].getPROD()[2],'',Kinetics["RXN6"].getPROD()[3],'',Kinetics["RXN6"].getPROD()[4],'',Kinetics["RXN6"].getPROD()[5],'',Kinetics["RXN6"].getPROD()[6],'',Kinetics["RXN6"].getPROD()[7],'',Kinetics["RXN6"].getPROD()[8], file = fileID)
        print('*RORDER ',Kinetics["RXN6"].getRORDER()[0],'',Kinetics["RXN6"].getRORDER()[1],'',Kinetics["RXN6"].getRORDER()[2],'',Kinetics["RXN6"].getRORDER()[3],'',Kinetics["RXN6"].getRORDER()[4],'',Kinetics["RXN6"].getRORDER()[5],'',Kinetics["RXN6"].getRORDER()[6],'',Kinetics["RXN6"].getRORDER()[7],'',Kinetics["RXN6"].getRORDER()[8], file = fileID)
        print('*FREQFAC ',Kinetics["RXN6"].getFREQFAC(), file = fileID)
        print('*EACT ',Kinetics["RXN6"].getEACT(), file = fileID)
        print('*RENTH ',Kinetics["RXN6"].getRENTH(), file = fileID)
        print("**O2PP 'O2'", file = fileID)
        




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
        

### BEGIN STARS BINARY READER
class STARSOutputFile:
    '''
    Class to hold output files from STARS
    '''
    def __init__(self, input_file_base, reaction_type = 'BO'):
    #def __init__(self, input_file_base, reaction_type = 'MURAT'):

        '''
        Input:
            input_file_base - base of file for the simulation

        '''

        # Open STARS output files
        FID = [line.decode('utf-8', errors='ignore') for line in open(input_file_base + '.irf', 'rb')]
        FID_bin = open(input_file_base + '.mrf', 'rb')

        # Select number of species
        #   NEED TO GENERALIZE ON FUTURE ITERATIONS
        if reaction_type == 'MURAT':
            numSPHIST = 22 # MURAT
        elif reaction_type == 'BO':
            numSPHIST = 18 # BO
        else:
            raise Exception('Invalid reaction type')
        num_grids = 22

        # Initialize variables/objects for parsing data
        i = 0
        self.UNIT = {} # Create units dictionary
        self.COMP = {} # Compositions dictionary
        self.TIME = {} # Time dictionary
        self.GRID = {} # Grid dictionary
        self.GRID['TEMP'] = []
        self.SP = {}
        self.SP['VAL'] = []
        self.SPHIST = {}

        self.TIME['VECT'] = []
        self.TIME['CHR'] = []
        self.TIME['CHR_UNIT'] = []

        self.REC = {}
        REC_list = ['WELL-REC', 'LAYER-REC', 'GROUP-REC', 'SECTOR-REC', 'RSTSPEC01-REC', 'RSTSPEC02-REC', 'RSTSPEC03-REC',
            'RSTSPEC04-REC', 'RSTSPEC05-REC', 'RSTSPEC06-REC', 'RSTSPEC07-REC', 'RSTSPEC08-REC', 'RSTSPEC09-REC',
            'RSTSPEC10-REC', 'RSTSPEC11-REC', 'RSTSPEC12-REC', 'RSTSPEC13-REC', 'RSTSPEC14-REC', 'RSTSPEC15-REC',
            'RSTSPEC16-REC', 'RSTSPEC17-REC', 'RSTSPEC18-REC', 'RSTSPEC19-REC', 'RSTSPEC20-REC', 'RSTSPEC21-REC',
            'RSTSPEC22-REC']
        
        skip_list = ['WELL-ARRAY', 'LAYER-ARRAY', 'GROUP-ARRAY']

        # Iterate over lines of file
        while i < len(FID):
            line_split = FID[i].split()
            
            # Parse case-by-case quantities
            if line_split[0] == 'INTERNAL-UNIT-TABLE':
                i+=1 
                self.UNIT['INT'] = [str(FID[i+j].split()[1]) for j in range(21)]
                self.UNIT['DESCP'] = [str(FID[i+j].split()[3]) for j in range(21)]
                i+=21 
                
            elif line_split[0] == 'OUTPUT-UNIT-TABLE':
                i+=1
                self.UNIT['OUT'] = [str(FID[i+j].split()[1]) for j in range(21)]
                i+=21

            elif line_split[0] == 'NCOMP':
                self.COMP['NUM'] = int(line_split[1])
                i+=1

            elif line_split[0] == 'COMPNAME':
                i+=1
                self.COMP['NAME'] = [str(FID[i+j].split()[1]).replace("'","") for j in range(self.COMP['NUM'])]
                i+=self.COMP['NUM']

            elif line_split[0] == 'COMP-PHASE-TEMPLATE':
                i+=1
                self.COMP['TEMPL'] = [str(FID[i+j].split()[2]) for j in range(self.COMP['NUM'])]
                i+=self.COMP['NUM']
                
            elif line_split[0] == 'TIME' and 'SPEC-HISTORY' in [FID[i-1].split()[0], FID[i-3].split()[0]]:
                self.TIME['VECT'].append(line_split[1:])
                i+=1
                
            elif line_split[0] == 'TIMCHR':
                self.TIME['CHR'].append(line_split[2])
                self.TIME['CHR_UNIT'].append(line_split[3].replace("'",""))
                i+=1
            

            # Parse grid values from binary file
            elif line_split[0] == 'GRID':
                item_num = int(FID[i].split()[2])
                for j in range(item_num):
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap() 
                _, i = self.parse_nobin(FID, i)

            elif line_split[0] == 'GRID-VALUE':
                props_list, i = self.parse_nobin(FID, i)
                print('self.parse_nobin is', self.parse_nobin(FID, i))
                for prop in props_list[3:]:
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    print('num_bytes is', num_bytes)
                    print('prop is', prop)
                    print('np.asscalar(num_bytes) is', np.asscalar(num_bytes))
                    if prop == 'TEMP':
                        grid_temp = np.fromfile(FID_bin, np.float64, count=num_grids).byteswap()
                        self.GRID['TEMP'].append(grid_temp)
                    else:
                        _ = np.fromfile(FID_bin, np.int8, count=np.asscalar(num_bytes)).byteswap()                        



            # Parse species names and values from binary file
            elif line_split[0] == 'SPHIST-NAMES':
                self.SPHIST['NUM'] = []
                self.SPHIST['NAME'] = []
                i+=1
                for j in range(numSPHIST):
                    line_split = FID[i+j].split()
                    self.SPHIST['NUM'].append(line_split[0])
                    self.SPHIST['NAME'].append(' '.join([line.replace("'","") for line in line_split[3:]]))
                i+=numSPHIST

            elif line_split[0] == 'SPEC-HISTORY':
                props_list, i = self.parse_nobin(FID, i)
                for prop in props_list[3:]:
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    # print(prop)
                    # print(np.asscalar(num_bytes))
                    if prop == 'SPVALS':
                        spvals_temp = np.fromfile(FID_bin, np.float64, count=numSPHIST).byteswap()
                        self.SP['VAL'].append(spvals_temp)
                    else:
                        _ = np.fromfile(FID_bin, np.byte, count=np.asscalar(num_bytes)).byteswap()
            

            # Variables stored in binary but skipped in parsing
            elif line_split[0] in skip_list:
                item_num = int(line_split[2])
                for j in range(item_num):
                    num_bytes = np.fromfile(FID_bin, np.int64, count=1).byteswap()
                    _ = np.fromfile(FID_bin, np.byte, count=np.asscalar(num_bytes)).byteswap() 
                i+=1
            

            # Other variables parsed from text file
            elif line_split[0] in REC_list:
                list_temp = FID[i].split()
                while list_temp[-1] != '/':
                    i+=1
                    list_temp += FID[i].split()
                self.REC[line_split[0]] = list_temp[1:-1]
                i+=1

            else:
                i+=1
        
        FID_bin.close()

        # OUTPUT VARIABLES MURAT
        if reaction_type == 'MURAT':
            self.t = (np.asarray(self.TIME['VECT'])[:,1]).astype(np.float)
            TEMP_VALS = np.asarray(self.GRID['TEMP'])
            self.SPEC_VALS = np.asarray(self.SP['VAL'])
            self.N2_PROD   = self.SPEC_VALS[:,8]
            self.O2_PROD   = self.SPEC_VALS[:,9] 
            self.H2O_PROD  = self.SPEC_VALS[:,10]
            self.CO_PROD   = self.SPEC_VALS[:,11]
            self.CO2_PROD  = self.SPEC_VALS[:,12]
            self.lin_HR = np.asarray(TEMP_VALS)[:,8]
        elif reaction_type == 'BO':
            # OUTPUT VARIABLES BO
            TEMP_VALS = np.asarray(self.GRID['TEMP'])
            SPEC_VALS = np.asarray(self.SP['VAL'])
            self.N2_PROD = SPEC_VALS[6,:]
            self.O2_PROD = SPEC_VALS[7,:] 
            self.H2O_PROD = SPEC_VALS[8,:]
            self.CO_PROD = SPEC_VALS[9,:]
            self.CO2_PROD = SPEC_VALS[10,:]
            self.lin_HR = TEMP_VALS[8,:]
    
    @staticmethod
    def parse_nobin(file_in, i):
        '''
        file_in - file to be parsed
        i - current line number
        '''
        
        list_temp = file_in[i].split()
        while list_temp[-1] != '/':
            i+=1
            list_temp += file_in[i].split()
        i+=1

        return list_temp[1:-1], i


"""
##############################################################################
Update of Tim's code begins from here
##############################################################################
"""
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
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
class Kinetics:
    '''
    Reaction definition for writing STARS runfile
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def print_rxn(self, filename):
        print('** {}'.format(self.NAME), file=filename)
        attrs_list = ['STOREAC', 'STOPROD', 'RORDER', 'FREQFAC', 'EACT', 'RENTH', 'O2PP']
        for attr in attrs_list:
            if hasattr(self, attr):
                print_ln = '*{}'.format(attr) + ' '
                if isinstance(getattr(self, attr), list):
                    print_ln += ' '.join(str(x) + ' ' for x in getattr(self, attr))
                else:
                    print_ln += str(getattr(self, attr)) + ' '
                print(print_ln, file=filename)

def Fluid_definition_opt(fluid_option):
    if fluid_option == 1:
        # ['NAME', 'CMM', 'PCRIT', 'TCRIT', 'ACEN', 'AVG', 'BVG', 'AVISC', 'BVISC', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'MOLDEN', 'CP', 'CT1', 'SOLID_DEN', 'SOLID_CP']
        Component_map = {}
        Component_map["WATER"] = Component(COMPNAME="WATER",CMM=[0.018],PCRIT=[21754.478],TCRIT=[647.4],ACEN=[0.344],AVG=[1.7e-5],BVG=[1.116],AVISC=[7.52e-5],BVISC=[1384.86],
                                                CPG1=[7.701],CPG2=[4.595e-4],CPG3=[2.521e-6],CPG4=[-0.859e-9],
                                                MOLDEN=[55520],CP=[4.352331606e-7],CT1=[2.16e-4])
        Component_map["HEVY OIL"] = Component(COMPNAME="HEVY OIL",CMM=[0.675],PCRIT=[830.865],TCRIT=[887.6],ACEN=[1.589],
                                                AVG=[7.503e-6],BVG=[1.102],AVISC=[4.02e-4],BVISC=[3400.89],
                                                CPG1=[-34.081],CPG2=[4.137],CPG3=[-2.279e-3],CPG4=[4.835e-7],
                                                MOLDEN=[1464],CP=[7.25388601e-7],CT1=[2.68e-4])
        Component_map["LITE OIL"] = Component(COMPNAME="LITE OIL",CMM=[0.157],PCRIT=[2107.56],TCRIT=[617.4],ACEN=[0.449],
                                                AVG=[3.77e-6],BVG=[0.943],AVISC=[4.02e-4],BVISC=[3400.89],
                                                CPG1=[-7.913],CPG2=[0.961],CPG3=[-5.29e-4],CPG4=[1.123e-7],
                                                MOLDEN=[5118],CP=[7.25388601e-7],CT1=[5.11e-4])
        Component_map["INRT GAS"] = Component(COMPNAME="INRT GAS",CMM=[0.041],PCRIT=[3445.05],TCRIT=[126.5],ACEN=[0.04],
                                                AVG=[3.213e-4],BVG=[0.702],
                                                CPG1=[31.150],CPG2=[-1.357e-2],CPG3=[2.679e-5],CPG4=[-1.167e-8])
        Component_map["OXYGEN"] = Component(COMPNAME="OXYGEN",CMM=[0.032],PCRIT=[5035.852],TCRIT=[154.8],ACEN=[0.022],
                                                AVG=[3.355e-4],BVG=[0.721],
                                                CPG1=[28.106],CPG2=[-3.68e-6],CPG3=[1.746e-5],CPG4=[-1.065e-8])
        Component_map["COKE"] = Component(COMPNAME="COKE",CMM=[0.013],SOLID_DEN=[916.344, 0, 0],SOLID_CP=[16, 0])
        Kinetics_map = {}
        Kinetics_map["Cracking"] = Kinetics(NAME="Cracking",STOREAC=[0, 1, 0, 0, 0, 0],STOPROD=[0, 0, 2.154, 0, 0, 25.96],
                                            FREQFAC=4.167e5,EACT=62802,RENTH=93000)
        Kinetics_map["HO burning"] = Kinetics(NAME="HO burning",STOREAC=[0, 1, 0, 0, 60.55, 0],STOPROD=[28.34, 0, 0, 51.53, 0, 0],
                                            FREQFAC=4.4394e11,EACT=138281,RENTH=29.1332e6)
        Kinetics_map["LO burning"] = Kinetics(NAME="LO burning",STOREAC=[0, 0, 1, 0, 14.06, 0],STOPROD=[6.58, 0, 0, 11.96, 0, 0],
                                            FREQFAC=4.4394e11,EACT=138281,RENTH=6.7625e6)
        Kinetics_map["Coke burning"] = Kinetics(NAME="Coke burning",STOREAC=[0, 0, 0, 0, 1.18, 1],STOPROD=[0.55, 0, 0, 1, 0, 0],
                                            FREQFAC=6123.8,EACT=58615,RENTH=523400)

    if fluid_option ==2:
        #(NAME, CMM, PCRIT, TCRIT, AVG, BVG, AVISC, BVISC, CPG1, CPG2, CPG3, CPG4, CPL1, CPL2, MASSDEN, CP, CT1, CT2, SOLID_DEN, SOLID_CP):
        Component_map = {}
        Component_map["H2O"] = Component(COMPNAME="H2O",CMM=[1.8e-2],PCRIT=[22083],TCRIT=[373.8],
                                        AVG=[0],BVG=[0],AVISC=[0],BVISC=[0],
                                        CPG1=[0],CPG2=[0],CPG3=[0],CPG4=[0],CPL1=[0],CPL2=[0],
                                        MASSDEN=[0.001],CP=[0],CT1=[0],CT2=[0],SOLID_DEN=[],SOLID_CP=[])
        Component_map["OIL"] = Component(COMPNAME="OIL",CMM=[4.73e-1],PCRIT=[890],TCRIT=[171],
                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],CPL1=[524.8821790],CPL2=[1.148635444845],
                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0])
        Component_map["N2"] = Component(COMPNAME="N2",CMM=[2.8e-2],PCRIT=[3392],TCRIT=[-147],
                                        AVG=[0.0003500869287],BVG=[0.6927470725],AVISC=[],BVISC=[],
                                        CPG1=[30.956477056957],CPG2=[-0.012716023994],CPG3=[0.000025490143],CPG4=[-0.000000011065],CPL1=[0],CPL2=[0])
        Component_map["O2"] = Component(COMPNAME="O2",CMM=[3.2e-2],PCRIT=[5033],TCRIT=[-118],
                                        AVG=[0.000362791571],BVG=[0.7120986013],AVISC=[],BVISC=[],
                                        CPG1=[28.600167325729],CPG2=[-0.003497011859],CPG3=[0.000024399453],CPG4=[-0.000000014928],CPL1=[0],CPL2=[0])
        Component_map["CO2"] = Component(COMPNAME="CO2",CMM=[4.4e-2],PCRIT=[7377],TCRIT=[31],
                                        AVG=[0.0001865724378],BVG=[0.7754816784],AVISC=[],BVISC=[],
                                        CPG1=[19.474325955388],CPG2=[0.075654731286],CPG3=[-0.000060750197],CPG4=[0.000000020109],CPL1=[0],CPL2=[0])
        Component_map["CO"] = Component(COMPNAME="CO",CMM=[2.8e-2],PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],CPL1=[0],CPL2=[0])
        Component_map["Coke1"] = Component(COMPNAME="Coke1",CMM=[1.88e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[8.3908475,0.0439425])
        Component_map["Coke2"] = Component(COMPNAME="Coke2",CMM=[1.36e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[6.96015,0.03645])
        Component_map["Ci"] = Component(COMPNAME="Ci",CMM=[2.08e-2],SOLID_DEN=[0.0014,0,0],SOLID_CP=[7.192155,0.037665])
        Kinetics_map = []
        Kinetics_map.append(Kinetics(NAME="RXN1",STOREAC=[0.00E+00, 1.00E+00, 0.00E+00, 9.50E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 0.00E+00],
                                    STOPROD=[22.292,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.0E+01,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,1.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=7E5,EACT=1E5,RENTH=3.98E+06, O2PP='\'O2\''))

        Kinetics_map.append(Kinetics(NAME="RXN2",STOREAC=[0.00E+00,0.00E+00,0.00E+00,1.50E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[1.3815E+00,0.00E+00,0.00E+00,0.00E+00,0.75E+00,0.3181E+00,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=1E6,EACT=1.14E5,RENTH=6.28E5, O2PP='\'O2\'')
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN3",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.3824E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=1.6E6,EACT=8E4,RENTH=0)
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN4",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.650000,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00],
                                    STOPROD=[0.239300,0.00E+00,0.00E+00,0.00E+00,0.565200,0.186200,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00],
                                    FREQFAC=4.5E6,EACT=1.4E5,RENTH=2.72E5, O2PP='\'O2\'')
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN5",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.9134615],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=5.334E4,EACT=5.099E4,RENTH=0)
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN6",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.900000,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,2.00E-01,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00],
                                    FREQFAC=1.621E10,EACT=100000,RENTH=1.514E6)
                                    )
    return Component_map, Kinetics_map

def print_fluid_opt(fileID, fluid_option, Components, Kinetics):
    print('**  ==============  FLUID DEFINITIONS  ======================', file = fileID)
    # Print fluid
    comp_names = Components.keys()
    def print_attrs(attrslist):
        for attr in attrslist:
            print_str='*' + attr
            for comp in comp_names:
                if hasattr(Components[comp], attr):
                    if attr =='COMPNAME':
                        print_str+= ' ' + '\'' + str(getattr(Components[comp], attr)) + '\'' 
                    else:
                        print_str+= ' ' + str(*getattr(Components[comp], attr)) 
            print(print_str, file=fileID)
    if fluid_option == 1:
        print("""
*MODEL 6 5 3   ** Number of noncondensible gases is numy-numx = 2
** Number of solid components is ncomp-numy = 1
              """, file = fileID)
        comp_attrs = ['COMPNAME', 'CMM', 'PCRIT', 'TCRIT', 'ACEN', 'AVG', 'BVG', 'AVISC', 'BVISC', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'MOLDEN', 'CP', 'CT1']
        print_attrs(comp_attrs)
        print('*SOLID_DEN ',"\'"+Components["COKE"].COMPNAME+"\'",*Components["WATER"].SOLID_DEN,*Components["HEVY OIL"].SOLID_DEN,*Components["LITE OIL"].SOLID_DEN,*Components["INRT GAS"].SOLID_DEN,*Components["OXYGEN"].SOLID_DEN,*Components["COKE"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["COKE"].COMPNAME+"\'",*Components["WATER"].SOLID_CP,*Components["HEVY OIL"].SOLID_CP,*Components["LITE OIL"].SOLID_CP,*Components["INRT GAS"].SOLID_CP,*Components["OXYGEN"].SOLID_CP,*Components["COKE"].SOLID_CP, file = fileID)
    elif fluid_option == 2:
        print("""
*MODEL 9 6 2 
** Number of noncondensible gases is numy-numx = 4
** Number of solid components is ncomp-numy = 4
              """, file = fileID)
        comp_attrs1 = ['COMPNAME', 'CMM', 'PCRIT', 'TCRIT', 'AVG', 'BVG', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'CPL1', 'CPL2']
        print_attrs(comp_attrs1)       
        print("""
*HVAPR 0 0 
*EV 0 0 
              """, file = fileID)
        comp_attrs2 = ['MASSDEN', 'CP', 'CT1', 'CT2'] # Add in 'AVISC', 'BVISC' if necessary
        print_attrs(comp_attrs2)
        print('*SOLID_DEN ',"\'"+Components["Coke1"].COMPNAME+"\'",*Components["Coke1"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Coke1"].COMPNAME+"\'",*Components["Coke1"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+Components["Coke2"].COMPNAME+"\'",*Components["Coke2"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Coke2"].COMPNAME+"\'",*Components["Coke2"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+Components["Ci"].COMPNAME+"\'",*Components["Ci"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+Components["Ci"].COMPNAME+"\'",*Components["Ci"].SOLID_CP, file = fileID)

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
#     if fluid_option == 2:
#         print("*O2PP 'O2'", file = fileID)
