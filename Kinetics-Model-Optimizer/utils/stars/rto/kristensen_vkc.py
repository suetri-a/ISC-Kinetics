import numpy as np 
import os
from .rto_base import RtoBase
from .reactions import Component, Kinetics

class KristensenVKC(RtoBase):

    @staticmethod
    def get_default_reaction():
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
        
        Kinetics_map = []
        Kinetics_map.append(Kinetics(NAME="Cracking",STOREAC=[0, 1, 0, 0, 0, 0],STOPROD=[0, 0, 2.154, 0, 0, 25.96],
                                    FREQFAC=4.167e5,EACT=62802,RENTH=93000)
                                    )
        Kinetics_map.append(Kinetics(NAME="HO burning",STOREAC=[0, 1, 0, 0, 60.55, 0],STOPROD=[28.34, 0, 0, 51.53, 0, 0],
                                    FREQFAC=4.4394e11,EACT=138281,RENTH=29.1332e6)
                                    )
        Kinetics_map.append(Kinetics(NAME="LO burning",STOREAC=[0, 0, 1, 0, 14.06, 0],STOPROD=[6.58, 0, 0, 11.96, 0, 0],
                                    FREQFAC=4.4394e11,EACT=138281,RENTH=6.7625e6)
                                    )
        Kinetics_map.append(Kinetics(NAME="Coke burning",STOREAC=[0, 0, 0, 0, 1.18, 1],STOPROD=[0.55, 0, 0, 1, 0, 0],
                                    FREQFAC=6123.8,EACT=58615,RENTH=523400)
                                    )

        return Component_map, Kinetics_map


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def print_IO_control(self, fileID):
        print('** ============== INPUT/OUTPUT CONTROL ======================', file=fileID)
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

    
    def print_grid(self, fileID):
        print('**  ==============  GRID AND RESERVOIR DEFINITION  =================', file = fileID)
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
    
    
    def print_fluid(self, fileID, components):
        print('**  ==============  FLUID DEFINITIONS  ======================', file = fileID)
        print("""
*MODEL 6 5 3   ** Number of noncondensible gases is numy-numx = 2
** Number of solid components is ncomp-numy = 1
              """, file = fileID)
        comp_attrs = ['COMPNAME', 'CMM', 'PCRIT', 'TCRIT', 'ACEN', 'AVG', 'BVG', 'AVISC', 'BVISC', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'MOLDEN', 'CP', 'CT1']
        self.print_attrs(components, comp_attrs, fileID)
        print('*SOLID_DEN ',"\'"+components["COKE"].COMPNAME+"\'",*components["WATER"].SOLID_DEN,*components["HEVY OIL"].SOLID_DEN,*components["LITE OIL"].SOLID_DEN,*components["INRT GAS"].SOLID_DEN,*components["OXYGEN"].SOLID_DEN,*components["COKE"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+components["COKE"].COMPNAME+"\'",*components["WATER"].SOLID_CP,*components["HEVY OIL"].SOLID_CP,*components["LITE OIL"].SOLID_CP,*components["INRT GAS"].SOLID_CP,*components["OXYGEN"].SOLID_CP,*components["COKE"].SOLID_CP, file = fileID)
    
    def print_ref_cond(self, fileID):
        print('**  ==============  Reference CONDITIONS  ======================', file = fileID)
        print("""
*PRSR 101.3529
*TEMR 298.15
*TSURF 293.15
*PSURF 101.3529
              """, file = fileID)

    
    def print_rock_fluid(self, fileID):
        print('**  ==============  ROCK-FLUID PROPERTIES  ======================', file = fileID)
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

    
    def print_initial_cond(self, fileID):
        print('**  ==============  INITIAL CONDITIONS  ======================', file = fileID)
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

    
    def print_numerical(self, fileID):
        print('**  ==============  NUMERICAL CONTROL  ======================', file = fileID)
        print("""
*NUMERICAL
*RUN
 
              """, file = fileID)

    
    def print_recurrent(self, fileID):
        print('**  ==============  RECURRENT DATA  ======================', file = fileID)
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

    
    def print_heater(self, fileID):
        print('**  ============== DEFINE HEATERS ======================', file = fileID)
        print("""
**Key_Word_for_Proportional_Heat_Transfer_Coefficient**
*UHTR   *IJK 1 1 1:12 3000
*TMPSET *IJK 1 1 1:12 273.15
**Key_Word_for_Constant_Heat_Transfer_Rate**
*HEATR  *IJK 1 1 1:12 30000
*AUTOHEATER *ON 1 1 1:12 
        
              """, file = fileID)

    
    def print_heating_ramp(self, fileID, HR):
        print('** ==========Linear Ramped Heating==========', file=fileID)
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