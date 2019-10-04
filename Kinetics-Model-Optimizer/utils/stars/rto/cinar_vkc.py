import numpy as np 
import os
from .rto_base import RtoBase
from .reactions import Component, Kinetics

class CinarVKC(RtoBase):

    @staticmethod
    def get_default_reaction():
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


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.numSPHIST = 22


    def get_O2_consumption(self):

        if not self.parsed:
            self.O2_PROD = self.SPEC_VALS[:,17] 
            self.TEMP_avg = self.SPEC_VALS[:,21]
            self.parsed = True

        return self.O2_PROD

    
    def print_IO_control(self, fileID):
        print('** ============== INPUT/OUTPUT CONTROL ======================', file=fileID)
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

    
    def print_grid(self, fileID):
        print('**  ==============  GRID AND RESERVOIR DEFINITION  =================', file = fileID)
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
    
    
    def print_fluid(self, fileID, components):
        print('**  ==============  FLUID DEFINITIONS  ======================', file = fileID)
        print("""
*MODEL 9 6 2 
** Number of noncondensible gases is numy-numx = 4
** Number of solid components is ncomp-numy = 4
              """, file = fileID)
        comp_attrs1 = ['COMPNAME', 'CMM', 'PCRIT', 'TCRIT', 'AVG', 'BVG', 'CPG1', 'CPG2', 'CPG3', 'CPG4', 'CPL1', 'CPL2']
        self.print_attrs(components, comp_attrs1, fileID)       
        print("""
*HVAPR 0 0 
*EV 0 0 
              """, file = fileID)
        comp_attrs2 = ['MASSDEN', 'CP', 'CT1', 'CT2'] # Add in 'AVISC', 'BVISC' if necessary
        self.print_attrs(components, comp_attrs2, fileID)
        print('*SOLID_DEN ',"\'"+components["Coke1"].COMPNAME+"\'",*components["Coke1"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+components["Coke1"].COMPNAME+"\'",*components["Coke1"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+components["Coke2"].COMPNAME+"\'",*components["Coke2"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+components["Coke2"].COMPNAME+"\'",*components["Coke2"].SOLID_CP, file = fileID)
        print('*SOLID_DEN ',"\'"+components["Ci"].COMPNAME+"\'",*components["Ci"].SOLID_DEN, file = fileID)
        print('*SOLID_CP ',"\'"+components["Ci"].COMPNAME+"\'",*components["Ci"].SOLID_CP, file = fileID)
        print("""
*VISCTABLE
10      0    100000
80      0      2084
100     0       580
1000    0         1
              """, file = fileID)

    
    def print_ref_cond(self, fileID):
        print('**  ==============  Reference CONDITIONS  ======================', file = fileID)
        print("""
*PRSR 101.325
*TEMR 25
*PSURF 101.325
*TSURF 20  
              """, file = fileID)

    
    def print_rock_fluid(self, fileID):
        print('**  ==============  ROCK-FLUID PROPERTIES  ======================', file = fileID)
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

    
    def print_initial_cond(self, fileID):
        print('**  ==============  INITIAL CONDITIONS  ======================', file = fileID)
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
    
    def print_numerical(self, fileID):
        print('**  ==============  NUMERICAL CONTROL  ======================', file = fileID)
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

    
    def print_recurrent(self, fileID):
        print('**  ==============  RECURRENT DATA  ======================', file = fileID)
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

    
    def print_heater(self, fileID):
        print('**  ============== DEFINE HEATERS ======================', file = fileID)
        print("""
*UHTR *IJK 2 1 1:11 3000
*TMPSET *IJK 2 1 1:11 25
*HEATR *IJK 2 1 1 30000
*AUTOHEATER *ON 2 1 1:11              

              """, file = fileID)

    
    def print_heating_ramp(self, fileID, HR):
        print('** ==========Linear Ramped Heating==========', file=fileID)

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