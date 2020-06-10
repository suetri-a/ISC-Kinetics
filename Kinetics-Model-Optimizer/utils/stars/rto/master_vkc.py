import numpy as np 

import os

from .rto_base import RtoBase

from .reactions import Component, Kinetics



class MasterVKC(RtoBase):



    @staticmethod

    def get_default_reaction():

        Component_map = {}

        Component_map["H2O"] = Component(COMPNAME="H2O",CMM=[1.8e-2],PCRIT=[22083],TCRIT=[373.8],

                                        AVG=[0],BVG=[0],AVISC=[0],BVISC=[0],

                                        CPG1=[0],CPG2=[0],CPG3=[0],CPG4=[0],CPL1=[0],CPL2=[0],

                                        MASSDEN=[0.001],CP=[0],CT1=[0],CT2=[0],SOLID_DEN=[],SOLID_CP=[],

                                        HVAPR=[0], EV=[0],

                                        phase=1)

        Component_map["OIL"] = Component(COMPNAME="Oil",CMM=[4.73e-1],PCRIT=[890],TCRIT=[171],
                                        AVG=[0.0001610891804],BVG=[0.7453161006],AVISC=[1.426417368e-11],BVISC=[10823.06574],
                                        CPG1=[26.804420692906],CPG2=[0.005649089963],CPG3=[0.000095012314],CPG4=[-0.000000054709],
                                        CPL1=[524.8821790],CPL2=[1.148635444845],
                                        MASSDEN=[0.000999798],CP=[7.25e-7],CT1=[0.00069242],CT2=[0],
                                        HVAPR=[0], EV=[0],
                                        phase=2)

        Component_map["N2"] = Component(COMPNAME="N2",CMM=[2.8e-2],PCRIT=[3392],TCRIT=[-147],
                                        AVG=[0.0003500869287],BVG=[0.6927470725],AVISC=[],BVISC=[],
                                        CPG1=[30.956477056957],CPG2=[-0.012716023994],CPG3=[0.000025490143],CPG4=[-0.000000011065],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)

        Component_map["O2"] = Component(COMPNAME="O2",CMM=[3.2e-2],PCRIT=[5033],TCRIT=[-118],
                                        AVG=[0.000362791571],BVG=[0.7120986013],AVISC=[],BVISC=[],
                                        CPG1=[28.600167325729],CPG2=[-0.003497011859],CPG3=[0.000024399453],CPG4=[-0.000000014928],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)

        Component_map["CO2"] = Component(COMPNAME="CO2",CMM=[4.4e-2],PCRIT=[7377],TCRIT=[31],
                                        AVG=[0.0001865724378],BVG=[0.7754816784],AVISC=[],BVISC=[],
                                        CPG1=[19.474325955388],CPG2=[0.075654731286],CPG3=[-0.000060750197],CPG4=[0.000000020109],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)

        Component_map["CO"] = Component(COMPNAME="CO",CMM=[2.8e-2],PCRIT=[3496],TCRIT=[-144],
                                        AVG=[0.0003315014585],BVG=[0.7037315714],AVISC=[],BVISC=[],
                                        CPG1=[30.990187019402],CPG2=[-0.01392019971],CPG3=[0.00003014996],CPG4=[-0.00000001415],
                                        CPL1=[0],CPL2=[0],
                                        phase=3)

        Component_map["Coke1"] = Component(COMPNAME="Coke1",CMM=[1.88e-2],
                                        SOLID_DEN=[0.0014,0,0],SOLID_CP=[8.3908475,0.0439425],
                                        phase=4)

        Component_map["Coke2"] = Component(COMPNAME="Coke2",CMM=[1.36e-2],
                                        SOLID_DEN=[0.0014,0,0],SOLID_CP=[6.96015,0.03645],
                                        phase=4)

        Component_map["Ci"] = Component(COMPNAME="Ci",CMM=[2.08e-2],
                                        SOLID_DEN=[0.0014,0,0],SOLID_CP=[7.192155,0.037665],
                                        phase=4)

        Kinetics_map = []

        Kinetics_map.append(Kinetics(NAME="RXN1",STOREAC=[0.00E+00, 1.00E+00, 0.00E+00, 0.00E+00, 0.00E+00, 4.50E+00, 0.00E+00, 0.00E+00, 0.00E+00],
                                    STOPROD=[22.292,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.0E+01,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=7E5,EACT=1E5,RENTH=3.98E+06, O2PP='\'O2\''))

        Kinetics_map.append(Kinetics(NAME="RXN2",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.50E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[1.3815E+00,0.00E+00,0.00E+00,0.00E+00,0.75E+00,0.3181E+00,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=1E6,EACT=1.14E5,RENTH=6.28E5, O2PP='\'O2\'')
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN3",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.3824E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=1.6E6,EACT=8E4,RENTH=0)
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN4",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.650000,0.00E+00,1.00E+00,0.00E+00],
                                    STOPROD=[0.239300,0.00E+00,0.565200,0.186200,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,1.00E+00,0.00E+00],
                                    FREQFAC=4.5E6,EACT=1.4E5,RENTH=2.72E5, O2PP='\'O2\'')
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN5",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.9134615],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,2.00E+00,0.00E+00,0.00E+00],
                                    FREQFAC=5.334E4,EACT=5.099E4,RENTH=0)
                                    )

        Kinetics_map.append(Kinetics(NAME="RXN6",STOREAC=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.900000,0.00E+00,0.00E+00,1.00E+00],
                                    STOPROD=[0.00E+00,0.00E+00,1.00E+00,2.00E-01,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00],
                                    RORDER=[0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00,1.00E+00,0.00E+00,0.00E+00,1.00E+00],

                                    FREQFAC=1.621E10,EACT=100000,RENTH=1.514E6)

                                    )

        return Component_map, Kinetics_map


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.numSPHIST = 7


    def get_reaction_dict(self):

        if not self.parsed:
            self.parse_stars_output()
        ydict = {}
        ydict['N2'] = self.SPEC_VALS[:,0][2:]
        ydict['O2'] = self.SPEC_VALS[:,1][2:]
        ydict['CO'] = self.SPEC_VALS[:,2][2:]
        ydict['CO2'] = self.SPEC_VALS[:,3][2:]
        ydict['H2O'] = self.SPEC_VALS[:,4][2:]
        ydict['Oil'] = self.SPEC_VALS[:,5][2:]
        ydict['Temp'] = self.SPEC_VALS[:,6][2:] - 273.15

        return 24*60*self.t[2:], ydict

    def get_O2_consumption(self):

        if not self.parsed:
            self.parse_stars_output()
        self.O2_PROD = self.SPEC_VALS[:,1][1:]
        self.TEMP_avg = self.SPEC_VALS[:,6][1:]

        return self.O2_PROD


    def print_IO_control(self, fileID):
        print("""
** ============== INPUT/OUTPUT CONTROL ======================

*CASEID 'Model 1-cartesian'
**CHECKONLY
*INUNIT *LAB
*OUTUNIT *LAB
*WPRN *GRID *TIME
*WPRN *ITER *TIME
*OUTPRN *GRID *ALL
*OUTPRN *WELL *ALL
*OUTPRN *ITER *BRIEF
*WSRF *GRID *TIME
*WSRF *WELL *TIME

*OUTSRF *WELL *MASS *COMPONENT *ALL
*OUTSRF *GRID *PRES *SW *SO *SG *TEMP *Y *X *MASS *SOLCONC
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'N2'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'O2'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'CO'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'CO2'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'H2O'
*OUTSRF *SPECIAL MOLEFRAC 'PROD' 'Oil'
*OUTSRF *SPECIAL AVGVAR TEMP
*INTERRUPT *STOP
              """, file = fileID)


    def print_grid(self, fileID):

        '''

        This should not have to be changed if the kinetic cell is designed the same way.

        '''

        print('**  ==============  GRID AND RESERVOIR DEFINITION  =================', file = fileID)
        print("""
*GRID *CART 1 1 3
*KDIR *DOWN
*DI *CON 1.32
*DJ *CON 1.32
*DK *CON 3
*POR *CON 0.36
*PERMI *CON 11400
*PERMJ *CON 11400
*PERMK *KVAR 11400 0 0
*END-GRID

**------------------------OTHER RESERVOIR PROPERTIES-----------------------------------
*CPOR 0
*CTPOR 0
*ROCKCP 1.20572 0.00236
*THCONR 2.5833
*THCONO 0.0
*THCONG 0.0
*THCONS 0.0

              """, file = fileID)

    def print_fluid(self, fileID, components):
        comp_names = components.keys()
        solid_pseudocomps = [c for c in comp_names if components[c].phase==4]

        numw = 1 # water
        numx = numw + sum([components[c].phase==2 for c in comp_names]) # parent fuel + liquid psuedo components
        numy = numx + sum([components[c].phase==3 for c in comp_names]) # add incompressible gases
        ncomp = len(components) # total number of components        

        print("""
**  ==============  FLUID DEFINITIONS  ======================
*MODEL {ncomp} {numy} {numx} {numw}

              """.format(ncomp=ncomp, numy=numy, numx=numx, numw=numw), 
            file = fileID)

        comp_attrs = ['COMPNAME', 'CMM', 'PCRIT', 'TCRIT', 'AVG', 'BVG', \
                        'CPG1', 'CPG2', 'CPG3', 'CPG4', 'CPL1', 'CPL2', \
                        'HVAPR', 'EV', 'MASSDEN', 'CP', 'CT1', 'CT2']
        self.print_attrs(components, comp_attrs, fileID)

        # Print properties of solid pseudo components
        for c in solid_pseudocomps:
            print("""
*SOLID_DEN \'{compname}\' {solid_den}
*SOLID_CP \'{compname}\' {solid_cp}
            """.format(compname=c, 
                    solid_den=' '.join(map(str, components[c].SOLID_DEN)),
                    solid_cp=' '.join(map(str, components[c].SOLID_CP))), 
                file=fileID)

        # Print fixed VISCTABLE instead of using AVISC and BVISC data
        visc1 = ['100000']*(numx - numw)
        visc2 = ['2084']*(numx - numw)
        visc3 = ['580']*(numx - numw)
        visc4 = ['1']*(numx - numw)

        print("""
*VISCTABLE
10      0    {visc1}
80      0    {visc2}
100     0    {visc3}
1000    0    {visc4}
              """.format(visc1=' '.join(visc1),visc2=' '.join(visc2),visc3=' '.join(visc3),visc4=' '.join(visc4)), file = fileID)

    def print_ref_cond(self, fileID):
        print("""
**  ==============  Reference CONDITIONS  ======================

*PRSR 101
*TEMR 15
*PSURF 101
*TSURF 25

              """, file = fileID)


    def print_rock_fluid(self, fileID):
        print("""
**  ==============  ROCK-FLUID PROPERTIES  ======================

*ROCKFLUID
*RPT 1 *LININTERP *OILWET
** Sw Krw Krow Pcow
** ---- --- ---- ----
*SWT
0 0 1 0
0.15 0.0249 0.6944 0
0.2 0.0442 0.6049 0
0.218 0.0525 0.5742 0
0.25 0.0691 0.5216 0
0.3 0.0995 0.4444 0
0.35 0.1354 0.3735 0
0.4 0.1769 0.3086 0
0.45 0.2239 0.25 0
0.5 0.2764 0.1975 0
0.55 0.3345 0.1512 0
0.6 0.398 0.1111 0
0.65 0.4671 0.0772 0
0.7 0.5418 0.0494 0
0.75 0.6219 0.0278 0
0.8 0.7076 0.0123 0
0.85 0.7989 0.0031 0
0.9 0.8956 0 0

*SLT
** Sl Krg Krog Pcgo
** -- --- ---- ----
0.1 0.8975 0 0
0.15 0.7914 0.001 0
0.2 0.6926 0.0051 0
0.25 0.6009 0.0133 0
0.3 0.5163 0.0261 0
0.35 0.4386 0.0441 0
0.4 0.3678 0.0677 0
0.45 0.3037 0.0973 0
0.5 0.2463 0.1332 0
0.55 0.1953 0.1757 0
0.6 0.1507 0.225 0
0.65 0.1124 0.2815 0
0.7 0.0801 0.3454 0
0.75 0.0536 0.4169 0
0.8 0.0328 0.4962 0
0.85 0.0174 0.5835 0
0.965 0.0007 0.8159 0
1 0 0.8956 0


              """, file = fileID)

    def print_initial_cond(self, fileID, IC_dict):
        print('**  ==============  INITIAL CONDITIONS  ======================', file = fileID)
        print("""
*INITIAL
*PRES *CON   100
*TEMP *CON   {T_init}
**VERTICAL OFF
*SW *CON {WAT_SAT}
*SO *CON {OIL_SAT}
*SG *CON {GAS_SAT}

*MFRAC_WAT 'H2O' *CON 1.00
*MFRAC_OIL 'Oil' *CON 1.00
*MFRAC_GAS 'N2' *CON {N2_con}
*MFRAC_GAS 'CO2' *CON 0.00
*MFRAC_GAS 'CO' *CON 0.00
*MFRAC_GAS 'O2' *CON {O2_con}

**INITREGION 1
**SW *IJK 
**WAT_SAT
**		 1 1 1:11 0.000
**W_SATEND
**         2 1 1:11 0.0

**SO *IJK 
**OIL_SAT
**		 1 1 1:11 1.0
**O_SATEND
**         2 1 1:11 0.0

**SG *IJK 
**GAS_SAT
**	     1 1 1:11 1.0
**G_SATEND
**         2 1 1:11 0.0

**MFRAC_GAS 'N2' CON       1
**MFRAC_GAS 'O2' *con 0.21
**MFRAC_GAS 'N2' *con 0.79      

              """.format(T_init=IC_dict['Temp'], WAT_SAT=0.0, OIL_SAT=IC_dict['Oil'], GAS_SAT=1-IC_dict['Oil'], 
                        O2_con=IC_dict['O2'], N2_con=1-IC_dict['O2']), 
                        file = fileID)

    def print_numerical(self, fileID):
        print("""
**  ==============  NUMERICAL CONTROL  ======================

*NUMERICAL
*MAXSTEPS 100000
**DTMAX 0.1
*DTMIN 0.000001
*MAXTEMP 6000   
*RUN    
              """, file = fileID)


    def print_recurrent(self, fileID, O2_con_in, components):

        comp_names = components.keys() # get component names
        phases = [components[c].phase for c in comp_names] # get phases
        comp_names = [c for _, c in sorted(zip(phases,comp_names))] # sort comp_names according to phase

        incomp = []
        for c in comp_names:
            if components[c].phase==1:
                incomp.append('0.0')
            elif components[c].phase==2:
                incomp.append('0.0')
            elif components[c].phase==3:
                if c=='N2':
                    incomp.append(str(1 - O2_con_in))
                elif c=='O2':
                    incomp.append(str(O2_con_in))
                else:
                    incomp.append('0.0')


        print("""
**  ==============  RECURRENT DATA  ======================
*TIME   0
*DTWELL 0.01
*WELL   'INJE'
*WELL   'PROD'
*INJECTOR UNWEIGHT 'INJE'
*INCOMP  GAS  {incomp}
*TINJW  25.
*PINJW 180
*OPERATE   STG  100
*GEOMETRY  K  0.26  1.  1.  0.
*PERF  TUBE-END  'INJE'
1 1 1  1.  OPEN    FLOW-FROM  'SURFACE'
*PRODUCER 'PROD'
*OPERATE  *BHP  100
*GEOMETRY  K  0.26  1.  1.  0.
*PERF  TUBE-END  'PROD'
1 1 1  1.  OPEN    FLOW-TO  'SURFACE'       

              """.format(incomp=' '.join(incomp)), file = fileID)


    def print_heater(self, fileID):
        print("""
**  ==============  DEFINE HEATERS  ======================
*UHTR *IJK 1 1 1 3000
*TMPSET *IJK 1 1 1 25
*HEATR *IJK 1 1 1 30000
*AUTOHEATER *ON 1 1 1              
              """, file = fileID)


    def print_heating_ramp(self, fileID, HR, TFINAL, TEMP0, TEMPMAX):
        print('** ==========  HEATING SCHEDULE  ==========', file=fileID)

        for t in range(int(TFINAL)-1):
            if type(HR) is dict:
                temp = np.interp(t, HR['Time'], HR['Temp'])
            else:
                temp = np.minimum(TEMPMAX, TEMP0 + HR*t)

            print('*TIME ', str(t+1), file = fileID)
            print('*TMPSET *IJK 1 1 1 ', str(temp), file = fileID)
            print("*INJECTOR 'INJE'", file = fileID)
            print('*TINJW ', str(temp), file = fileID)

        print('*TIME ', str(TFINAL), file=fileID)
        print('*STOP', file=fileID)
