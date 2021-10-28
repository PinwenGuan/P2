"""
This is the script for constructing pressure-potential phase diagrams (pressure-dependent Pourbaix diagrams 
at given pH) of binary metal-hydrogen systems. The input files are pressure-depedent Gibbs energy of the most 
stable structure for each considered composition, named such as 'hmin_Pd','hmin_PdH','hmin_Pd3H4', etc.. The 
common file for each system is hmin_H for pure H. The format of each input file is like:
#P (eV/A^3)   Hmin (eV/atom)
P1            G1
P2            G2
...
Pn            Gn
where the first column is the pressure, and the second is the correponding Gibbs energy. In cases where the 
finite-temperature effects are ignored, you can use enthalpy to approximate Gibbs energy. But for pure H, the 
finite-temperature effects usually cannot be ignored.
The code will create folder "pb" storing results, containg a "pbmap_pH" file for the Pourbaix diagram at given 
pH. The code will also plot and prompt the diagram.  
"""

from phasediagram import Pourbaix_H
import numpy as np
from scipy import interpolate
import os

# Define conditions.
kB=1.38065E-23;T=300;P0=1e5
plim=[1e-4,200] #GPa
#plim=[1e-4,200]
nnp=400
#HER=-0.09 #Pd
#HER=-0.611323393 #Y
#HER=-0.419984919 #La
HER=-0.1257775 #LaH3
#HER=-0.217096776 #Mg
#HER=-0.397163924 #Ca
#HER=0
pH=0 # set a pH value 
nu=400;nph=300;ulim=[-1,0.2];phlim=[-2,14]
scale='log' # log, linear
if scale=='linear':
	dp=(plim[1]-plim[0])/nnp 
	plim=np.array(plim)*1e9*1e-30/(1.6*1e-19)
	dp=dp*1e9*1e-30/(1.6*1e-19)
	p=[plim[0]+i*dp for i in range(nnp+1)]

if scale=='log': 
	plim=np.array(plim)*1e9*1e-30/(1.6*1e-19)
	p=[plim[0]*(plim[1]/plim[0])**(i/nnp) for i in range(nnp+1)]

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('---  There is this folder!  ---')

"""
Define studied compositions.
comp: compositions to be considered except the pure H, consistent with the input file names, e.g., 
comp=['Ca','CaH2','CaH6'] corresponds to input files 'hmin_Ca','hmin_CaH2','hmin_CaH6' in addition to 
'hmin_H'.
x: hydrogen content of the compositions defined in comp.
nat: number of atoms of the compositions defined in comp.
"""

"""Pd
comp=['Pd','PdH','Pd3H4','PdH6','PdH7','PdH8','PdH10','PdH12']
x=[0,0.5,4/7,6/7,7/8,8/9,10/11,12/13]
nat=[1,2,7,7,8,9,11,13]
"""
"""Y
comp=['Y','YH2','YH3','YH4','YH6','YH7','YH9','YH10']
x=[0,2/3,0.75,4/5,6/7,7/8,9/10,10/11]
nat=[1,3,4,5,7,8,10,11]
"""
"""Mg
comp=['Mg','MgH2','MgH4','MgH6','MgH12','MgH16']
x=[0,2/3,4/5,6/7,12/13,16/17]
nat=[1,3,5,7,13,17]
"""
"""La
comp=['La','LaH2','LaH3','LaH4','LaH5','LaH6','LaH8','LaH10','LaH11','LaH16']
x=[0,2/3,0.75,4/5,5/6,6/7,8/9,10/11,11/12,16/17]
nat=[1,3,4,5,6,7,9,11,12,17]
"""
#"""La-remove LaH11
comp=['La','LaH2','LaH3','LaH4','LaH5','LaH6','LaH8','LaH10','LaH16'] 
x=[0,2/3,0.75,4/5,5/6,6/7,8/9,10/11,16/17]
nat=[1,3,4,5,6,7,9,11,17]
#"""
"""Ca
comp=['Ca','CaH2','CaH6']
x=[0,2/3,6/7]
nat=[1,3,7]
"""

# read pressure-Gibbs energy data
p_each=[[] for i in comp]
h=[[] for i in comp]
h_ip=['' for i in comp]

for i in range(len(comp)):
	data=np.loadtxt('hmin_'+comp[i],usecols=(0,1))
	p_each[i]=data[:,0]
	h[i]=data[:,1]
	h_ip[i]=interpolate.interp1d(p_each[i], h[i],kind='cubic')

hset=[[] for i in p]
xset=[[] for i in p]

for i in range(len(p)):
	for j in range(len(comp)):
		try:
			hset[i].append(float(h_ip[j](p[i])))
			xset[i].append(x[j])
		except:
			print(comp[j])
			pass

data_H2=np.loadtxt('hmin_H',usecols=(0,1))
p_H2=data_H2[:,0]
G_H2=data_H2[:,1]
G300_H2=interpolate.interp1d(p_H2, G_H2)
h_H=[0 for i in p]
for j in range(len(p)):
	h_H[j]=float(G300_H2(p[j]))


#calculate Pourbaix diagrams
mkdir('pb')
d=['' for j in range(len(p))]
names=['' for j in range(len(p))]
text=['' for j in range(len(p))]
# generate a Pourbaix diagram for each pressure
for j in range(len(p)):
    try:
        print(p[j])
        y=hset[j]
        xx=xset[j]
        h_Pd=hset[j][0]
        y=[(y[k]-(1-xx[k])*h_Pd-xx[k]*h_H[j])*nat[k] for k in range(len(y))]
        d[j], names[j]=Pourbaix_H(comp,y,nu=nu,nph=nph,ulim=ulim,phlim=phlim)
        np.savetxt('pb/pbmap_'+str(p[j]/(1e-30)*(1.6*1e-19)/1e9)+'GPa',d[j],fmt='%5s',header=' P='+str(p[j])+'eV/A^3 '+str(names[j]))
    except:
        print(j)

# Merge the calculated phase equilibria at different pressures.
import glob

c=glob.glob('pb/pbmap*GPa')
s=[]
for i in range(len(c)):
	with open(c[i],'r') as f:
		cc=f.readline().split('eV/A^3')
		phases=cc[-1].strip('\n').replace('[','').replace(']','').replace('\'','').replace(' ','').split(',')
		p=float(cc[0].strip('#').replace('P=','').replace(' ',''))
	a0=np.loadtxt(c[i])[:,int(nph*(pH-phlim[0])/(phlim[1]-phlim[0]))] # Slice the Pourbaix diagram at given pH
	a=[phases[int(j)] for j in a0]
	s.append((a,p)) # Pourbaix diagrams at given pH paired with pressures

s=sorted(s, key=lambda x: x[1])
map=[i[0] for i in s]
phases=[]
for i in range(len(map)):
	phases=phases+map[i] # All the phases at given pH at all the pressures

phases_reduced = []
null=[phases_reduced.append(i) for i in phases if not i in phases_reduced]
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import get_el_sp, Element, Specie, DummySpecie
comps_reduced = [Composition(i) for i in phases_reduced]
comps_H=[i.get_atomic_fraction(Element("H")) for i in comps_reduced]
comps_phases_reduced=list(zip(comps_H,phases_reduced))
comps_phases_reduced.sort()
phases_reduced=[i[1] for i in comps_phases_reduced] # Unduplicated phases ordered by their hydrogen content

# Map the Pourbaix diagram at given pH to numbers corresponding to the order of phases by their hydrogen content.
map_number=[[] for i in map]
for i in range(len(map_number)):
	map_number[i]=[phases_reduced.index(j) for j in map[i]]

map_number=np.array(map_number)
map_number=map_number.transpose()
# save the Pourbaix diagram at given pH mapped to numbers corresponding to the order of phases listed in the header.
np.savetxt('pb/pbmap_pH'+str(pH),map_number,fmt='%5s',header=' pH='+str(pH)+' '+str(phases_reduced)) 

# plot the diagram.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker,cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

map_number=np.loadtxt('pb/pbmap_pH'+str(pH))
cmp=cm.get_cmap('rainbow')
# Linear or log scale in pressure:
if scale=='linear':
	plt.imshow(map_number,extent=[plim[0]/(1e-30)*(1.6*1e-19)/1e9,plim[1]/(1e-30)*(1.6*1e-19)/1e9,ulim[0],ulim[1]],origin='lower',aspect='auto',cmap=cmp)
	plt.hlines(HER, plim[0]/(1e-30)*(1.6*1e-19)/1e9,plim[1]/(1e-30)*(1.6*1e-19)/1e9, colors = "r", linestyles = "dashed")
	plt.xlabel("P (GPa)",fontsize=15)

if scale=='log':
	plt.imshow(map_number,extent=[np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9),np.log10(plim[1]/(1e-30)*(1.6*1e-19)/1e9),ulim[0],ulim[1]],origin='lower',aspect='auto',cmap=cmp)
	plt.hlines(HER, np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9),np.log10(plim[1]/(1e-30)*(1.6*1e-19)/1e9), 
		colors = "k", linestyles = "dashed")
	plt.xlabel("logP (GPa)",fontsize=15)

plt.ylabel('U_RHE (V)',fontsize=15)
plt.tick_params(axis='both',which='major',labelsize=13,width=2)
plt.rcParams["axes.labelweight"] = "bold"
plt.xticks(color='w')
plt.yticks(color='w')
plt.show()
