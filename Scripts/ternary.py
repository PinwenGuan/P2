from phasediagram import *
import numpy as np
from scipy import interpolate
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker,cm
import copy
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, \
    PhaseDiagram, PDPlotter
from cycler import cycler

plim=[1e-4,500] #GPa
nnp=400
HER=0
pH=0 
nu=400;nph=300;ulim=[-1,1];phlim=[-2,14]; #define axis ranges  
scale='log' # log, linear
if scale=='linear':
	dp=(plim[1]-plim[0])/nnp 
	plim=np.array(plim)*1e9*1e-30/(1.6*1e-19)
	dp=dp*1e9*1e-30/(1.6*1e-19)
	p=[plim[0]+i*dp for i in range(nnp+1)]

if scale=='log': 
	plim=np.array(plim)*1e9*1e-30/(1.6*1e-19)
	p=[plim[0]*(plim[1]/plim[0])**(i/nnp) for i in range(nnp+1)]

kB=1.38065E-23;T=300;P0=1e5

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('---  There is this folder!  ---')

directory='./'
f=glob.glob(directory+'hmin*') #the input files containing enthaly data should be named as "hmin_{composition}"

comp=[i.split('/')[-1].replace('hmin_','') for i in f]

# read p-H data
p_each=[[] for i in comp]
h=[[] for i in comp]
h_ip=['' for i in comp]
hH=''

for i in range(len(comp)):
	data=np.loadtxt(f[i],usecols=(0,1)) #load enthaly data
	p_each[i]=data[:,0]
	h[i]=data[:,1]
	if comp[i]=='H':
		h_ip[i]=interpolate.interp1d(p_each[i], h[i]+1e9) # effectively remove H
		hH=interpolate.interp1d(p_each[i], h[i]) # true enthalpy of H under pressure
	else:		
		h_ip[i]=interpolate.interp1d(p_each[i], h[i],kind='cubic')

hset=[[] for i in p]
xset=[[] for i in p]

for i in range(len(p)):
	for j in range(len(comp)):
		try:
			hset[i].append(float(h_ip[j](p[i])))
			xset[i].append(comp[j])
		except:
			print(comp[j])
			pass

#calculate Pourbaix diagrams
mkdir('pb')
with open('pb/pbmap','w') as f:
	f.write('# P (GPa)   U_low (V)   U_high (V)   x_low   x_high   Phases\n')

phb={} #phase boundaries
for n in range(len(p)):
	muH0=hH(p[n])
	entry=[PDEntry(xset[n][j],hset[n][j]*Composition(xset[n][j]).num_atoms) for j in range(len(xset[n]))]
	pd = PhaseDiagram(entry)
	lines=list(uniquelines(pd.facets))
	xH=[i.composition.fractional_composition['H'] for i in pd.qhull_entries]
	iH=[i for i in range(len(xH)) if xH[i]==1]
	lines=[i for i in lines if not set(iH) <= set(i)] # remove lines with H as endpoint
	mu=[[] for i in lines]
	cb=[[] for i in lines] # x(Mg)/(x(Li)+x(Mg))
	label=['' for i in lines]
	for i in range(len(lines)):
		compb=[pd.qhull_entries[k].composition.fractional_composition for k in lines[i]]
		label[i]=pd.qhull_entries[lines[i][0]].name
		x=['' for k in range(len(compb))]
		for k in range(len(compb)):
			if k!=0:
				label[i]=label[i]+'+'+pd.qhull_entries[lines[i][k]].name
			if (compb[k]['Li']+compb[k]['Mg'])!=0:
				x[k]=compb[k]['Mg']/(compb[k]['Li']+compb[k]['Mg'])
				xspare=x[k]
		for k in range(len(compb)):
			if x[k]=='':
				x[k]=xspare
		cb[i]=sorted(x)
		for j in range(len(pd.facets)):
			if set(lines[i]) <= set(pd.facets[j]):
				compf=[pd.qhull_entries[k].composition.fractional_composition for k in pd.facets[j]]
				compfa=compf[0]
				for k in range(1,len(compf)):
					compfa=compfa+compf[k]
				compfa=compfa/len(compf)
				muh=list(pd.get_all_chempots(compfa).values())[0][Element('H')]
				mu[i].append(muh)
		mu[i]=-(np.array(mu[i])-muH0) # convert H chemical potential to RHE electrode potential
		mu[i]=sorted(mu[i])
	mumin=min([min(i) for i in mu])
	mumax=max([max(i) for i in mu])
	muset=[]
	for i in range(len(mu)):
		muset=muset+mu[i]
	muset=sorted(muset)	
	cmp=cm.get_cmap('rainbow')
	data=list(zip(lines,mu,cb,label))
	data=sorted(data, key = lambda x: x[1], reverse=True)
	lines=[i[0] for i in data]
	mu=[i[1] for i in data]
	cb=[i[2] for i in data]
	label=[i[3] for i in data]
	for i in range(len(lines)):
		x=np.linspace(cb[i][0],cb[i][1],300)
		y1=[max(mu[i][0],ulim[0]) for j in x]
		y1=np.array(y1)
		if len(mu[i])==2:
			y2=[min(mu[i][1],ulim[1]) for j in x]
		else:
			if y1[0]==mumin and x[0]!=x[-1]:
				y2=copy.deepcopy(y1)
				y1=[ulim[0] for j in x]
			elif y1[0]==mumax and x[0]!=x[-1]:
				y2=[ulim[1] for j in x]
			else:
				y2=y1
				continue
		if y1[0]>y2[0]:
			continue
		if not label[i] in phb.keys():
			phb[label[i]]=[[p[n],y1[0],y2[0],x[0],x[-1],label[i]]]
		else:
			phb[label[i]].append([p[n],y1[0],y2[0],x[0],x[-1],label[i]])

with open('pb/pbmap','a') as f:
	for i in phb.keys():
		for j in range(len(phb[i])):
			z=[str(k) for k in phb[i][j]]
			z='   '.join(z)
			f.write(z+'\n')


# slice fixed x(Mg) planes
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 2
plt.rcParams["font.family"] = "Arial"

with open('pb/pbmap','r') as f:
	c=f.readlines()

c=[i.strip('\n').split('   ') for i in c if not '#' in i]

xbound0=[]
for i in range(len(c)):
	xbound0=xbound0+[float(j) for j in c[i][-3:-1]]

xbound = []
null=[xbound.append(i) for i in xbound0 if not i in xbound]
xbound.sort()

nxrange=len(xbound)-1
x_range=[[] for n in range(nxrange)]
phb=[{} for n in range(nxrange)]
for n in range(nxrange):
	x_range[n]=xbound[n:n+2]
	for i in range(len(c)):
		if float(c[i][-3])<=x_range[n][0] and float(c[i][-2])>=x_range[n][1]:
			if not c[i][-1] in phb[n].keys():
				phb[n][c[i][-1]]=[[float(j) for j in c[i][:-3]]]
			else:
				phb[n][c[i][-1]].append([float(j) for j in c[i][:-3]])


plt.rcParams['axes.prop_cycle']=cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
	'#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']+['xkcd:salmon','xkcd:light green','xkcd:yellow'
	,'xkcd:navy','xkcd:light pink','xkcd:gold','xkcd:neon pink','xkcd:baby blue','xkcd:forrest green'
	,'xkcd:cobalt blue','xkcd:reddish pink','xkcd:hot purple','xkcd:ecru','xkcd:rust red'
	,'xkcd:fern green','xkcd:rose red'])

color_table=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
	'#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']+['xkcd:salmon','xkcd:light green','xkcd:yellow'
	,'xkcd:navy','xkcd:light pink','xkcd:gold','xkcd:neon pink','xkcd:baby blue','xkcd:forrest green'
	,'xkcd:cobalt blue','xkcd:reddish pink','xkcd:hot purple','xkcd:ecru','xkcd:rust red'
	,'xkcd:fern green','xkcd:rose red']

#register color for each phase equilibrium
pheq=list(phb[0].keys())
for i in range(1,len(phb)):
	z=list(phb[i].keys())
	for j in range(len(z)):
		if not z[j] in pheq:
			pheq.append(z[j])

phase_colors={}
for i in range(len(pheq)):
	phase_colors[pheq[i]]=color_table[i]


for n in range(len(phb)):
	iphase=0
	plt.cla()
	#for n in range(nxrange):
	#plt.subplot(1,nxrange,n+1)
	for i in phb[n].keys():
		print(phb[n].keys())
		#ii=r'$\ce{'+i+'}$'
		#ii=i.translate(sub)
		data=np.array(phb[n][i])
		#plt.fill_between(np.log10(data[:,0]/(1e-30)*(1.6*1e-19)/1e9), data[:,1], data[:,2], color=cmp(iphase/len(list(phb.keys()))),label=i)
		plt.fill_between(data[:,0]/(1e-30)*(1.6*1e-19)/1e9, data[:,1], data[:,2],label=i,color=phase_colors[i])
		iphase=iphase+1
	plt.hlines(HER, plim[0]/(1e-30)*(1.6*1e-19)/1e9,plim[1]/(1e-30)*(1.6*1e-19)/1e9, 
			colors = "k", linestyles = "dashed")	
	plt.xlim(plim/(1e-30)*(1.6*1e-19)/1e9)
	plt.ylim(ulim)
	plt.xscale('log')
	plt.xlabel("P (GPa)",fontsize=15)
	plt.ylabel('$\mathrm{U}_\mathrm{RHE}$ (V)',fontsize=15)
	plt.tick_params(axis='both',which='major',labelsize=13,width=2)
	plt.rcParams["axes.labelweight"] = "bold"
	plt.legend(frameon=False,prop={'size': 9},loc='upper left',bbox_to_anchor=(1, 1.03),ncol=1,columnspacing=1,handletextpad=0.1,scatteryoffsets=[0.5])
	plt.subplots_adjust(left=0.14, right=0.74, bottom=0.13, top=0.95)
	#plt.show()
	plt.savefig('LiMgH-'+str(n)+'.pdf',format='pdf',dpi=350) #save Li-Mg-H phase daigrams


# plot phase diagram superimposed with instability of a compound
compound='LiMgH16' #the compound with instability
instability=np.zeros([nu+1,nnp+1])
iphase=-np.ones([nu+1,nnp+1])

def get_decomp_and_e(entry,entry_ref,per_atom=True):
	"""
	Calculate the decomposition of a compound against a set of references.
	Input: 
	entry:(composition, energy) of the compound, e.g., ('LiH',-3)
	entry_ref: list of (composition, energy) of the references
	Return: decomposition coefficient, decomposition energy
	"""
	c=Composition(entry[0])
	c_ref=[Composition(i[0]) for i in entry_ref]
	ele=[]
	for i in [c]+c_ref:
		ele=ele+i.elements
	el = []
	null=[el.append(i) for i in ele if not i in el]
	st=[c[i] for i in el]
	st_ref=[[c_ref[j][i] for i in el] for j in range(len(c_ref))]
	st=np.array(st);st_ref=np.transpose(np.array(st_ref))
	coeff=np.dot(np.linalg.inv(st_ref),st)
	e=entry[1]
	e_ref=[i[1] for i in entry_ref]
	e_ref=np.array(e_ref)
	de=e-np.dot(np.transpose(e_ref),coeff)
	if per_atom:
		de=de/c.num_atoms
		coeff=coeff/c.num_atoms
	coeffs={}
	for i in range(len(el)):
		coeffs[el[i].name]=coeff[i]
	return coeffs,de

for i in phb.keys():
	data=np.array(phb[i])
	x=np.log10(data[:,0]/(1e-30)*(1.6*1e-19)/1e9)
	#plt.fill_between(np.log10(data[:,0]/(1e-30)*(1.6*1e-19)/1e9), data[:,1], data[:,2], color=cmp(iphase/len(list(phb.keys()))),label=i)
	plt.plot(x, data[:,1],color='k')
	plt.plot(x, data[:,2],color='k')
	# label all the grid points belonging to this phase region
	u1=interpolate.interp1d(x, data[:,1],kind='cubic',fill_value="extrapolate")
	u2=interpolate.interp1d(x, data[:,2],kind='cubic',fill_value="extrapolate")
	ip=(x[[0,-1]]-np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9))/(np.log10(plim[1]/plim[0]))*nnp 
	ip1=int(np.floor(ip[0]));ip2=int(np.ceil(ip[1]))
	for j in range(ip1,ip2+1):
		log_p_map=np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9)+np.log10(plim[1]/plim[0])*j/nnp
		u1_map=u1(log_p_map)
		u2_map=u2(log_p_map)
		iu=(np.array([u1_map,u2_map])-ulim[0])/(ulim[1]-ulim[0])*nu
		iu1=int(np.floor(iu[0]));iu2=int(np.ceil(iu[1]))
		iu1=max(iu1,0);iu2=min(iu2,nu)
		iphase[iu1:(iu2+1),j]=list(phb.keys()).index(i)
		p_map=10**(log_p_map)*1e9*1e-30/(1.6*1e-19)
		compr=i.split('+')+['H']
		y=[h_ip[comp.index(k)](p_map) for k in compr if not k=='H']+[hH(p_map)]
		entry=[(compr[k],y[k]*Composition(compr[k]).num_atoms) for k in range(len(compr))]
		y_compound=h_ip[comp.index(compound)](p_map) 
		entry_compound=(compound,y_compound*Composition(compound).num_atoms)
		coeffs,de=get_decomp_and_e(entry_compound,entry)
		instability[iu1:(iu2+1),j]=np.array([de-coeffs['H']*(-1)*(k/nu*(ulim[1]-ulim[0])+ulim[0]) 
		for k in range(iu1,(iu2+1))]) 
		if j in range(283,287):
			print(de);print(coeffs['H'])

# find lowest instability and conditions
insmin=instability[:,int(nnp*0):int(nnp*1.1)].min() 
iumin, ipmin = np.where(np.isclose(instability, insmin))
umin=ulim[0]+iumin/nu*(ulim[1]-ulim[0])
pmin=plim[0]*(plim[1]/plim[0])**(ipmin/nnp)/(1e-30)*(1.6*1e-19)/1e9

cmp=cm.get_cmap('rainbow')
plt.imshow(instability,extent=[np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9),np.log10(plim[1]/(1e-30)*(1.6*1e-19)/1e9),ulim[0],ulim[1]],
	origin='lower',aspect='auto',cmap=cmp)
plt.hlines(HER, np.log10(plim[0]/(1e-30)*(1.6*1e-19)/1e9),np.log10(plim[1]/(1e-30)*(1.6*1e-19)/1e9), 
		colors = "k", linestyles = "dashed")

#plt.legend(frameon=False,prop={'size': 11},loc=(0,1),ncol=5,columnspacing=1,handletextpad=0.3,scatteryoffsets=[0.5])
plt.xlim(np.log10(plim/(1e-30)*(1.6*1e-19)/1e9))
plt.ylim(ulim)
plt.tick_params(axis='both',which='major',labelsize=13,width=2)
plt.rcParams["axes.labelweight"] = "bold"
plt.xticks(color='w')
plt.yticks(color='w')
plt.show()

sm = plt.cm.ScalarMappable(cmap=cmp)
sm.set_clim(instability.min(), instability.max())
cb=plt.colorbar(sm,orientation = "horizontal",aspect=15)
cb.set_label(label='Instability (eV/atom)',weight='bold',fontsize=15)
cb.ax.tick_params(labelsize=13) 
plt.show()

plt.scatter(np.log10(np.array(p)),h_ip[comp.index(compound)](np.array(p)))
plt.show()