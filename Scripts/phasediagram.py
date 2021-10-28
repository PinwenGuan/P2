"""
Pressure-independent Pourbaix calculator for hydrides.
formula: e.g., ['Pd', 'PdH', 'Pd3H4', 'PdH6', 'PdH7', 'PdH8', 'PdH10', 'PdH12'], the first one should be pure metal
E: e.g., [0.0, 0.03, 0.64, 0.96, 1.20, 1.09, 1.33, 1.71]
Return: Pourbaix map represented by an array; stable phase compositions  
"""
from pymatgen.analysis.phase_diagram import *
def Pourbaix_H(formula,E,nu=200,nph=300,ulim=[-1.1,0.2],phlim=[-2,14]):
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    comp=formula
    y=E
    nat=[Composition(i).num_atoms for i in comp]
    entry=[PDEntry(comp[i],y[i]) for i in range(len(y))]+[PDEntry('H',1e5)]
    stability=[PhaseDiagram(entry).get_decomp_and_e_above_hull(i,allow_negative=True)[1] for i in entry[:-1]]
    comp_convex=[comp[i] for i in range(len(comp)) if stability[i]==0]
    y_convex=[y[i]/nat[i] for i in range(len(y)) if stability[i]==0]
    x=[1-float(i.composition.fractional_composition.formula.split(' ')[0].strip(comp[0])) for i in entry[:-1]]
    x_convex=[x[i] for i in range(len(x)) if stability[i]==0]
    mutr=['' for i in range(len(comp_convex)-1)]
    for i in range(len(comp_convex)-1):
        mutr[i]=y_convex[i]+(1-x_convex[i])*(y_convex[i]-y_convex[i+1])/(x_convex[i]-x_convex[i+1])
    # T=300 K, U=-0.0596*pH-mutr
    phmin=phlim[0];phmax=phlim[1];umin=ulim[0];umax=ulim[1]
    a=-np.ones(shape=(nu+1,nph+1))
    for i in range(nu+1):
        u=umin+i/nu*(umax-umin)
        for j in range(len(mutr)+1):
            if j==0:
                ph1=phmax
            else:
                ph1=min((-u-mutr[j-1])/0.0596,phmax)
            if j==len(mutr):
                ph2=phmin
            else:
                ph2=max((-u-mutr[j])/0.0596,phmin)
            for k in range(int(np.ceil((ph2-phmin)/(phmax-phmin)*nph)),int(np.floor((ph1-phmin)/(phmax-phmin)*nph))+1):
                a[i,k]=j
    return a,comp_convex
