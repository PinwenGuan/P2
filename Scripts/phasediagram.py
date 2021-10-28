"""
Pourbaix calculator for hydrides.
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

"""
Modified ase pourbaix calculator. Slow and problematic.
"""

from ase.phasediagram import *
class Pourbaix_gpw:
    def __init__(self, references, formula=None, T=300.0, **kwargs):
        """Pourbaix object.

        references: list of (name, energy) tuples
            Examples of names: ZnO2, H+(aq), H2O(aq), Zn++(aq), ...
        formula: str
            Stoichiometry.  Example: ``'ZnO'``.  Can also be given as
            keyword arguments: ``Pourbaix(refs, Zn=1, O=1)``.
        T: float
            Temperature in Kelvin.
        """

        if formula:
            assert not kwargs
            kwargs = parse_formula(formula)[0]

        self.kT = units.kB * T
        self.references = []
        for name, energy in references:
            if name == 'O':
                continue
            count, charge, aq = parse_formula(name)
            for symbol in count:
                if aq:
                    if not (symbol in 'HO' or symbol in kwargs):
                        break
                else:
                    #if symbol not in kwargs:
                    if not (symbol in 'HO' or symbol in kwargs):
                        break
            else:
                self.references.append((count, charge, aq, energy, name))

        self.references.append(({}, -1, False, 0.0, 'e-'))  # an electron

        self.count = kwargs

        if 'O' not in self.count:
            self.count['O'] = 0

        self.N = {'e-': 0, 'H': 1}
        for symbol in kwargs:
            if symbol not in self.N:
                self.N[symbol] = len(self.N)

    def decompose(self, U, pH, verbose=True, concentration=1e-6):
        """Decompose material.

        U: float
            Potential in V.
        pH: float
            pH value.
        verbose: bool
            Default is True.
        concentration: float
            Concentration of solvated references.

        Returns optimal coefficients and energy.
        """

        alpha = np.log(10) * self.kT
        entropy = -np.log(concentration) * self.kT

        # We want to minimize np.dot(energies, x) under the constraints:
        #
        #     np.dot(x, eq2) == eq1
        #
        # with bounds[i,0] <= x[i] <= bounds[i, 1].
        #
        # First two equations are charge and number of hydrogens, and
        # the rest are the remaining species.
        if 'H' not in self.count:
            eq1 = [0, 0] + list(self.count.values())
        else:
            eq1 = [0] + list(self.count.values())
        eq2 = []
        energies = []
        bounds = []
        names = []
        for count, charge, aq, energy, name in self.references:
            eq = np.zeros(len(self.N))
            eq[0] = charge
            for symbol, n in count.items():
                eq[self.N[symbol]] = n
            eq2.append(eq)
            if name in ['H2O(aq)', 'H+(aq)', 'e-']:
                bounds.append((-np.inf, np.inf))
                if name == 'e-':
                    energy = -U
                elif name == 'H+(aq)':
                    energy = -pH * alpha
            else:
                bounds.append((0, 1))
                if aq:
                    energy -= entropy
            if verbose:
                print('{0:<5}{1:10}{2:10.3f}'.format(len(energies),
                                                     name, energy))
            energies.append(energy)
            names.append(name)

        try:
            from scipy.optimize import linprog
        except ImportError:
            from ase.utils._linprog import linprog
        result = linprog(energies, None, None, np.transpose(eq2), eq1, bounds)

        if verbose:
            print_results(zip(names, result.x, energies))

        return result.x, result.fun

    def diagram(self, U, pH, plot=True, show=True, ax=None):
        """Calculate Pourbaix diagram.

        U: list of float
            Potentials in V.
        pH: list of float
            pH values.
        plot: bool
            Create plot.
        show: bool
            Open graphical window and show plot.
        ax: matplotlib axes object
            When creating plot, plot onto the given axes object.
            If none given, plot onto the current one.
        """
        a = np.empty((len(U), len(pH)), int)
        a[:] = -1
        colors = {}
        f = functools.partial(self.colorfunction, colors=colors)
        bisect(a, U, pH, f)
        compositions = [None] * len(colors)
        names = [ref[-1] for ref in self.references]
        for indices, color in colors.items():
            compositions[color] = ' + '.join(names[i] for i in indices
                                             if names[i] not in
                                             ['H2O(aq)', 'H+(aq)', 'e-'])
        text = []
        for i, name in enumerate(compositions):
            b = (a == i)
            x = np.dot(b.sum(1), U) / b.sum()
            y = np.dot(b.sum(0), pH) / b.sum()
            name = re.sub('(\S)([+-]+)', r'\1$^{\2}$', name)
            name = re.sub('(\d+)', r'$_{\1}$', name)
            text.append((x, y, name))


        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            if ax is None:
                ax = plt.gca()

            # rasterized pcolormesh has a bug which leaves a tiny
            # white border.  Unrasterized pcolormesh produces
            # unreasonably large files.  Avoid this by using the more
            # general imshow.
            ax.imshow(a, cmap=cm.Accent,
                      extent=[min(pH), max(pH), min(U), max(U)],
                      origin='lower',
                      aspect='auto')

            for x, y, name in text:
                ax.text(y, x, name, horizontalalignment='center')
            ax.set_xlabel('pH')
            ax.set_ylabel('potential [V]')
            ax.set_xlim(min(pH), max(pH))
            ax.set_ylim(min(U), max(U))
            hx=[-2,16]
            hy=[0.028,-1.034]
            plt.plot(hx,hy,linestyle='--',color='r')
            if show:
                plt.show()

        return a, compositions, text

    def colorfunction(self, U, pH, colors):
        coefs, energy = self.decompose(U, pH, verbose=False)
        indices = tuple(sorted(np.where(abs(coefs) > 1e-7)[0]))
        color = colors.get(indices)
        if color is None:
            color = len(colors)
            colors[indices] = color
        return color

