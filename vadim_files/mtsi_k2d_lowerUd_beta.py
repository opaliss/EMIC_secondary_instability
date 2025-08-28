#!/usr/bin/env python3

# ---------------------------------------------
# Solves Electrostatic Dispersion Relation for
# unmagnetized ions + magnetized electrons
# accounting for a possible perpendicular drift between
# the two component
#
# Vadim Roytershteyn, 2022
# ---------------------------------------------

from queue import PriorityQueue
import numpy as np
import math
from scipy.special import jv
from scipy.special import ive
from scipy.special import wofz
from scipy.optimize import root
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('TkAgg')

import h5py

if len(sys.argv) > 1:
    mode = sys.argv[1]
else:
    mode = "mtsi"

cutoff = 0.2

# ---------------------------------------
# COLD ELECTRONS
# ---------------------------------------

class cold_e_mtsi:

    # initialization
    # U0 : magnitude of the flow
    # vthi = sqrt(Ti/mi)
    # wpewce
    def __init__(self, U0, ni, vthi, wpe_wce, mi_me, theta):

        # ion densities
        self.ni = ni

        # U0
        self.U0 = U0

        # Wce
        self.Wce = -1/wpe_wce

        # vthi
        self.vthi = vthi

        # mass ratio
        self.mi_me = mi_me

        # propagation angle
        self.theta = theta

    # ---------------------------
    # plasma dispersion function
    # ---------------------------
    def Z(self, xi):
        Z_ = 1j*np.sqrt(np.pi)*wofz(xi)
        return Z_

    # ---------------------------
    # determinant of the mode coupling
    # ---------------------------
    def determinant(self, w_2D):

        w = w_2D[0] + 1j*w_2D[1]

        k  = self.k_
        kper = k*np.sin(self.theta)
        #kpar = k*np.cos(self.theta)

        xi = (w-kper*self.U0)/(np.sqrt(2.0)*k*self.vthi)
        Z_   =  self.Z(xi)
        Zprime = -2.0*(1.0+xi*Z_)

        el_response = k**2*(np.cos(self.theta)**2/w**2 - np.sin(self.theta)**2/(self.Wce**2-w**2) )
        #el_response = k**2*(np.cos(self.theta)**2/w**2 - 1/self.Wce**2 )
        #res_ = 0.5*Zprime/self.mi_me/self.vthi**2 + el_response

        res_ = el_response - k**2
        for s in range(len(self.ni)):
            xi = (w-kper*self.U0[s])/(np.sqrt(2.0)*k*self.vthi[s])
            Z_   =  self.Z(xi)
            Zprime = -2.0*(1.0+xi*Z_)
            eps_i = 0.5*Zprime/self.mi_me[s]/self.vthi[s]**2 * ni[s]
            res_ += eps_i

        return (res_.real,res_.imag)


    # ---------------------------------------
    # Nonlinear solver: for a given k, return
    # the solution w
    # inputs : value of k and guess wg
    # ---------------------------------------
    def solve(self,k,wg):

        # store the value of k, so that it's accessible inside the function called by the solver
        self.k_ = k

        # initial guess
        wg_ = [wg.real,wg.imag]
        sol = root(self.determinant, wg_ )
        res_ = sol.x[0] + 1j*sol.x[1]

        return (res_,sol.success)


# ---------------------------------------
# WARM ELECTRONS
# ---------------------------------------
class warm_e_mtsi:

    # initialization
    # U0: magnitude of the flow speed
    # vthe = sqrt(Te/me)
    # vthi = sqrt(Ti/mi)
    # wpewce
    def __init__(self, U0, vthe, ni, vthi, wpe_wce, mi_me, theta):

        # number of bessel functions to include in the sims
        self.Nbessel = 10

        # (relative) ion densities
        self.ni = ni

        # U0
        self.U0 = U0

        # Wce
        self.Wce = -1/wpe_wce

        # vthi and vthe
        self.vthi = vthi
        self.vthe = vthe

        self.mi_me = mi_me

        self.theta = theta

    # ---------------------------
    # plasma dispersion function
    # ---------------------------
    def Z(self, xi):
        Z_ = 1j*np.sqrt(np.pi)*wofz(xi)
        return Z_


    # ---------------------------
    # computes the sum over Bessel
    # functions
    # input : frequency w, parameter lambda
    # ---------------------------
    def Isum( self, w, lambda_ ):

        Nb = self.Nbessel   # number of functions to include

        kpar  = self.k_*np.cos(self.theta)

        res_ = 0.
        for m in np.arange(-Nb,Nb):
            xim = (w - m*self.Wce)/(kpar*np.sqrt(2.0)*self.vthe)
            res_ += ive(m, lambda_)*self.Z(xim)

        return res_

    # ---------------------------
    # determinant of the mode coupling
    # ---------------------------
    def determinant( self, w_2D ):

        w = w_2D[0] + 1j*w_2D[1]

        k  = self.k_
        kpar = k*np.cos(self.theta)
        kper = k*np.sin(self.theta)
        lambda_ = kper**2*self.vthe**2/self.Wce**2


        xi = (w - kper*self.U0)/(k*np.sqrt(2.0)*self.vthi)
        Z_   =  self.Z(xi)
        Zprime = -2.0*(1.0+xi*Z_)

        esum_ = self.Isum(w,lambda_)*w/(np.sqrt(2.0)*self.vthe*kpar)
        el_response = 1.0/self.vthe**2*(1+esum_)

        res_ = - el_response - k**2
        for s in range(len(self.ni)):
            xi = (w-kper*self.U0[s])/(np.sqrt(2.0)*k*self.vthi[s])
            Z_   =  self.Z(xi)
            Zprime = -2.0*(1.0+xi*Z_)
            res_ += self.ni[s]*0.5*Zprime/self.mi_me[s]/self.vthi[s]**2

#        res_ = el_response
#        print(res_)

        return (res_.real,res_.imag)


    # ---------------------------------------
    # Nonlinear solver: for a given k, return
    # the solution w
    # inputs : value of k and guess wg
    # ---------------------------------------
    def solve(self,k,wg):

        # store the value of k, so that it's accessible inside the function called by the solver
        self.k_ = k

        # initial guess
        wg_ = [wg.real,wg.imag]
        sol = root(self.determinant, wg_ )
        res_ = sol.x[0] + 1j*sol.x[1]

        residual = self.determinant(w_2D=sol.x)
        print("residual = ", np.abs(residual[0] + 1j * residual[1]))
        return (res_,sol.success)

# ---------------------------------------
# Begin main section
# ---------------------------------------


# wpe/wce
wpewce = 2.0
# Electron cyclotron
Wce = 1/wpewce

# electron beta = 8*pi*n*Te/B_0^2
betae = 1E-4

# at reference conditions (i.e. without increase in beta), same as the simulations
# but unfortunately wrong by a factor of sqrt(2)
rhoesqrt2 = np.sqrt(betae)
# corrected value that matches re in VPIC
rhoe0 = np.sqrt(0.5*betae)

# temperature ratios of ions to electrons
tite = np.array([1., 1e4, 1.])

if mode == "mtsi2":
     A = 2.6264563
     betae *= A
     tite  /= A
elif mode == "ionion2":
     A = 2.6264563
     betae *= A
     tite  /= A
elif mode == "mtsi3":
     A = 3.0422365
     betae *= A
     tite  /= A
elif mode == "ionion3":
     A = 3.0422365
     betae *= A
     tite  /= A
elif mode == "mtsi4":
     A = 2.6264563
     betae *= A
     tite  /= A
elif mode == "ionion4":
     A = 2.6264563
     betae *= A
     tite  /= A


# electron thermal speed / c = sqrt(Te/me)/c at actual conditions
vte = np.sqrt(0.5*betae)/wpewce

# mass ratios of ions to electrons
mime = np.array([100., 100., 10*100.])
# relative drift speed of the ion species in the electron frame
if mode == "mtsi" or mode == "mtsi2" or mode == "mtsi3" or mode == "mtsi4":
    Ud = np.array([1.0e-3, 1.0e-3, -1.5e-3])
elif mode == "ionion" or mode == "ionion2" or mode == "ionion3" or mode == "ionion4":
    Ud = np.array([1.0e-3, 1.0e-3, -1.5e-3]) * -1.

# ion densitites
if mode == "mtsi4":
    nO = 0.00
    ni = np.array([0.9-nO, 0.1, nO])
elif mode == "ionion4":
    nO = 0.01 / 100.
    ni = np.array([0.9-nO, 0.1, nO])
else:
    ni = np.array([0.8, 0.1, 0.1])

Wci = Wce/mime[0]

# ion temperature
vti = vte*np.sqrt(tite/mime)
# lower-hybrid frequency/wpe
wLH = 1/np.sqrt(np.amin(mime))/wpewce

print(mode)
print("Ud = "+str(Ud))
print("wLH = "+str(wLH))
print("betae = "+str(betae))
print("ni = "+str(ni))
print("vti = "+str(vti))
print("mi_me = "+str(mime))

degree = 1/180*np.pi

#----------------------------------------

if mode == "mtsi":
    # at reference parameters
    # #Angle is 87.65 degree
    # kpar0 = 0.024048096192384773 / rhoesqrt2
    # kper0 = 0.4328657314629259 / rhoesqrt2
    # gamma0 = 0.17381379407474326 * wLH
    # omega0 = 0.35775905499439775 * wLH
    # Angle is 87.60 degree
    kpar0 = 0.018509254627313655 / rhoesqrt2
    kper0 = 0.44222111055527763 / rhoesqrt2
    gamma0 = 0.11598813520502371 * wLH
    omega0 = 0.23803350718440489 * wLH
elif mode == "ionion":
    # at reference parameters
    # kpar0 = 0.0 / rhoesqrt2
    # kper0 = 0.14428857715430862 / rhoesqrt2
    # gamma0 = 0.11073368712783938 * wLH
    # omega0 = 0.4194155664343345 * wLH
    kpar0 = 0.0 / rhoesqrt2
    kper0 = 0.17708854427213605 / rhoesqrt2
    gamma0 = 0.10782320595185518 * wLH
    omega0 = 0.4265399627367856 * wLH
elif mode == "mtsi2":
    # increase betae to 2.63 as we have by time 185 Wci^-1 in the simulation with only protons
    # note, this still contains 10% heavy ions
    # # Angle is 88.89
    # kpar0 = 0.008016032064128258 / rhoesqrt2
    # kper0 = 0.4148296593186373 / rhoesqrt2
    # gamma0 = 0.04874885667576077 * wLH
    # omega0 = 0.09251305433341739 * wLH
    # Angle is 88.76 degree
    kpar0 = 0.009004502251125562 / rhoesqrt2
    kper0 = 0.41620810405202596 / rhoesqrt2
    gamma0 = 0.049093779515762444 * wLH
    omega0 = 0.10398577256020898 * wLH
elif mode == "ionion2":
    # increase betae to 2.63 as we have by time 185 Wci^-1 in the simulation with only protons
    # kpar0 = 0.0 / rhoesqrt2
    # kper0 = 0.17835671342685375 / rhoesqrt2
    # gamma0 = 0.1086580303851905 * wLH
    # omega0 = 0.4297384720810467 * wLH
    kpar0 = 0.0 / rhoesqrt2
    kper0 = 0.17858929464732368 / rhoesqrt2
    gamma0 = 0.10865898889055448 * wLH
    omega0 = 0.42998016144217666 * wLH
elif mode == "mtsi3":
    # increase betae to 3.04 as we have by time 185 Wci^-1 in the simulation with heavy ions present
    # # Angle is 88.90
    # kpar0 = 0.008016032064128258 / rhoesqrt2
    # kper0 = 0.41883767535070143 / rhoesqrt2
    # gamma0 = 0.03851124437902947 * wLH
    # omega0 = 0.08907516307208045 * wLH
    # Angle is 88.98
    kpar0 = 0.007503751875937968 / rhoesqrt2
    kper0 = 0.4197098549274637 / rhoesqrt2
    gamma0 = 0.038559785244940886 * wLH
    omega0 = 0.08364457253647514 * wLH
elif mode == "ionion3":
    # increase betae to 3.04 as we have by time 185 Wci^-1 in the simulation with heavy ions present
    # kpar0 = 0.0 / rhoesqrt2
    # kper0 = 0.17835671342685375 / rhoesqrt2
    # gamma0 = 0.10886933212281688 * wLH
    # omega0 = 0.430223198813943 * wLH
    kpar0 = 0.0 / rhoesqrt2
    kper0 = 0.1790895447723862 / rhoesqrt2
    gamma0 = 0.10887642609257435 * wLH
    omega0 = 0.4309896514332275 * wLH
elif mode == "mtsi4":
    # increase betae to 2.63 as we have by time 185 Wci^-1 in the simulation with only protons
    # reduce nO to zero
    # # Angle is 88.97
    # kpar0 = 0.008016032064128258 / rhoesqrt2
    # kper0 = 0.4448897795591183 / rhoesqrt2
    # gamma0 = 0.04275962279250277 * wLH
    # omega0 = 0.08533044512476352 * wLH
    # Angle is 88.84
    kpar0 = 0.009004502251125562 / rhoesqrt2
    kper0 = 0.44372186093046523 / rhoesqrt2
    gamma0 = 0.043049077735282194 * wLH
    omega0 = 0.09549843156538732 * wLH
elif mode == "ionion4":
    # increase betae to 2.63 as we have by time 185 Wci^-1 in the simulation with only protons
    # reduce n)
    # kpar0 = 0.0 / rhoesqrt2
    # kper0 = 0.17835671342685375 / rhoesqrt2
    # gamma0 = 0.1086580303851905 * wLH
    # omega0 = 0.4297384720810467 * wLH
    kpar0 = 0.0 / rhoesqrt2
    kper0 = 0.18637274549098198 / rhoesqrt2
    gamma0 = 0.012443488126779664 * wLH
    omega0 = 0.5123110630535211 * wLH









N = 200
#N = 1000
#N = 500

omegaA = np.zeros((N,N))
gammaA = np.zeros((N,N))
k_ = np.linspace(0., 1.0/rhoesqrt2, N)

i0 = np.searchsorted(k_, kpar0)
j0 = np.searchsorted(k_, kper0)

q = PriorityQueue()
q.put((-gamma0,omega0,i0,j0))

while not q.empty():
    gamma,omega,i,j = q.get()
    gamma *= -1.

    if omegaA[i,j] == 0.:
        kpar = k_[i]
        kper = k_[j]

        theta = math.atan2(kper, kpar)
        k = np.sqrt(kpar**2 + kper**2)

        #cold_solver = cold_e_mtsi( wpe_wce=wpewce,           ni=ni, vthi=vti, U0=Ud, mi_me=mime, theta=theta)
        warm_solver = warm_e_mtsi( wpe_wce=wpewce, vthe=vte, ni=ni, vthi=vti, U0=Ud, mi_me=mime, theta=theta)

        #wc,ierr = cold_solver.solve(k, omega+gamma*1j)
        w, ierr = warm_solver.solve(k, omega+gamma*1j)

        if 0. < w.real < 4.*wLH and 0. < w.imag < 2.*gamma:
            omegaA[i,j] = w.real
            gammaA[i,j] = w.imag

            if w.imag > cutoff * np.amax(gammaA):
                if i > 0 and j > 0 and omegaA[i-1,j-1] == 0.:
                    q.put((-w.imag,w.real,i-1,j-1))
                if i > 0 and omegaA[i-1,j  ] == 0.:
                    q.put((-w.imag,w.real,i-1,j  ))
                if i > 0 and j < N-1 and omegaA[i-1,j+1] == 0.:
                    q.put((-w.imag,w.real,i-1,j+1))
                if j > 0 and j > 0 and omegaA[i  ,j-1] == 0.:
                    q.put((-w.imag,w.real,i  ,j-1))

                if j < N-1 and omegaA[i  ,j+1] == 0.:
                    q.put((-w.imag,w.real,i  ,j+1))
                if i < N-1 and j > 0 and omegaA[i+1,j-1] == 0.:
                    q.put((-w.imag,w.real,i+1,j-1))
                if i < N-1 and omegaA[i+1,j  ] == 0.:
                    q.put((-w.imag,w.real,i+1,j  ))
                if i < N-1 and j < N-1 and omegaA[i+1,j+1] == 0.:
                    q.put((-w.imag,w.real,i+1,j+1))

# dump to file in case we screw plotting up
f = h5py.File(mode+".hdf5",'w')
f.attrs["wLH"] = wLH
f.attrs["Wci"] = Wci
f.attrs["wpewce"] = wpewce
f.attrs["rhoe"] = rhoe0
f.attrs["betae"] = betae
f.attrs["betaiC"] = betae * tite[0]
f.attrs["betaiH"] = betae * tite[1]
f.attrs["betaO"] = betae * tite[2]
f.attrs["mi"] = mime[0]
f.attrs["mO"] = mime[2]
f.attrs["Ui"] = Ud[0]
f.attrs["UO"] = Ud[2]
f.attrs["niC"] = ni[0]
f.attrs["niH"] = ni[1]
f.attrs["nO"] = ni[2]
dset = f.create_dataset("omega", data=omegaA)
dset = f.create_dataset("gamma", data=gammaA)
dset = f.create_dataset("k", data=k_)
f.close()

# where is is maximum growth rate in k space?
idx = np.argmax(gammaA)
imax,jmax = np.unravel_index(idx, gammaA.shape)
#print("imax = "+str(imax))
#print("jmax = "+str(jmax))
kparmax = k_[imax]
kpermax = k_[jmax]
print("kpar0 = "+str(kparmax*rhoesqrt2)+" / rhoesqrt2")
print("kper0 = "+str(kpermax*rhoesqrt2)+" / rhoesqrt2")
print("kpar0 rho_e = "+str(kparmax*rhoe0))
print("kper0 rho_e = "+str(kpermax*rhoe0))
thetamax = math.atan2(kpermax, kparmax)
kmax = np.sqrt(kparmax**2 + kpermax**2)
omegamax = omegaA[imax,jmax]
gammamax = gammaA[imax,jmax]
print("theta = "+str(thetamax/degree)+" degree")
print("kbest = "+str(kmax*rhoe0)+" rho_e^-1")
print("gamma = "+str(gammamax/wLH)+" * wLH")
print("omega = "+str(omegamax/wLH)+" * wLH")

# there is the maximum growth rate if we limit ourselves to perpendicular propagation?
jmax2 = np.argmax(gammaA[0,:])
kparmax2 = 0.
kpermax2 = k_[jmax2]
omegamax2 = omegaA[0,jmax2]
gammamax2 = gammaA[0,jmax2]
print("compare with omega = "+str(omegamax2/wLH)+" gamma = "+str(gammamax2/wLH)+" wLH at kper rhoe = "+str(kpermax2*rhoe0))

# what is the smallest positive growth rate that we found?
gammamin = cutoff*gammamax
print("gammamin = "+str(gammamin/wLH)+" wLH")

if mode == "mtsi" or imax==0:
    levels = np.linspace(gammamin, gammamax, 10)
else:
    # what is the gamma of the saddle for small kpar?
    tmp = np.amax(gammaA, axis=1)
    gammaSaddle = np.amin(tmp[0:imax])
    print("gammaSaddle = "+str(gammaSaddle/wLH)+" wLH")

    # set size for gamma
    dgamma = gammamax/10.
    print("dgamma = "+str(dgamma))
    offset = 0.5*(gammamax2-gammaSaddle)
    print("offset = "+str(offset))

    # values for contour lines
    levels = [gammaSaddle+offset]
    g = gammaSaddle+offset
    while g+dgamma <= gammamax:
        levels.append(g+dgamma)
        g = g+dgamma
    g = gammaSaddle+offset
    while g-dgamma >= 0.9*gammamin:
        levels.append(g-dgamma)
        g = g-dgamma
    levels = sorted(levels)


im = plt.contourf(k_*rhoe0, k_*rhoe0, gammaA.T/wLH)
plt.plot(kparmax*rhoe0, kpermax*rhoe0, 'ko')
ann1 = r"""$\omega = """+str(round(omegamax/wLH,3))+r"""\omega_{LH}$
$\gamma = """+str(round(gammamax/wLH,3))+r"""\omega_{LH}$"""
plt.annotate(ann1, (kparmax*rhoe0, kpermax*rhoe0))
if mode == "ionion" and imax > 0:
    plt.plot(kparmax2*rhoe0, kpermax2*rhoe0, 'ko')
    ann2 = r"""$\omega = """+str(round(omegamax2/wLH,3))+r"""\omega_{LH}$
    $\gamma = """+str(round(gammamax2/wLH,3))+r"""\omega_{LH}$"""
    plt.annotate(ann2, (kparmax2*rhoe0, kpermax2*rhoe0))
plt.colorbar(im, label="$\gamma / \omega_{LH}$")
plt.xlabel(r"$k_\parallel\,\rho_e$")
plt.ylabel(r"$k_\perp\,\rho_e$")
plt.savefig(mode+"_gamma.png")
plt.xlim(0, 0.06)
plt.ylim(0, 0.6)
plt.show()
plt.close()



im = plt.contourf(k_*rhoe0, k_*rhoe0, omegaA.T/wLH)
plt.plot(kparmax*rhoe0, kpermax*rhoe0, 'ko')
ann1 = r"""$\omega = """+str(round(omegamax/wLH,3))+r"""\omega_{LH}$
$\gamma = """+str(round(gammamax/wLH,3))+r"""\omega_{LH}$"""
plt.annotate(ann1, (kparmax*rhoe0, kpermax*rhoe0))
plt.colorbar(im, label="$\gamma / \omega_{LH}$")
plt.xlabel(r"$k_\parallel\,\rho_e$")
plt.ylabel(r"$k_\perp\,\rho_e$")
plt.savefig(mode+"_gamma.png")
plt.xlim(0, 0.06)
plt.ylim(0, 0.6)
plt.show()
plt.close()

# im = plt.imshow(gammaA.T/wLH, extent=(k_[0]*rhoe0,k_[-1]*rhoe0,k_[0]*rhoe0,k_[-1]*rhoe0), origin='lower', aspect='auto')
# plt.contour(gammaA.T/wLH, extent=(k_[0]*rhoe0,k_[-1]*rhoe0,k_[0]*rhoe0,k_[-1]*rhoe0), origin='lower', colors='k', levels=levels/wLH, linewidths=0.5)
# plt.plot(kparmax*rhoe0, kpermax*rhoe0, 'ko')
# plt.annotate(ann1, (kparmax*rhoe0, kpermax*rhoe0))
# if mode == "ionion" and imax > 0:
#     plt.plot(kparmax2*rhoe0, kpermax2*rhoe0, 'ko')
#     plt.annotate(ann2, (kparmax2*rhoe0, kpermax2*rhoe0))
# plt.colorbar(im, label="$\gamma / \omega_{LH}$")
# plt.xlabel(r"$k_\parallel\,\rho_e$")
# plt.ylabel(r"$k_\perp\,\rho_e$")
#
# plt.ylim(0, 0.6)
# plt.xlim(0, 0.06)
# plt.savefig(mode+"_gamma_z1.png")
# plt.show()
# plt.close()
#
# im = plt.imshow(omegaA.T/wLH, extent=(k_[0]*rhoe0,k_[-1]*rhoe0,k_[0]*rhoe0,k_[-1]*rhoe0), origin='lower', aspect='auto')
# plt.colorbar(im, label="$\omega / \omega_{LH}$")
# plt.xlabel(r"$k_\parallel\,\rho_e$")
# plt.ylabel(r"$k_\perp\,\rho_e$")
# plt.savefig(mode+"_omega.png")
# plt.show()
# plt.close()


#mask = omegaA[0,:] > 0.
#print(np.amin(k_[mask])*rhoe0)
#print(np.amax(k_[mask])*rhoe0)
