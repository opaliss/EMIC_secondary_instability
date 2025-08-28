#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
import matplotlib
matplotlib.use('TkAgg')


#betae = 1.0
#mtsifile = "mtsi.hdf5"
#ionionfile = "ionion.hdf5"
#betae = 2.63
#mtsifile = "mtsi2.hdf5"
#ionionfile = "ionion2.hdf5"
#betae = 3.04
#mtsifile = "mtsi3.hdf5"
#ionionfile = "ionion3.hdf5"
#betae = 2.63
#mtsifile = "mtsi4.hdf5"
#ionionfile = None
betae = 1.0 # *1e-4
mtsifile = "mtsi.hdf5"
ionionfile = "mtsi.hdf5"

f1 = h5py.File(mtsifile,'r')
wLH  = f1.attrs["wLH"]
rhoe = f1.attrs["rhoe"]
omegaA = np.array(f1["omega"])
gammaA = np.array(f1["gamma"])
k = np.array(f1["k"])
print("gamma mtsi range: "+str(np.amin(gammaA/wLH))+".."+str(np.amax(gammaA/wLH)))
f1.close()
# where is is maximum growth rate in k space?
idx = np.argmax(gammaA)
imax,jmax = np.unravel_index(idx, gammaA.shape)
#print("imax = "+str(imax))
#print("jmax = "+str(jmax))
kparmax = k[imax]
kpermax = k[jmax]
print("kpar rho_e = "+str(kparmax*rhoe))
print("kper rho_e = "+str(kpermax*rhoe))
thetamax = math.atan2(kpermax, kparmax)
kmax = np.sqrt(kparmax**2 + kpermax**2)
omegamax = omegaA[imax,jmax]
gammamax = gammaA[imax,jmax]


if ionionfile is not None:
    f2 = h5py.File(ionionfile,'r')
    omegaB = np.array(f2["omega"])
    gammaB = np.array(f2["gamma"])
    print("gamma ionion range: "+str(np.amin(gammaB/wLH))+".."+str(np.amax(gammaB/wLH)))
    f2.close()

    vmin = min([np.amin(gammaA), np.amin(gammaB)]) / wLH
    vmax = max([np.amax(gammaA), np.amax(gammaB)]) / wLH
    gammamax = max([np.amax(gammaA), np.amax(gammaB)])
    gammamin = 0.1 *  min([np.amax(gammaA), np.amax(gammaB)])
    levels = np.linspace(gammamin, gammamax, 10)

    # where is is maximum growth rate in k space?
    idx2 = np.argmax(gammaB)
    imax2,jmax2 = np.unravel_index(idx2, gammaB.shape)
    #print("imax2 = "+str(imax2))
    #print("jmax2 = "+str(jmax2))
    kparmax2 = k[imax2]
    kpermax2 = k[jmax2]
    print("kpar rho_e = "+str(kparmax2*rhoe))
    print("kper rho_e = "+str(kpermax2*rhoe))
    thetamax2 = math.atan2(kpermax2, kparmax2)
    kmax2 = np.sqrt(kparmax2**2 + kpermax2**2)
    omegamax2 = omegaB[imax2,jmax2]
    gammamax2 = gammaB[imax2,jmax2]

else:
    vmin = np.amin(gammaA) / wLH
    vmax = np.amax(gammaA) / wLH
    gammamax = np.amax(gammaA)
    gammamin = 0.1 * np.amax(gammaA)
    levels = np.linspace(gammamin, gammamax, 10)

plt.subplots_adjust(left=0.125,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0)

plt.subplot(2, 1, 1)
plt.title("MTSI (Proton-electron)")
im = plt.imshow(gammaA.T/wLH, extent=(k[0]*rhoe,k[-1]*rhoe,k[0]*rhoe,k[-1]*rhoe), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar(im, label="$\gamma / \omega_{LH}$")
plt.ylabel(r"$k\perp\,\rho_e$")
plt.xlim(0.0,0.06)
plt.ylim(0.,0.7)

if ionionfile is not None:
    plt.subplot(2, 1, 2)
    plt.title("Oxygen related mode")
    im2 = plt.imshow(gammaB.T/wLH, extent=(k[0]*rhoe,k[-1]*rhoe,-k[-1]*rhoe,-k[0]*rhoe), origin='upper', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im2, label="$\gamma / \omega_{LH}$")
    plt.xlabel(r"$k\parallel\,\rho_e$")
    plt.ylabel(r"$k\perp\,\rho_e$")
plt.xlim(0.0,0.06)
plt.ylim(-0.7,0.)

plt.tight_layout()
plt.savefig("modecomparison.png")
plt.show()






if ionionfile is not None:
    plt.suptitle(r"$\beta_e = $"+str(betae)+"$\cdot10^{-4}$")
else:
    plt.suptitle(r"$\beta_e = $"+str(betae)+"$\cdot10^{-4}$, protons only")
plt.contour(gammaA.T/wLH, extent=(k[0]*rhoe,k[-1]*rhoe,k[0]*rhoe,k[-1]*rhoe), origin='lower', colors="#9400D3", levels=levels/wLH, linewidths=0.5)
plt.plot(kparmax*rhoe, kpermax*rhoe, 'ko')
ann1 = r"""  $\omega = """+str(round(omegamax/wLH,3))+r"""\omega_{LH}$
  $\gamma = """+str(round(gammamax/wLH,3))+r"""\omega_{LH}$"""
plt.annotate(ann1, (kparmax*rhoe, kpermax*rhoe))
if ionionfile is not None:
    plt.contour(gammaB.T/wLH, extent=(k[0]*rhoe,k[-1]*rhoe,k[0]*rhoe,-k[-1]*rhoe), origin='lower', colors="#009E73", levels=levels/wLH, linewidths=0.5)
    plt.plot(kparmax2*rhoe, -kpermax2*rhoe, 'ko')
    ann2 = r"""  $\omega = """+str(round(omegamax2/wLH,3))+r"""\omega_{LH}$
  $\gamma = """+str(round(gammamax2/wLH,3))+r"""\omega_{LH}$"""
    plt.annotate(ann2, (kparmax2*rhoe, -kpermax2*rhoe))
plt.xlim(-0.06,0.06)
plt.ylim(-0.7,0.7)
plt.xlabel(r"$k_\parallel \rho_e$")
plt.ylabel(r"$k_\perp \rho_e$")
fn = 'modecomp_beta%g.png' % betae
plt.savefig(fn)
plt.show()
plt.close()

