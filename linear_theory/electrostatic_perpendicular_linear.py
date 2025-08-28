import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import scipy
from QLT_equations.general_plasma_equations import Z, Z_prime, I, J

import matplotlib

font = {'family': 'sans-serif',
        'size': 14}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)


def cold_electron_response(k_perp, omega_pe, alpha_perp_c):
    lambda_ = (k_perp * alpha_perp_c / np.sqrt(2)) ** 2
    return 2 * (omega_pe ** 2) / (alpha_perp_c ** 2) * (1 - I(m=0, Lambda=lambda_))


def ion_response(omega_pi_, alpha_i_, k_perp, v_0_, n_, omega):
    return n_ * (omega_pi_ ** 2) / (alpha_i_ ** 2) * Z_prime(z=(omega - k_perp * v_0_) / (alpha_i_ * np.abs(k_perp)))


def disp_k_approx(k_perp,
                  npC_,
                  nO_,
                  nHe_,
                  omega_pe_,
                  omega_pi_,
                  VDp_,
                  VDHe_,
                  VDO_,
                  alpha_p_,
                  alpha_O_,
                  alpha_He_,
                  alpha_perp_c_):
    return lambda omega: k_perp ** 2 \
                         + cold_electron_response(k_perp=k_perp, omega_pe=omega_pe_,
                                                  alpha_perp_c=alpha_perp_c_) \
                         - ion_response(omega_pi_=omega_pi_, n_=npC_, alpha_i_=alpha_p_, k_perp=k_perp, v_0_=VDp_,
                                        omega=omega) \
                         - ion_response(omega_pi_=omega_pi_, n_=nO_, alpha_i_=alpha_O_, k_perp=k_perp, v_0_=VDO_,
                                        omega=omega) \
                         - ion_response(omega_pi_=omega_pi_, n_=nHe_, alpha_i_=alpha_He_, k_perp=k_perp, v_0_=VDHe_,
                                        omega=omega)


if __name__ == "__main__":
    # normalization
    # time is normalized to the proton cyclotron frequency 1/Omega_cp
    # space is normalized to ion inertial length d_i

    # cold electron density
    ne = 1  # ne
    # cold He+ density
    nHe = 0  # ne
    # cold O+ density
    nO = 0.1  # ne
    # hot proton density
    npH = 0.1  # ne
    # cold proton density
    npC = 1 - npH - nO - nHe  # ne (quasi-neutral)

    # driver frequency
    omega0 = 0.5  # Omega_cp

    # mass ratios
    mp_me = 100
    mO_mp = 16
    mHe_mp = 4

    omega_pe = mp_me * 2  # Omega_cp

    alpha_perp_c = 1e-4 * (mp_me ** 2)  # de x Omega_cp
    alpha_p = alpha_perp_c / np.sqrt(mp_me)  # de x Omega_cp
    alpha_He = alpha_perp_c / np.sqrt(mp_me * mHe_mp)  # de x Omega_cp
    alpha_O = alpha_perp_c / np.sqrt(mp_me * mO_mp)  # de x Omega_cp

    VDp = 3 * alpha_p  # de x Omega_cp
    VDO = VDp * (mO_mp * omega0 / (1 + omega0))
    VDHe = VDp * (mHe_mp * omega0 / (1 + omega0))

    omega_pi = omega_pe / np.sqrt(mp_me)  # Omega_cp

    k_perp = 150

    sol_approx = scipy.optimize.newton(disp_k_approx(k_perp=k_perp, npC_=npC,
                                                     nO_=nO,
                                                     nHe_=nHe,
                                                     omega_pe_=omega_pe,
                                                     omega_pi_=omega_pi,
                                                     VDp_=VDp,
                                                     VDHe_=VDHe,
                                                     VDO_=VDO,
                                                     alpha_p_=alpha_p,
                                                     alpha_O_=alpha_O,
                                                     alpha_He_=alpha_He,
                                                     alpha_perp_c_=alpha_perp_c),
                                       x0=2.38 + 0.0005j, tol=1e-16, maxiter=1000,
                                       x1=2.5 + 1e-2j)
    print(sol_approx)
    print("omega_k + i gamma = ", sol_approx)
    print("dispersion residual approx = ", abs(disp_k_approx(k_perp=k_perp)(sol_approx)))
