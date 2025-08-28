import sys, os

sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
import matplotlib.pyplot as plt
import scipy
from QLT_equations.dispersion_functions import *
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # normalization
    # time is normalized to the electron cyclotron frequency
    # space is normalized to electron inertial length d_e

    # match the values in khazanov et al. The nonlinear coupling of electromagnetic ion cyclotron and lower hybrid

    # cold electron density
    ne = 1  # ne
    # cold He+ density
    nHe = 0.  # ne
    # cold O+ density
    nO = 0.1  # ne
    # hot proton density
    npH = 0.1  # ne
    # cold proton density
    npC = 1 - npH - nO - nHe  # ne

    # mass ratios
    mp_me = 1836  # dimensionless
    mO_mp = 10  # dimensionless
    mHe_mp = 4  # dimensionless

    omega_pe = 2  # Omega_ce

    # assume the plasma is isothermal Te=Tp=TO+=THe+
    alpha_c_perp = np.sqrt(3e-4)  # d_e x Omega_ce
    alpha_c_par = alpha_c_perp  # d_e x Omega_ce
    alpha_p_par = alpha_c_perp / np.sqrt(mp_me)  # d_e x Omega_ce
    alpha_pH_par = np.sqrt(1e4) * alpha_c_perp / np.sqrt(mp_me)  # d_e x Omega_ce
    alpha_He_par = alpha_c_perp / np.sqrt(mp_me * mHe_mp)  # de x Omega_ce
    alpha_O_par = alpha_c_perp / np.sqrt(mp_me * mO_mp)  # de x Omega_ce

    wLH = 1 / np.sqrt(mp_me)  # Omega_ce
    rhoe = alpha_c_perp  # /np.sqrt(2) # de

    UDp = 1.63299 * alpha_p_par / np.sqrt(2)
    UDpH = 0.0163299 * alpha_pH_par / np.sqrt(2)
    UDO = -8.77876 * alpha_O_par / np.sqrt(2)
    UDHe = 0 * alpha_He_par / np.sqrt(2)


    def disp_k_(k,
                theta,
                npC_=npC,
                nO_=nO,
                nHe_=nHe,
                omega_pe_=omega_pe,
                mp_me_=mp_me,
                mO_mp_=mO_mp,
                mHe_mp_=mHe_mp,
                VDp_=UDp,
                VDHe_=UDHe,
                VDO_=UDO,
                alpha_c_par_=alpha_c_par,
                alpha_p_par_=alpha_p_par,
                alpha_O_par_=alpha_O_par,
                alpha_He_par_=alpha_He_par,
                approx="hot"):
        if approx == "hot":
            return lambda omega: 1 + electron_response(k_=k, omega=omega, omega_pe_=omega_pe_, theta_=theta,
                                                       alpha_c_par_=alpha_c_par_) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), n_=npC_, alpha_i_=alpha_p_par_,
                                                v_0_=VDp_, omega=omega, k_=k, theta_=theta) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mO_mp_), n_=nO_,
                                                alpha_i_=alpha_O_par_, k_=k, v_0_=VDO_, omega=omega, theta_=theta) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mHe_mp_), n_=nHe_,
                                                alpha_i_=alpha_He_par_, k_=k, v_0_=VDHe_, omega=omega, theta_=theta) \
                                - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mHe_mp_), n_=nHe_,
                                               alpha_i_=alpha_He_par_, k_=k, v_0_=VDHe_, omega=omega, theta_=theta)
        elif approx == "cold":
            return lambda omega: 1 + electron_response_cold(omega=omega, omega_pe_=omega_pe_, theta_=theta) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), n_=npC_, alpha_i_=alpha_p_par_,
                                                v_0_=VDp_, omega=omega, k_=k, theta_=theta) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mO_mp_), n_=nO_,
                                                alpha_i_=alpha_O_par_, k_=k, v_0_=VDO_, omega=omega, theta_=theta) \
                                 - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mHe_mp_), n_=nHe_,
                                                alpha_i_=alpha_He_par_, k_=k, v_0_=VDHe_, omega=omega, theta_=theta) \
                                - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mHe_mp_), n_=nHe_,
                                               alpha_i_=alpha_He_par_, k_=k, v_0_=VDHe_, omega=omega, theta_=theta)

    k_vec = np.linspace(0.28, 0.5, 100) / rhoe  # de
    angle = 1
    theta_test = -np.pi / 2 + angle * np.pi / 180
    omega_guess = np.sqrt(npC) * (1 - 1 / (2 ** (4 / 3)) * ((nO / npC / mO_mp) ** (1 / 3))) * wLH  # Omega_ce
    print("omega_guess = ", omega_guess / wLH)

    sol_approx_hot = np.zeros(len(k_vec), dtype="complex128")
    sol_approx_cold = np.zeros(len(k_vec), dtype="complex128")

    for ii in range(len(k_vec)):
        try:
            sol_approx_hot[ii] = scipy.optimize.newton(disp_k_(k=k_vec[ii], theta=theta_test, approx="hot"),
                                                       omega_guess + 0.005j, tol=1e-15,
                                                       maxiter=10000, x1=omega_guess * 1.1 + 1e-2j)

            if abs(disp_k_(k=k_vec[ii], theta=theta_test)(sol_approx_hot[ii])) > 1e-10:
                sol_approx_hot[ii] = np.nan

        except:
            print("hot failed to converge")
        try:
            sol_approx_cold[ii] = scipy.optimize.newton(disp_k_(k=k_vec[ii], theta=theta_test, approx="cold"),
                                                        omega_guess + 0.005j, tol=1e-15,
                                                        maxiter=10000, x1=omega_guess * 1.1 + 1e-2j)
        except:
            print("cold failed to converge")


    fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
    ax[0].plot(k_vec * rhoe, sol_approx_cold.imag / wLH, color="blue")
    ax[0].plot(k_vec * rhoe, sol_approx_hot.imag / wLH, color="red")
    ax[0].set_title(r"$\gamma/\omega_{LH}$")
    ax[0].set_xlabel(r"$|\vec{k}|\rho_{e}$")

    ax[1].plot(k_vec * rhoe, sol_approx_cold.real / wLH, color="blue")
    ax[1].plot(k_vec * rhoe, sol_approx_hot.real / wLH, color="red")
    ax[1].plot(k_vec * rhoe, (k_vec * UDO * np.sin(theta_test)) / wLH, linestyle='--', label=r'${k_\perp}U_{0}$')
    ax[1].plot(k_vec * rhoe, (k_vec * UDHe * np.sin(theta_test)) / wLH, linestyle='--', label=r'${k_\perp}U_{He}$')
    ax[1].plot(k_vec * rhoe, (k_vec * UDp * np.sin(theta_test)) / wLH, linestyle='--', label=r'${k_\perp}U_{p}$')
    ax[1].set_title(r"$\omega_{r}/\omega_{LH}$")
    ax[1].set_xlabel(r"$|\vec{k}|\rho_{e}$")

    ax[2].plot(k_vec * rhoe, abs(disp_k_(k=k_vec, theta=theta_test)(sol_approx_cold)), color="blue")
    ax[2].plot(k_vec * rhoe, abs(disp_k_(k=k_vec, theta=theta_test)(sol_approx_hot)), color="red")
    ax[2].set_title(r"residual $D(k, \omega^{*})$")
    ax[2].set_xlabel(r"$|\vec{k}|\rho_{e}$")
    ax[2].set_yscale("log")

    plt.show()
