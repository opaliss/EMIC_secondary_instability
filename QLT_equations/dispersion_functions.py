"""Module with dispersion relation functions

Last modified: Aug 23, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""

import numpy as np
from QLT_equations.general_plasma_equations import Z, Z_prime, I
import scipy.optimize as opt
from queue import PriorityQueue
import math


def ion_response(omega_pi_, alpha_i_, k_, theta_, v_0_, n_, omega):
    k_perp = np.sin(theta_) * np.abs(k_)
    k2 = (k_ ** 2)
    xi = (omega - k_perp * v_0_) / (alpha_i_ * np.abs(k_))
    coeff = n_ * (omega_pi_ ** 2) / (alpha_i_ ** 2) / k2
    return coeff * Z_prime(z=xi)


def sum_bessel(lambda_, omega_, k_par_, alpha_c_par_, n_max_, omega_ce_):
    res = 0
    for n in range(-n_max_, n_max_ + 1):
        xi_n = (omega_ + n * omega_ce_) / np.abs(k_par_) / alpha_c_par_
        res += I(m=n, Lambda=lambda_) * Z(z=xi_n)
    return res


def electron_response(k_, omega_, omega_pe_, theta_, alpha_c_par_, omega_ce_, n_max_=20):
    k_perp = np.sin(theta_) * np.abs(k_)
    k_par = np.cos(theta_) * np.abs(k_)
    k2 = k_ ** 2
    xi_0 = omega_ / np.abs(k_par) / alpha_c_par_
    lambda_ = 0.5 * (k_perp ** 2) * (alpha_c_par_ ** 2) / (omega_ce_ ** 2)
    return 2 * (omega_pe_ ** 2) / (alpha_c_par_ ** 2) / k2 * (
            1 + xi_0 * sum_bessel(lambda_=lambda_, omega_ce_=omega_ce_, omega_=omega_, k_par_=k_par,
                                  alpha_c_par_=alpha_c_par_, n_max_=n_max_))


def electron_response_cold(omega, omega_pe_, theta_, omega_ce_):
    term1 = - (np.cos(theta_) ** 2) * (omega_pe_ ** 2) / (omega ** 2)
    term2 = ((omega_pe_ / omega_ce_) ** 2) * (np.sin(theta_) ** 2)
    return term1 + term2


def ion_response_cold(omega_pi_, k_, v_0_, theta_, n_, omega):
    k_perp = np.sin(theta_) * np.abs(k_)
    return n_ * (omega_pi_ ** 2) / ((omega - k_perp * v_0_) ** 2)


def dispersion_relation(k_,
                        theta_,
                        n_p_,
                        n_O_,
                        n_pH_,
                        omega_pe_,
                        omega_ce_,
                        mp_me_,
                        mO_me_,
                        UD_p_,
                        UD_pH_,
                        UD_O_,
                        alpha_e_par_,
                        alpha_p_par_,
                        alpha_pH_par_,
                        alpha_O_par_,
                        n_max_=10,
                        electron_response_="hot"):
    if electron_response_ == "hot":
        return lambda omega: 1 + electron_response(k_=k_, omega_=omega, omega_ce_=omega_ce_, omega_pe_=omega_pe_,
                                                   theta_=theta_, alpha_c_par_=alpha_e_par_, n_max_=n_max_) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), alpha_i_=alpha_p_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_p_, n_=n_p_, omega=omega) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), alpha_i_=alpha_pH_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_pH_, n_=n_pH_, omega=omega) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mO_me_), alpha_i_=alpha_O_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_O_, n_=n_O_, omega=omega)
    elif electron_response_ == "cold":
        return lambda omega: 1 + electron_response_cold(omega=omega, omega_ce_=omega_ce_, omega_pe_=omega_pe_,
                                                        theta_=theta_) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), alpha_i_=alpha_p_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_p_, n_=n_p_, omega=omega) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), alpha_i_=alpha_pH_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_pH_, n_=n_pH_, omega=omega) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mO_me_), alpha_i_=alpha_O_par_,
                                            k_=k_, theta_=theta_, v_0_=UD_O_, n_=n_O_, omega=omega)


def solve_dispersion(func, x0, x1, maxiter=1000, tol_newton=1e-16, tol_root=1e-12):
    """
    Solve dispersion relation f(omega) = 0 for complex omega.
    Tries Newton first, falls back to root(hybr) if needed.
    """
    try:
        # --- Fall back to robust solver ---
        def f_wrapped(omega_vec):
            omega = omega_vec[0] + 1j * omega_vec[1]
            val = func(omega)
            return [val.real, val.imag]

        guess_vec = [x0.real, x0.imag]
        root_root = opt.root(f_wrapped, x0=guess_vec, method="hybr", tol=tol_root)
        if root_root.success:
            return root_root.x[0] + 1j * root_root.x[1]
        else:
            print("root did not work!")
            try:
                root_newton = opt.newton(func, x0=x0, x1=x1, tol=tol_newton, maxiter=maxiter)
                return root_newton
            except:
                print("newton did not work!")
    except:
        try:
            root_newton = opt.newton(func, x0=x0, x1=x1, tol=tol_newton, maxiter=maxiter)
            return root_newton
        except:
            print("newton did not work!")


def solve_dispersion_on_grid(k_perp_vec, k_par_vec, k_par_0, k_perp_0, omega_0, gamma_0,
                             n_p_, n_O_, n_pH_, omega_pe_, omega_ce_, mp_me_, mO_me_,
                             UD_p_, UD_pH_, UD_O_, alpha_e_par_, alpha_p_par_, alpha_pH_par_,
                             alpha_O_par_, n_max_=10, electron_response_="hot", cutoff=0.2):
    omegaA = np.zeros((len(k_perp_vec), len(k_par_vec)))
    gammaA = np.zeros((len(k_perp_vec), len(k_par_vec)))

    i0 = np.searchsorted(k_perp_vec, k_perp_0)
    j0 = np.searchsorted(k_par_vec, k_par_0)

    q = PriorityQueue()
    q.put((gamma_0, omega_0, i0, j0))

    while not q.empty():
        gamma, omega, ii, jj = q.get()

        if omegaA[ii, jj] == 0.:
            k_perp = k_perp_vec[ii]
            k_par = k_par_vec[jj]

            try:
                w = solve_dispersion(func=dispersion_relation(k_=np.sqrt(k_par ** 2 + k_perp ** 2),
                                                              theta_=math.atan2(k_perp, k_par),
                                                              n_p_=n_p_,
                                                              n_O_=n_O_,
                                                              n_pH_=n_pH_,
                                                              omega_pe_=omega_pe_,
                                                              omega_ce_=omega_ce_,
                                                              mp_me_=mp_me_,
                                                              mO_me_=mO_me_,
                                                              UD_p_=UD_p_,
                                                              UD_pH_=UD_pH_,
                                                              UD_O_=UD_O_,
                                                              alpha_e_par_=alpha_e_par_,
                                                              alpha_p_par_=alpha_p_par_,
                                                              alpha_pH_par_=alpha_pH_par_,
                                                              alpha_O_par_=alpha_O_par_,
                                                              n_max_=n_max_,
                                                              electron_response_=electron_response_),
                                     x1=omega * 1.5 + 1j * gamma,
                                     x0=omega + 1j * gamma)

                omegaA[ii, jj] = w.real
                gammaA[ii, jj] = w.imag

                if 0. < np.abs(w.real) < 4. / np.sqrt(mp_me_) and 0. < w.imag < 2. * gamma:
                    if w.imag > cutoff * np.amax(gammaA):
                        if ii > 0 and jj > 0 and omegaA[ii - 1, jj - 1] == 0.:
                            q.put((w.imag, w.real, ii - 1, jj - 1))
                        if ii > 0 and omegaA[ii - 1, jj] == 0.:
                            q.put((w.imag, w.real, ii - 1, jj))
                        if ii > 0 and jj < len(k_par_vec) - 1 and omegaA[ii - 1, jj + 1] == 0.:
                            q.put((w.imag, w.real, ii - 1, jj + 1))
                        if jj > 0 and jj > 0 and omegaA[ii, jj - 1] == 0.:
                            q.put((w.imag, w.real, ii, jj - 1))

                        if jj < len(k_par_vec) - 1 and omegaA[ii, jj + 1] == 0.:
                            q.put((w.imag, w.real, ii, jj + 1))
                        if ii < len(k_perp_vec) - 1 and jj > 0 and omegaA[ii + 1, jj - 1] == 0.:
                            q.put((w.imag, w.real, ii + 1, jj - 1))
                        if ii < len(k_perp_vec) - 1 and omegaA[ii + 1, jj] == 0.:
                            q.put((w.imag, w.real, ii + 1, jj))
                        if ii < len(k_perp_vec) - 1 and jj < len(k_par_vec) - 1 and omegaA[ii + 1, jj + 1] == 0.:
                            q.put((w.imag, w.real, ii + 1, jj + 1))
            except:
                print("kpar", k_par)
                print("kper", k_perp)

    return omegaA + 1j * gammaA
