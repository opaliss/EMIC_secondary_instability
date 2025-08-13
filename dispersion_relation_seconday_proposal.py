import numpy as np
import scipy
from QLT_equations.general_plasma_equations import Z_prime


def cold_electron_response(k_perp_, omega, omega_pe_, k_par_):
    return omega_pe_**2 - (k_par_**2)/(k_par_**2 + k_perp_**2) * (omega_pe_**2)/(omega**2)


def ion_response(omega_pi_, alpha_i_, k_perp_, v_0_, n_, omega):
    return n_ * (omega_pi_ ** 2) / (alpha_i_ ** 2) / (k_perp_**2) * Z_prime(
        z=(omega - k_perp_ * v_0_) / (alpha_i_ * k_perp_))


if __name__ == "__main__":
    # normalization
    # time is normalized to the electron cyclotron frequency
    # space is normalized to electron inertial length d_e

    # match the NASA proposal

    # cold electron density
    ne = 1  # ne
    # hot proton density
    npH = 0.2  # ne
    # cold proton density
    npC = 1 - npH # ne

    # mass ratios
    mp_me = 100  # dimensionless
    omega_pe = 4  # Omega_ce

    # assume the plasma is isothermal Te=Tp=TO+=THe+
    alpha_c_perp = np.sqrt(1e-4)  # d_e x Omega_ce
    alpha_c_par = alpha_c_perp  # d_e x Omega_ce
    alpha_p_par = alpha_c_perp / np.sqrt(mp_me)  # d_e x Omega_ce

    # relative drift
    UDp = 2.5 * alpha_p_par


    def disp_k_(k_perp,
                k_par,
                npC_=npC,
                omega_pe_=omega_pe,
                mp_me_=mp_me,
                VDp_=UDp,
                alpha_p_par_=alpha_p_par):
        return lambda omega: 1 + cold_electron_response(k_perp_=k_perp, k_par_=k_par, omega=omega, omega_pe_=omega_pe_) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), n_=npC_,
                                            alpha_i_=alpha_p_par_,
                                            k_perp_=k_perp, v_0_=VDp_, omega=omega)


    k_perp = 1 / alpha_c_perp   # d_e
    k_par = k_perp * 1e-1
    omega_guess = 0.2 / np.sqrt(mp_me) # Omega_ce

    sol_approx = scipy.optimize.newton(disp_k_(k_perp=k_perp, k_par=k_par), omega_guess + 0.05j, tol=1e-17,
                                       maxiter=10000,
                                       x1=omega_guess * 0.99 + 1e-2j)
    print(sol_approx)
    print("omega_k + i gamma = ", sol_approx * np.sqrt(mp_me))
    print("dispersion residual approx = ", abs(disp_k_(k_perp=k_perp, k_par=k_par)(sol_approx)) )
