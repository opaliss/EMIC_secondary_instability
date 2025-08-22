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
    npH = 0.2  # ne
    # cold proton density
    npC = 1 - npH - nO - nHe  # ne

    # mass ratios
    mp_me = 100  # dimensionless
    mO_mp = 16  # dimensionless
    # mHe_mp = 4  # dimensionless

    omega_pe = 4  # Omega_ce

    # assume the plasma is isothermal Te=Tp=TO+=THe+
    alpha_c_perp = np.sqrt(1e-4)  # d_e x Omega_ce
    alpha_p_par = alpha_c_perp / np.sqrt(mp_me)  # d_e x Omega_ce
    # alpha_He_par = alpha_c_perp / np.sqrt(mp_me * mHe_mp)  # de x Omega_ce
    alpha_O_par = alpha_c_perp / np.sqrt(mp_me * mO_mp)  # de x Omega_ce

    # relative drift is what matters
    UDp = -0.8 * alpha_p_par
    UDO = 1.4 * alpha_p_par
    # UDHe = 0.8 * alpha_p_par

    def disp_k_(k_perp,
                k_par,
                npC_=npC,
                nO_=nO,
                nHe_=nHe,
                omega_pe_=omega_pe,
                mp_me_=mp_me,
                mO_mp_=mO_mp,
                #mHe_mp_=mHe_mp,
                VDp_=UDp,
                # VDHe_=UDHe,
                VDO_=UDO,
                alpha_p_par_=alpha_p_par,
                alpha_O_par_=alpha_O_par):
        return lambda omega: 1 + cold_electron_response(k_perp_=k_perp, omega=omega, omega_pe_=omega_pe_, k_par_=k_par) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_), n_=npC_, alpha_i_=alpha_p_par_,
                                            k_perp_=k_perp, v_0_=VDp_, omega=omega) \
                             - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mO_mp_), n_=nO_,
                                            alpha_i_=alpha_O_par_, k_perp_=k_perp, v_0_=VDO_, omega=omega) \
                             # - ion_response(omega_pi_=omega_pe_ / np.sqrt(mp_me_ * mHe_mp_), n_=nHe_,
                             #                alpha_i_=alpha_He_par_, k_perp_=k_perp, v_0_=VDHe_, omega=omega)


    k_perp = 1 / alpha_c_perp # d_e
    k_par = k_perp * 0.1
    omega_guess = 0.2 / np.sqrt(mp_me)  # Omega_ce


    sol_approx = scipy.optimize.newton(disp_k_(k_perp=k_perp, k_par=k_par), omega_guess + 0.005j, tol=1e-16,
                                       maxiter=10000, x1=omega_guess*0.99 + 1e-2j)
    print(sol_approx)
    print("omega_k + i gamma = ", sol_approx)
    print("dispersion residual approx = ", abs(disp_k_(k_perp=k_perp, k_par=k_par)(sol_approx)))