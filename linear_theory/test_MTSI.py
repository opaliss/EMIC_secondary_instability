import numpy as np
import matplotlib.pyplot as plt
import scipy
from QLT_equations.dispersion_functions import *
import matplotlib
matplotlib.use('TkAgg')

# normalization
# time is normalized to the electron cyclotron frequency
# space is normalized to electron inertial length d_e

# match the NASA proposal
# cold electron density
ne = 1  # ne
# hot proton density
npH = 0.1  # ne
# cold proton density
npC = 1 - npH # ne

# mass ratios
mp_me = 100 # dimensionless
omega_pe = 2  # Omega_ce

# assume the plasma is isothermal Te=Tp=TO+=THe+
alpha_c_perp = np.sqrt(1e-4)  # d_e x Omega_ce
alpha_c_par = alpha_c_perp  # d_e x Omega_ce
alpha_p_par = alpha_c_perp / np.sqrt(mp_me)  # d_e x Omega_ce

# relative drift
UDp = -1.5 * np.sqrt(1e-4)


def disp_k_(k_,
            theta_,
            npC_=npC,
            omega_pe_=omega_pe,
            mp_me_=mp_me,
            VDp_=UDp,
            alpha_c_par_=alpha_c_par,
            alpha_p_par_=alpha_p_par,
            n_max_=10,
            electron_response_="hot"):
    if "hot" == electron_response_:
        return lambda omega: 1 + electron_response(k_=k_, omega_=omega, omega_pe_=omega_pe_, theta_=theta_,
                                                   alpha_c_par_=alpha_c_par_, n_max_=n_max_) \
                             - ion_response(omega_pi_=omega_pe_/np.sqrt(mp_me_), alpha_i_=alpha_p_par_,
                                            k_=k_, theta_=theta_, v_0_=VDp_, n_=npC_, omega=omega)
    elif "cold" == electron_response_:
        return lambda omega: 1 + electron_response_cold(omega=omega, omega_pe_=omega_pe_, theta_=theta_) \
                             - ion_response(omega_pi_=omega_pe_/np.sqrt(mp_me_), alpha_i_=alpha_p_par_,
                                            k_=k_, theta_=theta_, v_0_=VDp_, n_=npC_, omega=omega)


# quick check
k_test = np.abs(1 *  np.sqrt(1/mp_me) * np.sqrt(npC) / UDp )  # d_e
angle = 1 # deg
theta_test = -np.pi/2 + angle * np.pi/180

omega_guess = 1/(2**(3/4)) * ((mp_me * np.sqrt(npC)*(np.cos(theta_test)**2))**(1/3)) / np.sqrt(mp_me) # Omega_ce

sol_approx_hot = scipy.optimize.newton(disp_k_(k_=k_test, theta_=theta_test, electron_response_="hot"),
                                   omega_guess + 0.005j, tol=1e-16,
                                   maxiter=10000,
                                   x1=omega_guess * 0.99 + 1e-2j)

sol_approx_cold = scipy.optimize.newton(disp_k_(k_=k_test, theta_=theta_test, electron_response_="cold"),
                                   omega_guess + 0.005j, tol=1e-16,
                                   maxiter=10000,
                                   x1=omega_guess * 0.99 + 1e-2j)

print("k_ rho_{e}", k_test*alpha_c_perp/np.sqrt(2))
print("[hot] omega_k + i gamma = ", sol_approx_hot * np.sqrt(mp_me))
print("[hot] dispersion residual approx = ", abs(disp_k_(k_=k_test, theta_=theta_test, electron_response_="hot")(sol_approx_hot)) )

print("\n[cold] omega_k + i gamma = ", sol_approx_cold * np.sqrt(mp_me))
print("[cold] dispersion residual approx = ", abs(disp_k_(k_=k_test, theta_=theta_test, electron_response_="cold")(sol_approx_cold)) )