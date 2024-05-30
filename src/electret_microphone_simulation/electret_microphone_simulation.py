import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def h(t, nu):
    """Time-dependent distance between plates."""
    omega = 2 * np.pi * nu
    return h_0 + h_1 * np.cos(omega * t)

def h_dot(t, nu):
    """Derivative of h(t) with respect to time."""
    omega = 2 * np.pi * nu
    return -h_1 * omega * np.sin(omega * t)

def differential_equation(t, u, nu):
    """Differential equation of the electric potential for the model of the electret microphone."""
    h_t = h(t, nu)
    h_t_dot = h_dot(t, nu)
    term1 = -(h_t**2 - R * epsilon_0 * S * h_t_dot) / (epsilon_0 * S * R * h_t)
    term2 = (P_0 * d * h_t_dot ) / (epsilon_0 * h_t)
    return term1 * u + term2

def plot_potential(nu, periods, output_file):
    """Solve the differential equation and plot u(t) for a given frequency."""
    period = 1 / nu
    t_span = (0, periods * period)
    u0 = [0]  # Initial condition

    sol = solve_ivp(differential_equation, t_span, u0, args=(nu,), max_step=period/20, method='RK45')

    plt.figure(figsize=(12, 6))
    plt.plot(sol.t, sol.y[0], label=f'Frequency = {nu} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Potential u(t) (V)')
    plt.title(f'Potential u(t) over time for frequency {nu} Hz')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)

def plot_frequency_response():
    """Plot the frequency response of the microphone."""
    frequencies = np.geomspace(50, 20000, num=50)
    amplitudes = []
    for freq in frequencies:
        sol = solve_ivp(differential_equation, (0, 10 / freq), [0], args=(freq,), max_step=1/(20*freq), method='RK45')
        amplitudes.append(np.max(sol.y[0]))

    plt.figure(figsize=(12, 6))
    plt.semilogx(frequencies, amplitudes, marker='o')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude |U(ν)| (V)')
    plt.title('Frequency Response of the Electret Microphone')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'frequency_response.png'))

def main():
    # Constants
    global epsilon_0, P_0, h_0, h_1, d, R, S
    epsilon_0 = 8.854187817e-12 # Vacuum permittivity in F/m (farads per meter)
    P_0 = 10**-4                # Polarization in C/m^2
    h_0 = 1e-3                  # Initial distance in meters
    h_1 = 0.1e-3                # Amplitude of oscillation in meters
    d = 20e-6                   # Thickness of the electret in meters
    R = 5e9                     # Resistance in ohms
    S = 100e-6                  # Area in m^2 (100 mm^2)
    frequencies = [1000, 5000]  # Frequencies in Hz

    # Plot potential as function of time for input frequencies of 1000 Hz (25 periods) and 5000 Hz (80 periods)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    for frequency in frequencies:
        output_file = os.path.join(script_dir, f'potential_at_{frequency}Hz.png')
        periods = 25 if frequency == 1000 else 80
        plot_potential(frequency, periods, output_file)

    # Plot the frequency response
    plot_frequency_response()

if __name__ == "__main__":
    main()
