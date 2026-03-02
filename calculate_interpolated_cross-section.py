import numpy as np

# Target WDM Mass
TARGET_WDM_KEV = 5.9

# Format: { 'IDM_Mass': (x_halfmode, y_halfmode, x_envelope, y_envelope) }
# x = m_WDM [keV], y = cross-section [cm^2]

data_n2 = {
    '1e-4 GeV': (7.48,  4.2e-28, 4.14, 2.8e-27),
    '1e-2 GeV': (10.40, 1.3e-25, 2.67, 7.1e-24),
    '1 GeV':    (6.64,  1.6e-23, 2.16, 8.0e-22)
}

data_n4 = {
    '1e-4 GeV': (7.61, 2.2e-27, 4.50, 3.4e-26),
    '1e-2 GeV': (9.36, 1.7e-22, 3.47, 1.7e-19),
    '1 GeV':    (9.66, 8.6e-19, 2.84, 2.8e-16)
}

# The Interpolation Function
def calculate_59kev_match(x1, y1, x2, y2, target_x=TARGET_WDM_KEV):
    """
    Performs a log-linear interpolation to find the y-value at target_x.
    Because the cross-sections span orders of magnitude, we must interpolate
    the log10 of the cross-sections, not the raw values.
    """
    # Step A: Convert y-values (cross-sections) to log10 space
    log_y1 = np.log10(y1)
    log_y2 = np.log10(y2)
    
    # Step B: Calculate the slope (m) and intercept (b) of the line: y = mx + b
    m = (log_y2 - log_y1) / (x2 - x1)
    b = log_y1 - m * x1
    
    # Step C: Find the log(y) value at the target x (5.9 keV)
    log_y_target = m * target_x + b
    
    # Step D: Convert back to linear space
    y_target = 10**log_y_target
    
    return y_target, m, b

# Execution and Output Formatting
def run_interpolation(data_dict, model_name):
    print(f"\n{'='*65}")
    print(f" INTERPOLATION RESULTS: {model_name}")
    print(f"{'='*65}")
    
    for mass_label, coords in data_dict.items():
        x_half, y_half, x_env, y_env = coords
        
        # Run the math
        y_target, slope, intercept = calculate_59kev_match(x_half, y_half, x_env, y_env)
        
        print(f"IDM Mass: {mass_label}")
        print(f"  [-] Half-mode Star : ({x_half:>5.2f} keV, {y_half:.2e} cm^2)")
        print(f"  [-] Envelope Star  : ({x_env:>5.2f} keV, {y_env:.2e} cm^2)")
        print(f"  [-] Line Equation  : log10(sigma) = {slope:.4f} * m_WDM + {intercept:.4f}")
        print(f"  [>] 5.9 keV Target : {y_target:.4e} cm^2\n")

if __name__ == "__main__":
    run_interpolation(data_n2, "n = 2 Model")
    run_interpolation(data_n4, "n = 4 Model")
