import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from surrogate_lib import AcquisitionStrategies, SafeOptimizer

def true_function(x):
    """Hàm mục tiêu giả định: x*sin(x)"""
    return x * np.sin(x)

def run_strategy_comparison():
    print("So sánh các chiến lược khám phá (Section 16.1 - 16.5)")

    X_domain = np.linspace(0, 10, 500).reshape(-1, 1)
    y_true = true_function(X_domain).ravel()
    
    X_train = np.array([[1.0], [5.0], [9.0]]) 
    y_train = true_function(X_train).ravel()
    
    kernel = C(1.0, (0.1, 100.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.5, 20.0))

    gp = GaussianProcessRegressor(
      kernel=kernel, 
      n_restarts_optimizer=20, 
      alpha=1e-5,       
      normalize_y=True, 
      random_state=42
    )
    
    gp.fit(X_train, y_train)
    
    print(f"Learned Kernel: {gp.kernel_}")
    
    mu, sigma = gp.predict(X_domain, return_std=True)
    y_min = np.min(y_train)
    strategies = {
        "Prediction-Based": AcquisitionStrategies.prediction_based(mu, sigma),
        "Error-Based": AcquisitionStrategies.error_based(mu, sigma),
        "LCB (alpha=2)": AcquisitionStrategies.lower_confidence_bound(mu, sigma, alpha=2.0),
        "Prob. of Imp. (PI)": AcquisitionStrategies.probability_of_improvement(mu, sigma, y_min),
        "Exp. Imp. (EI)": AcquisitionStrategies.expected_improvement(mu, sigma, y_min)
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    axes[0].plot(X_domain, y_true, 'k--', label='True f(x)')
    axes[0].plot(X_train, y_train, 'ro', markersize=8, label='Data', zorder=5)
    axes[0].plot(X_domain, mu, 'b-', label='GP Mean')
    axes[0].fill_between(X_domain.ravel(), mu - 1.96*sigma, mu + 1.96*sigma, alpha=0.2, color='blue', label='95% Conf')
    axes[0].set_title("Gaussian Process Model (Corrected)")
    axes[0].legend()
    axes[0].set_ylim(-8, 8) 

    for i, (name, values) in enumerate(strategies.items(), 1):
        ax = axes[i]
        next_idx = np.argmin(values)
        
        ax.plot(X_domain, values, 'g-', label='Acquisition Val')
        ax.axvline(X_domain[next_idx], color='m', linestyle='--', linewidth=2, label='Next Point')
        ax.plot(X_domain[next_idx], values[next_idx], 'mo') 
        
        ax.set_title(name)
        ax.legend()
        
    plt.tight_layout()
    plt.show()

def run_safeopt_demo():
    print('Demo SafeOpt')
    X_grid = np.linspace(0, 10, 100).reshape(-1, 1) 
    y_max = 2.5 
    start_vals = [2.0, 2.2]
    
    X_sample = []
    y_sample = []
    visited_indices = []
    
    for val in start_vals:
        idx = np.abs(X_grid - val).argmin()
        if idx not in visited_indices: 
            X_sample.append(X_grid[idx])
            y_sample.append(true_function(X_grid[idx])[0])
            visited_indices.append(idx)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=0.5, length_scale_bounds=(0.1, 2.0))
    
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        n_restarts_optimizer=10, 
        alpha=1e-10,       
        normalize_y=True,  
        random_state=42
    )
    gp.fit(X_sample, y_sample)
    opt = SafeOptimizer(gp, X_grid, y_max, beta=3.0)
    
    iterations = 8
    fig, axes = plt.subplots(iterations, 1, figsize=(10, 3*iterations), sharex=True)
    if iterations == 1: axes = [axes]

    for k in range(iterations):
        ax = axes[k]
        
        opt.update_confidence_intervals()
        opt.compute_sets()
        next_idx = opt.get_new_query_point(visited_indices=visited_indices)
        
        mu, sigma = gp.predict(X_grid, return_std=True)
        
        ax.plot(X_grid, true_function(X_grid), 'k--', alpha=0.4, label='True f(x)')
        ax.axhline(y_max, color='r', linestyle=':', linewidth=1.5, label='Threshold')
        
        safe_indices = np.where(opt.S)[0]
        if len(safe_indices) > 0:
            ax.fill_between(X_grid[safe_indices, 0], -15, 15, color='green', alpha=0.1)
            ax.scatter(X_grid[safe_indices], opt.u[safe_indices], c='green', s=10, alpha=0.6, label='Safe Upper Bound')

        ax.plot(np.array(X_sample), np.array(y_sample), 'ro', markersize=6, label='Observed')

        if next_idx is not None:
            x_next_val = X_grid[next_idx].item()
            curr_x = X_sample[-1].item()
            curr_y = y_sample[-1].item()
            next_u_val = opt.u[next_idx].item()
            
            ax.plot(x_next_val, next_u_val, 'm*', markersize=14, label='Next Query')
            ax.arrow(curr_x, curr_y, 
                     x_next_val - curr_x, next_u_val - curr_y, 
                     head_width=0.15, color='m', alpha=0.3, length_includes_head=True)

            y_next_real = true_function(np.array([x_next_val]))[0]
            
            X_sample.append(X_grid[next_idx])
            y_sample.append(y_next_real)
            visited_indices.append(next_idx)
            
            gp.fit(X_sample, y_sample)
        else:
            ax.text(5, 0, "No safe candidates / Converged", ha='center', color='red')

        ax.set_ylabel(f"Iter {k+1}")
        ax.set_ylim(-6, 8)
        
        if k == 0:
            ax.legend(loc='upper right', ncol=2, fontsize='small')
            ax.set_title(f"SafeOpt: 2-Point Initialization Strategy")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_strategy_comparison()
    run_safeopt_demo()