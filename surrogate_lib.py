# FILE: surrogate_lib.py
import numpy as np
from scipy.stats import norm

class AcquisitionStrategies:
    """
    Cài đặt các chiến lược khám phá (Exploration Strategies) 
    dựa trên Chapter 16.1 - 16.5.
    """

    @staticmethod
    def prediction_based(mu, sigma):
        """Section 16.1: Minimizer of the surrogate mean."""
        return mu

    @staticmethod
    def error_based(mu, sigma):
        """Section 16.2: Maximize uncertainty (sigma)."""
        return -sigma

    @staticmethod
    def lower_confidence_bound(mu, sigma, alpha=2.0):
        """Section 16.3: Lower Confidence Bound (LCB)."""
        return mu - alpha * sigma

    @staticmethod
    def probability_of_improvement(mu, sigma, y_min):
        """Section 16.4: Probability of Improvement (PI)."""
        with np.errstate(divide='warn'):
            z = (y_min - mu) / sigma
            pi = norm.cdf(z)
            pi[sigma == 0.0] = 0.0
        return -pi

    @staticmethod
    def expected_improvement(mu, sigma, y_min):
        """Section 16.5: Expected Improvement (EI)."""
        with np.errstate(divide='warn'):
            z = (y_min - mu) / sigma
            ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma == 0.0] = 0.0
        return -ei


class SafeOptimizer:
    """
    Cài đặt thuật toán SafeOpt dựa trên Section 16.6 
    (Algorithms 16.3, 16.4, 16.5, 16.6).
    """
    def __init__(self, gp, X_domain, y_max, beta=3.0):
        self.gp = gp
        self.X = X_domain # Không gian thiết kế rời rạc
        self.y_max = y_max
        self.beta = beta
        self.m = len(X_domain)
        
        # Khởi tạo các vector biên và tập hợp
        self.u = np.full(self.m, np.inf)
        self.l = np.full(self.m, -np.inf)
        self.S = np.zeros(self.m, dtype=bool) # Safe set
        self.M = np.zeros(self.m, dtype=bool) # Potential minimizers
        self.E = np.zeros(self.m, dtype=bool) # Potential expanders

    def update_confidence_intervals(self):
        """Algorithm 16.4"""
        mu, sigma = self.gp.predict(self.X, return_std=True)
        beta_scale = np.sqrt(self.beta)
        self.u = mu + beta_scale * sigma
        self.l = mu - beta_scale * sigma
        return self.u, self.l

    def compute_sets(self):
        """Algorithm 16.5"""
        # 1. Update Safe set (S)
        self.S = self.u <= self.y_max
        
        if not np.any(self.S):
            self.M[:] = False
            self.E[:] = False
            return

        # 2. Update Potential Minimizers (M)
        min_u_in_S = np.min(self.u[self.S])
        self.M[:] = False
        self.M[self.S] = self.l[self.S] <= min_u_in_S

        # 3. Calculate max width in M for thresholding
        w = self.u - self.l
        w_max = np.max(w[self.M]) if np.any(self.M) else 0

        # 4. Update Expanders (E)
        self.E[:] = False
        potential_expanders = self.S & (~self.M)
        
        if np.any(potential_expanders):
            candidates = np.where(potential_expanders)[0]
            for idx in candidates:
                if w[idx] > w_max:
                    self.E[idx] = True

    def get_new_query_point(self, visited_indices=None):
        """
        Algorithm 16.6: Chọn điểm tiếp theo.
        Có bổ sung tham số visited_indices để tránh chọn lại điểm cũ.
        """
        # Hợp của tập Minimizers (M) và Expanders (E)
        candidates = self.M | self.E
        
        # Nếu có danh sách điểm đã thăm, loại bỏ chúng khỏi candidates
        if visited_indices is not None:
            candidates = candidates.copy() 
            for idx in visited_indices:
                if 0 <= idx < self.m:
                    candidates[idx] = False

        if not np.any(candidates):
            return None

        # Chọn điểm có độ bất định (width = u - l) lớn nhất
        candidate_indices = np.where(candidates)[0]
        w = self.u - self.l
        
        # Tìm index trong candidate_indices có w lớn nhất
        best_local_idx = np.argmax(w[candidate_indices])
        best_global_idx = candidate_indices[best_local_idx]
        
        return best_global_idx