#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

// ---- Scalar field ODE ----
// phi'' = -((d-1)/r) * phi' + phi(1 - 3αφ + 2φ²)
static inline double f_phi(double r, double phi, double phi_prime, int d, double s) {

    double alpha = std::cosh(2.0 * s) / std::sinh(2.0 * s);

    if (r < 1e-12) {
        return (phi * (1.0 - 3.0 * alpha * phi + 2.0 * phi * phi)) / d;
    }

    return phi * (1.0 - 3.0 * alpha * phi + 2.0 * phi * phi)
        - ((d - 1.0) / r) * phi_prime;
}


// High-precision RK4 with careful origin handling
static std::vector<double> solve_ode_high_precision(int d, double s, double phi0, double rmax, double h) {

    std::size_t size = static_cast<std::size_t>(rmax / h) + 1;

    std::vector<double> phi(size);
    std::vector<double> phi_prime(size);

    phi[0] = phi0;
    phi_prime[0] = 0.0;

    double alpha = std::cosh(2.0 * s) / std::sinh(2.0 * s);

    if (size > 1) {
        double phi_prime_prime_0 =
            (phi0 * (1.0 - 3.0 * alpha * phi0 + 2.0 * phi0 * phi0)) / d;

        phi[1] = phi0 + 0.5 * h * h * phi_prime_prime_0;
        phi_prime[1] = h * phi_prime_prime_0;
    }

    for (std::size_t i = 1; i + 1 < size; ++i) {

        double r = static_cast<double>(i) * h;

        double y1 = phi[i];
        double y2 = phi_prime[i];

        double k1_y1 = y2;
        double k1_y2 = f_phi(r, y1, y2, d, s);

        double k2_y1 = y2 + 0.5 * h * k1_y2;
        double k2_y2 = f_phi(r + 0.5 * h,
                             y1 + 0.5 * h * k1_y1,
                             y2 + 0.5 * h * k1_y2,
                             d, s);

        double k3_y1 = y2 + 0.5 * h * k2_y2;
        double k3_y2 = f_phi(r + 0.5 * h,
                             y1 + 0.5 * h * k2_y1,
                             y2 + 0.5 * h * k2_y2,
                             d, s);

        double k4_y1 = y2 + h * k3_y2;
        double k4_y2 = f_phi(r + h,
                             y1 + h * k3_y1,
                             y2 + h * k3_y2,
                             d, s);

        phi[i + 1] = y1 + (h / 6.0) *
            (k1_y1 + 2.0 * k2_y1 + 2.0 * k3_y1 + k4_y1);

        phi_prime[i + 1] = y2 + (h / 6.0) *
            (k1_y2 + 2.0 * k2_y2 + 2.0 * k3_y2 + k4_y2);

        if (phi[i + 1] < 1e-8 && phi_prime[i + 1] > -1e-8) {
            phi.resize(i + 2);
            break;
        }

        if (!std::isfinite(phi[i + 1]) || std::fabs(phi[i + 1]) > 1e8) {
            phi.resize(i + 2);
            break;
        }
    }

    return phi;
}


// Residual for shooting
static inline double shoot_residual(int d, double s, double phi0, double rmax, double h) {
    std::vector<double> phi = solve_ode_high_precision(d, s, phi0, rmax, h);
    return phi.back();
}


// Physics-motivated initial guess
static inline double guess_phi0(double s) {
    double alpha = std::cosh(2.0 * s) / std::sinh(2.0 * s);
    return (3.0 * alpha - std::sqrt(9.0 * alpha * alpha - 8.0)) / 4.0;
}


// Robust bracketing
static bool find_bracket(int d, double s, double rmax, double h, double& low, double& high) {

    double g = guess_phi0(s);

    double test_vals[] = { 0.2, 0.3, 0.5, 0.7, g, 1.0, 1.1, 1.2, 1.5, 2.0 };
    const int ntest = sizeof(test_vals) / sizeof(test_vals[0]);

    double residuals[ntest];

    for (int i = 0; i < ntest; i++) {
        residuals[i] = shoot_residual(d, s, test_vals[i], rmax, h);

        std::cout << "  s=" << std::fixed << std::setprecision(6) << s
                  << "  phi0=" << test_vals[i]
                  << "  -> phi(rmax)=" << residuals[i] << std::endl;
    }

    for (int i = 0; i < ntest - 1; i++) {
        if (residuals[i] * residuals[i + 1] <= 0.0) {
            low = test_vals[i];
            high = test_vals[i + 1];
            return true;
        }
    }

    low = 0.5;
    high = 2.0;
    return false;
}


// Bisection shooting
static double shooting_method(int d, double s, double rmax, double h, double tol = 1e-8) {

    double low, high;

    if (!find_bracket(d, s, rmax, h, low, high)) {
        std::cerr << "  Warning: fallback bracket used for s = " << s << "\n";
    }

    double f_low = shoot_residual(d, s, low, rmax, h);
    double f_high = shoot_residual(d, s, high, rmax, h);

    for (int iter = 0; iter < 80; ++iter) {
        double mid = 0.5 * (low + high);
        double f_mid = shoot_residual(d, s, mid, rmax, h);

        if (std::abs(f_mid) < tol) {
            return mid;
        }

        if (f_low * f_mid <= 0.0) {
            high = mid;
            f_high = f_mid;
        } else {
            low = mid;
            f_low = f_mid;
        }
    }

    return 0.5 * (low + high);
}


// Format s exactly as 0.500000 in filenames
static std::string format_s(double s) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << s;
    return ss.str();
}


int main(int argc, char const* argv[]) {

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " d delta_s s_max\n";
        std::cout << "Example: " << argv[0] << " 2 0.5 5.0\n";
        return 1;
    }

    int d = std::atoi(argv[1]);
    double delta_s = std::atof(argv[2]);
    double s_max = std::atof(argv[3]);

    if (d < 1) {
        std::cerr << "Error: d must be >= 1.\n";
        return 1;
    }

    if (delta_s <= 0.0) {
        std::cerr << "Error: delta_s must be > 0.\n";
        return 1;
    }

    if (s_max <= 0.0) {
        std::cerr << "Error: s_max must be > 0.\n";
        return 1;
    }

    if (s_max < delta_s) {
        std::cerr << "Error: s_max must be >= delta_s.\n";
        return 1;
    }

    double h = 1e-4;
    double rmax = 13.0;

    std::string summary_name =
        "phi0_summary_d" + std::to_string(d) +
        "_ds" + format_s(delta_s) +
        "_smax" + format_s(s_max) + ".csv";

    std::ofstream summary(summary_name);
    summary << "s,phi0\n";
    summary << std::fixed << std::setprecision(10);

    for (double s = delta_s; s <= s_max + 1e-12; s += delta_s) {

        double s_clean = std::round(s * 1e6) / 1e6;

        std::cout << "\n========================================\n";
        std::cout << "Solving for d = " << d
                  << ", s = " << std::fixed << std::setprecision(6) << s_clean
                  << "\n";

        double phi0 = shooting_method(d, s_clean, rmax, h);
        std::vector<double> phi = solve_ode_high_precision(d, s_clean, phi0, rmax, h);

        std::string filename =
            "phi_d" + std::to_string(d) +
            "_s" + format_s(s_clean) + ".dat";

        std::ofstream outfile(filename);
        outfile << std::fixed << std::setprecision(12);

        for (std::size_t i = 0; i < phi.size(); ++i) {
            double r = i * h;
            outfile << r << "\t" << phi[i] << "\n";
        }

        outfile.close();

        summary << s_clean << "," << phi0 << "\n";

        std::cout << "Saved: " << filename << "\n";
        std::cout << "phi(0) = " << std::setprecision(10) << phi0 << "\n";
    }

    summary.close();

    std::cout << "\nDone.\n";
    std::cout << "Summary saved in: " << summary_name << "\n";

    return 0;
}
