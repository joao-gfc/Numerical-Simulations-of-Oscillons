// spectrum_sphaleron_scan_positive_l.cpp
//
// Finds the positive eigenvalues omega^2 for the sphaleron background
// for a range of s values, in an arbitrary angular sector l.
//
// Stability equation:
//
//   -eta'' - (D-1)/r eta' + l(l+D-2)/r^2 eta + U(r) eta = omega^2 eta
//
// where
//   U(r) = 1 - 6 alpha phi_S(r) + 6 phi_S(r)^2,
//   alpha = coth(2s).
//
// This version:
//   • scans s from delta_s to s_max in steps of delta_s
//   • loads backgrounds named like phi_d2_s0.500000.dat
//   • searches in an arbitrary l sector
//   • finds up to max_modes positive eigenvalues in the scan window
//   • saves one summary file with columns:
//         s, omega2_mode1, omega2_mode2, ..., omega2_modeN
//
// Build:
//   g++ -O3 -std=gnu++17 spectrum_sphaleron_scan_positive_l.cpp -o sphaleron_spectrum_positive_l
//
// Run:
//   ./sphaleron_spectrum_positive_l D l delta_s s_max Rmax w2_min w2_max dw2 h max_modes
//
// Example:
//   ./sphaleron_spectrum_positive_l 2 1 0.01 1.0 13.0 0.0 10.0 0.01 0.001 5

#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <cmath>
#include <string>
#include <cctype>
#include <limits>

using namespace std;
using Real = long double;

// ------------------------------------------------------------
// Safe number-to-string for output filenames
// ------------------------------------------------------------
static std::string safe_num(long double x, int prec = 6) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << x;
    std::string s = os.str();

    if (!s.empty() && s[0] == '-') s[0] = 'm';
    for (char& c : s) if (c == '.') c = 'p';

    return s;
}

// ------------------------------------------------------------
// Format s exactly like 0.500000 for background filenames
// ------------------------------------------------------------
static std::string format_s(double s) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << s;
    return ss.str();
}

// ------------------------------------------------------------
// Background phi_S(r) table
// ------------------------------------------------------------
struct PhiTable {
    vector<Real> r, phi;

    bool load_dat(const string& path) {
        ifstream in(path);
        if (!in) return false;

        r.clear();
        phi.clear();

        string line;
        bool first = true;

        while (getline(in, line)) {
            if (line.empty()) continue;

            for (char& c : line) {
                if (c == ',' || c == ';') c = ' ';
            }

            if (first) {
                first = false;
                bool has_alpha = false;
                for (unsigned char c : line) {
                    if (std::isalpha(c)) {
                        has_alpha = true;
                        break;
                    }
                }
                if (has_alpha) continue;
            }

            Real rr, pp;
            stringstream ss(line);
            if (ss >> rr >> pp) {
                r.push_back(rr);
                phi.push_back(pp);
            }
        }

        if (r.empty()) return false;

        vector<size_t> idx(r.size());
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [&](size_t i, size_t j) { return r[i] < r[j]; });

        vector<Real> r2, p2;
        r2.reserve(r.size());
        p2.reserve(phi.size());

        for (auto k : idx) {
            r2.push_back(r[k]);
            p2.push_back(phi[k]);
        }

        r.swap(r2);
        phi.swap(p2);

        vector<Real> r3, p3;
        for (size_t i = 0; i < r.size(); ++i) {
            if (!r3.empty() && r[i] == r3.back()) {
                p3.back() = phi[i];
            } else {
                r3.push_back(r[i]);
                p3.push_back(phi[i]);
            }
        }

        r.swap(r3);
        phi.swap(p3);

        return true;
    }

    Real rmin() const { return r.empty() ? 0.0L : r.front(); }
    Real rmax() const { return r.empty() ? 0.0L : r.back(); }

    Real eval(Real x) const {
        if (r.empty()) return 0.0L;
        if (x <= r.front()) return phi.front();
        if (x >= r.back())  return phi.back();

        auto it = upper_bound(r.begin(), r.end(), x);
        size_t j = size_t(it - r.begin());
        size_t i = j - 1;

        Real t = (x - r[i]) / (r[j] - r[i]);
        return phi[i] + t * (phi[j] - phi[i]);
    }
};

// ------------------------------------------------------------
// Effective potential U(r)
// ------------------------------------------------------------
struct Ueff {
    const PhiTable* tab = nullptr;
    Real alpha = 1.0L;

    inline Real operator()(Real rr) const {
        Real ps = tab->eval(rr);
        return 1.0L - 6.0L * alpha * ps + 6.0L * ps * ps;
    }
};

// ------------------------------------------------------------
// ODE system for arbitrary l
//
// Equation:
//   -eta'' - (D-1)/r eta' + l(l+D-2)/r^2 eta + U eta = w2 eta
//
// rewritten as
//
//   eta'' + (D-1)/r eta' - l(l+D-2)/r^2 eta + (w2 - U) eta = 0
//
// so
//
//   g' = -(D-1)/r g + [ l(l+D-2)/r^2 + U - w2 ] f
// ------------------------------------------------------------
struct State {
    Real f, g;
};

struct RHS {
    Ueff U;
    int D = 2;
    int l = 0;

    inline State operator()(Real r, const State& y, Real w2) const {
        Real invr = (r > 0.0L) ? 1.0L / r : 0.0L;
        Real A = (Real)(D - 1) * invr;
        Real cent = (r > 0.0L) ? ((Real)l * (l + D - 2)) / (r * r) : 0.0L;
        Real Veff = U(r);

        return { y.g, -A * y.g + (cent + Veff - w2) * y.f };
    }
};

static inline void rk4_step(Real& r, State& y, Real h, const RHS& rhs, Real w2) {
    State k1 = rhs(r, y, w2);
    State k2 = rhs(r + h / 2.0L,
                   { y.f + h * k1.f / 2.0L, y.g + h * k1.g / 2.0L }, w2);
    State k3 = rhs(r + h / 2.0L,
                   { y.f + h * k2.f / 2.0L, y.g + h * k2.g / 2.0L }, w2);
    State k4 = rhs(r + h,
                   { y.f + h * k3.f, y.g + h * k3.g }, w2);

    y.f += (h / 6.0L) * (k1.f + 2.0L * k2.f + 2.0L * k3.f + k4.f);
    y.g += (h / 6.0L) * (k1.g + 2.0L * k2.g + 2.0L * k3.g + k4.g);
    r += h;
}

// ------------------------------------------------------------
// Regular Taylor start for arbitrary l
//
// eta(r) = r^l (a0 + a2 r^2 + ...)
//
// with
//
// a2 = (U(0) - w2) a0 / [ 2 (2l + D) ].
//
// We choose a0 = 1.
// ------------------------------------------------------------
static inline void taylor_start(Real r0, Real w2, const RHS& rhs, State& y_out) {
    Real U0 = rhs.U(0.0L);
    Real a0 = 1.0L;
    Real a2 = (U0 - w2) * a0 / (2.0L * (2.0L * rhs.l + rhs.D));

    Real rl = (rhs.l == 0) ? 1.0L : pow(r0, (Real)rhs.l);
    Real poly = a0 + a2 * r0 * r0;
    Real dpoly = 2.0L * a2 * r0;

    y_out.f = rl * poly;

    if (rhs.l == 0) {
        y_out.g = dpoly;
    } else {
        y_out.g =
            (Real)rhs.l * pow(r0, (Real)rhs.l - 1.0L) * poly
            + rl * dpoly;
    }
}

// ------------------------------------------------------------
// Shooting function eta(R; omega^2)
// ------------------------------------------------------------
static Real shoot(Real w2, Real r0, Real R, Real h, const RHS& rhs) {
    State y;
    taylor_start(r0, w2, rhs, y);

    Real r = r0;
    int steps = (int)floor((R - r0) / h);

    for (int i = 0; i < steps; ++i) {
        rk4_step(r, y, h, rhs, w2);

        if (!isfinite((double)y.f) || !isfinite((double)y.g)) {
            return numeric_limits<Real>::quiet_NaN();
        }
        if (fabsl(y.f) > 1e12L) {
            return y.f;
        }
    }

    return y.f;
}

// ------------------------------------------------------------
// Root refinement by bisection
// ------------------------------------------------------------
static bool refine_bisect(Real a, Real b, Real r0, Real R, Real h,
                          const RHS& rhs, Real& root,
                          int itmax = 80, Real tol = 1e-10L) {
    Real Fa = shoot(a, r0, R, h, rhs);
    Real Fb = shoot(b, r0, R, h, rhs);

    if (!isfinite((double)Fa) || !isfinite((double)Fb) || Fa * Fb > 0.0L) {
        return false;
    }

    for (int it = 0; it < itmax; ++it) {
        Real m = (a + b) / 2.0L;
        Real Fm = shoot(m, r0, R, h, rhs);

        if (!isfinite((double)Fm)) {
            double eps = nextafter((double)m, (double)a) - (double)m;
            Fm = shoot(m + (Real)eps, r0, R, h, rhs);
        }

        if (Fa * Fm <= 0.0L) {
            b = m;
            Fb = Fm;
        } else {
            a = m;
            Fa = Fm;
        }

        if (fabsl(b - a) < tol) break;
    }

    root = (a + b) / 2.0L;
    return true;
}

// ------------------------------------------------------------
// Find up to max_modes roots in the scan window
// ------------------------------------------------------------
static vector<Real> find_modes(Real w2_min, Real w2_max, Real dw2,
                               Real r0, Real Rmax, Real h,
                               const RHS& rhs, int max_modes) {
    vector<Real> roots;

    Real prev_w2 = w2_min;
    Real prev_F = shoot(prev_w2, r0, Rmax, h, rhs);

    if (!isfinite((double)prev_F)) {
        prev_F = copysign((Real)1.0L, (double)prev_F);
    }

    for (Real w2 = w2_min + dw2; w2 <= w2_max + 1e-18L; w2 += dw2) {
        Real F = shoot(w2, r0, Rmax, h, rhs);

        if (!isfinite((double)F)) {
            F = copysign((Real)1.0L, (double)F);
        }

        if (prev_F * F <= 0.0L) {
            Real root;
            if (refine_bisect(prev_w2, w2, r0, Rmax, h, rhs, root)) {
                bool duplicate = false;
                for (Real x : roots) {
                    if (fabsl(x - root) < 1e-8L) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    roots.push_back(root);
                    if ((int)roots.size() >= max_modes) break;
                }
            }
        }

        prev_w2 = w2;
        prev_F = F;
    }

    return roots;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc != 11) {
        cerr << "Usage:\n  " << argv[0]
             << " D l delta_s s_max Rmax w2_min w2_max dw2 h max_modes\n\n"
             << "Example:\n"
             << "  ./sphaleron_spectrum_positive_l 2 1 0.01 1.0 13.0 0.0 10.0 0.01 0.001 5\n";
        return 1;
    }

    int  D         = stoi(argv[1]);
    int  l         = stoi(argv[2]);
    Real ds        = stold(argv[3]);
    Real s_max     = stold(argv[4]);
    Real Rmax      = stold(argv[5]);
    Real w2_min    = stold(argv[6]);
    Real w2_max    = stold(argv[7]);
    Real dw2       = stold(argv[8]);
    Real h         = stold(argv[9]);
    int  max_modes = stoi(argv[10]);

    if (D < 2 || D > 3 || l < 0 || ds <= 0.0L || s_max <= 0.0L ||
        !(Rmax > 0.0L) || !(w2_min < w2_max) ||
        !(dw2 > 0.0L) || !(h > 0.0L) || max_modes < 1) {
        cerr << "Parameter error.\n";
        return 1;
    }

    string out_name =
        "spectrum_scan_positive"
        "_D" + to_string(D) +
        "_l" + to_string(l) +
        "_ds" + safe_num(ds) +
        "_smax" + safe_num(s_max) +
        "_R" + safe_num(Rmax) +
        "_w2m" + safe_num(w2_min) +
        "_w2M" + safe_num(w2_max) +
        "_dw" + safe_num(dw2) +
        "_h" + safe_num(h) +
        "_Nmodes" + to_string(max_modes) +
        ".csv";

    ofstream out(out_name);
    if (!out) {
        cerr << "Could not open output file: " << out_name << "\n";
        return 1;
    }

    out << setprecision(16);
    out << "s";
    for (int k = 1; k <= max_modes; ++k) {
        out << ",omega2_mode" << k;
    }
    out << ",background_file,status\n";

    cout << fixed << setprecision(8);

    for (Real s = ds; s <= s_max + 1e-18L; s += ds) {
        double s_clean = std::round((double)s * 1e6) / 1e6;

        string fname =
            "phi_d" + to_string(D) +
            "_s" + format_s(s_clean) +
            ".dat";

        cout << "\n----------------------------------------\n";
        cout << "Processing s = " << s_clean
             << " using " << fname
             << " in l = " << l << "\n";

        PhiTable tab;
        if (!tab.load_dat(fname)) {
            cerr << "  Failed to read background file.\n";
            out << s_clean;
            for (int k = 1; k <= max_modes; ++k) out << ",nan";
            out << "," << fname << ",missing_background\n";
            continue;
        }

        if (tab.rmax() < Rmax - 1e-12L) {
            cerr << "  Background rmax too small.\n";
            out << s_clean;
            for (int k = 1; k <= max_modes; ++k) out << ",nan";
            out << "," << fname << ",rmax_too_small\n";
            continue;
        }

        Real alpha = cosh(2.0L * s_clean) / sinh(2.0L * s_clean);

        Ueff U;
        U.tab = &tab;
        U.alpha = alpha;

        RHS rhs;
        rhs.U = U;
        rhs.D = D;
        rhs.l = l;

        Real r0 = max((Real)1e-6L, tab.rmin());

        vector<Real> modes = find_modes(w2_min, w2_max, dw2, r0, Rmax, h, rhs, max_modes);

        cout << "  found " << modes.size() << " modes\n";
        for (size_t i = 0; i < modes.size(); ++i) {
            cout << "    mode " << (i + 1) << ": omega^2 = " << (double)modes[i] << "\n";
        }

        out << s_clean;
        for (int k = 0; k < max_modes; ++k) {
            if (k < (int)modes.size()) out << "," << (double)modes[k];
            else                       out << ",nan";
        }
        out << "," << fname << ",ok\n";
    }

    cout << "\nSaved results to: " << out_name << "\n";
    return 0;
}
