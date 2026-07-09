// sanos_downstream_aadc.cpp — AAD downstream: ∂(barrier price)/∂(surface nodes)
// Requires: AADC library (https://matlogica.com/aadc)
//
// Inputs: local vol surface σ²(K_m, T_j) on a grid (from SANOS calibration)
// Output: barrier option price + ∂price/∂σ²(K_m, T_j) via reverse pass
//
// This is the "Factor 2" in the chain rule:
//   ∂V/∂quote = Σ_m (∂V/∂C_fit(K_m)) · (∂C_fit(K_m)/∂quote)
//                    ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^
//                    this program (AAD)      IFT (Python)

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <fstream>
#include <aadcNG/aadcNG.h>

using namespace aadcNG;
using idouble = idoubleNG;

int main(int argc, char** argv) {
    // Grid parameters (must match SANOS output)
    int N_strikes = 11;
    int N_expiries = 3;
    double K_min = 0.7, K_max = 1.3;
    double expiries[] = {0.25, 0.5, 1.0};

    // Barrier option parameters
    double S0 = 1.0;        // forward = 1 (normalized)
    double K_strike = 1.0;  // ATM call
    double barrier = 0.8;   // down-and-out
    double r = 0.0;
    double T = 1.0;
    int M_paths = 20000;
    int n_steps = 50;

    // Surface nodes: local variance σ²(K_m, T_j)
    // For testing: flat vol σ = 0.20 → σ² = 0.04
    int N_surface = N_strikes * N_expiries;  // 33 nodes
    std::vector<double> surface_vals(N_surface, 0.04);  // default flat

    // Read surface from file if provided
    if (argc > 1) {
        std::ifstream fin(argv[1]);
        if (fin.good()) {
            for (int i = 0; i < N_surface && fin >> surface_vals[i]; ++i);
            std::cerr << "Read " << N_surface << " surface values from " << argv[1] << std::endl;
        }
    }

    // Strike grid
    std::vector<double> K_grid(N_strikes);
    for (int i = 0; i < N_strikes; ++i)
        K_grid[i] = K_min + i * (K_max - K_min) / (N_strikes - 1);

    // Record kernel
    AADCFunctions<> fn;
    std::vector<AADCArgument> surface_args(N_surface);
    std::vector<AADCArgument> z_args(n_steps);
    AADCResult price_res;
    std::shared_ptr<ComputeBlock> root;

    {
        recording::RecordingGuard<> guard(fn);
        beginBlockRecording();

        // Surface nodes as inputs (these are what we differentiate w.r.t.)
        std::vector<idouble> sigma2(N_surface);
        for (int i = 0; i < N_surface; ++i) {
            sigma2[i] = idouble(surface_vals[i]);
            surface_args[i] = sigma2[i].markAsInput();
        }

        // Random normals
        std::vector<idouble> Z(n_steps);
        for (int j = 0; j < n_steps; ++j) {
            Z[j] = idouble(0.0);
            z_args[j] = Z[j].markAsInputNoDiff();
        }

        // MC path under local vol
        double dt = T / n_steps;
        double sqrt_dt = std::sqrt(dt);

        idouble S(S0);
        idouble alive(1.0);

        for (int step = 0; step < n_steps; ++step) {
            double t = step * dt;

            // Find expiry index (piecewise constant in T)
            int j_exp = 0;
            for (int j = 0; j < N_expiries - 1; ++j)
                if (t >= expiries[j]) j_exp = j;

            // Interpolate local variance from surface using iIf
            // Piecewise constant: pick the nearest grid node to current S
            // Use iIf chain to select the right node on tape
            int base = j_exp * N_strikes;
            idouble local_var = sigma2[base];  // start with lowest strike
            for (int i = 1; i < N_strikes; ++i) {
                // If S >= K_grid[i], use sigma2 at this node
                local_var = iIf(S >= idouble(K_grid[i]), sigma2[base + i], local_var);
            }

            idouble local_vol = std::sqrt(local_var);

            // GBM step
            S = S * std::exp((idouble(r) - local_var * idouble(0.5)) * idouble(dt)
                             + local_vol * idouble(sqrt_dt) * Z[step]);

            // Barrier check
            alive = alive * iIf(S > idouble(barrier), idouble(1.0), idouble(0.0));
        }

        // Payoff: down-and-out call
        idouble payoff_raw = S - idouble(K_strike);
        idouble payoff = alive * iIf(payoff_raw > idouble(0.0), payoff_raw, idouble(0.0))
                         * idouble(std::exp(-r * T));

        price_res = payoff.markAsOutput();
    }
    root = endBlockRecording(fn.tapeHandle());
    fn.setRootBlock(root);

    auto ws = fn.createWorkSpace();
    KernelRunner runner(fn.tapeHandle(), root);

    // Set surface values
    for (int i = 0; i < N_surface; ++i)
        ws->setVal(surface_args[i], surface_vals[i]);

    // MC loop
    std::mt19937_64 gen(42);
    std::normal_distribution<> N01;

    double sum_price = 0;
    std::vector<double> sum_grad(N_surface, 0.0);

    for (int m = 0; m < M_paths; ++m) {
        for (int j = 0; j < n_steps; ++j)
            ws->setVal(z_args[j], N01(gen));

        runner.forward(*ws, root);
        sum_price += ws->val(price_res);

        // Reverse
        auto& dbuf = ws->dBuffer();
        std::fill(dbuf.begin(), dbuf.end(), 0.0);
        ws->setDiff(price_res, 1.0);
        runner.reverse(*ws, root);

        for (int i = 0; i < N_surface; ++i)
            sum_grad[i] += ws->diff(surface_args[i]);
    }

    double price = sum_price / M_paths;

    // Output results
    std::cout << std::setprecision(8);
    std::cout << "PRICE " << price << std::endl;
    for (int i = 0; i < N_surface; ++i) {
        int j_exp = i / N_strikes;
        int i_str = i % N_strikes;
        std::cout << "GRAD " << i << " " << j_exp << " " << i_str
                  << " " << K_grid[i_str] << " " << expiries[j_exp]
                  << " " << sum_grad[i] / M_paths << std::endl;
    }

    // FD validation (3 random nodes)
    std::cout << "\nFD_VALIDATION" << std::endl;
    double h = 1e-4;
    int test_nodes[] = {5, 16, 27};  // ATM nodes at each expiry
    for (int node : test_nodes) {
        // Up
        ws->setVal(surface_args[node], surface_vals[node] + h);
        gen.seed(42);
        double s_up = 0;
        for (int m = 0; m < M_paths; ++m) {
            for (int j = 0; j < n_steps; ++j) ws->setVal(z_args[j], N01(gen));
            runner.forward(*ws, root);
            s_up += ws->val(price_res);
        }
        // Down
        ws->setVal(surface_args[node], surface_vals[node] - h);
        gen.seed(42);
        double s_dn = 0;
        for (int m = 0; m < M_paths; ++m) {
            for (int j = 0; j < n_steps; ++j) ws->setVal(z_args[j], N01(gen));
            runner.forward(*ws, root);
            s_dn += ws->val(price_res);
        }
        ws->setVal(surface_args[node], surface_vals[node]);  // restore

        double fd = (s_up - s_dn) / (2 * h * M_paths);
        double aad = sum_grad[node] / M_paths;
        double ratio = (std::abs(fd) > 1e-10) ? aad / fd : 0;

        std::cout << "NODE " << node << " AAD=" << aad << " FD=" << fd
                  << " RATIO=" << ratio << std::endl;
    }

    return 0;
}
