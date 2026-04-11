using SequentialQuadraticProgramming
using Test
using LinearAlgebra

@testset "Numerical Safeguards (Phase 8.2)" begin

    @testset "Zero-cost on well-behaved problems" begin
        # Rosenbrock: unconstrained, convex-ish — safeguards should never fire.
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [-1.9, 2.0];
                      options = SQPOptions(numerical_safeguards = true))
        @test r.converged
        @test r.n_steps_clamped == 0
        @test r.n_bfgs_skipped == 0
        @test r.lm_lambda_final == 0.0
    end

    @testset "HS037 — clean problem, safeguards no-op" begin
        f(x) = -x[1] * x[2] * x[3]
        g(x) = [-(x[1] + 2 * x[2] + 2 * x[3]);
                -72 + (x[1] + 2 * x[2] + 2 * x[3])]
        h(x) = zeros(0)
        r_off = sqp_solve(f, g, h, [1.0, 10.0, 10.0], zeros(3), 42.0 * ones(3);
                          options = SQPOptions(numerical_safeguards = false))
        r_on = sqp_solve(f, g, h, [1.0, 10.0, 10.0], zeros(3), 42.0 * ones(3);
                         options = SQPOptions(numerical_safeguards = true))
        @test r_off.converged
        @test r_on.converged
        @test r_on.n_steps_clamped == 0
        @test r_on.n_bfgs_skipped == 0
        @test r_on.lm_lambda_final == 0.0
        # Same iteration count — safeguards did not interfere
        @test r_on.iterations == r_off.iterations
    end

    @testset "Opt-out preserves v0.8.1 behaviour" begin
        # HS092 with safeguards off → should reproduce the old baseline
        n = 6
        mu = [8.6033358901938017e-01, 3.4256184594817283e+00, 6.4372981791719468e+00,
              9.5293344053619631e+00, 1.2645287223856643e+01, 1.5771284874815882e+01,
              1.8902409956860023e+01, 2.2036496727938566e+01, 2.5172446326646664e+01,
              2.8309642854452012e+01, 3.1447714637546234e+01, 3.4586424215288922e+01,
              3.7725612827776501e+01, 4.0865170330488070e+01, 4.4005017920830845e+01,
              4.7145097736761031e+01, 5.0285366337773652e+01, 5.3425790477394663e+01,
              5.6566344279821521e+01, 5.9707007305335459e+01, 6.2847763194454451e+01,
              6.5988598698490392e+01, 6.9129502973895256e+01, 7.2270467060308960e+01,
              7.5411483488848148e+01, 7.8552545984242926e+01, 8.1693649235601683e+01,
              8.4834788718042290e+01, 8.7975960552493220e+01, 9.1117161394464745e+01]
        A_coef = 2 * sin.(mu) ./ (mu .+ sin.(mu) .* cos.(mu))
        rho_fn(x) = -(exp.(-mu .^ 2 .* sum(x .^ 2)) .+
                      sum([2 * (-1)^(ii - 1) * exp.(-mu .^ 2 * sum([x[i]^2 for i in ii:n])) for ii in 2:n]) .+
                      (-1)^n) ./ mu .^ 2
        function firstsum(x)
            isum = 0.0
            mrho = rho_fn(x)
            for i in 1:29
                isum += sum([mu[i]^2 * mu[j]^2 * A_coef[i] * A_coef[j] * mrho[i] * mrho[j] *
                             (sin(mu[i] + mu[j]) / (mu[i] + mu[j]) + sin(mu[i] - mu[j]) / (mu[i] - mu[j])) for j in (i + 1):30])
            end
            return isum
        end
        f(x) = sum(x .^ 2)
        g(x) = [firstsum(x) +
                sum(mu .^ 4 .* A_coef .^ 2 .* rho_fn(x) .^ 2 .* (sin.(2 .* mu) ./ (2 * mu) .+ 1) ./ 2) -
                sum(mu .^ 2 .* A_coef .* rho_fn(x) .* (2 * sin.(mu) ./ mu .^ 3 - 2 * cos.(mu) ./ mu .^ 2)) +
                2 / 15 - 0.0001]
        h(x) = zeros(0)
        x0 = [0.5 * (-1)^(i + 1) for i in 1:n]
        r_off = sqp_solve(f, g, h, x0, -Inf * ones(n), Inf * ones(n);
                          options = SQPOptions(max_iterations = 2000,
                                               numerical_safeguards = false))
        # Safeguards-off path matches the v0.8.1 behaviour:
        # HS092 converges to the neighbourhood of the known optimum.
        # The tolerance is generous because v0.8.1's trajectory is unstable
        # and the final objective varies by ~0.01 across BLAS/solver versions.
        @test r_off.converged
        @test isapprox(r_off.objective, 1.36265681, atol = 0.02)
        @test r_off.n_steps_clamped == 0
        @test r_off.n_bfgs_skipped == 0
        @test r_off.lm_lambda_final == 0.0
    end

    @testset "HS092 — safeguards improve convergence" begin
        n = 6
        mu = [8.6033358901938017e-01, 3.4256184594817283e+00, 6.4372981791719468e+00,
              9.5293344053619631e+00, 1.2645287223856643e+01, 1.5771284874815882e+01,
              1.8902409956860023e+01, 2.2036496727938566e+01, 2.5172446326646664e+01,
              2.8309642854452012e+01, 3.1447714637546234e+01, 3.4586424215288922e+01,
              3.7725612827776501e+01, 4.0865170330488070e+01, 4.4005017920830845e+01,
              4.7145097736761031e+01, 5.0285366337773652e+01, 5.3425790477394663e+01,
              5.6566344279821521e+01, 5.9707007305335459e+01, 6.2847763194454451e+01,
              6.5988598698490392e+01, 6.9129502973895256e+01, 7.2270467060308960e+01,
              7.5411483488848148e+01, 7.8552545984242926e+01, 8.1693649235601683e+01,
              8.4834788718042290e+01, 8.7975960552493220e+01, 9.1117161394464745e+01]
        A_coef = 2 * sin.(mu) ./ (mu .+ sin.(mu) .* cos.(mu))
        rho_fn(x) = -(exp.(-mu .^ 2 .* sum(x .^ 2)) .+
                      sum([2 * (-1)^(ii - 1) * exp.(-mu .^ 2 * sum([x[i]^2 for i in ii:n])) for ii in 2:n]) .+
                      (-1)^n) ./ mu .^ 2
        function firstsum(x)
            isum = 0.0
            mrho = rho_fn(x)
            for i in 1:29
                isum += sum([mu[i]^2 * mu[j]^2 * A_coef[i] * A_coef[j] * mrho[i] * mrho[j] *
                             (sin(mu[i] + mu[j]) / (mu[i] + mu[j]) + sin(mu[i] - mu[j]) / (mu[i] - mu[j])) for j in (i + 1):30])
            end
            return isum
        end
        f(x) = sum(x .^ 2)
        g(x) = [firstsum(x) +
                sum(mu .^ 4 .* A_coef .^ 2 .* rho_fn(x) .^ 2 .* (sin.(2 .* mu) ./ (2 * mu) .+ 1) ./ 2) -
                sum(mu .^ 2 .* A_coef .* rho_fn(x) .* (2 * sin.(mu) ./ mu .^ 3 - 2 * cos.(mu) ./ mu .^ 2)) +
                2 / 15 - 0.0001]
        h(x) = zeros(0)
        x0 = [0.5 * (-1)^(i + 1) for i in 1:n]
        r = sqp_solve(f, g, h, x0, -Inf * ones(n), Inf * ones(n);
                      options = SQPOptions(max_iterations = 2000,
                                           numerical_safeguards = true))

        @test r.converged
        @test isapprox(r.objective, 1.36265681, atol = 0.01)
        # Safeguards actually fired on this problem
        @test r.n_steps_clamped > 0
        @test r.n_bfgs_skipped > 0
        # Iteration count should be at most the v0.8.1 baseline (671)
        # and typically much better. Keep the bound generous to stay
        # robust across solver/BLAS version drift.
        @test r.iterations < 671
    end

    @testset "Backward-compatible 6-arg SQPResult ctor still works" begin
        x = [1.0, 2.0]
        r = SQPResult(x, 3.14, 5, true, 1e-8, :converged)
        @test r.x == x
        @test r.n_steps_clamped == 0
        @test r.n_bfgs_skipped == 0
        @test r.lm_lambda_final == 0.0
        @test r.diagnostics === nothing
        @test r.n_soc_steps == 0
    end
end
