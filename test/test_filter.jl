using SequentialQuadraticProgramming
using Test
using LinearAlgebra

const SQP = SequentialQuadraticProgramming

@testset "Filter Line Search (Phase 9.1)" begin

    @testset "Filter data structure — dominance" begin
        F = SQP.Filter{Float64}()
        @test isempty(F)
        @test length(F) == 0
        # Empty filter dominates nothing
        @test !SQP.is_dominated(F, 1.0, 0.5)

        # Seed entry (-Inf, 10.0) should dominate anything with θ > 10
        push!(F.entries, (-Inf, 10.0))
        @test SQP.is_dominated(F, 5.0, 100.0)
        @test SQP.is_dominated(F, -1e9, 20.0)
        # But nothing with θ ≤ 10 is dominated by (-Inf, 10)
        @test !SQP.is_dominated(F, 5.0, 5.0)
    end

    @testset "Filter augmentation" begin
        F = SQP.Filter{Float64}()
        # Add entry with margin: (f_k - γ_f·θ_k, (1-γ_θ)·θ_k)
        SQP.augment!(F, 10.0, 1.0, 1e-2, 1e-2)
        @test length(F) == 1
        f_e, θ_e = F.entries[1]
        @test f_e ≈ 10.0 - 1e-2 * 1.0
        @test θ_e ≈ (1 - 1e-2) * 1.0

        # Adding an entry that's dominated by the existing one is a no-op
        SQP.augment!(F, 100.0, 100.0, 1e-2, 1e-2)
        @test length(F) == 1  # nothing changed

        # Adding an entry that dominates the existing one prunes it
        SQP.augment!(F, 5.0, 0.5, 1e-2, 1e-2)
        @test length(F) == 1
        f_e2, θ_e2 = F.entries[1]
        @test f_e2 ≈ 5.0 - 1e-2 * 0.5
        @test θ_e2 ≈ (1 - 1e-2) * 0.5
    end

    @testset "Filter FIFO eviction" begin
        F = SQP.Filter{Float64}(; max_size = 3)
        # Add entries with non-dominating different points
        SQP.augment!(F, 1.0, 10.0, 0.0, 0.0)
        SQP.augment!(F, 5.0, 5.0, 0.0, 0.0)
        SQP.augment!(F, 10.0, 1.0, 0.0, 0.0)
        @test length(F) == 3
        # Fourth non-dominating entry evicts the oldest
        SQP.augment!(F, 20.0, 0.5, 0.0, 0.0)
        @test length(F) == 3
        # The oldest (1.0, 10.0) should no longer be present
        @test !((1.0, 10.0) in F.entries)
    end

    @testset "theta_constraint_violation" begin
        # Only positive g values contribute; all h values contribute in abs
        g_val = [-1.0, 2.0, 0.5, -0.3]
        h_val = [0.4, -0.6]
        θ = SQP.theta_constraint_violation(g_val, h_val)
        @test θ ≈ 2.0 + 0.5 + 0.4 + 0.6

        # Empty constraints → zero
        @test SQP.theta_constraint_violation(Float64[], Float64[]) == 0.0
    end

    @testset "Reset" begin
        F = SQP.Filter{Float64}()
        push!(F.entries, (1.0, 2.0))
        push!(F.entries, (3.0, 4.0))
        SQP.reset!(F)
        @test isempty(F)
    end

    # ─────────────────────────────────────────────────────────────────
    # Integration tests — filter globalization in the SQP solver
    # ─────────────────────────────────────────────────────────────────

    @testset "Default globalization is unchanged (:line_search)" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [-1.9, 2.0])
        @test r.converged
        @test r.iterations == 24
        # Filter counters are zero when not using filter
        @test r.n_filter_f_steps == 0
        @test r.n_filter_h_steps == 0
        @test r.n_filter_fallbacks == 0
    end

    @testset "Rosenbrock with :filter_line_search (sanity)" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [-1.9, 2.0];
                      options = SQPOptions(globalization = :filter_line_search))
        @test r.converged
        @test r.objective < 1e-6
        # All unconstrained → f-steps only, filter only has the seed entry
        @test r.n_filter_f_steps > 0
        @test r.n_filter_h_steps == 0
    end

    @testset "HS071 with :filter_line_search" begin
        f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
        g(x) = [-prod(x) + 25.0]
        h(x) = [dot(x, x) - 40.0]
        r = sqp_solve(f, g, h, [1.0, 5.0, 5.0, 1.0], ones(4), 5.0 * ones(4);
                      options = SQPOptions(max_iterations = 2000,
                                           globalization = :filter_line_search))
        @test r.converged
        @test isapprox(r.objective, 17.014, atol = 0.1)
    end

    @testset "HS037 with :filter_line_search" begin
        f(x) = -x[1] * x[2] * x[3]
        g(x) = [-(x[1] + 2 * x[2] + 2 * x[3]); -72 + (x[1] + 2 * x[2] + 2 * x[3])]
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [1.0, 10.0, 10.0], zeros(3), 42.0 * ones(3);
                      options = SQPOptions(max_iterations = 1000,
                                           globalization = :filter_line_search))
        @test r.converged
        @test isapprox(r.objective, -3456.0, atol = 10.0)
    end

    @testset "HS118 with :filter_line_search" begin
        f(x) = sum((2.3 * x[3 * k + 1] + 0.0001 * x[3 * k + 1]^2 +
                     1.7 * x[3 * k + 2] + 0.0001 * x[3 * k + 2]^2 +
                     2.2 * x[3 * k + 3] + 0.00015 * x[3 * k + 3]^2) for k in 0:4)
        g(x) = [([x[3 * j + 1] - x[3 * j - 2] + 7 for j in 1:4] .- 13);
                -([x[3 * j + 1] - x[3 * j - 2] + 7 for j in 1:4]);
                ([x[3 * j + 2] - x[3 * j - 1] + 7 for j in 1:4] .- 14);
                -([x[3 * j + 2] - x[3 * j - 1] + 7 for j in 1:4]);
                ([x[3 * j + 3] - x[3 * j] + 7 for j in 1:4] .- 13);
                -([x[3 * j + 3] - x[3 * j] + 7 for j in 1:4]);
                -(x[1] + x[2] + x[3] - 60); -(x[4] + x[5] + x[6] - 50);
                -(x[7] + x[8] + x[9] - 70); -(x[10] + x[11] + x[12] - 85);
                -(x[13] + x[14] + x[15] - 100)]
        h(x) = zeros(0)
        x0 = Float64[20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20]
        lb = zeros(15); lb[1] = 8; lb[2] = 43; lb[3] = 3
        ub = 60.0 * ones(15); ub[1] = 16; ub[2] = 57; ub[3] = 16
        for k in 1:4; ub[3 * k + 1] = 90; ub[3 * k + 2] = 120; end
        r = sqp_solve(f, g, h, x0, lb, ub;
                      options = SQPOptions(max_iterations = 1000,
                                           globalization = :filter_line_search))
        @test r.converged
        @test isapprox(r.objective, 664.82045, atol = 1.0)
    end

    @testset "HS092 — filter line search converges dramatically faster" begin
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
                      options = SQPOptions(max_iterations = 500,
                                           globalization = :filter_line_search))
        @test r.converged
        @test isapprox(r.objective, 1.36265681, atol = 0.02)
        # HS092 should benefit dramatically from the filter —
        # baseline (:line_search) takes 352 iters with Phase 8.2 defaults.
        @test r.iterations < 200
        # At least some f-steps AND some h-steps (mixed mode)
        @test r.n_filter_f_steps + r.n_filter_h_steps > 0
    end

    @testset "HS015 — filter line search fixes previously-broken case" begin
        x0 = [-2.0, 1.0]
        f(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
        g(x) = [-x[1] * x[2] + 1; -(x[1] + x[2]^2)]
        h(x) = zeros(0)
        lb = [-Inf, -Inf]
        ub = [0.5, Inf]
        r = sqp_solve(f, g, h, x0, lb, ub;
                      options = SQPOptions(max_iterations = 100,
                                           globalization = :filter_line_search))
        @test r.converged
        @test isapprox(r.objective, 306.5, atol = 1.0)
        @test r.iterations < 50
    end

    @testset "Backward-compatible 6-arg SQPResult ctor still works" begin
        x = [1.0, 2.0]
        r = SQPResult(x, 3.14, 5, true, 1e-8, :converged)
        @test r.n_filter_f_steps == 0
        @test r.n_filter_h_steps == 0
        @test r.n_filter_fallbacks == 0
        @test r.filter_size_final == 0
    end
end
