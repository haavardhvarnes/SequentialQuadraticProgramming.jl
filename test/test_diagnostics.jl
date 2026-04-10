using SequentialQuadraticProgramming
using Test
using LinearAlgebra

@testset "Problem Diagnostics" begin

    @testset "Linear problem → zero nonlinearity" begin
        # f(x) = sum(x), constraints linear — diagnose_problem should report
        # minimal nonlinearity and recommend plain line search.
        f(x) = sum(x)
        g(x) = [x[1] + x[2] - 2.0]
        h(x) = [x[1] - x[2]]
        prob = NLPProblem(f, g, h, [0.5, 0.5])
        diag = diagnose_problem(prob)
        @test diag isa ProblemDiagnostics
        @test diag.n_variables == 2
        @test diag.n_ineq == 1
        @test diag.n_eq == 1
        @test diag.constraint_nonlinearity < 1e-6
        @test diag.recommended_strategy == :line_search
        @test isempty(diag.warnings) || !any(startswith(w, "High constraint nonlinearity") for w in diag.warnings)
    end

    @testset "HS071 diagnostics" begin
        f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
        g(x) = [-prod(x) + 25.0]
        h(x) = [dot(x, x) - 40.0]
        prob = NLPProblem(f, g, h, [1.0, 5.0, 5.0, 1.0], ones(4), 5.0 * ones(4))
        diag = diagnose_problem(prob)
        @test diag.n_variables == 4
        # HS071 has bounds → n_ineq includes the bound-as-inequality rows
        @test diag.n_ineq == 1 + 2 * 4   # 1 ineq + 4 lower + 4 upper
        @test diag.n_eq == 1
        @test diag.initial_feasibility > 0  # sum(1^2+5^2+5^2+1^2)=52 vs 40 → violation = 12
        @test isfinite(diag.constraint_nonlinearity)
        @test diag.jacobian_condition > 0
    end

    @testset "HS092 → detects high nonlinearity" begin
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
        prob = NLPProblem(f, g, h, x0, -Inf * ones(n), Inf * ones(n))
        diag = diagnose_problem(prob)
        @test diag.constraint_nonlinearity > 0.3
        @test diag.recommended_strategy == :line_search_soc
        @test !isempty(diag.warnings)
        @test any(startswith(w, "High constraint nonlinearity") for w in diag.warnings)
    end

    @testset "Unconstrained problem" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        prob = NLPProblem(f, g, h, [-1.9, 2.0])
        diag = diagnose_problem(prob)
        @test diag.n_ineq == 0
        @test diag.n_eq == 0
        @test diag.constraint_nonlinearity == 0
        @test diag.initial_feasibility == 0
        @test diag.recommended_strategy == :line_search
    end

    @testset "Diagnostics via SQPOptions(diagnose=true)" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        result = sqp_solve(f, g, h, [-1.9, 2.0];
                           options = SQPOptions(diagnose = true))
        @test result.converged
        @test result.diagnostics !== nothing
        @test result.diagnostics.n_variables == 2
        @test result.n_soc_steps == 0  # SOC not enabled in this phase
    end

    @testset "Backward-compatible SQPResult constructor" begin
        # Verify that the old 6-arg constructor still works
        x = [1.0, 2.0]
        r = SQPResult(x, 3.14, 5, true, 1e-8, :converged)
        @test r.x == x
        @test r.objective == 3.14
        @test r.iterations == 5
        @test r.converged == true
        @test r.constraint_violation == 1e-8
        @test r.status == :converged
        @test r.diagnostics === nothing
        @test r.n_soc_steps == 0
    end
end
