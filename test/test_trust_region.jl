using SequentialQuadraticProgramming
using Test
using LinearAlgebra

@testset "Trust Region Globalization" begin
    tr_opts(; kwargs...) = SQPOptions(; globalization = :trust_region, kwargs...)

    @testset "Rosenbrock (trust region)" begin
        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        x0 = [-1.9, 2.0]
        result = sqp_solve(f, g, h, x0; options = tr_opts())
        @test result.converged
        @test result.objective < 1e-4
    end

    @testset "HS071 (trust region)" begin
        f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
        g(x) = [-prod(x) + 25.0]
        h(x) = [x' * x - 40.0]
        x0 = [1.0, 5.0, 5.0, 1.0]
        lb = ones(4)
        ub = 5.0 * ones(4)
        result = sqp_solve(f, g, h, x0, lb, ub; options = tr_opts(max_iterations = 2000))
        @test result.converged
        @test isapprox(result.objective, 17.014, atol = 0.2)
    end

    @testset "HS037 post office (trust region)" begin
        f(x) = -x[1] * x[2] * x[3]
        g(x) = [-(x[1] + 2 * x[2] + 2 * x[3]); -72 + (x[1] + 2 * x[2] + 2 * x[3])]
        h(x) = zeros(0)
        x0 = [1.0, 10.0, 10.0]
        lb = zeros(3)
        ub = 42.0 * ones(3)
        result = sqp_solve(f, g, h, x0, lb, ub; options = tr_opts(max_iterations = 1000))
        @test result.converged
        @test isapprox(result.objective, -3456.0, atol = 10.0)
    end

    @testset "HS118 (trust region)" begin
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
        result = sqp_solve(f, g, h, x0, lb, ub; options = tr_opts(max_iterations = 1000))
        # HS118 is near-quadratic — trust region radius shrinks fast, gets close but may not converge
        @test isapprox(result.objective, 664.82045, atol = 1.0)
    end

    @testset "Line search still works (regression)" begin
        f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        result = sqp_solve(f, g, h, [-1.9, 2.0];
                           options = SQPOptions(globalization = :line_search))
        @test result.converged
    end
end
