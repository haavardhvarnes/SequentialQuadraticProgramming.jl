using SequentialQuadraticProgramming
using Test
using LinearAlgebra

@testset "Second-Order Correction" begin

    @testset "SOC is a no-op on well-behaved problems" begin
        # Rosenbrock: no constraints, SOC cannot fire.
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r0 = sqp_solve(f, g, h, [-1.9, 2.0])
        r1 = sqp_solve(f, g, h, [-1.9, 2.0]; options = SQPOptions(use_soc = true))
        @test r0.converged == r1.converged
        @test r0.iterations == r1.iterations
        @test r1.n_soc_steps == 0
    end

    @testset "SOC does not break HS071" begin
        f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
        g(x) = [-prod(x) + 25.0]
        h(x) = [dot(x, x) - 40.0]
        x0 = [1.0, 5.0, 5.0, 1.0]
        lb = ones(4); ub = 5.0 * ones(4)
        r = sqp_solve(f, g, h, x0, lb, ub;
                      options = SQPOptions(use_soc = true, max_iterations = 2000))
        @test r.converged
        @test isapprox(r.objective, 17.014, atol = 0.1)
    end

    @testset "SOC does not break HS037 (pure inequality + bounds)" begin
        f(x) = -x[1] * x[2] * x[3]
        g(x) = [-(x[1] + 2 * x[2] + 2 * x[3]);
                -72 + (x[1] + 2 * x[2] + 2 * x[3])]
        h(x) = zeros(0)
        x0 = [1.0, 10.0, 10.0]
        r = sqp_solve(f, g, h, x0, zeros(3), 42.0 * ones(3);
                      options = SQPOptions(use_soc = true, max_iterations = 1000))
        @test r.converged
        @test isapprox(r.objective, -3456.0, atol = 10.0)
    end

    @testset "SOC does not break HS118" begin
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
                      options = SQPOptions(use_soc = true, max_iterations = 1000))
        @test r.converged
        @test isapprox(r.objective, 664.82045, atol = 1.0)
    end

    @testset "SOC does not break Maratos" begin
        f(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
        g(x) = zeros(0)
        h(x) = [x[1]^2 + x[2]^2 - 1.0]
        for t_deg in (10, 45, 60)
            t = t_deg * π / 180
            x0 = [cos(t), sin(t)]
            r0 = sqp_solve(f, g, h, x0; options = SQPOptions(use_soc = false))
            r1 = sqp_solve(f, g, h, x0; options = SQPOptions(use_soc = true))
            @test r0.converged
            @test r1.converged
            @test isapprox(r0.objective, r1.objective, atol = 1e-4)
        end
    end

    @testset "SOC correction QP — solve_qp_correction direct test" begin
        # Set up a tiny test: linearize a nonlinear equality constraint
        # and verify that solve_qp_correction drives the residual to zero.
        import SequentialQuadraticProgramming: solve_qp_correction
        H = [2.0 0.0; 0.0 2.0]
        J = [1.0 1.0]              # single equality constraint: x1 + x2 = 0
        c_trial = [0.5]            # violated by 0.5
        df_x = [0.0, 0.0]
        d_c = solve_qp_correction(COSMOQPSolver(), H, c_trial, J, df_x, 0)  # 0 ineq rows
        @test d_c !== nothing
        # Verify J*d_c ≈ -c_trial (equality constraint satisfied)
        @test isapprox((J * d_c)[1], -c_trial[1], atol = 1e-6)
    end

    @testset "Non-COSMO solver with use_soc=true emits warning" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        # Clarabel is loaded via test/Project.toml, so ClarabelQPSolver is available
        @test_logs (:warn, r"Second-Order Correction.*only supported with COSMOQPSolver"i) begin
            r = sqp_solve(f, g, h, [-1.9, 2.0];
                          qp_solver = ClarabelQPSolver(),
                          options = SQPOptions(use_soc = true))
            @test r.converged
            @test r.n_soc_steps == 0
        end
    end
end
