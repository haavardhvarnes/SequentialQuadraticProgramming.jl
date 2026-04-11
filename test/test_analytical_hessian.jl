using SequentialQuadraticProgramming
using Test
using LinearAlgebra

@testset "Analytical Hessian (Phase 9.0)" begin

    @testset "modify_eigenvalues! :abs method" begin
        # Indefinite matrix: one negative eigenvalue, one positive
        H = [1.0 2.0; 2.0 1.0]    # eigenvalues are -1 and 3
        H_orig = copy(H)
        _, corrected, λmin = SequentialQuadraticProgramming.modify_eigenvalues!(H;
                                method = :abs)
        @test corrected
        @test λmin ≈ -1.0
        @test isposdef(H)
        # Absolute-value method: eigenvalues become 1 and 3 (preserving magnitudes)
        λ_new = sort(eigvals(H))
        @test isapprox(λ_new[1], 1.0, atol = 1e-10)
        @test isapprox(λ_new[2], 3.0, atol = 1e-10)
    end

    @testset "modify_eigenvalues! :clip method" begin
        # Same indefinite matrix
        H = [1.0 2.0; 2.0 1.0]
        _, corrected, _ = SequentialQuadraticProgramming.modify_eigenvalues!(H;
                            method = :clip, floor = 1e-4)
        @test corrected
        @test isposdef(H)
        # :clip leaves +3 alone, replaces -1 with 1e-4
        λ_new = sort(eigvals(H))
        @test isapprox(λ_new[1], 1e-4, atol = 1e-10)
        @test isapprox(λ_new[2], 3.0, atol = 1e-10)
    end

    @testset "modify_eigenvalues! on PD matrix is a no-op" begin
        H = [2.0 0.0; 0.0 3.0]
        H_orig = copy(H)
        _, corrected, λmin = SequentialQuadraticProgramming.modify_eigenvalues!(H)
        @test !corrected
        @test λmin ≈ 2.0
        @test H ≈ H_orig
    end

    @testset "Default :bfgs strategy matches v0.8.2 behavior" begin
        # Rosenbrock — same iteration count as v0.8.2
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [-1.9, 2.0])
        @test r.converged
        @test r.iterations == 24
        @test r.n_hessian_corrections == 0

        # HS071 — same iteration count as v0.8.2
        f2(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
        g2(x) = [-prod(x) + 25.0]
        h2(x) = [dot(x, x) - 40.0]
        r2 = sqp_solve(f2, g2, h2, [1.0, 5.0, 5.0, 1.0], ones(4), 5.0 * ones(4);
                       options = SQPOptions(max_iterations = 2000))
        @test r2.converged
        @test r2.iterations == 6
        @test r2.n_hessian_corrections == 0
    end

    @testset "Explicit :analytical strategy on Rosenbrock" begin
        f(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        g(x) = zeros(0)
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, [-1.9, 2.0];
                      options = SQPOptions(hessian_strategy = :analytical))
        @test r.converged
        @test r.objective < 1e-6
        # Rosenbrock's Hessian is PD everywhere along the iterates —
        # no correction needed
        @test r.n_hessian_corrections == 0
    end

    @testset "Explicit :analytical fixes HS015 (currently @test_broken)" begin
        x0 = [-2.0, 1.0]
        f(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
        g(x) = [-x[1] * x[2] + 1; -(x[1] + x[2]^2)]
        h(x) = zeros(0)
        lb = [-Inf, -Inf]
        ub = [0.5, Inf]
        r = sqp_solve(f, g, h, x0, lb, ub;
                      options = SQPOptions(hessian_strategy = :analytical,
                                           max_iterations = 100))
        @test r.converged
        @test isapprox(r.objective, 306.5, atol = 1.0)
        @test r.n_hessian_corrections > 0
    end

    @testset "Explicit :analytical fixes HS108 (currently @test_broken)" begin
        x0 = ones(9)
        f(x) = -0.5 * (x[1] * x[4] - x[2] * x[3] + x[3] * x[9] - x[5] * x[9] +
                       x[5] * x[8] - x[6] * x[7])
        g(x) = [-(1 - x[3]^2 - x[4]^2);
                -(1 - x[5]^2 - x[6]^2);
                -(1 - x[9]^2);
                -(1 - x[1]^2 - (x[2] - x[9])^2);
                -(1 - (x[1] - x[5])^2 - (x[2] - x[6])^2);
                -(1 - (x[1] - x[7])^2 - (x[2] - x[8])^2);
                -(1 - (x[3] - x[5])^2 - (x[4] - x[6])^2);
                -(1 - (x[3] - x[7])^2 - (x[4] - x[8])^2);
                -(1 - x[7]^2 - (x[8] - x[9])^2);
                -(x[1] * x[4] - x[2] * x[3]);
                -x[3] * x[9];
                x[5] * x[9];
                -(x[5] * x[8] - x[6] * x[7]);
                -x[9]]
        h(x) = zeros(0)
        r = sqp_solve(f, g, h, x0, zeros(9), 80000.0 * ones(9);
                      options = SQPOptions(hessian_strategy = :analytical,
                                           max_iterations = 500))
        @test r.converged
        @test isapprox(r.objective, -0.8660254038, atol = 0.01)
        @test r.n_hessian_corrections > 0
    end

    @testset "Explicit :analytical dramatically speeds up HS105" begin
        y = zeros(235)
        y[1] = 95; y[2] = 105
        y[3:6] .= 110; y[7:10] .= 115
        y[11:25] .= 120; y[26:40] .= 125
        y[41:55] .= 130; y[56:68] .= 135
        y[69:89] .= 140; y[90:101] .= 145
        y[102:118] .= 150; y[119:122] .= 155
        y[123:142] .= 160; y[143:150] .= 165
        y[151:167] .= 170; y[168:175] .= 175
        y[176:181] .= 180; y[182:187] .= 185
        y[188:194] .= 190; y[195:198] .= 195
        y[199:201] .= 200; y[202:204] .= 205
        y[205:212] .= 210; y[213] = 215
        y[214:219] .= 220; y[220:224] .= 230
        y[225] = 235; y[226:232] .= 240
        y[233] = 245; y[234:235] .= 250
        a(x) = x[1] ./ x[6] .* exp.(-(y .- x[3]) .^ 2 ./ (2 * x[6]^2))
        b(x) = x[2] ./ x[7] .* exp.(-(y .- x[4]) .^ 2 ./ (2 * x[7]^2))
        c(x) = (1 - x[2] - x[1]) / x[8] .* exp.(-(y .- x[5]) .^ 2 / (2 * x[8]^2))
        d(x) = max.(a(x) .+ b(x) .+ c(x), 1e-9)
        f(x) = -sum(log.((d(x)) ./ sqrt(2 * pi)))
        g(x) = [-(1 - x[1] - x[2])]
        h(x) = zeros(0)
        x0 = [0.1, 0.2, 100.0, 125.0, 175.0, 11.2, 13.2, 15.8]
        lb = zeros(8); lb[1:2] .= 0.001; lb[3] = 100.0; lb[4] = 130.0; lb[5] = 170.0; lb[6:8] .= 5.0
        ub = zeros(8); ub[1:2] .= 0.499; ub[3] = 180.0; ub[4] = 210.0; ub[5] = 240.0; ub[6:8] .= 21.9
        r = sqp_solve(f, g, h, x0, lb, ub;
                      options = SQPOptions(hessian_strategy = :analytical,
                                           max_iterations = 100))
        @test r.converged
        @test isapprox(r.objective, 1156.48, atol = 2.0)
        # v0.8.2 needed 545 iters; analytical should do it in well under 50
        @test r.iterations < 50
    end

    @testset "Large problem: :auto picks :bfgs" begin
        # n = 60 problem — :auto should NOT use analytical (auto_hessian_max_n = 50)
        f(x) = sum(x .^ 2) + 0.01 * sum(i * x[i] for i in 1:length(x))
        g(x) = zeros(0)
        h(x) = [sum(x) - 1.0]
        x0 = 0.01 * ones(60)
        r = sqp_solve(f, g, h, x0;
                      options = SQPOptions(hessian_strategy = :auto,
                                           max_iterations = 200))
        @test r.converged
        # auto picks bfgs for n=60 (> auto_hessian_max_n=50)
        @test r.n_hessian_corrections == 0
    end

    @testset "Backward-compatible 6-arg SQPResult ctor still works" begin
        x = [1.0, 2.0]
        r = SQPResult(x, 3.14, 5, true, 1e-8, :converged)
        @test r.n_hessian_corrections == 0
    end
end
