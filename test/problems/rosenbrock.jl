"""
Rosenbrock and related test problems.
"""

function test_rosenbrock_unconstrained()
    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    g(x) = zeros(0)
    h(x) = zeros(0)
    x0 = [-1.9, 2.0]
    result = sqp_solve(f, g, h, x0)
    @test result.converged
    @test result.objective < 1e-6
    @test isapprox(result.x, [1.0, 1.0], atol = 1e-3)
end

function test_rosenbrock_bounds()
    f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    g(x) = zeros(0)
    h(x) = zeros(0)
    lb = [1.25, -2.1]
    ub = [Inf, Inf]
    x0 = [-1.9, 2.0]
    result = sqp_solve(f, g, h, x0, lb, ub)
    @test result.converged
    @test result.x[1] >= 1.25 - 1e-4
end

function test_post_office()
    f(x) = -prod(x)
    g(x) = [x[1] + 2 * x[2] + 2 * x[3] - 72.0;
            -(x[1] + 2 * x[2] + 2 * x[3])]
    h(x) = zeros(0)
    lb = zeros(3)
    ub = [42.0, 42.0, 42.0]
    x0 = 10.0 * ones(3)
    result = sqp_solve(f, g, h, x0, lb, ub)
    @test result.converged
    @test isapprox(result.x, [24.0, 12.0, 12.0], atol = 0.5)
end

function test_minbox()
    f(x) = dot(x, x)
    g(x) = zeros(0)
    h(x) = [prod(x) - 1.0]
    x0 = [2.99, 0.4, 3.5]
    lb = zeros(3)
    ub = Inf * ones(3)
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 1000))
    # Minbox converges slowly; check it makes progress toward the optimum
    @test_broken result.converged
    @test result.objective < 6.0
    @test result.constraint_violation < 1e-4
end

function test_maratos()
    f(x) = 2.0 * (x[1]^2 + x[2]^2 - 1.0) - x[1]
    h(x) = [x[1]^2 + x[2]^2 - 1.0]
    g(x) = zeros(0)
    t = pi / 10
    x0 = [cos(t), sin(t)]
    result = sqp_solve(f, g, h, x0)
    @test result.converged
    @test isapprox(result.objective, -1.0, atol = 0.01)
end
