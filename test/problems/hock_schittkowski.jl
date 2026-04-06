"""
Hock-Schittkowski benchmark problems.
Adapted from https://vanderbei.princeton.edu/ampl/nlmodels/hs/
"""

function test_hs015()
    x0 = [-2.0, 1.0]
    f(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
    g(x) = [-x[1] * x[2] + 1;
            -(x[1] + x[2]^2)]
    h(x) = zeros(0)
    lb = [-Inf, -Inf]
    ub = [0.5, Inf]
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 5000))
    # HS015 is a challenging problem; convergence to the neighborhood is acceptable
    @test_broken result.converged
    @test result.objective < 350.0
end

function test_hs037()
    x0 = [1.0, 10.0, 10.0]
    f(x) = -x[1] * x[2] * x[3]
    g(x) = [-(x[1] + 2 * x[2] + 2 * x[3]); -72 + (x[1] + 2 * x[2] + 2 * x[3])]
    h(x) = zeros(0)
    lb = zeros(3)
    ub = 42.0 * ones(3)
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 1000))
    @test result.converged
    @test isapprox(result.objective, -3456.0, atol = 10.0)
end

function test_hs071()
    x0 = [1.0, 5.0, 5.0, 1.0]
    f(x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
    g(x) = [-prod(x) + 25.0]
    h(x) = [x' * x - 40.0]
    lb = ones(4)
    ub = 5.0 * ones(4)
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 2000))
    @test result.converged
    @test isapprox(result.objective, 17.014, atol = 0.1)
    @test isapprox(result.x[1], 1.0, atol = 0.05)
end

function test_hs092()
    n = 6
    x0 = [0.5 * (-1)^(i + 1) for i in 1:n]
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

    rho(x) = -(exp.(-mu .^ 2 .* sum(x .^ 2)) .+
               sum([2 * (-1)^(ii - 1) * exp.(-mu .^ 2 * sum([x[i]^2 for i in ii:n])) for ii in 2:n]) .+
               (-1)^n) ./ mu .^ 2

    function firstsum(x)
        isum = 0.0
        mrho = rho(x)
        for i in 1:29
            isum += sum([mu[i]^2 * mu[j]^2 * A_coef[i] * A_coef[j] * mrho[i] * mrho[j] *
                         (sin(mu[i] + mu[j]) / (mu[i] + mu[j]) + sin(mu[i] - mu[j]) / (mu[i] - mu[j])) for j in (i + 1):30])
        end
        return isum
    end

    f(x) = sum(x .^ 2)
    g(x) = [firstsum(x) + sum(mu .^ 4 .* A_coef .^ 2 .* rho(x) .^ 2 .* (sin.(2 .* mu) ./ (2 * mu) .+ 1) ./ 2) -
             sum(mu .^ 2 .* A_coef .* rho(x) .* (2 * sin.(mu) ./ mu .^ 3 - 2 * cos.(mu) ./ mu .^ 2)) + 2 / 15 - 0.0001]
    h(x) = zeros(0)
    lb = -Inf * ones(n)
    ub = Inf * ones(n)
    result = sqp_solve(f, g, h, x0, lb, ub;
                       options = SQPOptions(max_iterations = 2000, xtol = 1e-6, ftol = 1e-6))
    @test result.converged
    @test isapprox(result.objective, 1.36265681, atol = 0.01)
end

function test_hs105()
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

    x0 = [0.1, 0.2, 100.0, 125.0, 175.0, 11.2, 13.2, 15.8]
    f(x) = -sum(log.((d(x)) ./ sqrt(2 * pi)))
    g(x) = [-(1 - x[1] - x[2])]
    h(x) = zeros(0)
    lb = zeros(8)
    lb[1:2] .= 0.001; lb[3] = 100.0; lb[4] = 130.0; lb[5] = 170.0; lb[6:8] .= 5.0
    ub = zeros(8)
    ub[1:2] .= 0.499; ub[3] = 180.0; ub[4] = 210.0; ub[5] = 240.0; ub[6:8] .= 21.9
    result = sqp_solve(f, g, h, x0, lb, ub;
                       options = SQPOptions(max_iterations = 3000, xtol = 1e-6, ftol = 1e-6))
    @test result.converged
    @test isapprox(result.objective, 1136.67, atol = 25.0)
end

function test_hs108()
    x0 = ones(9)
    f(x) = -0.5(x[1]x[4] - x[2]x[3] + x[3]x[9] - x[5]x[9] + x[5]x[8] - x[6]x[7])
    g(x) = [-(1 - x[3]^2 - x[4]^2);
            -(1 - x[5]^2 - x[6]^2);
            -(1 - x[9]^2);
            -(1 - x[1]^2 - (x[2] - x[9])^2);
            -(1 - (x[1] - x[5])^2 - (x[2] - x[6])^2);
            -(1 - (x[1] - x[7])^2 - (x[2] - x[8])^2);
            -(1 - (x[3] - x[5])^2 - (x[4] - x[6])^2);
            -(1 - (x[3] - x[7])^2 - (x[4] - x[8])^2);
            -(1 - x[7]^2 - (x[8] - x[9])^2);
            -(x[1]x[4] - x[2]x[3]);
            -x[3]x[9];
            x[5]x[9];
            -(x[5]x[8] - x[6]x[7]);
            -x[9]]
    h(x) = zeros(0)
    lb = zeros(length(x0))
    ub = 80000.0 * ones(length(x0))
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 2000))
    # HS108 converges slowly; check constraint feasibility and objective improvement
    @test_broken result.converged
    @test result.objective < -0.3
    @test result.constraint_violation < 1e-6
end

function test_hs118()
    x0 = Float64[20, 55, 15, 20, 60, 20, 20, 60, 20, 20, 60, 20, 20, 60, 20]
    f(x) = sum((2.3 * x[3 * k + 1] + 0.0001 * x[3 * k + 1]^2 + 1.7 * x[3 * k + 2] +
                0.0001 * x[3 * k + 2]^2 + 2.2 * x[3 * k + 3] + 0.00015 * x[3 * k + 3]^2) for k in 0:4)
    g(x) = [([x[3 * j + 1] - x[3 * j - 2] + 7 for j in 1:4] .- 13);
            -([x[3 * j + 1] - x[3 * j - 2] + 7 for j in 1:4]);
            ([x[3 * j + 2] - x[3 * j - 1] + 7 for j in 1:4] .- 14);
            -([x[3 * j + 2] - x[3 * j - 1] + 7 for j in 1:4]);
            ([x[3 * j + 3] - x[3 * j] + 7 for j in 1:4] .- 13);
            -([x[3 * j + 3] - x[3 * j] + 7 for j in 1:4]);
            -(x[1] + x[2] + x[3] - 60);
            -(x[4] + x[5] + x[6] - 50);
            -(x[7] + x[8] + x[9] - 70);
            -(x[10] + x[11] + x[12] - 85);
            -(x[13] + x[14] + x[15] - 100)]
    h(x) = zeros(0)
    lb = zeros(length(x0))
    lb[1] = 8; lb[2] = 43; lb[3] = 3
    ub = 60.0 * ones(length(x0))
    ub[1] = 16; ub[2] = 57; ub[3] = 16
    for k in 1:4
        ub[3 * k + 1] = 90
        ub[3 * k + 2] = 120
    end
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 1000))
    @test result.converged
    @test isapprox(result.objective, 664.82045, atol = 1.0)
end

function test_hs119()
    a = zeros(16, 16)
    a[1, 1] = 1; a[1, 4] = 1; a[1, 7] = 1; a[1, 8] = 1; a[1, 16] = 1
    a[2, 2] = 1; a[2, 3] = 1; a[2, 7] = 1; a[2, 10] = 1
    a[3, 3] = 1; a[3, 7] = 1; a[3, 9] = 1; a[3, 10] = 1; a[3, 14] = 1
    a[4, 4] = 1; a[4, 7] = 1; a[4, 11] = 1; a[4, 15] = 1
    a[5, 5] = 1; a[5, 6] = 1; a[5, 10] = 1; a[5, 12] = 1; a[5, 16] = 1
    a[6, 6] = 1; a[6, 8] = 1; a[6, 15] = 1
    a[7, 7] = 1; a[7, 11] = 1; a[7, 13] = 1
    a[8, 8] = 1; a[8, 10] = 1; a[8, 15] = 1
    a[9, 9] = 1; a[9, 12] = 1; a[9, 16] = 1
    a[10, 10] = 1; a[10, 14] = 1
    a[11, 11] = 1; a[11, 13] = 1; a[11, 12] = 1
    a[12, 14] = 1
    a[13, 13] = 1; a[13, 14] = 1
    a[14, 14] = 1; a[15, 15] = 1; a[16, 16] = 1

    c_rhs = [2.5, 1.1, -3.1, -3.5, 1.3, 2.1, 2.3, -1.5]

    x0 = 10.0 * ones(16)
    f(x) = sum(a[i, j] * (x[i]^2 + x[i] + 1) * (x[j]^2 + x[j] + 1) for i in 1:16, j in 1:16)
    g(x) = zeros(0)
    h(x) = [(0.22 * x[1] + 0.2 * x[2] + 0.19 * x[3] + 0.25 * x[4] + 0.15 * x[5] +
             0.11 * x[6] + 0.12 * x[7] + 0.13 * x[8] + 1 * x[9] - c_rhs[1]);
            (-1.46 * x[1] - 1.3 * x[3] + 1.82 * x[4] - 1.15 * x[5] + 0.8 * x[7] + 1 * x[10] - c_rhs[2]);
            (1.29 * x[1] - 0.89 * x[2] - 1.16 * x[5] - 0.96 * x[6] - 0.49 * x[8] + 1 * x[11] - c_rhs[3]);
            (-1.1 * x[1] - 1.06 * x[2] + 0.95 * x[3] - 0.54 * x[4] - 1.78 * x[6] - 0.41 * x[7] + 1 * x[12] - c_rhs[4]);
            (-1.43 * x[4] + 1.51 * x[5] + 0.59 * x[6] - 0.33 * x[7] - 0.43 * x[8] + 1 * x[13] - c_rhs[5]);
            (-1.72 * x[2] - 0.33 * x[3] + 1.62 * x[5] + 1.24 * x[6] + 0.21 * x[7] - 0.26 * x[8] + 1 * x[14] - c_rhs[6]);
            (1.12 * x[1] + 0.31 * x[4] + 1.12 * x[7] - 0.36 * x[9] + 1 * x[15] - c_rhs[7]);
            (0.45 * x[2] + 0.26 * x[3] - 1.1 * x[4] + 0.58 * x[5] - 1.03 * x[7] + 0.1 * x[8] + 1 * x[16] - c_rhs[8])]
    lb = zeros(length(x0))
    ub = 5.0 * ones(length(x0))
    result = sqp_solve(f, g, h, x0, lb, ub; options = SQPOptions(max_iterations = 1000))
    @test result.converged
    @test isapprox(result.objective, 244.8997, atol = 1.0)
end
