using SequentialQuadraticProgramming
using JuMP
import MathOptInterface as MOI
using Test
using LinearAlgebra

@testset "MOI / JuMP Integration" begin
    @testset "JuMP HS071" begin
        model = Model(SequentialQuadraticProgramming.Optimizer)
        set_silent(model)
        set_optimizer_attribute(model, "max_iterations", 2000)

        @variable(model, 1 <= x[i = 1:4] <= 5, start = [1, 5, 5, 1][i])
        @NLobjective(model, Min, x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3])
        @NLconstraint(model, x[1] * x[2] * x[3] * x[4] >= 25)
        @NLconstraint(model, x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 == 40)

        optimize!(model)

        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test isapprox(objective_value(model), 17.014, atol = 0.1)
        @test isapprox(value(x[1]), 1.0, atol = 0.05)
    end

    @testset "JuMP HS037 (post office)" begin
        model = Model(SequentialQuadraticProgramming.Optimizer)
        set_silent(model)

        @variable(model, 0 <= x[1:3] <= 42, start = 10)
        @NLobjective(model, Min, -x[1] * x[2] * x[3])
        @constraint(model, x[1] + 2 * x[2] + 2 * x[3] >= 0)
        @constraint(model, x[1] + 2 * x[2] + 2 * x[3] <= 72)

        optimize!(model)

        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test isapprox(objective_value(model), -3456.0, atol = 10.0)
        @test isapprox(value(x[1]), 24.0, atol = 1.0)
    end

    @testset "JuMP unconstrained Rosenbrock" begin
        model = Model(SequentialQuadraticProgramming.Optimizer)
        set_silent(model)

        @variable(model, x[1:2], start = 0)
        set_start_value(x[1], -1.9)
        set_start_value(x[2], 2.0)
        @NLobjective(model, Min, (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)

        optimize!(model)

        @test termination_status(model) == MOI.LOCALLY_SOLVED
        # MOI path uses FiniteDiff (no ForwardDiff through evaluator), so slightly less precise
        @test isapprox(objective_value(model), 0.0, atol = 1e-2)
        @test isapprox(value(x[1]), 1.0, atol = 0.05)
        @test isapprox(value(x[2]), 1.0, atol = 0.05)
    end

    @testset "JuMP maximization" begin
        model = Model(SequentialQuadraticProgramming.Optimizer)
        set_silent(model)

        @variable(model, 0 <= x[1:2] <= 10, start = 1)
        @NLobjective(model, Max, -(x[1] - 3)^2 - (x[2] - 4)^2)

        optimize!(model)

        @test termination_status(model) == MOI.LOCALLY_SOLVED
        @test isapprox(value(x[1]), 3.0, atol = 0.1)
        @test isapprox(value(x[2]), 4.0, atol = 0.1)
    end

    @testset "Solver name and attributes" begin
        opt = SequentialQuadraticProgramming.Optimizer()
        @test MOI.get(opt, MOI.SolverName()) == "SequentialQuadraticProgramming"
        @test MOI.is_empty(opt)

        MOI.set(opt, MOI.Silent(), true)
        @test MOI.get(opt, MOI.Silent()) == true
    end
end
