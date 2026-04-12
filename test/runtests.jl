using SequentialQuadraticProgramming
using Test
using LinearAlgebra

include("problems/rosenbrock.jl")
include("problems/hock_schittkowski.jl")

@testset "SequentialQuadraticProgramming.jl" begin
    @testset "Rosenbrock" begin
        test_rosenbrock_unconstrained()
        test_rosenbrock_bounds()
        test_maratos()
    end

    @testset "Post office / minbox" begin
        test_post_office()
        test_minbox()
    end

    @testset "Hock-Schittkowski" begin
        @testset "HS015" begin test_hs015() end
        @testset "HS037" begin test_hs037() end
        @testset "HS071" begin test_hs071() end
        @testset "HS108" begin test_hs108() end
        @testset "HS118" begin test_hs118() end
        @testset "HS119" begin test_hs119() end
    end

    @testset "Hock-Schittkowski (harder)" begin
        @testset "HS092" begin test_hs092() end
        @testset "HS105" begin test_hs105() end
    end
end

include("test_moi.jl")
include("test_clarabel.jl")
include("test_trust_region.jl")
include("test_highs.jl")
include("test_diagnostics.jl")
include("test_soc.jl")
include("test_numerical_safeguards.jl")
include("test_analytical_hessian.jl")
include("test_filter.jl")
