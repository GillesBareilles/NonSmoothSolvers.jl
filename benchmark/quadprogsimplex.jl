using NonSmoothSolvers
using StatProfilerHTML
using BenchmarkTools


function test1()
    Random.seed!(1645)
    P = getbundlelikeproblem(n=20, ngroups=4, nvecpergroup=6)
    a = zeros(6*4)
    NSS.quadprogsimplex(P, a; show_trace = false)

    function toto(n)
        for i in 1:n
            quadprogsimplex(P, a; show_trace = false)
        end
        return
    end

    @profilehtml toto(1000)
end

@benchmark isnextsystempsd($P, $P̂, $J, $l)


test1()
