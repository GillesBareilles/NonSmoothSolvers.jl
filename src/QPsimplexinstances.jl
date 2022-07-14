"""
    $TYPEDSIGNATURES

Difficult instances used by Kiwiel in his paper.
"""
function getKiwieltestpb(; n=5, Tf=Float64, b = 1e3, jₐ = 2)
    m = 2n+2
    P = Tf[j / (i + j) for i in 1:n, j in 1:m]
    Ĵ = 1:floor(n/2)

    ĵ = 1 + mod(jₐ-1, m)
    Ĵ = ĵ:ĵ+m
    if ĵ > n+2
        Ĵ = union(1:ĵ-n-2, ĵ:m)
    end
    x̄ = Tf[ j in Ĵ ? 1/(n+1) : 0 for j in 1:m]
    v̄ = minimum([- P[:, j]' * P * x̄ for j in 1:m])
    a = Tf[ -v̄ - P[:, j]' * P * x̄ + (j in Ĵ ? 0 : b) for j in 1:m]

    return P, a, x̄, Ĵ
end

"""
    $TYPEDSIGNATURES

Form QP without linear terms close to the ill-conditioned ones appearing in
the subproblem of the Gradient Sampling.
"""
function getbundlelikeproblem(;n=20, ngroups=4, nvecpergroup=6)
    basevecs = rand(n, ngroups)
    P = zeros(n, ngroups * nvecpergroup)
    for i = 1:ngroups
        for j = 1+(i-1)*nvecpergroup:i*nvecpergroup
            P[:, j] .= basevecs[:, i] + 1e-6 * randn(n)
        end
    end
    return P
end

