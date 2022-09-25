function φ(bundle, pb, y, xcenter, Fxcenter)
    val1 = maximum([F(pb, b.yᵢ) + dot(b.gᵢ, y - b.yᵢ) for b in bundle ])
    val2 = Fxcenter + maximum([-e.eᵢ + dot(e.gᵢ, y - xcenter) for e in bundle])
    if !isapprox(val1, val2; rtol = 1e-2)
        @warn "model values disagree here" val1 val2 length(bundle)
    end
    return val1
end

raw"""
    $TYPEDSIGNATURES

Compute one serious step of the proximal bundle algorithm:
$\arg\min_z F(z) + 0.5 * μ \|z - x\|^2$.
Return ...
"""
function bundlesubroutine(pb, μ::Tf, x::Vector{Tf}, σ::Tf, ϵglobal, bundle; printlev=0, testlevel=0) where Tf
    updatebundle!(bundle, pb, x)
    push!(bundle, bundlepoint(pb, x, x))

    ᾱ = nothing
    ᾱ_nullcoords = nothing

    ϵ̂ = Inf
    p̂ = similar(x)
    p̂prev = copy(p̂)
    ŝ = similar(x)

    phist = []
    Fx = F(pb, x)

    (printlev>0) && @printf "it F(p̂)       |B| |Bact|          |ŝ|        |ϵ̂|  tol(μ, σ)\n"
    it = 0
    while true
        ## FIXME bundle contains info on x?
        if findfirst(e -> norm(e.yᵢ - x) < 1e-14, bundle) === nothing
            push!(bundle, bundlepoint(pb, x, x))
        end

        ## NOTE: χ-QP
        α̂, α_nullcoords = solve_χQP(pb, μ, x, bundle)
        ĝ = sum(α̂[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(bundle))
        p̂ = x - (1/μ) * ĝ

        # TODO All remaining (active) linear models should be equal at p̂
        r̂ = Fx + maximum(-bndl.eᵢ + dot(bndl.gᵢ, p̂ - x) for (i, bndl) in enumerate(bundle)) # model value
        ϵ̂ = F(pb, p̂) - r̂                                                                   # model accuracy
        (testlevel > 0) && @assert isapprox(r̂, φ(bundle, pb, p̂, x, Fx); rtol = 1e-2)

        ## NOTE deleting non-active entries
        deleteat!(bundle, α_nullcoords)
        deleteat!(α̂, α_nullcoords)
        push!(bundle, bundlepoint(pb, p̂, x))

        ## NOTE: γ-QP
        ᾱ, ᾱ_nullcoords = solve_γQP(bundle)
        ŝ = sum(ᾱ[i] * bndlelt.gᵢ for (i, bndlelt) in enumerate(bundle))

        push!(phist, p̂)
        p̂prev = copy(p̂)

        (testlevel > 0) && check_bundle(pb, bundle, x, Fx)
        (printlev>0) && @printf "%2i %.4e  %2i  %2i        %.2e  % .2e  %.2e     %.2e\n" it F(pb, p̂) length(bundle) length(bundle) norm(ŝ) ϵ̂ (σ / μ * norm(ŝ)^2) ϵ̂

        ## NOTE: stopping criterion
        if (ϵ̂ < (σ / μ) * norm(ŝ)^2) || max(norm(ŝ)^2, μ / σ * ϵ̂) < ϵglobal
            if !(F(pb, p̂) - F(pb, x) ≤ -inv(2μ) * norm(ĝ)^2)
                @warn "Serious step does not provide theoretical sufficient decrease" F(pb, p̂) - F(pb, x) -inv(2μ) * norm(ĝ)^2
            end
            break
        end
        if  ϵ̂ < 1e2*eps(Tf) && norm(ŝ) < 1e2*eps(Tf)
            @warn "breaking here: both ŝ, the ϵ̂-subgradient of F at p̂, and ϵ̂, the error beetwen f(p̂) and the cutting planes model are null up to machine precision" it norm(ŝ) ϵ̂
            break
        end

        it += 1
        (it > 500) && throw(error("too much null steps, exiting to serious step"))
    end

    Û = get_Uorthonormalbasis(bundle, ᾱ, ᾱ_nullcoords)
    return ϵ̂, p̂, ŝ, Û, bundle, (; nnullsteps = it, phist)
end


function get_Uorthonormalbasis(activebundle, α̂minnormelt, α_nullcoords)
    n = length(first(activebundle).gᵢ)
    actindices = (!).(α_nullcoords)
    nactgᵢ = sum(actindices)

    if nactgᵢ == 1
        return zeros(n, 0)
    end

    actgᵢs = map(belt -> belt.gᵢ, activebundle[actindices])

    V̂ = zeros(n, nactgᵢ-1)
    for i in 1:nactgᵢ-1
        V̂[:, i] .= actgᵢs[i] - actgᵢs[end]
    end

    Û = nullspace(V̂')
    return Û
end

