raw"""
    $TYPEDSIGNATURES

Compute one serious step of the proximal bundle algorithm:
$\arg\min_z F(z) + 0.5 * μ \|z - x\|^2$.
Return ...
"""
function bundlesubroutine(pb, μ::Tf, x::Vector{Tf}, σ::Tf, ϵglobal, bundle; printlev=0) where Tf

    updatebundle!(bundle, pb, x)
    push!(bundle, bundlepoint(pb, x, x))

    α̂minnormelt = nothing

    ϵ̂ = Inf
    p̂ = similar(x)
    p̂prev = copy(p̂)
    ŝ = similar(x)

    phist = []

    (printlev>0) && @printf "it |B| |Bact|          |ŝ|        |ϵ̂|  tol(μ, σ)\n"
    it = 0
    while true
        r̂, p̂, α̂ = solve_χQP(pb, μ, x, bundle)
        ϵ̂ = F(pb, p̂) - r̂
        # @printf "%.2e  %.2e  %.2e   %.2e\n" F(pb, p̂) norm(p̂ - p̂prev) norm(ŝ) ϵ̂
        push!(phist, p̂)
        p̂prev = copy(p̂)

        # removing entries of the bundle corresponding to null coordinates of α̂
        deleteat!(bundle, findall(t->t==0, α̂))
        push!(bundle, bundlepoint(pb, p̂, x))

        ŝ, α̂minnormelt = solve_γQP(bundle)

        (printlev>0) && @printf "%2i  %2i  %2i        %.2e  % .2e  %.2e\n" it length(bundle) length(bundle) norm(ŝ) ϵ̂ (σ / μ * norm(ŝ)^2)

        if (ϵ̂ < (σ / μ) * norm(ŝ)^2) #|| max(norm(ŝ)^2, μ / σ * ϵ̂) < ϵglobal # NOTE: stopped here, this stopping criterion is unclear
            break
        end
        if  norm(ŝ) < 10*eps(Tf) && ϵ̂ < 10*eps(Tf)
            @warn "breaking here: both ŝ, the ϵ̂-subgradient of F at p̂, and ϵ̂, the error beetwen f(p̂) and the cutting planes model are null up to machine precision" it norm(ŝ) ϵ̂
            break
        end

        it += 1
        if it > 500
            @warn "too much null steps, exiting to serious step" it
            break
        end
    end

    Û = get_Uorthonormalbasis(bundle, α̂minnormelt)
    return ϵ̂, p̂, ŝ, Û, bundle, (; nnullsteps = it, phist)
end


function get_Uorthonormalbasis(activebundle, α̂minnormelt)
    n = length(first(activebundle).gᵢ)
    actindices = α̂minnormelt .> 0.
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

struct BundlePoint{Tf}
    fᵢ::Tf
    gᵢ::Vector{Tf}
    eᵢ::Tf
    yᵢ::Vector{Tf}
end

function updatebundle!(bundle, pb, x)
    Fx = F(pb, x)
    for (i, belt) in enumerate(bundle)
        e = Fx - belt.fᵢ - dot(belt.gᵢ, x - belt.yᵢ)
        bundle[i] = BundlePoint(belt.fᵢ, belt.gᵢ, e, belt.yᵢ)
    end
    return
end

function bundlepoint(pb, yᵢ, x)
    fᵢ = F(pb, yᵢ)
    gᵢ = ∂F_elt(pb, yᵢ)
    eᵢ = F(pb, x) - fᵢ - dot(gᵢ, x - yᵢ)

    return BundlePoint(fᵢ, gᵢ, eᵢ, yᵢ)
end
