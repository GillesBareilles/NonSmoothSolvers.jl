raw"""
    $TYPEDSIGNATURES

Compute one serious step of the proximal bundle algorithm:
$\arg\min_z F(z) + 0.5 * μ \|z - x\|^2$.
Return ...
"""
function bundlesubroutine(pb, μ::Tf, x::Vector{Tf}, σ::Tf; printlev=0) where Tf
    bundle = [bundlepoint(pb, x, x)]

    α̂minnormelt = nothing
    activebundle = nothing

    ϵ̂ = Inf
    p̂ = similar(x)
    ŝ = similar(x)


    # @show bundle
    (printlev>0) && @printf "it |B| |Bact|          |ŝ|        |ϵ̂|  tol(μ, σ)\n"
    it = 0
    while true
        r̂, p̂, α̂ = solve_χQP(pb, μ, x, bundle)
        ϵ̂ = F(pb, p̂) - r̂

        activebundle = form_active_bundle(pb, bundle, α̂, p̂, x, r̂)

        ŝ, α̂minnormelt = solve_γQP(activebundle)

        (printlev>0) && @printf "%2i  %2i  %2i        %.2e  % .2e  %.2e\n" it length(bundle) length(activebundle) norm(ŝ) ϵ̂ (σ / μ * norm(ŝ)^2)
        bundle = deepcopy(activebundle)

        if (ϵ̂ < σ / μ * norm(ŝ)^2) || norm(ŝ) < 10*eps(Tf)
            break
        end
        it += 1
        if it > 50
            @warn "too much null steps, exiting to serious step" it
            break
        end
    end

    Û = get_Uorthonormalbasis(activebundle, α̂minnormelt)
    return ϵ̂, p̂, ŝ, Û, (; nnullsteps = it)
end


function get_Uorthonormalbasis(activebundle, α̂minnormelt)
    n = length(first(activebundle).gᵢ)
    actindices = α̂minnormelt .> 0.
    nactgᵢ = sum(actindices)

    if nactgᵢ == 1
        return zeros(n, 0)
    end

    actgᵢs = map(bndl -> bndl.gᵢ, activebundle[actindices])

    V̂ = zeros(n, nactgᵢ-1)
    for i in 1:nactgᵢ-1
        V̂[:, i] .= actgᵢs[i] - actgᵢs[end]
    end

    Û = nullspace(V̂')
    return Û
end

function bundlepoint(pb, yᵢ, x)
    fᵢ = F(pb, yᵢ)
    gᵢ = ∂F_elt(pb, yᵢ)
    eᵢ = F(pb, x) - fᵢ - dot(gᵢ, x - yᵢ)

    return (; fᵢ, gᵢ, eᵢ, yᵢ)
end


function form_active_bundle(pb, bundle, α̂, p̂, x, r̂)
    actbndl = []
    for (i, bndlelt) in enumerate(bundle)
        αcond = α̂[i] > 1e-14
        modelcond = norm(r̂ - (F(pb, x) - bndlelt.eᵢ  + dot(bndlelt.gᵢ, p̂ - x))) < 1e-13
        # @show αcond, modelcond

        if αcond
            push!(actbndl, bndlelt)
        end
    end
    push!(actbndl, bundlepoint(pb, p̂, x))
    return actbndl
end

