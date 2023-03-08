raw"""
    $TYPEDSIGNATURES

Compute one serious step of the proximal bundle algorithm:
$\arg\min_z F(z) + 0.5 * μ \|z - x\|^2$.
Return ...
"""
function bundlesubroutine!(bundle::Bundle{Tf}, pb, μ::Tf, x::Vector{Tf}, σ::Tf, ϵglobal, haveinv; printlev=0, testlevel=0, nullstepshist = []) where Tf
    printlev = 0

    (printlev > 2) && printstyled(" === Bundle subroutine computation === \n", color = :blue)
    (printlev > 2) && @show μ, σ, haveinv
    (printlev > 2) && @show x

    ᾱ = nothing
    ᾱ_nullcoords = nothing

    ϵ̂ = Inf
    p̂ = similar(x)
    Fp̂ = Tf(0)
    gp̂ = similar(x)
    p̂prev = similar(x)
    p̂prev .= p̂
    ŝ = similar(x)

    phist = []
    Fx = F(pb, x)

    subroutinestatus = :Unfinished

    (printlev>3) && @printf "it F(p̂)       |B| |Bact|          |ŝ|        |ϵ̂|  tol(μ, σ)\n"
    it = 1
    while true
        printlev > 3 && println("\n----------------- it: $it")
        printlev > 3 && display(bundle)

        ## NOTE χ-QP
        ## bsolve procedure
        α̂, α_nullcoords = solve_χQP(pb, μ, x, bundle) # dual
        α̂_nonullcoords = findall(t -> t > 1e-15, α̂)

        ĝ = similar(x) # greg
        ĝ .= 0
        for i in α̂_nonullcoords
            ĝ .+= α̂[i] * bundle.bpts[i].gᵢ
        end
        p̂ = x - (1/μ) * ĝ # primal = phat

        # @show α̂_nonullcoords, ᾱ_nullcoords, α̂


        ϵ = sum(α̂[i] * bundle.bpts[i].eᵢ for i in α̂_nonullcoords)
        Δ = ϵ + 1/(2μ) * norm(ĝ)^2
        Δv = ϵ + 1/(μ) * norm(ĝ)^2

        # NOTE Bundle compression: keep only elements active in χ-QP
        deleteat!(bundle.bpts, α_nullcoords)
        deleteat!(α̂, α_nullcoords)
        printlev > 1 && @show p̂
        printlev > 1 && display(bundle)
        # done with bsolve procedure

        printlev > 3 && println(" xxx augmenting bundle")
        Fp̂, gp̂ = blackbox_oracle(pb, p̂)
        push!(nullstepshist, p̂)
        push!(bundle.bpts, BundlePoint(
            Fp̂,
            gp̂,
            bundle.Frefpoint - Fp̂ -1/μ*dot(gp̂, ĝ), # HACK Why ??
            p̂
        ))

        r̂ = bundle.Frefpoint - Δv
        ϵ̂ = Fp̂ - r̂

        printlev > 3 && display(bundle)
        printlev > 3 && @show ϵ̂, Fp̂, r̂, Δv

        ## NOTE: γ-QP
        ᾱ, ᾱ_nullcoords = solve_γQP(bundle) # dualsh
        ŝ = zeros(size(x))
        for (i, bndlelt) in enumerate(bundle.bpts)
            ŝ .+= ᾱ[i] .* bndlelt.gᵢ
        end

        printlev > 3 && @show ŝ

        push!(phist, p̂)
        p̂prev = copy(p̂)

        (testlevel > 0) && check_bundle(pb, bundle, x, Fx)
        # (printlev>0) && @printf "%2i %.4e  %2i  %2i        %.2e  % .2e  %.2e     %.2e\n" it F(pb, p̂) length(bundle) length(bundle) norm(ŝ) ϵ̂ (σ / μ * norm(ŝ)^2) ϵ̂

        ## NOTE: stopping criterion for global optimal point
        isglobalopt = (ϵ̂ + haveinv * norm(ŝ)^2 ≤ ϵglobal^2) || max(norm(ŝ)^2, μ/σ*ϵ̂) < max(1e-9, ϵglobal^2)
        if isglobalopt
            @info "Found optimal point: " Fp̂
            @warn "missing part of code here (solution polish?)" # TODO ask Claudia
            subroutinestatus = :ApproxMinimizerFound
            break
        end

        ## NOTE test to exit null steps
        seriousstep = ϵ̂ < σ/μ * norm(ŝ)^2
        if seriousstep
            # if !(F(pb, p̂) - F(pb, x) ≤ -inv(2μ) * norm(ĝ)^2)
            #     @warn "Serious step does not provide theoretical sufficient decrease" F(pb, p̂) - F(pb, x) -inv(2μ) * norm(ĝ)^2
            #     throw(ErrorException("In null step sequence, weird behavior."))
            # end
            subroutinestatus = :SeriousStepFound
            break
        end
        # if  ϵ̂ < 1e2*eps(Tf) && norm(ŝ) < 1e2*eps(Tf)
        #     @warn "breaking here: both ŝ, the ϵ̂-subgradient of F at p̂, and ϵ̂, the error beetwen f(p̂) and the cutting planes model are null up to machine precision" it norm(ŝ) ϵ̂
        #     throw(ErrorException("In null step sequence, weird behavior."))
        #     break
        # end

        it += 1
        if it > 500
            @error "too much null steps, exiting to serious step"
            throw(ErrorException("Too much steps in bundle procedure"))
        end
    end

    Û = get_Uorthonormalbasis(bundle, ᾱ, ᾱ_nullcoords)

    (printlev > 2) && printstyled(" =====================\n", color = :blue)
    (printlev > 2) && @show p̂
    (printlev > 2) && @show ŝ
    (printlev > 2) && @show ϵ̂
    (printlev > 2) && @show Û
    (printlev > 2) && printstyled(" === Bundle subroutine computation end\n", color = :blue)
    return ϵ̂, p̂, Fp̂, gp̂, ŝ, Û, (; nnullsteps = it, phist, subroutinestatus)
end

function get_Uorthonormalbasis(activebundle::Bundle{Tf}, α̂minnormelt, α_nullcoords) where Tf
    n = length(first(activebundle.bpts).gᵢ)
    actindices = (!).(α_nullcoords)
    nactgᵢ = sum(actindices)

    if nactgᵢ == 1
        @info "only one active elt here"
        return Matrix{Tf}(I, n, n)
    end

    actgᵢs = map(belt -> belt.gᵢ, activebundle.bpts[actindices])

    V̂ = zeros(n, nactgᵢ-1)
    for i in 1:nactgᵢ-1
        V̂[:, i] .= actgᵢs[i] - actgᵢs[end]
    end

    Û = nullspace(V̂')
    return Û
end

