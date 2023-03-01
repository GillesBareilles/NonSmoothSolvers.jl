mutable struct BundlePoint{Tf}
    fᵢ::Tf
    gᵢ::Vector{Tf}
    eᵢ::Tf
    yᵢ::Vector{Tf}
end

Base.show(io::IO, bp::BundlePoint) = print(io, "fᵢ: ", bp.fᵢ, " eᵢ: ", bp.eᵢ, " gᵢ: ", bp.gᵢ, ", yᵢ", bp.yᵢ)

function bundlepoint(pb, yᵢ, x)
    fᵢ = F(pb, yᵢ)
    gᵢ = ∂F_elt(pb, yᵢ)
    eᵢ = F(pb, x) - fᵢ - dot(gᵢ, x - yᵢ)

    return BundlePoint(fᵢ, gᵢ, eᵢ, copy(yᵢ))
end



mutable struct Bundle{Tf}
    bpts::Vector{BundlePoint{Tf}}
    refpoint::Vector{Tf}
    Frefpoint::Tf
end

function Base.show(io::IO, bundle::Bundle)
    println(io, " + eᵢ: ", map(bp -> bp.eᵢ, bundle.bpts))
    print(io, " + gᵢ: ", map(bp -> bp.gᵢ, bundle.bpts))
    # println(io, "   + fᵢ: ", map(bp -> bp.fᵢ, bundle.bpts))
    # println(io, "   + yᵢ: ", map(bp -> bp.yᵢ, bundle.bpts))
    return
end

function initial_bundle(pb, xcenter::Vector{Tf}) where {Tf}
    Fx = F(pb, xcenter)
    gx = ∂F_elt(pb, xcenter)
    return Bundle(
        [BundlePoint(Fx, gx, 0., xcenter)],
        copy(xcenter),
        Fx,
    )
end

function add_point!(bundle::Bundle{Tf}, pb, yᵢ) where Tf
    push!(bundle.bpts, bundlepoint(pb, yᵢ, bundle.refpoint))
    return
end

function updatebundle!(bundle, pb, x)
    Fx = F(pb, x)
    for (i, belt) in enumerate(bundle)
        e = Fx - belt.fᵢ - dot(belt.gᵢ, x - belt.yᵢ)
        bundle[i] = BundlePoint(belt.fᵢ, belt.gᵢ, e, belt.yᵢ)
    end
    return nothing
end



function change_bundle_center!(bundle::Bundle{Tf}, newcenter, Fnewcenter) where {Tf}
    oldcenter = bundle.refpoint
    Foldcenter = bundle.Frefpoint
    for i in axes(bundle.bpts, 1)
        b = bundle.bpts[i]
        b.eᵢ += Fnewcenter - Foldcenter + dot(b.gᵢ, oldcenter - newcenter)
    end
    bundle.refpoint .= newcenter
    bundle.Frefpoint = Fnewcenter
    return
end





function check_bundle(pb, bundle, x::Vector{Tf}, Fx) where Tf
    for belt in bundle
        @assert belt.fᵢ == F(pb, belt.yᵢ)
        @assert belt.gᵢ == ∂F_elt(pb, belt.yᵢ)
        @assert belt.eᵢ == Fx - belt.fᵢ - dot(belt.gᵢ, x - belt.yᵢ)
        @assert belt.eᵢ ≥ -1e2 * eps(Tf)
    end
    return
end



function update_mu(mu, k, mufirst, muave, ν, νold, haveinv, nsrch, sclst, mulast, sscale, sₖ, mumin, mumax, firstsn)
    @show mu
    @show mufirst
    @show muave
    @show mumax
    @show mumin
    @show mulast
    @show firstsn
    @show sₖ

    # @show k
    # @show ν
    # @show νold
    # @show haveinv
    # @show nsrch
    # @show sclst
    # @show sscale

    if k==1     # NOTE pass here
        mu=min(mufirst,muave)
        mumax=1e4*mu
    end # smallest muu sometimes first
    # if k>2,[muave,1.d0/haveinv,mulast,sclst,mu];end
    #if newt > 0 & mineigH < 5.d-15, haveinv=1.d0/muave, end

    if k == 2   && ν == νold
        @assert false
        mu=min([mu,1/haveinv,muave])
    end # especially small when nu = n
    #or run>1 worse
    # mu before
    if k > 2 && ν == νold && nsrch > 0
        @assert false
        mu=median([muave,1e0/haveinv,sclst,mulast]); # sclst=sscale*mulast;
        if sscale < 1e0 && mu < sclst
            mu=sclst
        end # relate to nulo?
    end
    if k > 2 && ν > νold && sscale < 1.0
        @assert false
        mu=min(mu,sclst)
    end # OK on foverten
    #if k>2&nu>nuold&sscale<1.d0,mu=median([muave,1.d0/haveinv,sclst,mulast]),end
    # replacing sclst with mu below gave same results
    if k > 2 && ν < νold
        @assert false
        if run>1 || nsrch > 0
            mu=median([muave,1/haveinv,sclst,mulast])#keyboard,
        end #if nu<nulo&nsrch>0&run>1, mu=mulast, end# else mu=mulast bad on foverten
    end
    if k > 2 && ν == size(sₖ, 1)
        @assert false
        mu=max([min(muave,1/haveinv),mulast])
    end # questionable? OK foverten

    if k > 1    # NOTE pass here
        mu=max(mu,mumin,1e-2*mulast)
    end # mumin for rate proof; OK foverten
    if k > 2    # NOTE pass here
        mu=min(mu, mumax*firstsn/norm(sₖ))
    end # for convergence proof; OK foverten
    #if k > 2 & newt < 2 & mu > mumax*firstsn/norm(sk), mu=min(mu,sscale*mulast), end # mu could be larger?

    #fprintf('\n    xc=(');for i=1:min(n,3),fprintf('#3.1e  ',xc(i));end;fprintf(')'); # ?print nxhi too?

    println("-----")

    @show mu
    @show mumax
    return mu, mumax
end
