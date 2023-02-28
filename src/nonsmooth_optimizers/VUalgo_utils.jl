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






function check_bundle(pb, bundle, x::Vector{Tf}, Fx) where Tf
    for belt in bundle
        @assert belt.fᵢ == F(pb, belt.yᵢ)
        @assert belt.gᵢ == ∂F_elt(pb, belt.yᵢ)
        @assert belt.eᵢ == Fx - belt.fᵢ - dot(belt.gᵢ, x - belt.yᵢ)
        @assert belt.eᵢ ≥ -1e2 * eps(Tf)
    end
    return
end
