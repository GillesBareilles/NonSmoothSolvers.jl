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
    return nothing
end

function bundlepoint(pb, yᵢ, x)
    fᵢ = F(pb, yᵢ)
    gᵢ = ∂F_elt(pb, yᵢ)
    eᵢ = F(pb, x) - fᵢ - dot(gᵢ, x - yᵢ)

    return BundlePoint(fᵢ, gᵢ, eᵢ, yᵢ)
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
