"""
    $TYPEDSIGNATURES

Update the quasi-Newton `H` matrix from updates `pₖ`, `pₖ₋₁`, `sₖ`, `sₖ₋₁` with
BFGS or SR1 scheme.
"""
function qNewtonupdate!(H, pₖ, pₖ₋₁, sₖ, sₖ₋₁, U, k, curvmin, ν, νlow, μ, kase; printlev = 0)
    n = length(pₖ)

    printlev > 0 && printstyled(" === qNewton step computation === \n", color = :yellow)
    printlev > 1 && @show pₖ
    printlev > 1 && @show pₖ₋₁
    printlev > 1 && @show sₖ
    printlev > 1 && @show sₖ₋₁
    printlev > 1 && @show U
    printlev > 1 && @show H
    printlev > 1 && @show k
    printlev > 1 && @show curvmin
    printlev > 1 && @show ν
    printlev > 1 && @show νlow
    printlev > 1 && @show μ
    printlev > 1 && @show kase
    printlev > 0 && printstyled(" ============================== \n", color = :yellow)

    if k > 1
        dp = pₖ-pₖ₋₁
        ds = sₖ-sₖ₋₁
        dpds= dot(dp, ds)

        curv = dpds / dot(dp, dp) # possible BFGS or SR1 update of nxn Hessian

        if dpds > 1e-8 * norm(dp)*norm(ds) && curv > curvmin # curvature is large enough
            if k == 2 || ν < νlow
                # NOTE initialize (or reinitialize) approximate hessian
                kase=1

                scale=max(curv,μ)
                if k == 2
                    scale=min(curv,μ)
                end
                H .= scale * Diagonal(ones(n))
            else
                if kase == 1
                    H .= curv * Diagonal(ones(n))
                    kase=2
                end

                Hndp=H*dp
                dpHndp=dp'*Hndp
                Wds=H\ds
                dsWds=ds'*Wds

                if dsWds > (1-1e-6)*dpds
                    # BFGS update
                    H .= H .- (Hndp*Hndp')/dpHndp .+ (ds*ds')/dpds
                else
                    # SR1 update
                    r=ds-Hndp
                    H .= H .+ (r*r')/(r'*dp)
                end
            end
        else
            @warn "No update, curvature too small"
        end
    end

    Hreduced=U'*H*U
    # hmin = min(minimum(Hreduced[i, i] for i in axes(Hreduced, 1)), 0)
    hsum = sum(max(Hreduced[i, i], 0) for i in axes(Hreduced, 1))
    haveinv = ν/hsum
    du=-U*(Hreduced\(U'*sₖ))

    printlev > 0 && printstyled(" ============================== \n", color = :yellow)
    printlev > 1 && @show du
    printlev > 1 && @show haveinv
    # printlev > 1 && @show hmin
    printlev > 1 && @show H
    printlev > 1 && @show kase
    printlev > 0 && printstyled(" === qNewton step computation end \n", color = :yellow)

    return du, haveinv, kase
end
