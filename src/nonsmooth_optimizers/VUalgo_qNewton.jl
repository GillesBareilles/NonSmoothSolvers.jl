function qNewtonupdate(pₖ, pₖ₋₁, sₖ, sₖ₋₁, U, Hin, k, curvmin, ν, νlow, μ, kase)
    n = length(pₖ)


    printstyled(" === qNewton step computation === \n", color = :yellow)
    @show pₖ
    @show pₖ₋₁
    @show sₖ
    @show sₖ₋₁
    @show U
    @show Hin
    @show k
    @show curvmin
    @show ν
    @show νlow
    @show μ
    @show kase
    printstyled(" ============================== \n", color = :yellow)

    # TODO H in place
    H = copy(Hin)
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

                if dsWds > (1-.9999999e-7)*dpds
                    # NOTE: BFGS update
                    H=H -(Hndp*Hndp')/dpHndp +(ds*ds')/dpds
                else
                    # SR1 update
                    r=ds-Hndp
                    H=H  +(r*r')/(r'*dp)
                end
            end
        else
            @error "No update, curvature too small"
        end
    end

    Hreduced=U'*H*U
    hmin = min(minimum(Hreduced[i, i] for i in axes(Hreduced, 1)), 0)
    hsum = sum(max(Hreduced[i, i], 0) for i in axes(Hreduced, 1))
    haveinv = ν/hsum
    du=-U*(Hreduced\(U'*sₖ))


    printstyled(" ============================== \n", color = :yellow)
    @show du
    @show haveinv
    @show hmin
    @show H
    @show kase
    printstyled(" === qNewton step computation end \n", color = :yellow)

    return du, hmin, haveinv, H, kase
end
