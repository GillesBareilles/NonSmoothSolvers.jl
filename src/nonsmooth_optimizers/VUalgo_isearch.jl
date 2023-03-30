# function isearch(pb, pₖ, Fpₖ, gpₖ, bundle, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, state.k, σₖ, vound, nrow, ncol, mufirst)
"""
    $TYPEDSIGNATURES


"""
function isearch!(bundle::Bundle{Tf}, pb,
                  p, fp, gp,
                  phat, fphat, gphat,
                  k, vound, mufirst;
                  printlev = 0, nullstepshist = []) where Tf

    # TODO fix variable names in this function.
    # TODO remove redundant point info from bundle point.

    printlev > 0 && printstyled(" === qNewton step computation === \n", color = :light_yellow)
    printlev > 0 && @show p
    printlev > 0 && @show fp
    printlev > 0 && @show gp
    printlev > 0 && @show bundle
    printlev > 0 && @show phat
    printlev > 0 && @show fphat
    printlev > 0 && @show gphat
    printlev > 0 && @show k
    printlev > 0 && @show vound
    printlev > 0 && @show mufirst
    printlev > 0 && printstyled(" ============================== \n", color = :light_yellow)

    nsim = 0

    # NOTE add point p if not in bundle
    if isnothing(findfirst(bpt -> bpt.gᵢ == gp, bundle.bpts))
        push!(bundle.bpts, BundlePoint(-1., gp, 0., [-1., -1.]))
    else
        throw(error("Reached this part of code, deemed unreachable.."))
    end

    # NOTE set x high and x low
    if fphat >= fp
        xhi, fxhi, gxhi = phat, fphat, gphat
        xlo, fxlo, gxlo = p, fp, gp
    else
        xhi, fxhi, gxhi = p, fp, gp
        xlo, fxlo, gxlo = phat, fphat, gphat
    end
    xdiff=xhi-xlo
    dxlo=dot(gxlo, xdiff)
    dxhi=dot(gxhi, xdiff)

    # NOTE Jump linesearch in some cases
    muu = (dxhi-dxlo) / norm(xdiff)^2
    muave=muu
    performlinesearch = true
    if dxlo >= 0
        # No point in doing linesearch: xdiff increases fun vals from  point xlow
        muave=min(muave,mufirst)
        performlinesearch = false
    elseif dxlo < 0 && dxhi <= 0
        # Function values decrease from xlo and xhigh both. Seems unlikely with convex functions.
        # NOTE: ask Claudia
        xlo=xhi;
        muave=min(muave,mufirst)
        performlinesearch = false
    end

    # NOTE run interpolation linesearch
    d2=dxhi-dxlo; #muu=d2/xdiff2;
    fxls=fxhi; nxhi=0; nxlo=0;
    nave=0
    ldescent=0;
    while performlinesearch
        tv=(fxlo-fxhi+dxhi)/(dxhi-dxlo); # ? max(0, ?
        # NOTE ask Claudia what model this tv minimizes
        if -dxlo >= tv*d2
            tvu=tv; tu=1.;
        else
            tu=-dxlo/d2; tvu=tu;
        end
        ttest=tvu;
        if ldescent == 0
            ttest=tv
        end
        vtest=ttest*(fxhi-fxlo-dxlo); muu=d2/norm(xdiff)^2;
        if (vtest < vound)
            performlinesearch = false
            if tu < tv && ldescent > 0
                break
            end
        end
        nave=nave+1; muave=((nave-1)*muave+muu)/nave
        xls = xlo + tvu * xdiff

        fxls, gxls = blackbox_oracle(pb, xls)
        push!(nullstepshist, copy(xls)); toto(nullstepshist, loc = "isearch")

        (printlev > 2) && @printf("\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,nsim,dxlo,dxhi,tv,tu,fxls,muu);
        if fxls >= fxlo
            # Update xhi as xls
            nxhi=nxhi+1;
            xhi, fxhi, gxhi = xls, fxls, gxls

            dxhi = dot(gxls, xdiff)
            dxlo = tvu*dxlo;
            xdiff = tvu*xdiff

            if ldescent==0
                d2=dxhi-dxlo;
            else
                d2=tvu*d2;
            end
        else
            # Update xlo as xls, or xlo, xhi as xls, xlo
            nxlo=nxlo+1; ldescent += 1
            dxls = dot(gxls, xdiff)
            tvucomp= 1.0 - tvu
            if dxls <= 0
                xlo, fxlo, gxlo = xls, fxls, gxls

                dxhi=tvucomp*dxhi
                dxlo=tvucomp*dxls
                xdiff=tvucomp*xdiff

                dxold=tvucomp*dxlo
                d2=(dxlo-dxold)*(tvucomp/tvu);
            else
                xhi, fxhi, gxhi = xlo, fxlo, gxlo
                xlo, fxlo, gxlo = xls, fxls, gxls

                dxhi=-tvu*dxlo
                dxlo=-tvu*dxls
                xdiff = -tvu*xdiff

                dxold=-tvu*dxhi
                d2=(dxlo-dxold)*(tvu/tvucomp); # check somehow
            end
        end
    end

    # NOTE: Add relevant points to bundle, centered at p
    if nxlo > 0
        push!(bundle.bpts, BundlePoint(-1., gxlo, fp-fxlo-gxlo'*(p-xlo), [-1., -1.]))
    end
    if nxhi > 0
        push!(bundle.bpts, BundlePoint(-1., gxhi, fp-fxhi-gxhi'*(p-xhi), [-1., -1.]))
    end

    printlev > 0 && printstyled(" ============================== \n", color = :light_yellow)
    printlev > 0 && @show xlo
    printlev > 0 && @show fxlo
    printlev > 0 && @show bundle
    printlev > 0 && @show muave
    printlev > 0 && printstyled(" === qNewton step computation end \n", color = :light_yellow)

    return xlo,fxlo,muave
end
