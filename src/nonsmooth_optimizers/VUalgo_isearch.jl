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

    if fphat >= fp # ? is xhi needed; yes if bundle only 1 or 2 more
        xhi=phat; gxhi=gphat; xlo=p; gxlo=gp; xdiff=xhi-xlo; dxlo=gp'*xdiff;
        fxhi=fphat; fxlo=fp; dxhi=gphat'*xdiff;
    else
        xhi=p; gxhi=gp; xlo=phat; gxlo=gphat; xdiff=xhi-xlo;
        fxhi=fp; fxlo=fphat; dxhi=gp'*xdiff; dxlo=gphat'*xdiff;
    end

    d2=dxhi-dxlo; xdiff2=xdiff'*xdiff; muu=d2/xdiff2;
    nave=0; muave=muu
    lstop=0; ldescent=0; fxls=fxhi; nxhi=0; nxlo=0;
    if dxlo >= 0
        tv=d2; tu=-1.; vtest=vound; lstop=1; muave=min(muave,mufirst)
    end # why? lower bound too?
    if dxlo < 0 && dxhi <= 0
        xlo=xhi; tv=d2; tu=-1.; vtest=vound; lstop=1;
        muave=min(muave,mufirst)
    end

    while lstop==0
        tv=(fxlo-fxhi+dxhi)/(dxhi-dxlo); # ? max(0, ?
        if -dxlo >= tv*d2
            tvu=tv; tu=1.;
        else
            tu=-dxlo/d2; tvu=tu;
        end
        ttest=tvu;
        if ldescent == 0
            ttest=tv
        end
        vtest=ttest*(fxhi-fxlo-dxlo); muu=d2/xdiff2;
        if (vtest < vound)
            lstop=2
        end
        if lstop == 2 && (tu < tv && ldescent > 0)
            break
        end
        nave=nave+1; muave=((nave-1)*muave+muu)/nave
        xls=xlo+tvu*xdiff;

        fxls, gxls = blackbox_oracle(pb, xls)
        push!(nullstepshist, copy(xls)); toto(nullstepshist, loc = "isearch")

        (printlev > 2) && @printf("\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,nsim,dxlo,dxhi,tv,tu,fxls,muu);
        if fxls >= fxlo
            nxhi=nxhi+1;
            xhi=xls; xdiff=tvu*xdiff;
            fxhi=fxls; dxhi=gxls'*xdiff; dxlo=tvu*dxlo; xdiff2=(tvu^2)*xdiff2;
            gxhi=gxls;
            if ldescent==0
                d2=dxhi-dxlo;
            else
                d2=tvu*d2;
            end
        else
            nxlo=nxlo+1; ldescent=ldescent+1; # fxls < fxlo
            dxls=gxls'*xdiff; tvucomp= 1.0 - tvu;#'
            if dxls <= 0
                xlo=xls;  # xhi,fxhi are same
                fxlo=fxls; xdiff2=(tvucomp^2)*xdiff2; xdiff=tvucomp*xdiff;
                gxlo=gxls; #hessxlo=hessxls;
                dxold=tvucomp*dxlo; dxlo=tvucomp*dxls; dxhi=tvucomp*dxhi;
                d2=(dxlo-dxold)*(tvucomp/tvu);
            else
                xhi=xlo; xlo=xls;  # dxls > 0
                fxhi=fxlo; fxlo=fxls; xdiff=-tvu*xdiff; xdiff2=(tvu^2)*xdiff2;
                gxhi=gxlo; gxlo=gxls; #hessxhi=hessxlo; hessxlo=hessxls;
                dxold=-tvu*dxhi; dxhi=-tvu*dxlo; dxlo=-tvu*dxls;
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
