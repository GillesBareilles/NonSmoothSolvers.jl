"""
    $TYPEDSIGNATURES

Extrapolation search from point `xc`, and `p`.
"""
function esearch!(bundle::Bundle{Tf},
                  xc, fxc, gxc, erxc,
                  p, fp, gp,
                  pb, k, μ, σ, sₖ;
                  printlev = 0, nullstepshist = []) where Tf
    nsrch = 0

    xᶜₖ₊₁, pₖ = xc, p
    gxᶜₖ₊₁ = gxc
    printlev = 0
    printlev > 0 &&
        printstyled(" xxx esearch xxx \n", color = :magenta)
    printlev > 0 && @show xc
    printlev > 0 && @show gxc
    printlev > 0 && @show p
    printlev > 0 && @show fp
    printlev > 0 && @show gp
    printlev > 0 && @show k
    printlev > 0 && @show μ
    printlev > 0 && @show σ
    printlev > 0 && @show sₖ
    printlev > 0 && @show bundle
    printlev > 0 && printstyled(" xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n", color = :magenta)

    dxc = xc - p
    gxcdxc = dot(gxc, dxc)

    ### extrapolation line search if wolfe test not satisfied at xc
    esearch=0
    mwolfe=1e-1
    gxoldxc=dot(gp, dxc)
    wolfetest = mwolfe*gxoldxc
    nxhi=0
    # in case extrapolation drops p from bundle

    vtest = -1

    nave = Tf(0)
    tu = Tf(0)
    tv = Tf(0)
    fxtrap = Tf(0)
    xtrap = similar(xc)
    gxtrap = similar(xc)
    erxtrap = Tf(0)
    vound = Tf(0)


    xlo, fxlo, gxlo, erxlo = similar(p), Tf(0), similar(gp), Tf(0)
    xhi, fxhi, gxhi, erxhi = similar(p), Tf(0), similar(gp), Tf(0)


    ##%#    if gxcdxc >= wolfetest    # xc is OK
    gxtdxc=gxcdxc
    dt=1.0
    t=1.0

    # NOTE extrapolate: find dt > 1 : xtrap = xc + dt * dxc meets wolfe
    # note that this implicitly changes the value of xc
    while gxtdxc <  wolfetest
        esearch=1
        nsrch += 1

        # NOTE dtcand minimizes t -> q(0) + q'(0) + 0.5t² (q'(1) - q'(0)),
        # the quadratic that interpolates f at xlo (order 0 and 1), and at xhi (order 1 only)
        dtcand = -gxcdxc/(gxcdxc-gxoldxc)
        dt = max(1.0, min(4.0, dtcand))
        xtrap .= xc .+ dt*dxc
        t += dt

        fxtrap, gxtrap = blackbox_oracle(pb, xtrap)
        push!(nullstepshist, copy(xtrap))

        gxtdxc = dot(gxtrap, dxc)
        erxtrap = fp-fxtrap+gxtdxc*t # center at p

        if gxtdxc < wolfetest
            gxoldxc = gxcdxc
            gxcdxc = gxtdxc
            xc .= xtrap
            fxc = fxtrap
            gxc .= gxtrap
            erxc = erxtrap

            printstyled("xxxxx\n")
        end
    end # of while not wolfe and of extrapolation

    if esearch == 1
        @show t, dt
        printstyled("---->\n", color=:yellow)
        @show xc, fxc, gxc, erxc
        @show xtrap, fxtrap, gxtrap, erxtrap
        printstyled("<----\n", color=:yellow)
    end

    ## NOTE prepare for possible interpolation
    # set [xlo, xhi]
    if esearch==1
        nxhi=nxhi+1
        if fxtrap ≤ fxc
            xhi .= xc
            fxhi = fxc
            gxhi .= gxc
            erxhi = erxc

            xlo .= xtrap
            fxlo = fxtrap
            gxlo .= gxtrap
            erxlo = erxtrap

            dxlo=-dt*gxtdxc
            dxhi=-dt*gxcdxc
        else
            xhi .= xtrap
            fxhi = fxtrap
            gxhi .= gxtrap
            erxhi = erxtrap

            xlo .= xc
            fxlo = fxc
            gxlo .= gxc
            erxlo = erxc

            dxlo= dt*gxcdxc;
            dxhi= dt*gxtdxc;
        end
    else # esearch=0
        if fxc ≤ fp
            xhi .= p
            fxhi = fp
            gxhi .= gp
            erxhi = Tf(0)

            xlo .= xc
            fxlo = fxc
            gxlo .= gxc
            erxlo = erxc


            dxlo=-gxcdxc;
            dxhi=-gp'*dxc;
        else
            xhi .= xc
            fxhi = fxc
            gxhi .= gxc
            erxhi = erxc

            xlo .= p
            fxlo = fp
            gxlo .= gp
            erxlo = Tf(0)

            dxlo=gp'*dxc;
            dxhi=gxcdxc;
        end
    end
    xdiff = xhi - xlo
    d2 = dxhi - dxlo;
    xdiff2=norm(xdiff)^2;
    muu=d2/xdiff2;
    mufirst=muu;
    nave=0;
    muave=muu;

    @assert !isnan(muu) # trouble if xc == p

    if k == 1
        vound=5e-2*(σ/mufirst)*sₖ'*sₖ;
    end # ? smaller factor?
    #   if (k > 2 & esearch > 0), mu=min(mu,mufirst),  end #
    isearch=0;
    ##%# possible interpolation to find new xc; ? what if dimU=n ?
    (printlev > 3) && @show dxlo
    (printlev > 3) && @show dxhi
    (printlev > 3) && @show k
    (printlev > 3) && @show esearch

    if (dxlo<0 && dxhi>0) && (k==1 && esearch==0)
        isearch=1 # June08 version
        # next is Apr08 version
        #if (dxlo<0 & dxhi>0) & ((k==1 & esearch==0) | (k>1 & (k==2 | run>2) & mufirst<mu)), isearch=1,

        lstop=0; ldescent=0; fxls=fxhi; nb=0; nave=0; muubig=muu;
        while lstop==0
            tv = (fxlo-fxhi+dxhi)/(dxhi-dxlo); # max with 0?
            if -dxlo ≥ tv*d2
                tvu=tv; tu=min(1.0,-dxlo/d2);
            else
                tu=-dxlo/d2; tvu=tu;
            end
            ttest=tvu; # if ldescent == 0, ttest=tv; end
            vtest=ttest*(fxhi-fxlo-dxlo);
            muu=d2/xdiff2;
            muubig=max(muubig,muu);
            # ? nave,muave here ?
            if (vtest < vound)
                lstop=2
            end # ? what if tu=0 and ldescent=0 ?
            if lstop == 2 && (tu < tv)
                break
            end
            nave=nave+1;
            muave=((nave-1)*muave+muu)/nave;
            #fprintf('\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e muu %7.4e muubig %7.4e',...
            #                     k,dxlo,dxhi,tv,tu,muu,muubig),
            if k>1
                break
            end
            # evaluate if 3rd condition
            xls=xlo+tvu*xdiff

            fxls, gxls = blackbox_oracle(pb, xls); nb=nb+1;
            push!(nullstepshist, copy(xls))

            erxls=fp-fxls-gxls'*(p-xls); # gxls'* repeated below
            #fprintf('\n  %i dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e muubig %7.4e', k,dxlo,dxhi,tv,tu,fxls,muu,muubig);
            if fxls >= fxlo
                # Update xhi as xls
                nxhi = nxhi + 1
                xhi .= xls
                fxhi = fxls
                gxhi .= gxls

                dxhi = dot(gxls, xdiff)
                dxlo = tvu * dxlo
                xdiff = tvu * xdiff

                xdiff2=(tvu^2)*xdiff2;
                erxhi=erxls;

                if ldescent==0
                    d2=dxhi-dxlo;
                else
                    d2=tvu*d2;
                end
            else
                ldescent=ldescent+1; # fxls < fxlo
                dxls=gxls'*xdiff; tvucomp= 1.0-tvu;
                if dxls <= 0
                    xlo .= xls
                    fxlo = fxls
                    gxlo .= gxls

                    xdiff2=(tvucomp^2)*xdiff2;
                    xdiff=tvucomp*xdiff;
                    dxold=tvucomp*dxlo;
                    dxlo=tvucomp*dxls;
                    dxhi=tvucomp*dxhi;
                    d2=(dxlo-dxold)*(tvucomp/tvu); # both d2s need more checking ?
                    erxlo=erxls;
                else
                    xlo .= xls
                    fxlo = fxls
                    gxlo .= gxls
                    xhi .= xlo
                    fxhi = fxlo
                    gxhi .= gxlo

                    nxhi=nxhi+1; # dxls > 0
                    xdiff=-tvu*xdiff;
                    xdiff2=(tvu^2)*xdiff2;
                    erxhi=erxlo;
                    erxlo=erxls;
                    dxold=-tvu*dxhi;
                    dxhi=-tvu*dxlo;
                    dxlo=-tvu*dxls;
                    d2=(dxlo-dxold)*(tvu/tvucomp); # checked well enough ?
                end
            end
        end
        (printlev > 0) && @printf("\n  %i dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,dxlo,dxhi,tv,tu,fxls,muu);
        (printlev > 0) && @printf("\n  %i fxlo %7.4e vtest %7.4e vound %7.4e muu %7.4e muave %7.4e nb %i", k,fxlo,vtest,vound,muu,muave,nb);
    end

    if k==1 || esearch>0 # if esearch > 0 | isearch > 0
        xc=xlo; fxc=fxlo; gxc=gxlo; erxc=erxlo; # center at p
    else # k>1 and esearch=0, does not occur in June08 version
        if k>2 && isearch==1 && tu<tv
            muave=μ
        end # so that mu will be unchanged mu=muave
    end

    push!(bundle.bpts, BundlePoint(-1., gxc, erxc, [-1., -1.]))

    if nxhi > 0
        push!(bundle.bpts, BundlePoint(-1., gxhi, erxhi, [-1., -1.]))
    end

    printlev > 0 && printstyled(" xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \n", color = :magenta)
    printlev > 0 && @show xc
    printlev > 0 && @show fxc
    printlev > 0 && @show gxc
    printlev > 0 && @show erxc
    printlev > 0 && @show muave
    printlev > 0 && @show mufirst
    printlev > 0 && @show nsrch
    printlev > 0 &&
        printstyled(" xxx esearch end \n", color = :magenta)

    return xc, fxc, gxc, erxc, muave, mufirst, nsrch
end
