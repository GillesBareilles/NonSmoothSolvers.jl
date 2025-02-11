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

    dxc = xᶜₖ₊₁ - pₖ
    gxcdxc = dot(gxᶜₖ₊₁, dxc)

    ### extrapolation line search if wolfe test not satisfied at xc
    esearch=0
    mwolfe=1e-1
    gxoldxc=dot(gp, dxc)
    wolfetest = mwolfe*gxoldxc
    nxhi=0
    # in case extrapolation drops p from bundle

    nsim = 0
    vtest = -1

    nave = Tf(0)
    tu = Tf(0)
    tv = Tf(0)
    fxtrap = Tf(0)
    xtrap = similar(xc)
    gxtrap = similar(xc)
    erxtrap = Tf(0)
    vound = Tf(0)


    ##%#    if gxcdxc >= wolfetest    # xc is OK
    gxtdxc=gxcdxc
    dt=1.0
    t=1.0
    mextrap=4.0
    mextra=1.0

    # NOTE extrapolate: find dt > 1 : xtrap = xc + dt * dxc meets wolfe
    while gxtdxc <  wolfetest
        esearch=1;
        nsrch=nsrch+1; # extrapolate
        d2=gxcdxc-gxoldxc;

        # BUG this is fishy, always *4
        if -gxcdxc > d2*mextrap
            dt=mextrap*dt
        else
            dt=max((-gxcdxc/d2),mextra)*dt
        end
        xtrap .= xc .+ dt*dxc
        t=t+dt

        # NOTE: oracle call
        fxtrap, gxtrap = blackbox_oracle(pb, xtrap); nsim += 1
        push!(nullstepshist, copy(xtrap))
        toto(nullstepshist, loc = "esearch1")

        gxtdxc = gxtrap'*dxc
        erxtrap = fp-fxtrap+gxtdxc*t # center at p

        if gxtdxc < wolfetest
            gxoldxc = gxcdxc
            gxcdxc = gxtdxc
            xc .= xtrap
            fxc=fxtrap
            gxc .= gxtrap
            erxc = erxtrap
            # hessxc=hessxtrap
        end
    end # of while not wolfe and of extrapolation

    ## NOTE prepare for possible interpolation
    # set [xhi, xlo] as an interval ok with Wolfe
    if esearch==1
        nxhi=nxhi+1
        if fxtrap ≤ fxc
            fxhi=fxc;    xdiff=xc-xtrap;
            xlo=xtrap; fxlo=fxtrap; dxlo=-dt*gxtdxc; dxhi=-dt*gxcdxc;

            erxhi=erxc; gxhi=gxc; #hessxhi=hessxc;
            erxlo=erxtrap; gxlo=gxtrap; #hessxlo=hessxtrap;
        else
            fxhi=fxtrap; xdiff=xtrap-xc;
            xlo=xc;    fxlo=fxc;    dxlo= dt*gxcdxc; dxhi= dt*gxtdxc;

            erxhi=erxtrap; gxhi=gxtrap;# hessxhi=hessxtrap;
            erxlo=erxc; gxlo=gxc; #hessxlo=hessxc;
        end
    else # esearch=0
        if fxc ≤ fp
            fxhi=fp; xdiff=p-xc;
            erxhi=0; gxhi=gp; #hessxhi=hess;
            xlo=xc; fxlo=fxc; dxlo=-gxcdxc; dxhi=-gp'*dxc;
            erxlo=erxc; gxlo=gxc; #hessxlo=hessxc;
        else
            fxhi=fxc; xdiff=xc-p; # fxc>fp
            erxhi=erxc; gxhi=gxc; #hessxhi=hessxc;
            xlo=p; fxlo=fp; dxlo=gp'*dxc; dxhi=gxcdxc;
            erxlo=Tf(0); gxlo=gp; #hessxlo=hess;
        end
    end
    d2=dxhi-dxlo; xdiff2=norm(xdiff)^2; muu=d2/xdiff2; mufirst=muu;
    nave=0; muave=muu;

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
            vtest=ttest*(fxhi-fxlo-dxlo); muu=d2/xdiff2; muubig=max(muubig,muu);
            # ? nave,muave here ?
            if (vtest < vound)
                lstop=2
            end # ? what if tu=0 and ldescent=0 ?
            if lstop == 2 && (tu < tv)
                break
            end
            #     if lstop == 2 & (tu < tv & ldescent > 0), break; end # ? or tv<=tu?
            nave=nave+1; muave=((nave-1)*muave+muu)/nave;
            #fprintf('\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e muu %7.4e muubig %7.4e',...
            #                     k,nsim,dxlo,dxhi,tv,tu,muu,muubig),
            if k>1
                break
            end
            # evaluate if 3rd condition
            xls=xlo+tvu*xdiff

            fxls, gxls = blackbox_oracle(pb, xls); nsim=nsim+1; nb=nb+1;
            push!(nullstepshist, copy(xls)); toto(nullstepshist, loc = "esearch2")

            erxls=fp-fxls-gxls'*(p-xls); # gxls'* repeated below
            #fprintf('\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e muubig %7.4e', k,nsim,dxlo,dxhi,tv,tu,fxls,muu,muubig);
            if fxls >= fxlo
                xhi=xls; xdiff=tvu*xdiff; xdiff2=(tvu^2)*xdiff2;
                fxhi=fxls; dxhi=gxls'*xdiff; dxlo=tvu*dxlo; nxhi=nxhi+1;
                erxhi=erxls; gxhi=gxls; #hessxhi=hessxls;
                if ldescent==0
                    d2=dxhi-dxlo;
                else
                    d2=tvu*d2;
                end
            else
                ldescent=ldescent+1; # fxls < fxlo
                dxls=gxls'*xdiff; tvucomp= 1.0-tvu;
                if dxls <= 0
                    xlo=xls;   # xhi,fxhi are same
                    fxlo=fxls; xdiff2=(tvucomp^2)*xdiff2; xdiff=tvucomp*xdiff;
                    dxold=tvucomp*dxlo; dxlo=tvucomp*dxls; dxhi=tvucomp*dxhi;
                    d2=(dxlo-dxold)*(tvucomp/tvu); # both d2s need more checking ?
                    erxlo=erxls; gxlo=gxls;# hessxlo=hessxls;
                else
                    xhi=xlo; xlo=xls; nxhi=nxhi+1; # dxls > 0
                    fxhi=fxlo; fxlo=fxls; xdiff=-tvu*xdiff; xdiff2=(tvu^2)*xdiff2;
                    erxhi=erxlo; gxhi=gxlo; #hessxhi=hessxlo;
                    erxlo=erxls; gxlo=gxls; #hessxlo=hessxls;
                    dxold=-tvu*dxhi; dxhi=-tvu*dxlo; dxlo=-tvu*dxls;
                    d2=(dxlo-dxold)*(tvu/tvucomp); # checked well enough ?
                end
            end
        end # while of xc search inner loop
        (printlev > 0) && @printf("\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,nsim,dxlo,dxhi,tv,tu,fxls,muu);
        (printlev > 0) && @printf("\n  %i(%i) fxlo %7.4e vtest %7.4e vound %7.4e muu %7.4e muave %7.4e nb %i", k,nsim,fxlo,vtest,vound,muu,muave,nb);
    end # of interpolation
    ########

    if k==1 || esearch>0 # if esearch > 0 | isearch > 0
        xc=xlo; fxc=fxlo; gxc=gxlo; erxc=erxlo; # center at p
    else # k>1 and esearch=0, does not occur in June08 version
        if k>2 && isearch==1 && tu<tv
            muave=μ
        end # so that mu will be unchanged mu=muave
    end
    # in future if dropp == 1, overwrite last col of fais if p is there
    # and  consider size(fais,2)?

    push!(bundle.bpts, BundlePoint(-1., gxc, erxc, [-1., -1.]))
    # erlin=[erlin;erxc]; fais=[fais gxc]; # check for 0 in last erlin col?

    if nxhi > 0
        # erlin=[erlin;erxhi]; fais=[fais gxhi];
        push!(bundle.bpts, BundlePoint(-1., gxhi, erxhi, [-1., -1.]))
    end

    return xc, fxc, gxc, erxc, muave, mufirst, nsrch
end
