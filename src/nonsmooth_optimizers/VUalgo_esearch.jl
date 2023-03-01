function esearch!(bundle::Bundle{Tf}, xc, fxc, gxc, erxc, p, fp, gp, dxc, gxcdxc, pb, k, μ, σ, sₖ) where Tf
    nsrch = 0

    ### extrapolation line search if wolfe test not satisfied at xc
    esearch=0
    mwolfe=1e-1
    gxoldxc=dot(gp, dxc)
    wolfetest = mwolfe*gxoldxc
    nxhi=0
    dropp=0
    # in case extrapolation drops p from bundle

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
    dropp=1
    mextrap=4.0
    mextra=1.0

    while gxtdxc <  wolfetest
        esearch=1;
        nsrch=nsrch+1; # extrapolate
        d2=gxcdxc-gxoldxc;
        if -gxcdxc > d2*mextrap
            dt=mextrap*dt
        else
            dt=max((-gxcdxc/d2),mextra)*dt
        end
        xtrap=xc+dt*dxc
        t=t+dt

        fxtrap = F(pb, xtrap)
        gxtrap = ∂F_elt(pb, xtrap)

        gxtdxc=gxtrap'*dxc
        erxtrap=fp-fxtrap+gxtdxc*t # center at p

        if gxtdxc < wolfetest
            gxoldxc=gxcdxc
            gxcdxc=gxtdxc
            xc=xtrap
            fxc=fxtrap
            gxc=gxtrap
            erxc=erxtrap
            # hessxc=hessxtrap
        end
    end # of while not wolfe and of extrapolation

    ## prepare for possible interpolation
    if esearch==1
        nxhi=nxhi+1
        if fxtrap ≤ fxc
            fxhi=fxc;    xdiff=xc-xtrap;
            erxhi=erxc; gxhi=gxc; #hessxhi=hessxc;
            xlo=xtrap; fxlo=fxtrap; dxlo=-dt*gxtdxc; dxhi=-dt*gxcdxc;
            erxlo=erxtrap; gxlo=gxtrap; #hessxlo=hessxtrap;
        else
            fxhi=fxtrap; xdiff=xtrap-xc;
            erxhi=erxtrap; gxhi=gxtrap;# hessxhi=hessxtrap;
            xlo=xc;    fxlo=fxc;    dxlo= dt*gxcdxc; dxhi= dt*gxtdxc;
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
            erxlo=0; gxlo=gp; #hessxlo=hess;
        end
    end
    d2=dxhi-dxlo; xdiff2=xdiff'*xdiff; muu=d2/xdiff2; mufirst=muu;
    nave=0; muave=muu;
    if k == 1
        vound=5e-2*(σ/mufirst)*sₖ'*sₖ;
    end # ? smaller factor?
    #   if (k > 2 & esearch > 0), mu=min(mu,mufirst),  end #
    isearch=0;
    ##%# possible interpolation to find new xc; ? what if dimU=n ?
    @show dxlo
    @show dxhi
    @show k
    @show esearch
    if (dxlo<0 && dxhi>0) && (k==1 && esearch==0)
        throw(ErrorException("Not implemented"))
    end

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
        @info "further augmenting bundle"
        # erlin=[erlin;erxhi]; fais=[fais gxhi];
        push!(bundle.bpts, BundlePoint(-1., gxhi, erxhi, [-1., -1.]))
    end

    return xc, fxc, gxc, erxc, muave, mufirst, nsrch
end
