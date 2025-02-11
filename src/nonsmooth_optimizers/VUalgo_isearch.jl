# function isearch(pb, pₖ, Fpₖ, gpₖ, bundle, pᶜₖ₊₁, Fpᶜₖ₊₁, gpᶜₖ₊₁, state.k, σₖ, vound, nrow, ncol, mufirst)
function isearch(pb, p,fp,gp,bundle, phat,fphat,gphat, k,sigma,vound,mufirst; nullstepshist = [])

    # TODO fix variable names in this function.
    # TODO remove redundant point info from bundle point.
    # TODO add ! to fct name

    printlev = 0
    printlev > 0 &&
        printstyled(" --- isearch --- \n", color = :light_yellow)
    printlev > 0 && @show p
    printlev > 0 && @show fp
    printlev > 0 && @show gp
    printlev > 0 && @show bundle
    printlev > 0 && @show phat
    printlev > 0 && @show fphat
    printlev > 0 && @show gphat
    printlev > 0 && @show k
    printlev > 0 && @show sigma
    printlev > 0 && @show vound
    printlev > 0 && @show mufirst
    printlev > 0 && printstyled(" ============================== \n", color = :light_yellow)

    nsim = 15

    # input mufirst is output from esearch
    # TODO clean up this part of the code
    np=0 #; npbest=0;
    # need a logical variable to tell if gp is still in bundle
    # rather than the following
    for bpt in bundle.bpts
        if bpt.gᵢ == gp
            np+=1
        end
    end
    # bundl=faisp; berlin=erlinp; #bhessians=hessiansp;# berlin center is p
    if np == 0
        push!(bundle.bpts, BundlePoint(-1., gp, 0., [-1., -1.]))
        # bundl=[bundl,gp]; berlin=[berlin;0];
        # bhessians(1:nrow,1:ncol,length(berlin))= hess
    else
        throw(error("Reached this part of code deemed unreachable.."))
    end


    #? will pbest be used? maybe use V cut plane step instead?
    #        if npbest == 0; bundl=[bundl,gpbest];
    #                       berlin=[berlin;fp-fpbest-gpbest'*(p-pbest)];
    #                       bhessians(1:nrow,1:ncol,length(berlin))= hesspbest; end
    #        xls=pbest; xlsp=xls-p; dp=gp'*xlsp; xlsp2=xlsp'*xlsp;
    #        fxls=fpbest; gxls=gpbest; dxls=gxls'*xlsp; d2=dxls-dp;
    #        bundl=[gp gphat]; berlin=[0;fp-fphat+dxls];
    #        nb=1; bhessians(1:nrow,1:ncol,nb)= hess;
    #        nb=2; bhessians(1:nrow,1:ncol,nb)= hessphat;

    if fphat >= fp # ? is xhi needed; yes if bundle only 1 or 2 more
        xhi=phat; gxhi=gphat; xlo=p; gxlo=gp; xdiff=xhi-xlo; dxlo=gp'*xdiff;
        fxhi=fphat; fxlo=fp; dxhi=gphat'*xdiff;
        # hessxhi=hessphat; hessxlo=hess;
    else
        xhi=p; gxhi=gp; xlo=phat; gxlo=gphat; xdiff=xhi-xlo;
        fxhi=fp; fxlo=fphat; dxhi=gp'*xdiff; dxlo=gphat'*xdiff;
        # hessxhi=hess; hessxlo=hessphat;
    end
    d2=dxhi-dxlo; xdiff2=xdiff'*xdiff; muu=d2/xdiff2;
    nave=0; muave=muu; mubig=muu; muu1=muu;
    nb=2; # ?needed?
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
        # ?        nave=nave+1; muave=((nave-1)*muave+muu)/nave;
        if (vtest < vound)
            lstop=2
        end # ? what if tvu=0 ?
        if lstop == 2 && (tu < tv && ldescent > 0)
            break
        end
        nave=nave+1; muave=((nave-1)*muave+muu)/nave; mubig=max(mubig,muu);
        xls=xlo+tvu*xdiff;

        fxls, gxls = blackbox_oracle(pb, xls); nsim=nsim+1; nb=nb+1;
        push!(nullstepshist, copy(xls)); toto(nullstepshist, loc = "isearch")

        # [fxls,gxls,hessxls]=eval([simul,'(xls,Par)']); nsim=nsim+1; nb=nb+1;

        # at end put only the last one or two into bundle; possibly delete p?
        #        bundl=[bundl gxls]; bhessians(:,:,length(berlin)+1)=hessxls;
        #        berlin=[berlin; fp-fxls-gxls'*(p-xls)]; #%' centered at p
        (printlev > 2) && @printf("\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,nsim,dxlo,dxhi,tv,tu,fxls,muu);
        if fxls >= fxlo
            nxhi=nxhi+1; xhi=xls; xdiff=tvu*xdiff;
            fxhi=fxls; dxhi=gxls'*xdiff; dxlo=tvu*dxlo; xdiff2=(tvu^2)*xdiff2;
            gxhi=gxls; #hessxhi=hessxls;
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

    end # while of isearch loop; next [nxlo,nxhi] increases bundle
    # size by 1 or 2; ? should p be deleted sometimes ?

    # NOTE: Add relevant points to bundle, centered at p
    if nxlo > 0
        push!(bundle.bpts, BundlePoint(-1., gxlo, fp-fxlo-gxlo'*(p-xlo), [-1., -1.]))

        # bundl=[bundl gxlo]; #bhessians(:,:,length(berlin)+1)=hessxlo;
        # berlin=[berlin; fp-fxlo-gxlo'*(p-xlo)]; # centered at p
    end
    if nxhi > 0
        push!(bundle.bpts, BundlePoint(-1., gxhi, fp-fxhi-gxhi'*(p-xhi), [-1., -1.]))

        # bundl=[bundl gxhi]; #bhessians(:,:,length(berlin)+1)=hessxhi;
        # berlin=[berlin; fp-fxhi-gxhi'*(p-xhi)]; # centered at p
    end

    (printlev > 2) && @printf("\n  %i(%i) dxlo %7.4e dxhi %7.4e tv %7.4e tu %7.4e fxls %7.4e muu %7.4e", k,nsim,dxlo,dxhi,tv,tu,fxls,muu);
    #fprintf('\n  %i(%i) fxlo %7.4e vtest %7.4e vound %7.4e muu %7.4e muave %7.4e nb %i', k,nsim,fxlo,vtest,vound,muu,muave,nb);
    (printlev > 2) && @printf("\n  %i(%i) fxlo %7.4e vtest %7.4e vound %7.4e muu %7.4e muave %7.4e mubig %7.4e\n", k,nsim,fxlo,vtest,vound,muu,muave,mubig);


    printlev > 0 && printstyled(" ============================== \n", color = :light_yellow)
    printlev > 0 && @show xlo
    printlev > 0 && @show fxlo
    printlev > 0 && @show bundle
    # printlev > 0 && @show bhessians
    # printlev > 0 && @show nsim
    # printlev > 0 && @show muu1
    printlev > 0 && @show muave
    # printlev > 0 && @show mubig
    printlev > 0 && printstyled(" === qNewton step computation end \n", color = :light_yellow)

    # TODO remove muu1, muubig computation
    return xlo,fxlo,muave
end
