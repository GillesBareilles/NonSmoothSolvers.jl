using PlotsOptim: get_abscisses, get_ordinates, COLORS_7, COLORS_10
using PlotsOptim
using PGFPlotsX
using DelimitedFiles


function extract_seriousnullsteps(pb, tr)
    xsnull, ysnull = Int64[], Float64[]
    xsser, ysser = Int64[], Float64[]
    function subopt(p)
        return F(pb, p)
    end

    iser = 1
    for (inull, pt) in enumerate(tr[end].additionalinfo.nullstepshist)
        Fpt = subopt(pt)

        # Null steps
        push!(ysnull, Fpt)
        push!(xsnull, inull)

        # Serious step
        if pt == tr[iser].additionalinfo.p
            push!(ysser, Fpt)
            push!(xsser, inull)
            iser += 1
        end
    end
    @assert iser ≥ length(tr)

    return xsnull, ysnull, xsser, ysser
end

function plot_seriousnullsteps(pb, tr, Fopt; title)
    xsnull, ysnull, xsser, ysser = extract_seriousnullsteps(pb, tr)

    model_to_curve = [
        ("null", xsnull, ysnull .- Fopt, "x", COLORS_7[3]),
        ("serious", xsser, ysser .- Fopt, "pentagon", COLORS_7[5])
    ]

    # Building plot
    plotdata = []
    for (mod, xs, ys, marker, color) in model_to_curve
        points = Tuple{Float64, Float64}[(xs[i], ys[i]) for i in axes(xs, 1)]
        push!(
            plotdata,
            PlotInc(
                PGFPlotsX.Options(
                    "mark" => marker,
                    "color" => color,
                    (mod == "null" ? "" : "only marks") => nothing
                ),
                Coordinates(points),
            ),
        )
        push!(plotdata, LegendEntry(mod))
    end

    fig = TikzPicture(@pgf Axis(
        {
            ymode = "log",
            xlabel = "bbox calls",
            ylabel = "subopt",
            legend_pos = "north east",
            legend_style = "font=\\footnotesize",
            legend_cell_align = "left",
            unbounded_coords = "jump",
            title = title,
            xmin = 0,
            # xmax = 200,
            ymin = 1e-16,
            ymax = 1e6,
            width = "8cm",
            height = "6cm",
        },
        plotdata...,
    ))
    return fig
end

function write_nullsteps(pb, tr, name)
    xsnull, ysnull, xsser, ysser = extract_seriousnullsteps(pb, tr)

    open(name, "w") do io
        writedlm(io, ysnull)
    end

    return nothing
end
