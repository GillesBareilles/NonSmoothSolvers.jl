using PlotsOptim: get_abscisses, get_ordinates, COLORS_7, COLORS_10
using PlotsOptim
using PGFPlotsX
using DelimitedFiles


function extract_seriousnullsteps(pb, tr, Fopt)
    ind = 2
    xsnull, ysnull = [], []
    xsser, ysser = [], []
    function subopt(p)
        # return F(pb, p) - Fopt
        return F(pb, p)
    end

    # initial point (no bundle info)
    push!(xsser, ind - 1)
    push!(ysser, subopt(first(tr).additionalinfo.p))
    push!(xsnull, ind - 1)
    push!(ysnull, subopt(first(tr).additionalinfo.p))

    for o in tr[2:end]
        if haskey(o.additionalinfo, :nullsteps)
            nullsteps = o.additionalinfo.nullsteps
            for p in nullsteps
                push!(xsnull, ind)
                ind += 1
                push!(ysnull, subopt(p))
            end
        end
        push!(xsser, ind - 1)
        push!(ysser, subopt(o.additionalinfo.p))
    end

    return xsnull, ysnull, xsser, ysser
end

function plot_seriousnullsteps(pb, tr, Fopt; title)
    xsnull, ysnull, xsser, ysser = extract_seriousnullsteps(pb, tr, Fopt)

    model_to_curve = [
        ("null", xsnull, ysnull, "x", COLORS_7[3]),
        ("serious", xsser, ysser, "pentagon", COLORS_7[5])
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
            xlabel = "iteration",
            ylabel = "subopt",
            legend_pos = "north east",
            legend_style = "font=\\footnotesize",
            legend_cell_align = "left",
            unbounded_coords = "jump",
            title = title,
            xmin = 0,
            xmax = 200,
            ymin = 1e-15,
            ymax = 1e6,
            width = "8cm",
            height = "6cm",
        },
        plotdata...,
    ))
    return fig
end

function write_nullsteps(pb, tr, Fopt, name)
    ysnull = []
    function subopt(p)
        # return F(pb, p) - Fopt
        return F(pb, p)
    end

    for o in tr
        if haskey(o.additionalinfo, :nullsteps)
            nullsteps = o.additionalinfo.nullsteps
            for p in nullsteps
                push!(ysnull, subopt(p))
            end
        end
        push!(ysnull, subopt(o.additionalinfo.p))
    end

    open(name, "w") do io
        writedlm(io, ysnull)
    end

    return nothing
end
