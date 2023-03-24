using DataStructures
using JLD2

function matlabbaseline()
    pb_data = OrderedDict{String, Any}()
    pb_data["f2d"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 14,
        "niter         " => 4,
        "x             " => [ 6.2572313763823855e-08, -2.0201657621461943e-13 ],
        "F(x)          " => 2.0397422343951906e-13,
        "||x - xopt||  " => 6.257231e-08,
        "F(x) - F(xopt)" => 2.039742e-13,
        "time          " => 8.391060e-01,
    )
    pb_data["f3d-U3"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 9,
        "niter         " => 5,
        "x             " => [ 4.5970172113385388e-17, 1.0000000000000002e+00, 1.0000000000000005e+01 ],
        "F(x)          " => 1.7763568394002505e-15,
        "||x - xopt||  " => 5.333893e-15,
        "F(x) - F(xopt)" => 1.776357e-15,
        "time          " => 8.647300e-02,
    )
    pb_data["f3d-U2"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 35,
        "niter         " => 10,
        "x             " => [ 2.2170280697369117e-06, 1.2283233813513554e-12, 1.0000000000001227e+01 ],
        "F(x)          " => 1.2310152897043736e-12,
        "||x - xopt||  " => 2.217028e-06,
        "F(x) - F(xopt)" => 1.231015e-12,
        "time          " => 9.700720e-01,
    )
    pb_data["f3d-U1"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 27,
        "niter         " => 9,
        "x             " => [ -2.9409235431498033e-05, 1.4386856270343101e-10, 1.4386999353706855e-10 ],
        "F(x)          " => 1.4471300810352407e-10,
        "||x - xopt||  " => 2.940924e-05,
        "F(x) - F(xopt)" => 1.447130e-10,
        "time          " => 8.591260e-01,
    )
    pb_data["f3d-U0"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 34,
        "niter         " => 7,
        "x             " => [ 1.0000000000000018e+00, -2.0539125955565396e-15, -1.7902346272080649e-15 ],
        "F(x)          " => 5.6205040621648550e-15,
        "||x - xopt||  " => 3.252528e-15,
        "F(x) - F(xopt)" => 5.620504e-15,
        "time          " => 8.726140e-01,
    )
    pb_data["maxquad"] = OrderedDict{String, Any}(
        "nbboxcalls    " => 74,
        "niter         " => 12,
        "x             " => [ -1.2625649384872650e-01, -3.4377939491394220e-02, -6.8575654616000155e-03, 2.6360474639563470e-02, 6.7295087816740068e-02, -2.7839930872611890e-01, 7.4218899160101479e-02, 1.3852427812582696e-01, 8.4031639351898696e-02, 3.8580530630748608e-02 ],
        "F(x)          " => -8.4140833459047126e-01,
        "||x - xopt||  " => 8.374539e-07,
        "F(x) - F(xopt)" => 5.332068e-12,
        "time          " => 8.943050e-01,
    )
    # pb_data["f3dBis"] = OrderedDict{String, Any}(
    #     "nbboxcalls    " => 37,
    #     "niter         " => 9,
    #     "x             " => [ 1.6807836040854054e-06, 1.4129565973473147e-12, 1.4140350932573579e-12 ],
    #     "F(x)          " => 1.4125167618811623e-12,
    #     "||x - xopt||  " => 1.680784e-06,
    #     "F(x) - F(xopt)" => 1.412517e-12,
    #     "time          " => 3.040790e-01,
    # )

    jldfile = "benchmark/VUmatlab.jld2"
    gitcommit = "--"
    date = "--"
    jldsave(jldfile; pb_data, gitcommit, date)
    write_data_txt(pb_data, gitcommit, date, "benchmark/VUmatlab.txt")
    return
end
