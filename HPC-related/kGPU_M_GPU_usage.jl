using CUDA
using Pkg
using BenchmarkTools
using FFTW
using Adapt
using CSV
using DataFrames
using Dates

println("loaded standard packages");

#Pkg.add(path="/mainfs/scratch/hc8g23/GPU_CFD_solver/kGPU_M")

import kGPU_M

println("loaded all packages");


dt = 0.01;
Re = 40.0;
n_iter = 1000;

#lens = [30, 70, 80, 100, 128, 160, 200, 256, 300, 350, 400, 450, 512, 650, 800]
#padded_lens = [45, 105, 120, 150, 192, 240, 300, 384, 450, 525, 600, 675, 768, 975, 1200]

lens = [800]
padded_lens = [1200]

#lens = [400]
#padded_lens=[600]

#lens = [70, 100, 128, 200, 256, 300, 400, 512, 800, 896]
#padded_lens = [105, 150, 192, 300, 384, 450, 600, 768, 1200, 1344]

#lens = [896]
#padded_lens = [1344]

forcing_freq = 4;

function source_fn(x, y)
    return -4*cos(4*y);
end;

function benchmark_test(solver_GPU::kGPU_M.solver, iters::Int64)
    kGPU_M.run(solver_GPU, iters);
    solver_GPU.current_iter = 1;
end;

for i in 1:size(lens, 1)
    println("-------------------------------------");

    x_len = copy(padded_lens[i]);
    y_len = copy(padded_lens[i]);

    x_cutoff = copy(lens[i]);
    y_cutoff = copy(lens[i]);

    xs_full = LinRange(0, 2pi, x_len+1);
    ys_full = LinRange(0, 2pi, y_len+1);

    xs = Array{Float64}(xs_full[1:x_len]);
    ys = Array{Float64}(ys_full[1:y_len]);

    ω0 = rand(x_len, y_len);

    source = [source_fn(x, y) for x in xs, y in ys];
    source_hat = rfft(source);

    solver_GPU = kGPU_M.solver{CuArray, kGPU_M.GPU_rfft_type64, kGPU_M.GPU_irfft_type64}()
    kGPU_M.init_solver(solver_GPU, dt, Re, n_iter, xs, ys, x_cutoff, y_cutoff, ω0, source_hat, forcing_freq);

    kGPU_M.iter(solver_GPU, 1);

    println("executed 1 iteration for $(x_len)x$(y_len) simulation");

    #test_fn = kGPU_M.run(solver_GPU, 1000);

    
    b = @benchmarkable benchmark_test($solver_GPU, 1000);
    b.params.seconds = 3600;
    b.params.samples = 200;

    println("start benchmark")
    println(now());
    results = run(b);
    println("end benchmark")
    println(now());

    dframe = DataFrame(hcat(Array{Float64}(1:b.params.samples), results.times), :auto);
    open("kGPU_M_$(x_len)x$(y_len).csv", "a");
    CSV.write("kGPU_M_$(x_len)x$(y_len).csv", dframe);
    println("executed $(x_len)x$(y_len) benchmark");

    solver_GPU = nothing; # aim to free up memory
    GC.gc();
    CUDA.reclaim();

    println("$(CUDA.memory_status())")

end;

#=
x_len = 150;
y_len = 150;

x_cutoff = 100;
y_cutoff = 100;

xs_full = LinRange(0, 2pi, x_len+1);
ys_full = LinRange(0, 2pi, y_len+1);

xs = Array{Float64}(xs_full[1:x_len]);
ys = Array{Float64}(ys_full[1:y_len]);

ω0 = rand(x_len, y_len);

function source_fn(x, y)
    return -4*cos(4*y);;
end;

source = [source_fn(x, y) for x in xs, y in ys];
source_hat = rfft(source)

forcing_freq = 4;




solver_CPU = kGPU_M.solver{Array}()

kGPU_M.init_solver(solver_CPU, dt, Re, n_iter, xs, ys, x_cutoff, y_cutoff, ω0, source_hat, forcing_freq);

solver_GPU = Adapt.adapt_structure(CuArray, solver_CPU);

println("created GPU solver");

measures = kGPU_M.flow_measures{CuArray}(solver_GPU);

println("created flow measures struct")

kGPU_M.iter(solver_GPU, measures, 1);

println("executed 1 iteration");

b = @benchmarkable kGPU_M.iter(solver_GPU, measures, 1);
b.params.seconds = 1000;
b.params.samples = 100000;

results = run(b)

dframe = DataFrame(hcat(Array{Float64}(1:b.params.samples), results.times), :auto);
open("kGPU_M_02.csv", "a");
CSV.write("kGPU_M_02.csv", dframe);

println("executed benchmark");
=#



