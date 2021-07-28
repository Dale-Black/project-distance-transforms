### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a2688830-ea64-11eb-3a54-534fc3ee29f6
begin
	let
		using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
		Pkg.add("BenchmarkTools")
		Pkg.add("Plots")
		Pkg.add("Revise")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using Revise
	using PlutoUI
	using BenchmarkTools
	using Plots
	using DistanceTransforms
end

# ╔═╡ 7110c950-752d-4a6b-90a5-a827519b321c
using DataFrames

# ╔═╡ 381f2f6f-3152-4bd8-b3bf-15851030fcba
using CSV

# ╔═╡ 8ee25991-4258-40e1-b581-3dc85efb0cc3
TableOfContents()

# ╔═╡ f99d2caa-c29d-40a7-afe1-d9486275051c
md"""
## Benchmark
"""

# ╔═╡ dea471a1-027a-490b-b52e-d877cdd89bd9
begin
	edt_mean = []
	edt_std = []
	
	sedt_mean = []
	sedt_std = []
	
	sedtP_mean = []
	sedtP_std = []
	
	cdt_mean = []
	cdt_std = []
	
	for n in 1:1000:10000
		arr = rand([0, 1], n, n)
		
		# EDT
		edt = @benchmark euclidean_distance_transform($arr)
		push!(edt_mean, BenchmarkTools.mean(edt).time)
		push!(edt_std, BenchmarkTools.std(edt).time)
		
		# Vanilla SEDT
		arr_bool = boolean_indicator(arr)
        dt = Array{Float32}(undef, size(arr))
        v = ones(Int64, size(arr))
        z = zeros(Float32, size(arr) .+ 1)
		sedt = @benchmark squared_euclidean_distance_transform($arr_bool, $dt, $v, $z)
		push!(sedt_mean, BenchmarkTools.mean(sedt).time)
		push!(sedt_std, BenchmarkTools.std(sedt).time)
		
		# Parallel SEDT
		nthreads = Threads.nthreads()
		sedtP = @benchmark squared_euclidean_distance_transform(
			$arr_bool, $dt, $v, $z, $nthreads
			)
		push!(sedtP_mean, BenchmarkTools.mean(sedtP).time)
		push!(sedtP_std, BenchmarkTools.std(sedtP).time)
		
		# CDT
		cdt = @benchmark chamfer_distance_transform($arr)
		push!(cdt_mean, BenchmarkTools.mean(cdt).time)
		push!(cdt_std, BenchmarkTools.std(cdt).time)
	end
end

# ╔═╡ aad68125-5287-4cd3-83a2-508c87682f2b
edt_mean[1]

# ╔═╡ f3ecec91-f836-4872-9dbd-5e001b1d01b8
begin
	x = collect(1:length(edt_mean))
	x_new = ones(Int32, size(x))
	for i in 2:length(x)
		x_new[i] = x[i] * (200^2)
	end
end

# ╔═╡ e7b0f68d-c7b9-414d-b98f-86fe46b3715c
begin
	Plots.scatter(
		x_new,
		edt_mean, 
		label="euclidean DT",
		xlabel = "Array size (elements)",
		ylabel = "Time (ns)"
		)
	Plots.scatter!(x_new, sedt_mean, label="squared euclidean DT")
	Plots.scatter!(x_new, sedtP_mean, label="parallel squared euclidean DT")
	Plots.scatter!(x_new, cdt_mean, label="chamfer DT")
end

# ╔═╡ bafd4bbc-b6a1-458e-a064-5dd940bd897d
df = DataFrame(
	edt_mean = edt_mean,
	sedt_mean = sedt_mean,
	sedtP_mean = sedtP_mean,
	cdt_mean = cdt_mean,
	edt_std = edt_std,
	sedt_std = sedt_std,
	sedtP_std = sedtP_std,
	cdt_std = cdt_std
	)

# ╔═╡ cd0f6bce-ab33-47d3-a767-ec9269d4e39e
path = "/Users/daleblack/Google Drive/dev/julia/project-distance-transforms/julia/timing/pluto_notebooks/cpu_dt.csv"

# ╔═╡ 879af282-2f61-436a-bc68-6d6f1db6e069
CSV.write(path, df)

# ╔═╡ Cell order:
# ╠═a2688830-ea64-11eb-3a54-534fc3ee29f6
# ╠═8ee25991-4258-40e1-b581-3dc85efb0cc3
# ╟─f99d2caa-c29d-40a7-afe1-d9486275051c
# ╠═dea471a1-027a-490b-b52e-d877cdd89bd9
# ╠═aad68125-5287-4cd3-83a2-508c87682f2b
# ╠═f3ecec91-f836-4872-9dbd-5e001b1d01b8
# ╠═e7b0f68d-c7b9-414d-b98f-86fe46b3715c
# ╠═7110c950-752d-4a6b-90a5-a827519b321c
# ╠═bafd4bbc-b6a1-458e-a064-5dd940bd897d
# ╠═381f2f6f-3152-4bd8-b3bf-15851030fcba
# ╠═cd0f6bce-ab33-47d3-a767-ec9269d4e39e
# ╠═879af282-2f61-436a-bc68-6d6f1db6e069
