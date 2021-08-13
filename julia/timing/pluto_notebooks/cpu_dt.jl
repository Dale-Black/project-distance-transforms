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
		Pkg.add("PlutoUI")
		Pkg.add("DataFrames")
		Pkg.add("CSV")
		Pkg.add("CUDA")
		Pkg.add(url="https://github.com/JuliaFolds/FoldsCUDA.jl")
		Pkg.add("FLoops")
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
	end
	
	using PlutoUI
	using BenchmarkTools
	using Plots
	using DataFrames
	using CSV
	using DistanceTransforms
	using CUDA
	using FoldsCUDA
	using FLoops
end

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
	
	for n in 2:50:100
		# EDT
		x1 = rand([0, 1], n, n)
		edt = @benchmark euclidean($x1)
		
		push!(edt_mean, BenchmarkTools.mean(edt).time)
		push!(edt_std, BenchmarkTools.std(edt).time)
		
		# SEDT
		x2 = DistanceTransforms.boolean_indicator(rand([0, 1], n, n))
		tfm2 = DistanceTransforms.SquaredEuclidean(x2)
		sedt = @benchmark DistanceTransforms.transform($x2, $tfm2)
		
		push!(sedt_mean, BenchmarkTools.mean(sedt).time)
		push!(sedt_std, BenchmarkTools.std(sedt).time)
		
		# SEDT threaded
		x3 = DistanceTransforms.boolean_indicator(rand([0, 1], n, n))
		tfm3 = DistanceTransforms.SquaredEuclidean(x3)
		nthreads = Threads.nthreads()
		sedtP = @benchmark DistanceTransforms.transform!($x3, $tfm3, $nthreads)
		
		push!(sedtP_mean, BenchmarkTools.mean(sedtP).time)
		push!(sedtP_std, BenchmarkTools.std(sedtP).time)
		
		# CDT
		x5 = rand([0, 1], n, n)
		tfm5 = DistanceTransforms.Chamfer(x5)
		cdt = @benchmark DistanceTransforms.transform($x5, $tfm5)
		
		push!(cdt_mean, BenchmarkTools.mean(cdt).time)
		push!(cdt_std, BenchmarkTools.std(cdt).time)
	end
end

# ╔═╡ ebeb7b55-6004-4132-8560-0c5f6c58dc16
begin
	for n in 4:6
		x4 = CuArray(rand([0, 1], n, n))
		sz = size(x4)
		dt = CUDA.zeros(Float32, sz),
		v = CUDA.ones(Int64, sz),
		z = CUDA.zeros(Float32, sz .+ 1)
	end
end

# ╔═╡ 785866a9-bde6-4108-bc08-f5dd8a856f1e

		tfm4 = DistanceTransforms.SquaredEuclidean(x4, dt, v, z)
		sedtGPU = @benchmark DistanceTransforms.trasform!($x4, $tfm4)
		
		push!(sedtGPU_mean, BenchmarkTools.mean(sedtGPU).time)
		push!(sedtGPU_std, BenchmarkTools.std(sedtGPU).time)

# ╔═╡ c8178d6b-b6d3-4bf6-b715-bd251986d8eb
l = CuArray(DistanceTransforms.boolean_indicator(rand([0, 1], 2, 2)))

# ╔═╡ c75cd743-8739-49fb-afe8-797c111091aa
# dt = CUDA.zeros(Float32, size(l) .+ 1)

# ╔═╡ 0acb0c3d-7f6c-49f8-915e-7088b64e2b84
# v = CUDA.ones(Int64, size(l))

# ╔═╡ c80eeb49-2902-4429-9135-e193847f502c
md"""
## Visualize
"""

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

# ╔═╡ 1f2be9ca-0569-4f3c-9f25-8cb6a1b8fb94
md"""
## Save
"""

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
# CSV.write(path, df)

# ╔═╡ Cell order:
# ╠═a2688830-ea64-11eb-3a54-534fc3ee29f6
# ╠═8ee25991-4258-40e1-b581-3dc85efb0cc3
# ╟─f99d2caa-c29d-40a7-afe1-d9486275051c
# ╠═dea471a1-027a-490b-b52e-d877cdd89bd9
# ╠═ebeb7b55-6004-4132-8560-0c5f6c58dc16
# ╠═785866a9-bde6-4108-bc08-f5dd8a856f1e
# ╠═c8178d6b-b6d3-4bf6-b715-bd251986d8eb
# ╠═c75cd743-8739-49fb-afe8-797c111091aa
# ╠═0acb0c3d-7f6c-49f8-915e-7088b64e2b84
# ╟─c80eeb49-2902-4429-9135-e193847f502c
# ╠═f3ecec91-f836-4872-9dbd-5e001b1d01b8
# ╠═e7b0f68d-c7b9-414d-b98f-86fe46b3715c
# ╠═1f2be9ca-0569-4f3c-9f25-8cb6a1b8fb94
# ╠═bafd4bbc-b6a1-458e-a064-5dd940bd897d
# ╠═cd0f6bce-ab33-47d3-a767-ec9269d4e39e
# ╠═879af282-2f61-436a-bc68-6d6f1db6e069
