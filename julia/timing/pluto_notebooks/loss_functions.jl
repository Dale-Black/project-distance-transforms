### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 10227fe0-fbd6-11eb-3d08-a3d424535e44
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
		Pkg.add(url="https://github.com/Dale-Black/DistanceTransforms.jl")
		Pkg.add(url="https://github.com/Dale-Black/Losers.jl")
	end
	
	using PlutoUI
	using BenchmarkTools
	using Plots
	using DataFrames
	using CSV
	using CUDA
	using DistanceTransforms
	using Losers
end

# ╔═╡ 8556a10f-0a3d-4572-85e1-89eb53689ba9
TableOfContents()

# ╔═╡ 0a794f81-96c0-4c93-800a-9a7f002db06a
md"""
## Benchmarks
"""

# ╔═╡ 693fb012-f5f5-4df8-a438-85cd09d746cb
begin
	hd_mean = []
	hd_std = []
	
	dice_mean = []
	dice_std = []
	
	hdGPU_mean = []
	hdGPU_std = []
	
	diceGPU_mean = []
	diceGPU_std = []
	
	for n in 1:100:1000
		
		# HD CPU
		x1 = DistanceTransforms.boolean_indicator(rand([0, 1], n, n))
		tfm1 = DistanceTransforms.SquaredEuclidean(x1)
		x1_dtm = DistanceTransforms.transform(x1, tfm1)
		hd = @benchmark hausdorff($x1, $x1, $x1_dtm, $x1_dtm)
		
		push!(hd_mean, BenchmarkTools.mean(hd).time)
		push!(hd_std, BenchmarkTools.std(hd).time)
		
		# DICE CPU
		x2 = rand([0, 1], n, n)
		dice = @benchmark dice($x1, $x1)
		
		push!(dice_mean, BenchmarkTools.mean(dice).time)
		push!(dice_std, BenchmarkTools.std(dice).time)
	end
end

# ╔═╡ 61a58dd6-8f00-40cb-b1bc-657d066cc1ff
md"""
## Visualize
"""

# ╔═╡ d35c8aaa-d55e-4771-be0c-ff90bfdc7034
begin
	x = collect(1:length(hd_mean))
	x_new = ones(Int32, size(x))
	for i in 2:length(x)
		x_new[i] = x[i] * (200^2)
	end
end

# ╔═╡ 524499d2-b293-4d2f-bc92-c9ec33efa74c
begin
	Plots.scatter(
		x_new,
		hd_mean, 
		label="Hausdorff CPU",
		xlabel = "Array size (elements)",
		ylabel = "Time (ns)"
		)
	Plots.scatter!(x_new, dice_mean, label="Dice CPU")
	# Plots.scatter!(x_new, sedtP_mean, label="parallel squared euclidean DT")
	# Plots.scatter!(x_new, cdt_mean, label="chamfer DT")
end

# ╔═╡ 39e44a16-0054-46d2-8af2-11f68c061388
md"""
## Save
"""

# ╔═╡ a7f90612-9a98-4239-acd8-235a2d13cf9e
df = DataFrame(
	hd_mean = hd_mean,
	dice_mean = dice_mean,
	# hdGPU_mean = hdGPU_mean,
	# diceGPU_mean = diceGPU_mean,
	hd_std = hd_std,
	dice_std = dice_std,
	# hdGPU_std = hdGPU_std,
	# diceGPU_std = diceGPU_std
	)

# ╔═╡ 6d8e381c-4e3a-4926-b691-75b4d522d13c
path = "/Users/daleblack/Google Drive/dev/julia/project-distance-transforms/julia/timing/pluto_notebooks/loss_function.csv"

# ╔═╡ 11028f5f-262d-4756-b9ad-219c51c5ff11
# CSV.write(path, df)

# ╔═╡ Cell order:
# ╠═10227fe0-fbd6-11eb-3d08-a3d424535e44
# ╠═8556a10f-0a3d-4572-85e1-89eb53689ba9
# ╟─0a794f81-96c0-4c93-800a-9a7f002db06a
# ╠═693fb012-f5f5-4df8-a438-85cd09d746cb
# ╟─61a58dd6-8f00-40cb-b1bc-657d066cc1ff
# ╠═d35c8aaa-d55e-4771-be0c-ff90bfdc7034
# ╠═524499d2-b293-4d2f-bc92-c9ec33efa74c
# ╟─39e44a16-0054-46d2-8af2-11f68c061388
# ╠═a7f90612-9a98-4239-acd8-235a2d13cf9e
# ╠═6d8e381c-4e3a-4926-b691-75b4d522d13c
# ╠═11028f5f-262d-4756-b9ad-219c51c5ff11
