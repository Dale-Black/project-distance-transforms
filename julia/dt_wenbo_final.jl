### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# ╔═╡ aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("/Users/wenboli/Desktop/ssd/dependency")
	Pkg.Registry.update()
end

# ╔═╡ b05b4228-fc30-4a74-8a99-f9b8d53aff83
begin
	using PlutoUI
	using CairoMakie
	using LinearAlgebra
	using Luxor
	using BenchmarkTools
	using ColorTypes
	using ActiveContours
	using Printf
	using FLoops
	using ImageMorphology
	using LoopVectorization
end

# ╔═╡ 45c7f27a-b43c-4781-918a-51aebd273014
TableOfContents()

# ╔═╡ 45b90a66-956a-4b35-9e88-7621fd5c9c1a
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)

# ╔═╡ 2d1b3771-0158-44ed-b47c-f091622c17af
function boolean_indicator(img::BitArray)
	f = similar(img, Float32)
	@turbo for i in CartesianIndices(f)
           f[i] = img[i] ? 0f0 : 1f10
    end
	return f
end

# ╔═╡ 0e4d3711-f29b-4833-8a3e-a91e4375eec0
md"""
## Multi-Thded and Nonmulti-Thded
"""

# ╔═╡ e75c9e76-3634-4bde-abb8-5b4827eb089e
md"""
## 1D
"""

# ╔═╡ 93de6d70-46ff-414f-9185-72975b252cbe
begin
	function DT1helperA!(f)
		# i == -1 && j == -1
		i=length(f) 
		while i>0
			f[i]=1f10
			i-=1
		end
	end
	function DT1helperB!(f, j)
		# i == -1 && j != -1
		temp=1
		while j>0
			f[j]=temp^2
			j-=1
			temp+=1
		end
	end
	function DT1helperC!(f, i)
		# i != -1 && j == -1
		temp=1
		l = length(f)
		while i<=l
			f[i]=temp^2
			i+=1
			temp+=1
		end
	end
	function DT1helperD!(f, i, j)
		# i != -1 && j != -1
		temp=1
		while(i<=j)
			f[i]=f[j]=temp^2
			temp+=1
			i+=1
			j-=1
		end
	end
	function DT1Wenbo!(f)
		pointerA = 1
		l = length(f)
		while pointerA <= l
			while pointerA <= l && f[pointerA] == 0
				pointerA+=1
			end
			pointerB = pointerA
			while pointerB <= l && f[pointerB] == 1f10
				pointerB+=1
			end
			if pointerB > length(f)
				if pointerA == 1
					DT1helperA!(f)
				else
					DT1helperC!(f, pointerA)
				end
			else
				if pointerA == 1
					DT1helperB!(f, pointerB-1)
				else
					DT1helperD!(f, pointerA, pointerB-1)
				end
			end
			pointerA=pointerB
		end
	end
	function DT1Wenbo(f)
		f = boolean_indicator(f)
		DT1Wenbo!(f)
		return f
	end
end

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ 0509a408-2dbc-483f-bd5f-b9832268845a
begin
	function encode(leftD, rightf)
		if rightf == 1f10
			return -leftD
		end
		idx = 0
		while(rightf>1)
			rightf  /=10
			idx+=1 
		end
		return -leftD-idx/10-rightf/10
	end
	function decode(curr)	
		curr *= -10   				
		temp = Int(floor(curr))		
		curr -= temp 				
		if curr == 0
			return 1f10
		end
		temp %= 10
		while temp > 0
			temp -= 1
			curr*=10
		end
		return round(curr)
	end
	function DT2Helper!(f)
		l = length(f)
		pointerA = 1
		while pointerA<=l && f[pointerA] <= 1
			pointerA += 1
		end
		p = 0
		while pointerA<=l
			curr = f[pointerA]
			prev = curr
			temp = min(pointerA-1, p+1)
			p = 0
			while (0 < temp)
				fi = f[pointerA-temp]
				fi = fi < 0 ? decode(fi) : fi
				newDistance = muladd(temp, temp, fi)
				if newDistance < curr
					curr = newDistance
					p = temp
				end
				temp -= 1
			end
			temp = 1
			templ = length(f) - pointerA
			while (temp <= templ && muladd(temp, temp, -curr) < 0)
				curr = min(curr, muladd(temp, temp, f[pointerA+temp]))
				temp += 1
			end
			f[pointerA] = encode(curr, prev)
			# end
			pointerA+=1
			while pointerA<=l && f[pointerA] <= 1
				pointerA += 1
			end
		end
		i = 0
		while i<l
			i+=1
			f[i] = floor(abs(f[i]))
		end
	end
	function DT2WenboA!(f)
		for i in axes(f, 1)
			DT1Wenbo!(@view(f[i, :]))
		end
		for i in axes(f, 2)
			DT2Helper!(@view(f[:,i]))
		end
	end
	function DT2WenboB!(f)
		Threads.@threads for i in axes(f, 1)
			DT1Wenbo!(@view(f[i, :]))
		end
		Threads.@threads for i in axes(f, 2)
			DT2Helper!(@view(f[:,i]))
		end
	end
	function DT2Wenbo(f)
		f = boolean_indicator(f)
		DT2tf! = length(f) > 2700 && Threads.nthreads()>1 ?  DT2WenboB! : DT2WenboA!
		DT2tf!(f)
		return f
	end
end

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ 3ce8b2a0-21ba-4b37-8560-60b22b8772a9
begin
	function DT3WenboA!(f)
		for i in axes(f, 3)
		    DT2WenboA!(@view(f[:, :, i]))
		end
		for i in CartesianIndices(f[:,:,1])
			DT2Helper!(@view(f[i, :]))
		end
	end 
	function DT3WenboB!(f)
		Threads.@threads for i in axes(f, 3)
		    DT2WenboB!(@view(f[:, :, i]))
		end
		Threads.@threads for i in CartesianIndices(f[:,:,1])
			DT2Helper!(@view(f[i, :]))
		end
	end
	function DT3Wenbo(f)
		f = boolean_indicator(f)
		DT3tf! = length(f) > 2700 && Threads.nthreads()>1 ?  DT3WenboB! : DT3WenboA!
		DT3tf!(f)
		return f
	end
end

# ╔═╡ 8f46b1a3-09e4-44b8-8589-d8aa7fe381d1
md"""
## Timing: Wenbo vs ImageMorphology
"""

# ╔═╡ 14964c1d-32ef-48f0-a30c-5ad3d1faa34e
euclideanImageMorphology(img::BitArray) = distance_transform(feature_transform(img))

# ╔═╡ ac93baef-cd5f-4c4d-bd18-4d234e9b5883
euclideanImageMorphology(img) = distance_transform(feature_transform(Bool.(img)))

# ╔═╡ c4f655be-8b8d-44c1-869a-50a8aabb36bb
begin
	img1D = rand([0, 1], 200)
	img2D = rand([0, 1], 200, 400)
	img3D = rand([0, 1], 200, 400, 600)
	img2D4k = rand([0, 1], 3840, 2160)
	img1Df = Bool.(rand([0, 1], 200))
	img2Df = Bool.(rand([0, 1], 200, 800))
	img3Df = Bool.(rand([0, 1], 200, 400, 600))
	img2D4kf = Bool.(rand([0, 1], 3840, 2160))
	img2DSmallf = Bool.(rand([0, 1], 50, 50))
	img2D4krf = Bool.(rand([0, 1], 2160, 3840))
	"Test inputs created."
end

# ╔═╡ 1600c1dd-c4f4-4f3c-90ef-77684c34ecb1
md"""
### 1D
"""

# ╔═╡ 9fb81dce-10cc-4eee-80ba-3e98c4671c3f
let
	rslt1 = euclideanImageMorphology(img1Df);
	rslt1 .^ 2
	rslt2 = DT1Wenbo(img1Df)
	for i in CartesianIndices(rslt1)
		if rslt1[i] - rslt2[i] != 0.0
			"failed"
			break
		end
	end
	"1D: Test Passed!"
end

# ╔═╡ c935ea04-1905-4aff-98ea-1ce3afe0e53b
@benchmark DT1Wenbo($img1Df)

# ╔═╡ 685c72e3-24bc-4f3a-9dbb-d4c1d70d8166
@benchmark euclideanImageMorphology($img1Df)

# ╔═╡ d0c0ebdf-ff7b-4604-8c9a-acc77bc2cee0
md"""
### 2D
"""

# ╔═╡ c12b2514-bd9e-466c-a796-f7a2dd772891
let
	rslt1 = euclideanImageMorphology(img2Df);
	rslt1 .^ 2
	rslt2 = DT2Wenbo(img2Df)
	
	for i in CartesianIndices(rslt1)
		if rslt1[i] - rslt2[i] != 0.0
			"failed"
			break
		end
	end
	"2D: Test Passed!"
end

# ╔═╡ 026f5171-5206-4beb-abd9-cb8b27402e2a
@benchmark DT2Wenbo($img2Df)

# ╔═╡ 994460cf-cba5-4a8f-8018-da5aa63c3b97
@benchmark euclideanImageMorphology($img2Df)

# ╔═╡ d0f6676f-ecd9-4a8d-8f23-0ae656dd84dc
md"""
### 3D
"""

# ╔═╡ 20a72ac8-d547-432c-b979-878f5caecfe2
let
	n = 200
	f = Bool.(rand([0, 1], n, n, n))
	rslt1 = euclideanImageMorphology(f);
	rslt1 .^ 2
	rslt2 = DT3Wenbo(f)
	
	
	for i in CartesianIndices(rslt1)
		if rslt1[i] - rslt2[i] != 0.0
			"failed"
			break
		end
	end
	"3D: Test Passed!"
end

# ╔═╡ d3d0d39f-381e-4e0b-a0e7-10441cba9956
@benchmark DT3Wenbo($img3Df)

# ╔═╡ a149cb4c-ec20-40db-a6d6-02bb8bca4eb7
@benchmark euclideanImageMorphology($img3Df)

# ╔═╡ Cell order:
# ╟─aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╟─b05b4228-fc30-4a74-8a99-f9b8d53aff83
# ╟─45c7f27a-b43c-4781-918a-51aebd273014
# ╟─45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╟─2d1b3771-0158-44ed-b47c-f091622c17af
# ╟─0e4d3711-f29b-4833-8a3e-a91e4375eec0
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╟─93de6d70-46ff-414f-9185-72975b252cbe
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╟─0509a408-2dbc-483f-bd5f-b9832268845a
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╟─3ce8b2a0-21ba-4b37-8560-60b22b8772a9
# ╟─8f46b1a3-09e4-44b8-8589-d8aa7fe381d1
# ╟─14964c1d-32ef-48f0-a30c-5ad3d1faa34e
# ╟─ac93baef-cd5f-4c4d-bd18-4d234e9b5883
# ╟─c4f655be-8b8d-44c1-869a-50a8aabb36bb
# ╟─1600c1dd-c4f4-4f3c-90ef-77684c34ecb1
# ╟─9fb81dce-10cc-4eee-80ba-3e98c4671c3f
# ╠═c935ea04-1905-4aff-98ea-1ce3afe0e53b
# ╠═685c72e3-24bc-4f3a-9dbb-d4c1d70d8166
# ╟─d0c0ebdf-ff7b-4604-8c9a-acc77bc2cee0
# ╟─c12b2514-bd9e-466c-a796-f7a2dd772891
# ╠═026f5171-5206-4beb-abd9-cb8b27402e2a
# ╠═994460cf-cba5-4a8f-8018-da5aa63c3b97
# ╟─d0f6676f-ecd9-4a8d-8f23-0ae656dd84dc
# ╟─20a72ac8-d547-432c-b979-878f5caecfe2
# ╠═d3d0d39f-381e-4e0b-a0e7-10441cba9956
# ╠═a149cb4c-ec20-40db-a6d6-02bb8bca4eb7
