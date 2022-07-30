### A Pluto.jl notebook ###
# v0.19.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate("/Volumes/980ProSSD/dependency")
	# Pkg.instantiate()
	# Pkg.Registry.update()
	# Pkg.add("PlutoUI")
	# Pkg.add("CairoMakie")
	# Pkg.add("Luxor")
	# Pkg.add("BenchmarkTools")
	# Pkg.add("ColorTypes")
	# Pkg.add(url="https://github.com/Dale-Black/ActiveContours.jl")
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
end

# ╔═╡ 45c7f27a-b43c-4781-918a-51aebd273014
TableOfContents()

# ╔═╡ 45b90a66-956a-4b35-9e88-7621fd5c9c1a
boolean_indicator(f) = @. ifelse(f == 0, 1f10, 0f0)
#boolean_indicator(f) = replace!(@view(f[:,:]), 0=>1f10, 1=>0)

# ╔═╡ a6c9dc03-74ce-4174-8fd9-6895d117aa4a
function boolean_indicator_try(f)
	for i = 1 : length(f)
		if (f[i] == 0)
			f[i] = 10
			for j = 1:9
				f[i] *= 10
			end
		else
			f[i] = 0f0
		end
	end
end

# ╔═╡ 0c577fc5-d45d-448a-82c0-6f1abc9577c3
begin
	bitest = [1 1 0 0 0 1 1]
	boolean_indicator_try(bitest)
	bitest
end

# ╔═╡ f873e7c2-7dac-4925-b6b5-5887b461418b
@benchmark boolean_indicator([1 1 0 0 0 1 1])

# ╔═╡ 6e2adb2c-93f4-4a19-b93c-75b75a9116f3
@benchmark boolean_indicator_try([1 1 0 0 0 1 1])

# ╔═╡ 8aab09c7-07fe-4691-9e1d-35e2e215bc5d
md"""
## TODO
- Make sure the 3D distance transform is working
- Make sure the distance transforms (1D, 2D, 3D) all have a working non-allocating version (use BenchmarkTools and a profiler to validate this)
"""

# ╔═╡ e75c9e76-3634-4bde-abb8-5b4827eb089e
md"""
## 1D
"""

# ╔═╡ 17d3777b-69d4-40c1-84c1-2bc6a737d52a
"""
One-dimensional generalized distance transform
Input: f - the sampled function
Output: D - distance transform
Based on the paper:
P. Felzenszwalb, D. Huttenlocher
Distance Transforms of Sampled Functions
Cornell Computing and Information Science Technical Report TR2004-1963, September 2004
"""

# ╔═╡ 258346da-e792-11ec-06dc-5906e95d24e2
function DT1(f; D=zeros(length(f)), v=ones(Int32, length(f)), z=ones(length(f)+1))
	z[1] = -Inf32
	z[2] = Inf32
	k = 1; # Index of the rightmost parabola in the lower envelope
	for q = 2:length(f)
		s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    while s ≤ z[k]
	        k -= 1
	        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2*q - 2*v[k])
	    end
	    k += 1
	    v[k] = q
	    z[k] = s
		z[k+1] = Inf32
		# if k ≤ length(f) - 1
		# 	z[k+1] = Inf32
		# else
		# 	z[k] = Inf32
		# end
	end
	k = 1
	for q in 1:length(f)
	    while z[k+1] < q
	        k = k+1
	    end
	    D[q] = (q-v[k])^2 + f[v[k]]
	end
	return D
end

# ╔═╡ 098b8119-26e8-4002-84c4-c8478d3c5019
begin
	function DT1helper(input, output, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(output))
				output[i]=1f10
				i=i+1
			end
		elseif(i==-1)
			temp=1
			while(j>0)
				output[j]=temp^2
				j=j-1
				temp=temp+1
			end
		elseif(j==-1)
			temp=1
			while(i<=length(output))
				output[i]=temp^2
				i=i+1
				temp=temp+1
			end
		else
			temp=1
			while(i<=j)
				output[i]=output[j]=temp^2
				temp=temp+1
				i=i+1
				j=j-1
			end
		end
		return output
	end
	function DT1Wenbo(f; output=zeros(length(f)), pointerA=1, pointerB=1)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		while (pointerA<=length(f))
			if(f[pointerA] == 0)
				output[pointerA]=0
				pointerA=pointerA+1
				pointerB=pointerB+1
			else
				while(pointerB <= length(f) && f[pointerB]==1f10)
					pointerB=pointerB+1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						output = DT1helper(f, output, -1, -1)
					else
						output = DT1helper(f, output, pointerA, -1)
					end
				else
					if (pointerA == 1)
						output = DT1helper(f, output, -1, pointerB-1)
					else
						output = DT1helper(f, output, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
		return output
	end
end

# ╔═╡ 93de6d70-46ff-414f-9185-72975b252cbe
begin
	function DT1helper!(f, i, j)
		if (i==-1 && j==-1)
			i=1
			while(i<=length(f))
				f[i]=1f10
				i+=1
			end
		elseif(i==-1)
			temp=1
			while(j>0)
				f[j]=temp^2
				j-=1
				temp+=1
			end
		elseif(j==-1)
			temp=1
			while(i<=length(f))
				f[i]=temp^2
				i+=1
				temp+=1
			end
		else
			temp=1
			while(i<=j)
				f[i]=f[j]=temp^2
				temp+=1
				i+=1
				j-=1
			end
		end
	end
	function DT1Wenbo!(f)
		#assume length(f)>0
		#This is a one pass algorithm
		#time complexity=O(n), Space complexity=O(1)
		pointerA = pointerB = 1
		while (pointerA<=length(f))
			if(f[pointerA] == 0)
				pointerA+=1
				pointerB+=1
			else
				while(pointerB <= length(f) && f[pointerB]==1f10)
					pointerB+=1
				end
				if (pointerB > length(f))
					if (pointerA == 1)
						DT1helper!(f, -1, -1)
					else
						DT1helper!(f, pointerA, -1)
					end
				else
					if (pointerA == 1)
						DT1helper!(f, -1, pointerB-1)
					else
						DT1helper!(f, pointerA, pointerB-1)
					end
				end
				pointerA=pointerB
			end
		end
	end
end

# ╔═╡ 2280361b-068d-4d5e-ae30-7e66c42ee665
function testDT1Wenbo!(size,numCases)
	for i in 1:numCases
		testInput=boolean_indicator(rand([0, 1], size));
		rslt = DT1(testInput);
		DT1Wenbo!(testInput);
		if (testInput != rslt)
			println(testInput);
			println(rslt);
			println();
			return "Failed.";
		end 
	end
	return "Passed.";
end

# ╔═╡ c8e739c0-940d-47b4-be2e-b0345a7e362f
testDT1Wenbo!(15,5000)

# ╔═╡ 0469116a-7e10-44c5-801d-f5ce683dc486
begin
	f = [1 1 0 0 0 1 0 0 1 1 1 0 1 1];
	D, v, z =zeros(length(f)), ones(Int32, length(f)), ones(length(f));
	arg1 = zeros(length(f));
end

# ╔═╡ 30ae8059-4b4e-4026-966e-7d6320912b5c
@benchmark DT1($boolean_indicator($f); D=$D, v=$v, z=$z) 

# ╔═╡ 7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
@benchmark DT1Wenbo($boolean_indicator($f); output=$arg1, pointerA=1, pointerB=1) 

# ╔═╡ 5f9287b3-c7c3-4c0d-bf3b-d3161c4bb249
@benchmark DT1Wenbo!($boolean_indicator($f))

# ╔═╡ 2d02a6fd-7a23-4790-b8da-7b7150482a85
md"""
## 2D
"""

# ╔═╡ 29cb3c08-a702-4564-ad41-664f312913b0
begin
	img2d1 = [
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0 0 0 0
		0 0 0 0 0 0 0 0	0 0 0
		0 0 0 1 1 1 0 0 0 0 0
		0 0 1 0 0 1 0 0 0 0 0
		0 0 1 0 0 1 1 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 1 0 0 0 0 1 0 0 0
		0 0 0 1 1 1 1 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0	
		0 0 0 0 0 0 0 0 0 0 0
	];
	img2d2 = [
		 0 1 1 1 0 0 0 1 1
		 1 1 1 1 1 0 0 0 1
		 1 0 0 0 1 0 0 1 1
		 1 0 0 0 1 0 1 1 0
		 1 0 0 0 1 1 0 1 0
		 1 1 1 1 1 0 0 1 0
		 0 1 1 1 0 0 0 0 1
	];
	img2d3 = [
		0	0	1	0	0	1	0	
		1	0	0	1	1	0	1	
		0	0	0	1	0	1	0	
		0	0	0	0	0	0	0	
		1	0	1	0	1	1	1	
		0	0	1	1	1	1	0	
		0	0	0	1	1	0	0	
	];
end

# ╔═╡ 2d7da285-7078-4567-8cdd-a8cefd737f57
function DT2(img; D=zeros(size(img)))
	for i = 1:size(img, 1)
	    D[i, :] = DT1(img[i, :])
	end
	for j = 1:size(img, 2)
	    D[:, j] = DT1(D[:, j])
	end
	return D
end

# ╔═╡ 535c4318-d29e-4640-9105-66b16d13bf88
begin
	# june 20 new implementation
	function DT2Helper(f, output)
		pointerA = 1;
		while (pointerA<=length(f))
			output[pointerA]=f[pointerA]
			#it is needless to deal with distance <= 1
			if(f[pointerA] > 1)
				temp = 1
				if (2 * pointerA > length(f))
					#2nd half
					while (output[pointerA]>1 && temp <= length(f) - pointerA)
						#right
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						#left
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
							temp += 1
					end
					#continue searching for left
					if(output[pointerA]>1)
						while (output[pointerA]>1 && temp < pointerA)
							if (f[pointerA-temp]<output[pointerA])
								output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
							end
							temp += 1
						end
					end
				else
					#1st half
					while (output[pointerA]>1 && temp < pointerA)
						#right
						if (f[pointerA+temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
						end
						#left
						if (f[pointerA-temp]<output[pointerA])
							output[pointerA]=min(output[pointerA], f[pointerA-temp]+temp^2)
						end
						temp += 1
					end
					#continue searching for right
					if(output[pointerA]>1)
						while (output[pointerA]>1 && temp <= length(f) - pointerA)
							if (f[pointerA+temp]<output[pointerA])
								output[pointerA]=min(output[pointerA], f[pointerA+temp]+temp^2)
							end
							temp += 1
						end
					end
				end
			end
			pointerA+=1
		end
		return output
	end
	function DT2Wenbo(img; D=zeros(size(img)), pointerA=1, pointerB=1)
		# This is a worst case = O(n^3) implementation
		for i = 1:size(img, 1)
		    D[i, :] = DT1Wenbo(img[i, :]; output=D[i, :], pointerA=pointerA, pointerB=pointerB) 
		end
		# june 20 new implementation
		for j = 1:size(img, 2)
		    D[:, j] = DT2Helper(D[:, j], D[:, j]) 
		end
		return D
	end
end

# ╔═╡ 56a72a97-1505-4927-9106-9e4e0111dba9
begin
	function encode(curr, prev)
		idx = 0
		while(prev>1)
			prev/=10
			idx+=1 
		end
		return -curr-idx/10-prev/10
	end
	function decode(curr)
		curr *= -10
		temp = Int(floor(curr))
		curr -= temp
		for i = 1 : temp%10
			curr*=10
		end
		return round(curr)
	end
	# july 27 modified
	function DT2Helper!(f)
		pointerA = 1
		l = length(f) # faster?*************
		while (pointerA<=l)
			#it is needless to deal with distance <= 1
			if(f[pointerA] > 1)
				curr = f[pointerA]
				prev = curr
				#left bound : temp < pointerA
				#right bound : temp <= l - pointerA
				#left
				temp = 1
				while (temp < pointerA && curr>temp*temp)
					#left
					if (f[pointerA-temp] >= 0)
						curr = min(curr, f[pointerA-temp]+temp*temp)
					else
						prevprev = decode(f[pointerA-temp])
						if (prevprev > 0)
							curr = min(curr, prevprev+temp*temp)
						end
					end
					temp += 1
				end
				#right
				temp = 1
				while (temp <= l - pointerA && curr>temp*temp)
					if (f[pointerA+temp] >= 0)
						curr = min(curr, f[pointerA+temp]+temp*temp)
					else
						prevprev = decode(f[pointerA+temp])
						if (prevprev > 0)
							curr = min(curr, prevprev+temp*temp)
						end
					end
					temp += 1
				end
				if (prev != curr && prev >1)
					f[pointerA] = encode(curr, prev)
				end
			end
			pointerA+=1
		end
		for i = 1:l
			f[i] = floor(abs(f[i]))
		end
	end
	function DT2Wenbo!(f2d)
		for i = 1:size(f2d, 1)
			DT1Wenbo!(@view(f2d[i, :]))
		end
		for i = 1:size(f2d, 2)
			DT2Helper!(@view(f2d[:, i])) 
		end
	end
end

# ╔═╡ 81472647-35a8-489e-a65d-b44586ddd179
begin
	function printErrorArea(input, output1, output2, x, y)
		x=min(x, size(input, 1)-4)
		x=max(x, 4)
		y=min(y, size(input, 2)-4)
		y=max(y, 4)
		for i = x-3:x+3
			for j = y-3:y+3
				if (input[i, j] == 0)				
					print("1");
				else							
					print("0");
				end

				print("\t");
			end
			println();
		end
		println();
		for i = x-3:x+3
			for j = y-3:y+3
				print(output1[i, j]);
				print("\t");
			end
			println();
		end
		println();
		for i = x-3:x+3
			for j = y-3:y+3
				print(output2[i, j]);
				print("\t");
			end
			println();
		end
		println();
	end
	function testDT2Wenbo!(size1, size2 , numCases)
		for i in 1:numCases
			randomInput = rand([0, 1], size1, size2);
			testInput = boolean_indicator(randomInput);
			rslt = DT2(testInput);
			DT2Wenbo!(testInput);
			for j = 1:size1
				for k = 1:size2
					if (testInput[j, k] != rslt[j, k])
						printErrorArea(boolean_indicator(randomInput), rslt, testInput, j, k);
						return "Failed.";
					end 
				end
			end
		end
		return "Passed.";
	end
	test2d4kresolu = rand([0, 1], 3840, 2160);
	arg2d1= zeros(size(test2d4kresolu));
	"Caught in 4k? So let's test in 4k resolution!" 
end

# ╔═╡ f80f3573-68a1-4732-be81-ffb8d4a45017
testDT2Wenbo!(3840, 2160, 10)

# ╔═╡ 8a5666f7-ee15-4f8c-847f-9fb998a89c7e
@benchmark DT2($boolean_indicator($test2d4kresolu); D=$arg2d1)

# ╔═╡ ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
@benchmark DT2Wenbo($boolean_indicator($test2d4kresolu); D=$arg2d1, pointerA=1, pointerB=1)

# ╔═╡ 2fd6d12f-5013-41ee-97b2-085d795ef3bd
@benchmark DT2Wenbo!($boolean_indicator($test2d4kresolu))

# ╔═╡ 71072c67-6584-4fc7-ad4d-b3d362333562
md"""
## 3D
"""

# ╔═╡ 4731b952-c5aa-4c56-93ef-806e260eef73
function DT3(f; D=zeros(size(f)))
	for i = 1:size(f, 3)
	    D[:, :, i] = DT2(f[:, :, i]; D=D[:, :, i])
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
	    	D[i, j, :] = DT1(D[i, j, :]; D= D[i, j, :])
		end
	end
	return D
end

# ╔═╡ f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
function DT3Wenbo(f; D=zeros(size(f)), pointerA=1, pointerB=1)
	for i = 1:size(f, 3)
	    D[:, :, i] = DT2Wenbo(f[:, :, i]; D=D[:, :, i], pointerA=pointerA, pointerB=pointerB)
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
	    	D[i, j, :] = DT2Helper(D[i, j, :], D[i, j, :])
		end
	end
	return D
end

# ╔═╡ 803376a5-96fa-4a15-a354-6a98940768e0
function DT3Wenbo!(f)
	for i = 1:size(f, 3)
	    DT2Wenbo!(@view(f[:, :, i]))
	end
	for i = 1:size(f, 1)
		for j = 1:size(f, 2)
			DT2Helper!(@view(f[i, j, :]))
		end
	end
end

# ╔═╡ aeabedd0-5719-46f2-9715-7b21d8e5bf3d
begin
	container=[]
	vol1 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol2 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol3 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol4 = [
	 0 0 0 0 0 0 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 1 0 0 0 0 0 1 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
	 0 0 0 0 0 0 0 0 0
]
	vol5 = [
	 0 0 0 0 1 0 0 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 1 0 0 0 1 0 0
	 0 1 0 0 0 0 0 1 0
	 1 0 0 0 0 0 0 0 1
	 0 1 0 0 0 0 0 1 0
	 0 0 1 0 0 0 1 0 0
	 0 0 0 1 0 1 0 0 0
	 0 0 0 0 1 0 0 0 0
]
	#level 1
	push!(container, vol1)
	#level 2
	push!(container, vol2)
	#level 3
	push!(container, vol3)
	#level 4
	push!(container, vol4)
	#level 5
	push!(container, vol5)
	#level 6
	push!(container, vol4)
	#level 7
	push!(container, vol3)
	#level 8
	push!(container, vol2)
	#level 9
	push!(container, vol1)
	test_vol1 = cat(container..., dims=3)
end;

# ╔═╡ ac8ed96d-bf48-4a2e-81f0-ae612024177c
@bind c PlutoUI.Slider(1:size(test_vol1, 3), default=1, show_value=true)

# ╔═╡ fad6db76-5bd3-4350-8180-eb9735f6a749
heatmap(test_vol1[:, :, c])

# ╔═╡ da9b8ae0-d446-47f6-8c03-2e145e83b7ed
begin
	function testDT3Wenbo!(size1, size2, size3, numCases)
		for i in 1:numCases
			randomInput = rand([0, 1], size1, size2, size3);
			testInput = boolean_indicator(randomInput);
			rslt = DT3(testInput);
			DT3Wenbo!(testInput);
			if (testInput != rslt)
				return "Failed.";
			end 
		end
		return "Passed.";
	end
	test3d1kresolu = rand([0, 1], 1080, 1080, 150);
	arg3d1= zeros(size(test3d1kresolu));
end

# ╔═╡ 3f1064a6-63e0-401b-8b57-7aaa4e097ccf
testDT3Wenbo!(1080, 1080, 150,5)

# ╔═╡ 137ec483-53d4-4ce2-826b-af9d06fe9c62
@benchmark DT3($boolean_indicator($test3d1kresolu); D=$arg3d1)

# ╔═╡ a590e6c0-2f97-4866-bd0d-08b7758a347c
@benchmark DT3Wenbo($boolean_indicator($test3d1kresolu); D=$arg3d1, pointerA=1, pointerB=1)

# ╔═╡ 67a91884-5169-4a32-aa56-35800a965e77
@benchmark DT3Wenbo!($boolean_indicator($test3d1kresolu))

# ╔═╡ 0e4d3711-f29b-4833-8a3e-a91e4375eec0
md"""
## Multi-Thded
"""

# ╔═╡ 9a0d9463-0002-4d0e-a368-3d7bba542e3d
nthreads = Threads.nthreads()

# ╔═╡ fb595c55-2995-462f-8a7f-982022370bd9
md"""
### 2D
"""

# ╔═╡ ae9df0c7-3dd4-4e11-8440-acba1faeb61b


# ╔═╡ 36a60c88-2c32-4d21-80c3-ce965a49438b
md"""
### 3D
"""

# ╔═╡ 406fcc98-5117-48d3-911f-91b350e6f9df
md"""
## GPU
"""

# ╔═╡ 4c14ef1c-887a-44e3-84f6-9812bd433858
md"""
### 2D
"""

# ╔═╡ ceabaea4-17af-4fc2-8af4-b52c5fc886f5
md"""
### 3D
"""

# ╔═╡ Cell order:
# ╠═aec0d43c-16e0-4092-b81e-6dddbe41d3db
# ╠═b05b4228-fc30-4a74-8a99-f9b8d53aff83
# ╠═45c7f27a-b43c-4781-918a-51aebd273014
# ╠═45b90a66-956a-4b35-9e88-7621fd5c9c1a
# ╠═a6c9dc03-74ce-4174-8fd9-6895d117aa4a
# ╠═0c577fc5-d45d-448a-82c0-6f1abc9577c3
# ╠═f873e7c2-7dac-4925-b6b5-5887b461418b
# ╠═6e2adb2c-93f4-4a19-b93c-75b75a9116f3
# ╟─8aab09c7-07fe-4691-9e1d-35e2e215bc5d
# ╟─e75c9e76-3634-4bde-abb8-5b4827eb089e
# ╟─17d3777b-69d4-40c1-84c1-2bc6a737d52a
# ╟─258346da-e792-11ec-06dc-5906e95d24e2
# ╟─098b8119-26e8-4002-84c4-c8478d3c5019
# ╟─93de6d70-46ff-414f-9185-72975b252cbe
# ╟─2280361b-068d-4d5e-ae30-7e66c42ee665
# ╟─c8e739c0-940d-47b4-be2e-b0345a7e362f
# ╟─0469116a-7e10-44c5-801d-f5ce683dc486
# ╠═30ae8059-4b4e-4026-966e-7d6320912b5c
# ╠═7a1a556f-fa9e-4cec-88e8-5c74c10e0c4e
# ╠═5f9287b3-c7c3-4c0d-bf3b-d3161c4bb249
# ╟─2d02a6fd-7a23-4790-b8da-7b7150482a85
# ╟─29cb3c08-a702-4564-ad41-664f312913b0
# ╟─2d7da285-7078-4567-8cdd-a8cefd737f57
# ╟─535c4318-d29e-4640-9105-66b16d13bf88
# ╟─56a72a97-1505-4927-9106-9e4e0111dba9
# ╟─81472647-35a8-489e-a65d-b44586ddd179
# ╟─f80f3573-68a1-4732-be81-ffb8d4a45017
# ╠═8a5666f7-ee15-4f8c-847f-9fb998a89c7e
# ╠═ef8cde04-6e7c-4b2a-864b-e3fee501d6a3
# ╠═2fd6d12f-5013-41ee-97b2-085d795ef3bd
# ╟─71072c67-6584-4fc7-ad4d-b3d362333562
# ╠═4731b952-c5aa-4c56-93ef-806e260eef73
# ╠═f5ff25be-94d2-4e97-bb4d-df36abb4c0a8
# ╟─803376a5-96fa-4a15-a354-6a98940768e0
# ╟─aeabedd0-5719-46f2-9715-7b21d8e5bf3d
# ╠═ac8ed96d-bf48-4a2e-81f0-ae612024177c
# ╠═fad6db76-5bd3-4350-8180-eb9735f6a749
# ╟─da9b8ae0-d446-47f6-8c03-2e145e83b7ed
# ╠═3f1064a6-63e0-401b-8b57-7aaa4e097ccf
# ╠═137ec483-53d4-4ce2-826b-af9d06fe9c62
# ╠═a590e6c0-2f97-4866-bd0d-08b7758a347c
# ╠═67a91884-5169-4a32-aa56-35800a965e77
# ╠═0e4d3711-f29b-4833-8a3e-a91e4375eec0
# ╠═9a0d9463-0002-4d0e-a368-3d7bba542e3d
# ╠═fb595c55-2995-462f-8a7f-982022370bd9
# ╠═ae9df0c7-3dd4-4e11-8440-acba1faeb61b
# ╠═36a60c88-2c32-4d21-80c3-ce965a49438b
# ╠═406fcc98-5117-48d3-911f-91b350e6f9df
# ╠═4c14ef1c-887a-44e3-84f6-9812bd433858
# ╠═ceabaea4-17af-4fc2-8af4-b52c5fc886f5
