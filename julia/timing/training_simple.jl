### A Pluto.jl notebook ###
# v0.18.1

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

# ╔═╡ 7d6fbe8d-7f02-483c-a8b1-6c0287dcd06b
begin
    let
        using Pkg
		Pkg.activate(mktempdir())
		Pkg.Registry.update()
        Pkg.add("PlutoUI")
        Pkg.add("Tar")
        Pkg.add("MLDataPattern")
        Pkg.add("Glob")
        Pkg.add("NIfTI")
        Pkg.add("CairoMakie")
        Pkg.add("ImageCore")
		Pkg.add("ImageFiltering")
		Pkg.add("Images")
        Pkg.add("DataLoaders")
        Pkg.add("CUDA")
        Pkg.add(PackageSpec(;name="FastAI", version="0.4.0"))
    end

    using PlutoUI
    using Tar
    using MLDataPattern
    using Glob
    using NIfTI
    using CairoMakie
    using ImageCore
	using ImageFiltering
	using Images
    using DataLoaders
    using CUDA
    using FastAI
end

# ╔═╡ 7cd97284-f4ad-4a55-a57d-e8f2666ad086
TableOfContents()

# ╔═╡ 6d825ccd-8aa8-4e7b-930d-839887bd7971
md"""
## Load data
Part of the [Medical Decathlon Dataset](http://medicaldecathlon.com/)
"""

# ╔═╡ 24bbd77c-c9cf-49f9-adaf-b7d308b80a35
data_dir = "/Users/daleblack/Google Drive/Datasets/Task02_Heart"

# ╔═╡ a1d7217f-f0ec-48c7-8bcf-7320050b3ca6
function loadfn_label(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{UInt8}, a)
    convert_a = convert_a .+ 1
    return convert_a
end

# ╔═╡ 8f93a981-6b2a-4736-bd3d-5d27b8ad834c
function loadfn_image(p)
    a = NIfTI.niread(string(p)).raw
    convert_a = convert(Array{Float32}, a)
    convert_a = convert_a / max(convert_a...)
    return convert_a
end

# ╔═╡ 43392f62-1903-4f1f-b8e5-44db9539553f
begin
    images(dir) = mapobs(loadfn_image, Glob.glob("*.nii*", dir))
    masks(dir) =  mapobs(loadfn_label, Glob.glob("*.nii*", dir))
    data = (
        images(joinpath(data_dir, "imagesTr")),
        masks(joinpath(data_dir, "labelsTr")),
    )
end

# ╔═╡ 866fbacd-39fe-4c53-b6f5-cb1d41592d73
train_files, val_files = MLDataPattern.splitobs(data, 0.8)

# ╔═╡ 42fc9a6b-8d0a-4b0b-9eef-b8200b56bc15
image, mask = sample = getobs(data, 1);

# ╔═╡ f622d64f-3ffc-40a4-b7ed-f4b0224cf7f6
@bind a PlutoUI.Slider(1:size(image, 3), default=50, show_value=true)

# ╔═╡ 1dad006e-67aa-4d90-beee-9904a32f73d0
heatmap(image[:, :, a], colormap=:grays)

# ╔═╡ cd9ed3c4-8f5c-4d94-9385-175f87b83d06
heatmap(mask[:, :, a], colormap=:grays)

# ╔═╡ 81543df5-d741-4b1f-a208-612f7c865117
md"""
## Presize and cache data
"""

# ╔═╡ 417d910f-130d-4ca8-b766-6c7fa997136c
image_size = (64, 64, 64)

# ╔═╡ d28bb876-7652-49a9-9ebb-84baf99d4047
function presize(image, mask, files)
	container_images = Array{eltype(image)}(undef, image_size..., nobs(train_files))
	container_masks = Array{eltype(mask)}(undef, image_size..., nobs(train_files))
	for i in 1:nobs(train_files)
		image, mask = getobs(data, i)
		img = imresize(image, image_size)
		msk = round.(imresize(mask, image_size))
		container_images[:, :, :, i] = img
		container_masks[:, :, :, i] = msk
	end
	return container_images, container_masks
end

# ╔═╡ f52234c5-418a-4355-b1f8-4cb13dbabbc6
train_files_cache = presize(image, mask, train_files);

# ╔═╡ 957d4cdf-e128-4cbb-99a2-95b22b026c52
val_files_cache = presize(image, mask, val_files);

# ╔═╡ ad6c8a75-649e-41ae-8f8f-d5cda9fe75f9
md"""
As you can see below, we need to modify the pre-size function to accurately scale the images better but we will stick with this for right now
"""

# ╔═╡ e8068eb0-88b1-40b8-a4aa-b768e712cbfe
heatmap(train_files_cache[1][:, :, a, 1], colormap=:grays)

# ╔═╡ cc5e263a-b6aa-40b0-bb38-37ed6e0596df
heatmap(train_files_cache[2][:, :, a, 1], colormap=:grays)

# ╔═╡ 78c679b6-d729-493c-9a06-1eb558a0e00b
md"""
## Load data pt. 2
"""

# ╔═╡ d56afc84-6392-472b-95fa-734e54fe9dd3
function loadfn_pre(a)
    return a
end

# ╔═╡ 5c612287-852c-401a-9b76-933713f47a47
dt = collect(1:8)

# ╔═╡ f2feffe0-9f3a-409f-b275-6fb56984a2fe
getobs(dt, 8)

# ╔═╡ b2fbc741-b742-428d-8316-2dcbdb24ea16
mdata = mapobs(-, dt)

# ╔═╡ e610c5cd-5a5b-42d9-987c-f7f1d2943d08
begin
    images_pre(arr) = mapobs(loadfn_pre, arr)
    masks_pre(arr) =  mapobs(loadfn_pre, arr)
    data_pre = (
        images_pre(train_files_cache[1]),
        masks_pre(train_files_cache[2]),
    )
end

# ╔═╡ d109be13-a297-42ac-a845-6ee43aa0aef9
image_new, mask_new = getobs(data_pre, 1);

# ╔═╡ a220cc80-65b6-451e-b06f-0a42f58445a0
heatmap(image_new[:, :, a], colormap=:grays)

# ╔═╡ 11478b1b-e6d2-4bf7-b062-8046e3bf867a
heatmap(mask_new[:, :, a], colormap=:grays)

# ╔═╡ fbecc171-abad-47af-9edb-0e162bdde4ca
md"""
## Create learning task
"""

# ╔═╡ 22b3d560-7b72-465f-be7b-88634ed842b2
task = SupervisedTask(
           (FastAI.Vision.Image{3}(), Mask{3}(1:2)),
           (
               ProjectiveTransforms((image_size)),
               ImagePreprocessing(),
               OneHot()
           )
       )

# ╔═╡ f6716c80-947b-40eb-a60c-c86b7eb60bb7
describetask(task)

# ╔═╡ 9f10c793-0ef5-4bef-86b3-42d05ee4e640
md"""
## Visualize
Notice the random cropping
"""

# ╔═╡ 21963862-4c40-42b6-b6ab-2b8ab51fb888
xs, ys = FastAI.makebatch(task, data_pre, 1:3);

# ╔═╡ 5ee84fe2-8545-41d5-af11-c653eb6b9cc5
@bind b PlutoUI.Slider(1:size(xs, 3), default=50, show_value=true)

# ╔═╡ ff50e706-2fbf-4427-8f49-779138dfceba
heatmap(xs[:, :, b, 1, 2], colormap=:grays)

# ╔═╡ 0b0348c2-1f97-4d62-9244-b9fa734dd302
heatmap(ys[:, :, b, 2, 2], colormap=:grays)

# ╔═╡ a5cfce65-9d35-40a9-8de3-8ec736a738d5
md"""
## Dataloader
"""

# ╔═╡ e29e70f2-dcad-4c1b-b817-abefed94dd61
traindl, validdl = taskdataloaders(data_pre, task, 1)

# ╔═╡ 07b6336f-0cd5-476e-a877-ccf2fb384c6c
md"""
## Model
"""

# ╔═╡ 25ccf8ac-a8e0-4c5b-bfaa-7753726cbc6b
begin
    # 3D layer utilities
    conv = (stride, in, out) -> Conv((3, 3, 3), in=>out, stride=stride, pad=(1, 1, 1))
    tran = (stride, in, out) -> ConvTranspose((4, 4, 4), in=>out, stride=stride, pad=1)

    conv1 = (in, out) -> Chain(conv(1, in, out), BatchNorm(out), x -> leakyrelu.(x))
    conv2 = (in, out) -> Chain(conv(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
    tran2 = (in, out) -> Chain(tran(2, in, out), BatchNorm(out), x -> leakyrelu.(x))
end

# ╔═╡ 2f2c2307-8b08-4416-9eed-87f8fe354555
function unet3D(in_chs, lbl_chs)
    # Contracting layers
    l1 = Chain(conv1(in_chs, 4))
    l2 = Chain(l1, conv1(4, 4), conv2(4, 16))
    l3 = Chain(l2, conv1(16, 16), conv2(16, 32))
    l4 = Chain(l3, conv1(32, 32), conv2(32, 64))
    l5 = Chain(l4, conv1(64, 64), conv2(64, 128))

    # Expanding layers
    l6 = Chain(l5, tran2(128, 64), conv1(64, 64))
    l7 = Chain(Parallel(+, l6, l4), tran2(64, 32), conv1(32, 32))       # Residual connection between l6 & l4
    l8 = Chain(Parallel(+, l7, l3), tran2(32, 16), conv1(16, 16))       # Residual connection between l7 & l3
    l9 = Chain(Parallel(+, l8, l2), tran2(16, 4), conv1(4, 4))          # Residual connection between l8 & l2
    l10 = Chain(l9, conv1(4, lbl_chs))
end

# ╔═╡ 0f8c3c16-af2f-4521-bfe7-08d0445c1032
model = unet3D(1, 2) |> gpu;

# ╔═╡ 73368331-3e26-4420-a801-e1c7f3f7eefe
md"""
## Helper functions
"""

# ╔═╡ c3eef752-5838-4bf7-b788-2ec12e8ea480
function dice_metric(ŷ, y)
    dice = 2 * sum(ŷ .& y) / (sum(ŷ) + sum(y))
    return dice
end

# ╔═╡ 7cca5167-39c7-4da3-962c-7aa07f045d38
function as_discrete(array, logit_threshold)
    array = array .>= logit_threshold
    return array
end

# ╔═╡ f0ac298d-4c1f-45c3-9f15-a8d5d6098816
md"""
## Loss functions
"""

# ╔═╡ b411b979-f13b-4563-99c3-5af715f2ff97
function dice_loss(ŷ, y)
    ϵ = 1e-5
    return loss = 1 - ((2 * sum(ŷ .* y) + ϵ) / (sum(ŷ .* ŷ) + sum(y .* y) + ϵ))
end

# ╔═╡ 74f82e25-97f2-4961-86e4-81341d25e40c
function hd_loss(ŷ, y, ŷ_dtm, y_dtm)
    M = (ŷ .- y) .^ 2 .* (ŷ_dtm .^ 2 .+ y_dtm .^ 2)
    return loss = mean(M)
end

# ╔═╡ 84371d14-7002-499e-bdb8-d6c1cd91930e
md"""
## Training
"""

# ╔═╡ e9f360cc-5e1a-43e4-a269-2b7a1192eaf0
ps = Flux.params(model);

# ╔═╡ 367958fe-915f-4405-8352-9d2700ff038d
loss_function = dice_loss

# ╔═╡ 1ff09f97-d4cc-4615-b160-7f7f3379c293
optimizer = Flux.ADAM(0.01)

# ╔═╡ 5401da72-f7b3-4570-b14c-9afa87374aa4
begin
	max_epochs = 2
	val_interval = 1
	epoch_loss_values = []
	val_epoch_loss_values = []
	dice_metric_values = []
end

# ╔═╡ 7a27470c-8f34-4c9b-b422-82a91bd41914
for (xs, ys) in validdl
	@info size(xs)
	@info size(ys)
end

# ╔═╡ dbb882dd-d876-4ab7-93c4-3193e9e8ac4b
# begin
# 	for epoch in 1:max_epochs
# 		step = 0
# 		@show epoch
		
# 		# Loop through training data
# 		for (xs, ys) in traindl
# 			xs, ys = xs |> gpu, ys |> gpu
# 			step += 1
# 			@show step
# 			gs = Flux.gradient(ps) do
# 				ŷs = model(xs)
# 				loss = loss_function(ŷs[:, :, :, 2, :], ys[:, :, :, 2, :])
# 				return loss
# 			end
# 			Flux.update!(optimizer, ps, gs)
# 		end

# 		# Loop through validation data
# 		if (epoch + 1) % val_interval == 0
# 			val_step = 0
# 			for (val_xs, val_ys) in validdl
# 				val_xs, val_ys = val_xs |> gpu, val_ys |> gpu 
# 				val_step += 1
# 				@show val_step

# 				local val_ŷs = model(val_xs)
# 				local val_loss = loss_function(val_ŷs[:, :, :, 2, :], val_ys[:, :, :, 2, :])
# 				val_ŷs, val_ys = as_discrete(val_ŷs, 0.5), as_discrete(val_ys, 0.5)
# 			end
# 		end
# 	end
# end

# ╔═╡ 16fb3df2-5c36-439a-9a09-4a541363e867


# ╔═╡ Cell order:
# ╠═7d6fbe8d-7f02-483c-a8b1-6c0287dcd06b
# ╠═7cd97284-f4ad-4a55-a57d-e8f2666ad086
# ╟─6d825ccd-8aa8-4e7b-930d-839887bd7971
# ╠═24bbd77c-c9cf-49f9-adaf-b7d308b80a35
# ╠═a1d7217f-f0ec-48c7-8bcf-7320050b3ca6
# ╠═8f93a981-6b2a-4736-bd3d-5d27b8ad834c
# ╠═43392f62-1903-4f1f-b8e5-44db9539553f
# ╠═866fbacd-39fe-4c53-b6f5-cb1d41592d73
# ╠═42fc9a6b-8d0a-4b0b-9eef-b8200b56bc15
# ╟─f622d64f-3ffc-40a4-b7ed-f4b0224cf7f6
# ╠═1dad006e-67aa-4d90-beee-9904a32f73d0
# ╠═cd9ed3c4-8f5c-4d94-9385-175f87b83d06
# ╟─81543df5-d741-4b1f-a208-612f7c865117
# ╠═417d910f-130d-4ca8-b766-6c7fa997136c
# ╠═d28bb876-7652-49a9-9ebb-84baf99d4047
# ╠═f52234c5-418a-4355-b1f8-4cb13dbabbc6
# ╠═957d4cdf-e128-4cbb-99a2-95b22b026c52
# ╟─ad6c8a75-649e-41ae-8f8f-d5cda9fe75f9
# ╠═e8068eb0-88b1-40b8-a4aa-b768e712cbfe
# ╠═cc5e263a-b6aa-40b0-bb38-37ed6e0596df
# ╟─78c679b6-d729-493c-9a06-1eb558a0e00b
# ╠═d56afc84-6392-472b-95fa-734e54fe9dd3
# ╠═5c612287-852c-401a-9b76-933713f47a47
# ╠═f2feffe0-9f3a-409f-b275-6fb56984a2fe
# ╠═b2fbc741-b742-428d-8316-2dcbdb24ea16
# ╠═e610c5cd-5a5b-42d9-987c-f7f1d2943d08
# ╠═d109be13-a297-42ac-a845-6ee43aa0aef9
# ╠═a220cc80-65b6-451e-b06f-0a42f58445a0
# ╠═11478b1b-e6d2-4bf7-b062-8046e3bf867a
# ╟─fbecc171-abad-47af-9edb-0e162bdde4ca
# ╠═22b3d560-7b72-465f-be7b-88634ed842b2
# ╠═f6716c80-947b-40eb-a60c-c86b7eb60bb7
# ╟─9f10c793-0ef5-4bef-86b3-42d05ee4e640
# ╠═21963862-4c40-42b6-b6ab-2b8ab51fb888
# ╟─5ee84fe2-8545-41d5-af11-c653eb6b9cc5
# ╠═ff50e706-2fbf-4427-8f49-779138dfceba
# ╠═0b0348c2-1f97-4d62-9244-b9fa734dd302
# ╟─a5cfce65-9d35-40a9-8de3-8ec736a738d5
# ╠═e29e70f2-dcad-4c1b-b817-abefed94dd61
# ╟─07b6336f-0cd5-476e-a877-ccf2fb384c6c
# ╠═25ccf8ac-a8e0-4c5b-bfaa-7753726cbc6b
# ╠═2f2c2307-8b08-4416-9eed-87f8fe354555
# ╠═0f8c3c16-af2f-4521-bfe7-08d0445c1032
# ╟─73368331-3e26-4420-a801-e1c7f3f7eefe
# ╠═c3eef752-5838-4bf7-b788-2ec12e8ea480
# ╠═7cca5167-39c7-4da3-962c-7aa07f045d38
# ╟─f0ac298d-4c1f-45c3-9f15-a8d5d6098816
# ╠═b411b979-f13b-4563-99c3-5af715f2ff97
# ╠═74f82e25-97f2-4961-86e4-81341d25e40c
# ╟─84371d14-7002-499e-bdb8-d6c1cd91930e
# ╠═e9f360cc-5e1a-43e4-a269-2b7a1192eaf0
# ╠═367958fe-915f-4405-8352-9d2700ff038d
# ╠═1ff09f97-d4cc-4615-b160-7f7f3379c293
# ╠═5401da72-f7b3-4570-b14c-9afa87374aa4
# ╠═7a27470c-8f34-4c9b-b422-82a91bd41914
# ╠═dbb882dd-d876-4ab7-93c4-3193e9e8ac4b
# ╠═16fb3df2-5c36-439a-9a09-4a541363e867
