### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 824ed240-c3b6-11eb-12af-3feff4fbb636
begin
	using LinearAlgebra
	using Plots
	using Random
	using Statistics
	using Distributions
	using CSV
	using DataFrames
	using PlutoUI
	using GaussianProcesses
end

# ╔═╡ 7d382877-f8cc-47c7-929d-d7619b6afb9b
md"## Contagios de COVID-19 en la CDMX desde el 31/12/2019 al 02/06/2021

Datos recuperados de [Covid-19 México Trámites Gobierno](https://datos.covid-19.conacyt.mx/#DOView)
"

# ╔═╡ b84e47a9-698d-4581-8213-b4f2087cc4c0
df = CSV.read("covid19.csv", DataFrame)

# ╔═╡ b2202756-558d-4fb4-9d76-991063fb9643
plot(1:520, df[:, 2], xlabel="Día", ylabel="Contagios",
	title="Número de contagios por COVID-19 en la CDMX",
	color="skyblue", marker="o", markersize=1, label="Datos")

# ╔═╡ 37cd598d-cf19-4320-ba44-99b2c0364b44
md"Definimos algunos parámetros"

# ╔═╡ 58afdfa3-bf50-4547-b7c5-09b54f3f80df
a, b, n = 0, 350, 1000

# ╔═╡ c89563fb-395f-4714-b0c6-398fd5477b85
N = 520

# ╔═╡ 22286080-bb03-4b17-bdf0-50720f2b629d
x_train = collect(1:b) / 100

# ╔═╡ 45dbca16-b254-42d3-91f4-a344bc8bdb56
y_train = df[1:b, 2] / 1000

# ╔═╡ 8e06b9bd-25ef-41b5-9c1a-57eed237e823
x_test = collect(range(a, N, length=n)) / 100

# ╔═╡ 1de84c3c-f6ce-4914-a392-7192f7a04fe6
md"
Definimos el Kernel

$$K_{i, j} = k(x_i, x_j) = \eta \cdot e^{-\frac{1}{2l^2}|x_i - x_j|^2}$$
"

# ╔═╡ 47ea355a-1728-4d1a-9cd8-0f77e5c26f34
η = 8e-3

# ╔═╡ 1e8db85a-b446-4d09-904e-5898507a9aea
kernel(A, B, l) = [η * exp(-(1 / (2 * l^2)) * 
		norm(i - j)^2) + 1e-4 for i in A, j in B]

# ╔═╡ a74511a9-3a76-4972-a101-b70c017fb6b5
md"
Definimos las matrices de covarianza entre `x_train` y `x_test`
"

# ╔═╡ d30a43d2-0ff2-4bef-831d-9edd44bb0978
md"
Creamos las funciones previas, para ello, obtenemos la descomposición de Cholesky de K_xx
"

# ╔═╡ 560db1bc-9f31-4530-9257-dc6d689810d6
md"Creamos la función previa sobre el conjunto de prueba antes de observar los datos"

# ╔═╡ 494601da-bb45-4d4d-8dde-060b016a87bb
md"Calaculamos la funcion $\mu(x)$ y $\Sigma(x)$ del Proceso Gaussiano $\mathcal{GP}(\mu, \Sigma)$"

# ╔═╡ 8726bd07-64e4-4685-877c-8c261ab632a8
md"Una vez que tenemos la funcón media y covarianza, le mostramos al modelo los nuevos datos,  para luego crear las funciones posteriores"

# ╔═╡ df43b15d-72c7-403c-a633-872c2b00fcab
md"""
ι $(@bind l Slider(1e-6:1e-2:2))
σ² $(@bind σ² Slider(1e-4:1e-2:2))
"""

# ╔═╡ 1e7dd2de-59d0-46c7-b7a9-cdaad5b2f145
begin
	K = kernel(x_train, x_train, l) + σ²*I(b)
	K_x = kernel(x_train, x_test, l)
	K_xx = kernel(x_test, x_test, l)
end

# ╔═╡ ad5eb30e-2f37-44ed-bb97-b524432dbe21
L_xx = cholesky(K_xx + 1e-10 * I(n))

# ╔═╡ 933aadf0-1557-46da-9ca7-a14a470fd0e4
f_prior  = L_xx.U * rand(Normal(), (n, 5))

# ╔═╡ 2624c228-9e98-4788-a9e2-61e08463d8c9
plot(x_test, f_prior[:, :], label="", lw=1.2, xlabel="x", ylabel="y",
	title="Funciones previas")

# ╔═╡ 620e6d30-544d-4e43-8243-071918dc3aab
μ = K_x' * inv(K) * y_train

# ╔═╡ 362ae7a5-83b6-409f-818a-3606c204c723
Σ = K_xx - K_x' * inv(K) * K_x

# ╔═╡ a5d08d2f-8d07-48e0-a48d-86c9ca87219a
stdv = [sum(Σ[i, :]) for i in 1:n]

# ╔═╡ 4838585e-5782-4444-89aa-003d177aa3ae
f_post = μ .+ Σ * rand(Normal(), (n, 5))

# ╔═╡ 308d9838-0d53-4c27-a371-02365507fbc1
md"
Valores de los híperparametros $l, \sigma^2 , \eta$ son $l, $σ² y $η respectivamente.

Los valores que se adaptan a la regresión que estamos buscando son `l = 0.49` y `σ² = 0.0001`
"

# ╔═╡ 2ae4b517-05f8-425e-99bf-11d83174d3d1
begin
	plot(x_test, f_post[:, :], label="", color="skyblue", legend=:topleft)
	plot!(x_test, μ,
		ribbon=(2 * abs.(stdv)), fillalpha=.07, fillcolor="blue",
		lw=1.5, 
		color="crimson", 
		label="μ(x)",
		title="Regresión del proceso Gaussiano")
	plot!(x_test, μ, label="", lw=0.1,
		ribbon=(abs.(stdv)), fillalpha=.1, fillcolor="blue")
	scatter!(collect(1:520)/100, df[:, 2]/1000,
			 marker=".", markersize=0.85, label="Datos")
	vline!([3.5], label="", color="orange")
end

# ╔═╡ 18c532c0-a410-4b63-ba1d-cd21364e8186
md"Definimos ahora la función de máxima verosimilitud"

# ╔═╡ ceb84df8-16c9-455d-bad1-d830b32c2aee
function Marginal_likelihood(l, σ²)
	
	K = kernel(x_train, x_train, l) + σ²*I(b)
	
	return (0.5 * y_train' * inv(K) * y_train + 
			0.5 * log(det(K)) + (b/2) * log(2π))^0.5
end

# ╔═╡ 152cedf5-a1e1-46f4-84e4-6c14a3d36797
begin   
	contour((range(0.8, 5, length=50)), 
			collect(range(0.5, 1.5, length=50)),
			levels=30,
			Marginal_likelihood,
			title="Log Marginal Likelihood",
			xlabel="Amplitud-escala", ylabel="Ruido agregado")
	scatter!([1.60141], [0.69306], color="Crimson",
			 label="Parámetros óptimos", markersize=3)
end

# ╔═╡ 06cca80a-7dd8-494c-acbf-dd5a662181b7
function ∇(f, x₀, x₁, h)
	(1/2h) * [f(x₀ + h, x₁) - f(x₀ - h, x₁), f(x₀, x₁ + h) - f(x₀, x₁ - h)]
end

# ╔═╡ bb694350-9891-4581-b380-0ea123c6bd0d
function gradient_descent(f, α, ϵ, n)
	
	x₀, x₁ = 5rand(2) #Punto inicial del algoritmo
	h = 1e-7
	
	for i in 1:n
		x₀, x₁ = [x₀, x₁] - α * ∇(f, x₀, x₁, h)
	end
	
	x₀, x₁, f(x₀, x₁), norm(∇(f, x₀, x₁, h))
	
end

# ╔═╡ d8b55101-f0bd-4da6-ab48-6f548f1d80c3
opt = gradient_descent(Marginal_likelihood, 7e-2, 1e-4, 1000)

# ╔═╡ aa787f5f-a393-41ce-bd86-41dc27a58398
begin
	#Select mean and covariance function
	mZero = MeanZero()                   #Zero mean function
	kern = SE(0.0,0.0)                   #Squared exponential kernel (note that hyperparameters are on the log scale)
	
	#logObsNoise = -1.0                        # log standard deviation of observation noise (this is optional)
	gp = GP(x_train,y_train,mZero,kern)#,logObsNoise) 
end

# ╔═╡ 7271c8e7-5bc0-474b-a727-459a441c903e
optimize!(gp)

# ╔═╡ 11dba11c-7692-471c-b5b6-72a9b69340e0
mu, s2 = predict_y(gp, x_test);

# ╔═╡ bac927bd-d014-4314-85a1-1d21fce69772
begin
	plot(x_test, mu,
		ribbon=(2 * abs.(s2)), fillalpha=.07, fillcolor="blue",
		lw=1.5, 
		color="crimson", 
		label="μ(x)",
		title="Regresión del proceso Gaussiano")
	plot!(x_test, mu, label="", lw=0.1,
		ribbon=(abs.(s2)), fillalpha=.1, fillcolor="blue")
	scatter!(collect(1:520)/100, df[:, 2]/1000, color="yellow",
			 marker=".", markersize=0.85, label="Datos")
end

# ╔═╡ Cell order:
# ╟─7d382877-f8cc-47c7-929d-d7619b6afb9b
# ╟─b84e47a9-698d-4581-8213-b4f2087cc4c0
# ╟─b2202756-558d-4fb4-9d76-991063fb9643
# ╟─37cd598d-cf19-4320-ba44-99b2c0364b44
# ╠═58afdfa3-bf50-4547-b7c5-09b54f3f80df
# ╟─c89563fb-395f-4714-b0c6-398fd5477b85
# ╟─22286080-bb03-4b17-bdf0-50720f2b629d
# ╟─45dbca16-b254-42d3-91f4-a344bc8bdb56
# ╟─8e06b9bd-25ef-41b5-9c1a-57eed237e823
# ╟─1de84c3c-f6ce-4914-a392-7192f7a04fe6
# ╠═47ea355a-1728-4d1a-9cd8-0f77e5c26f34
# ╠═1e8db85a-b446-4d09-904e-5898507a9aea
# ╟─a74511a9-3a76-4972-a101-b70c017fb6b5
# ╠═1e7dd2de-59d0-46c7-b7a9-cdaad5b2f145
# ╟─d30a43d2-0ff2-4bef-831d-9edd44bb0978
# ╟─ad5eb30e-2f37-44ed-bb97-b524432dbe21
# ╟─560db1bc-9f31-4530-9257-dc6d689810d6
# ╟─933aadf0-1557-46da-9ca7-a14a470fd0e4
# ╟─494601da-bb45-4d4d-8dde-060b016a87bb
# ╟─620e6d30-544d-4e43-8243-071918dc3aab
# ╟─362ae7a5-83b6-409f-818a-3606c204c723
# ╟─a5d08d2f-8d07-48e0-a48d-86c9ca87219a
# ╟─8726bd07-64e4-4685-877c-8c261ab632a8
# ╟─4838585e-5782-4444-89aa-003d177aa3ae
# ╟─2624c228-9e98-4788-a9e2-61e08463d8c9
# ╟─308d9838-0d53-4c27-a371-02365507fbc1
# ╟─df43b15d-72c7-403c-a633-872c2b00fcab
# ╠═2ae4b517-05f8-425e-99bf-11d83174d3d1
# ╟─18c532c0-a410-4b63-ba1d-cd21364e8186
# ╠═ceb84df8-16c9-455d-bad1-d830b32c2aee
# ╟─152cedf5-a1e1-46f4-84e4-6c14a3d36797
# ╠═06cca80a-7dd8-494c-acbf-dd5a662181b7
# ╠═bb694350-9891-4581-b380-0ea123c6bd0d
# ╠═d8b55101-f0bd-4da6-ab48-6f548f1d80c3
# ╟─aa787f5f-a393-41ce-bd86-41dc27a58398
# ╟─7271c8e7-5bc0-474b-a727-459a441c903e
# ╠═11dba11c-7692-471c-b5b6-72a9b69340e0
# ╟─bac927bd-d014-4314-85a1-1d21fce69772
# ╟─824ed240-c3b6-11eb-12af-3feff4fbb636
