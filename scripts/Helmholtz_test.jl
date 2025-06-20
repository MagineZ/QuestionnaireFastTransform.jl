using QuestionnaireFastTransform
using Distances
using Plots
using MultiscaleGraphSignalTransforms
using LinearAlgebra

println("Create Helmholtz Kernel")
p = 10
N = 2^p

# --- Create Plane ---
x_plane = 0 .+ 2 .* rand(N)        # x between -1 and 1
y_plane = 1 .+ 2 .* rand(N)             # y between -10 and 0
z_plane = 0.01 .* rand(N)      # z between -5 and slightly above -5 (very thin plane)

# --- Create Spiral ---
theta = range(0, stop=6pi, length=N) # angle values
z_spiral = range(-15, stop=15, length=N) # height values
r_spiral = 1                            # spiral radius
x_spiral = r_spiral .* cos.(theta)   # x of spiral
y_spiral = r_spiral .* sin.(theta)   # y of spiral

z1 = vcat(x_plane', y_plane', z_plane') # 3×1024
z2 = vcat(x_spiral', y_spiral', z_spiral') # 3×1024

# Pairwise distance
D = pairwise(Euclidean(), eachcol(z1), eachcol(z2)) # 1024×1024 matrix

# Inverse distance matrix
#data_todo = 1.0 ./ D
v = 1
data = cos.(2π * v .* D) ./ D 

println("Run Questionnaire")
#Run Questionnaire_Roseland
data_af,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations_new(data,false,0)
#Run Questionnaire
#data_af,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations(data,false,0)

println("Run Butterfly")
#Butterfly Factorization
acc = 1E-10
Cmax = 16 
dyadic_blocks,dyadic_infos = dyadic_partition(data_af, Cmax)
BL, P = compress_dyadic_blocks(dyadic_blocks, acc)
BF = CompressedOperator(BL, P, coifman_col_order, coifman_row_order, dyadic_infos)

println("Run Test: l2 error =")
#Test with random vector
x = randn(2^(p))
x = x/norm(x)

#Product by Butterfly
f1 = BF * x
#Direct Mutilication
f = data * x
err = norm(f-f1)/norm(f)
println(err)