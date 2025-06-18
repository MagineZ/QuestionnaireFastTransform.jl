module QuestionnaireFastTransform


using MultiscaleGraphSignalTransforms, Plots, LinearAlgebra
using NPZ
using PyCall
using Plots
using Random
using Distances
using ProgressMeter
using StatsBase
using SparseArrays
using Base.Threads

np=pyimport("numpy")
pysys = pyimport("sys")
push!(pysys["path"], @__DIR__)
export get_Walsh_partition, build_kernel_permutations_new, ghwt_synthesis_aftermultiplication, Walsh_Multiplication, ghwt_2d_sparse, Walsh_Multiplication_fast, Walsh_Multiplication_fast2D, compress_dyadic_blocks, apply_compressed_operator 


function get_Walsh_partition(A)
    matrix=A
    m,n = size(matrix);
    Gm = gpath(m);
    GProws = partition_tree_fiedler(Gm; swapRegion = false);
    # column tree
    Gn = gpath(n);
    GPcols = partition_tree_fiedler(Gn; swapRegion = false);
    return GProws,GPcols
end

function get_Walsh_partition1D(A)
    m = length(A);
    Gm = gpath(m);
    W = Gm.W;
    L = diagm(sum(W; dims = 1)[:]) - W;
    𝛌, 𝚽 = eigen(L);
    𝚽 = 𝚽 .* sign.(𝚽[1,:])';
    GProws = partition_tree_fiedler(Gm; swapRegion = false);
    return GProws
end


function get_Walsh_coefficients1D(A,GP,c)
  A = reshape(A, :, 1)  # Convert 1D array to 2D
  N = length(A)
  G = gpath(N)
  G.f = A;
  dmatrix = ghwt_analysis!(G, GP = GP)
  dvec, BS = ghwt_tf_bestbasis(dmatrix, GP)
    return dmatrix, dvec, BS
end

function get_Walsh_coefficients1D_fast(matrix,GP)
   (N, jmax_row) = size(GP.rs)
    N = N-1
    (frows, fcols) = size(matrix)
    dmatrix = zeros((N,jmax_row,fcols))
    dmatrix[:,jmax_row,:] = matrix
   ghwt_core!(GP, dmatrix)

     return dmatrix
end

function get_Walsh_coefficients(matrix,GProws,GPcols)
    m,n=size(matrix)
    dvec, BSrows, BScols = ghwt_bestbasis_2d(matrix,GProws,GPcols);
    indices = sortperm(reshape(abs.(dvec[:]),m*n,1), rev=true,dims = 1);
    return dvec, indices, BSrows, BScols
end

function get_BS_coefficients_1D(data,GP,BS)
    n=length(data)
  G = gpath(n)
  G.f = data
  dmatrix = ghwt_analysis!(G, GP = GP)
    dvec = dmatrix2dvec(dmatrix, GP, BS)
  return dvec
end

function get_ghwt_synthesis(dvec,GP,BS)
      	f = ghwt_synthesis(dvec, GP, BS)
	return f
end


function get_ghwt_synthesis2D(dvec,GProws,GPcols,BSrows,BScols)
      	matrix = ghwt_synthesis_2d(dvec,GProws,GPcols,BSrows,BScols)
	return matrix
end
 

function get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,compare_matrix,GP_cols,GP_rows,BS_cols,BS_rows)
    dvec_rec=zeros(size(dvec))
    this_index=indices[1:finalindex]
    dvec_rec[this_index]=dvec[this_index]
    rec=ghwt_synthesis_2d(dvec_rec,GP_rows,GP_cols,BS_rows,BS_cols)
    score[1]=opnorm(rec)
    score[2]=opnorm(compare_matrix-rec[listinds,listinds])
end

function from_matrix_get_score(finalindex,listinds,datamatrix,GProws,GPcols,normalization)
    score=zeros(2)
    dvec,indices,BScols,BSrows=get_Walsh_coefficients(datamatrix,GProws,GPcols)
    matrix_main=datamatrix[listinds,listinds]
    get_scores_thresholded_coeffs!(score,finalindex,listinds,dvec,indices,matrix_main,GPcols,GProws,BScols,BSrows)
    return score./normalization
end

function get_inverse_permutation(perm)
    inverse=np.argsort(np.asarray(perm)) .+ 1
    return inverse
end
function build_kernel_permutations(A,oldtreebuilding::Bool = false,switchcos::Int = 0)
    #heatmap(data_permuted)
    if oldtreebuilding
        qrun_permuted=quaest.main(A,oldtreebuilding,switchcos)
    else
        qrun_permuted=quaest.main(A,oldtreebuilding,switchcos)
    end

    row_tree=qrun_permuted.row_trees[end]
    col_tree=qrun_permuted.col_trees[end]

    coifman_col_order=[x.elements[1] for x in col_tree.dfs_leaves()] .+1
    coifman_row_order=[x.elements[1] for x in row_tree.dfs_leaves()] .+1

    data_afterquest=A[coifman_row_order,coifman_col_order]
    return data_afterquest,coifman_col_order,coifman_row_order,qrun_permuted
end

function build_kernel_permutations_new(A,oldtreebuilding::Bool = false,switchcos::Int = 0)
    #heatmap(data_permuted)
    if oldtreebuilding
        qrun_permuted,row_order,col_order=quaest.main(A,oldtreebuilding,switchcos)
    else
        qrun_permuted,row_order,col_order=quaest.main(A,oldtreebuilding,switchcos)
    end

    coifman_col_order= col_order.+1
    coifman_row_order= row_order.+1
    data_afterquest=A[coifman_row_order,coifman_col_order]
    
    return data_afterquest,coifman_col_order,coifman_row_order,qrun_permuted
end

function plot_single(xdata,threeDtensor,arr_w,switch)
    nplots=size(threeDtensor)[2]

    # Generate 5 colors from the "hot" palette
    c = palette(:hot, nplots+3)

    # Initialize a plot
    p = plot()
    if switch==1
        titlelabel="relative norm of reconstruction"
    elseif switch ==2
        titlelabel="relative norm of residual"
    end

    # Plot each column (each series) with a different color
    for i in 1:nplots
        plot!(p, xdata, threeDtensor[:, i,switch], 
        color = c[i+1], 
        label = "frequency w = $(arr_w[i])",
        xlabel= "fraction of coefficients kept",
        title= titlelabel
        )
    end
    display(p)
end


function plot_sidebyside(xdata,threeDtensor,arr_w)
    nplots=size(threeDtensor)[2]
    c = palette(:hot, nplots+3)

    # Build a 1×2 layout, i.e. two side-by-side subplots
    p = plot(layout=(1,2), size=(800,450))

    for i in 1:nplots
        # Plot "first output" (A[:, i, 1]) in left subplot
        plot!(p[1],
          xdata,
          threeDtensor[:, i, 1],
          color = c[i+1],
          label = "frequency w=$(arr_w[i])",
          xlabel="fraction of coefficients kept",
          title= "relative norm of reconstruction"
        )

        # Plot "second output" (A[:, i, 2]) in right subplot
        plot!(p[2],
          xdata,
          threeDtensor[:, i, 2],
          color = c[i+1],
          label = "frequency w=$(arr_w[i])",
          xlabel="fraction of coefficients kept",
          title= "relative norm of the residual"
        )
    end
    display(p)
end


function plot_singlefrequency(freq_index,switch, logscale::Bool = false)
    if switch==1
        titlelabel="Frequency w=$(powers[freq_index])"
    elseif switch ==2
        titlelabel="Frequency w=$(powers[freq_index])"
    end
    yscale_choice = logscale ? :log10 : :identity
    plot(pertoplot,score_original[:,freq_index,switch],yscale=yscale_choice,label="original",xlabel="fraction of coefficients kept")
    plot!(pertoplot,score_permuted[:,freq_index,switch],yscale=yscale_choice,label="after applying permutation")
    plot!(pertoplot,score_afterquest[:,freq_index,switch],yscale=yscale_choice,label="after applying the questionnaire")
    plot!(title=titlelabel)
end


function single_freq_heatmaps(index_freq,switch)

    data_osc=np.imag(exp.(1im*2*pi * powers[index_freq] .* Dm))
    data_decay= np.nan_to_num(data_osc ./ Dm)
    npoints=size(data_decay)[1]
    perm=sample(range(1,npoints),npoints,replace=false)
    ### Change this line to go switch between pure oscillation and oscillation+decay
    if switch>0
        matrix_main=data_osc
        C=1
    else
        matrix_main=data_decay
        asas=vec(np.nan_to_num(log.(abs.(data_decay)),posinf=0,neginf=0))
        C=exp(mean(asas)+3*std(asas))
        println("color cutoff= ",C)
        #C=maximum(abs.(data_decay))
    end
    data_permuted=matrix_main[perm,perm]
    data_afterquest,coifman_col_order,coifman_row_order=build_kernel_permutations(matrix_main)
    clrange = (-C, C)
    pcolor = heatmap( fill(NaN,1,1);
                  clims=clrange,  # same color limits
                  colorbar=true,
                  framestyle=:none,  # hide axes
                  )

    # Custom layout: 
    #  - The first part is a grid(1,3) for the 3 heatmaps
    #  - {0.8w} means "occupy 80% of the total width"
    #  - Then a second cell for the colorbar subplot, with {0.2w} = 20% width
    layout_ = @layout([ grid(1,3){0.95w}  c{0.05w} ])
    p = plot(
        heatmap(matrix_main, colorbar=false,title="Original data",clims=clrange),
        heatmap(data_permuted, colorbar=false,title="After scrambling",clims=clrange),
        heatmap(data_afterquest, colorbar=false,title="After questionnaire",clims=clrange),
        pcolor,
        layout = layout_,
        size = (1200, 300)    # optional size
    )
end

function ghwt_synthesis_aftermultiplication(dmatrix::Matrix{Float64},GP::GraphPart)
    rs = GP.rs;
    tag = GP.tag;
    jmax = Base.size(rs,2)    
    for j = 1:(jmax - 1)    # from top to bottom-1
            regioncount = count(!iszero, rs[:, j]) - 1
            for r = 1:regioncount
                rs1 = rs[r, j]      # the start of the 1st subregion
                rs3 = rs[r + 1, j]  # 1 + the end of the 2nd subregion
                n = rs3 - rs1       # # of points in the current region
                # only proceed forward if coefficients do not exist
                #if count(!iszero, dmatrix[rs1:(rs3 - 1), j + 1, :]) == 0 &&
                #    count(!iszero, dmatrix[rs1:(rs3 - 1), j, :]) > 0
                #if count(!iszero, dvec_loc[rs1:(rs3-1),j]) > 0
                    if n == 1   # single node region
                            dmatrix[rs1, j + 1, :] += dmatrix[rs1, j, :]
                    elseif n > 1
                        rs2 = rs1 + 1 # the start of the 2nd subregion
                        while rs2 < rs3 && tag[rs2, j + 1] != 0 ### && rs2 < N+1
                            rs2 += 1
                        end
                        if rs2 == rs3 # the parent is a copy of the subregion      
                                dmatrix[rs1:(rs3 - 1), j + 1, :] += dmatrix[rs1:(rs3 - 1), j, :]
                        else   # the parent region has 2 child regions
                            n1 = rs2 - rs1 # # of pts in the 1st subregion
                            n2 = rs3 - rs2 # # of pts in the 2nd subregion

                            # SCALING COEFFICIENTS (n > 1)
                            
                                dmatrix[rs1, j + 1, :] +=
                                    ( sqrt(n1) * dmatrix[rs1, j, :] +
                                      sqrt(n2) * dmatrix[rs1 + 1, j, :] ) / sqrt(n)

                            # HAAR COEFFICIENTS
                                dmatrix[rs2, j + 1, :] +=
                                    ( sqrt(n2) * dmatrix[rs1, j, :] -
                                      sqrt(n1) * dmatrix[rs1 + 1, j, :] ) / sqrt(n)

                          
                            # WALSH COEFFICIENTS
                            # search through the remaining coefs in each subregion
                            parent = rs1 + 2
                            child1 = rs1 + 1
                            child2 = rs2 + 1
                            while child1 < rs2 || child2 < rs3
                                # subregion 1 has the smaller tag
                                if child2 == rs3 ||
                                    (tag[child1, j + 1] < tag[child2, j + 1] &&
                                     child1 < rs2)

                                 
                                        dmatrix[child1, j + 1, :] +=dmatrix[parent, j, :]
                                      
                                    child1 += 1; parent += 1

                                # subregion 2 has the smaller tag
                                elseif child1 == rs2 ||
                                    (tag[child2, j + 1] < tag[child1, j + 1] &&
                                     child2 < rs3)

                                   
                                        dmatrix[child2, j + 1, :] += dmatrix[parent, j, :]
                    
                                    child2 += 1; parent += 1

                                # both subregions have the same tag
                                else
                        
                                        dmatrix[child1, j + 1, :] +=
                                            ( dmatrix[parent, j, :] +
                                              dmatrix[parent + 1, j, :] ) / sqrt(2)
                                        dmatrix[child2, j + 1, :] +=
                                            ( dmatrix[parent, j, :] -
                                              dmatrix[parent + 1, j, :] ) / sqrt(2)
                                   
                                    child1 += 1; child2 += 1; parent += 2
                                end # of if child2 == r3 ...
                            end # of while child1 < rs2 ...
                        end # of if rs2 == rs3 ... else
                    end # of if n == 1 ... elseif n > 1 ...
                #end # of if countnz(...)
            end # of for r = 1:regioncount
        end # of for j = 1:(jmax - 1)
    f = dmatrix[:,jmax]
    invID = invperm(GP.ind)
    f = f[invID]
    return f
end

function Walsh_Multiplication(signal, dmatrix_rec2D, GProws,GPcols)
    #dm_x,~,~ = Main.get_Walsh_coefficients1D(signal, GPcols, 1)
    n=length(signal)
    G = gpath(n)
    G.f = reshape(signal,length(signal),1)
    dm_x = ghwt_analysis!(G, GP = GPcols)
    dm_x = vec(dm_x)
    dm_g = dmatrix_rec2D *dm_x;
    N = size(GProws.tag,1)
    jmax = Int(length(dm_g)/N)
    dm_g = reshape(dm_g,N,jmax);
    f = ghwt_synthesis_aftermultiplication(dm_g,GProws)
    return f
end

function ghwt_2d_sparse(A::Matrix{Float64},threshold::Float64,GProws::GraphPart,GPcols::GraphPart)
    m,n = Base.size(A)
    jmax_row = Base.size(GProws.rs,2)
    jmax_col = Base.size(GPcols.rs,2)
    dvec,infovec = eghwt_bestbasis_2d(A, GProws, GPcols)
    dvec = dvec'
    A_flat = vec(dvec)  # Ensure it's a 1D array
    dvec_rec = zeros(m*n)
    sorted_indices = sortperm(abs.(A_flat), rev=true)
    E = cumsum(abs.(A_flat[sorted_indices]) .^ 2) / sum(abs.(A_flat) .^ 2)    
    # Find index where energy exceeds threshold
    n_coeff = findfirst(x -> x > threshold, E)
    dvec_rec[sorted_indices[1:n_coeff]] = A_flat[sorted_indices[1:n_coeff]]
    dmatrix_rec2D = zeros(m *(jmax_row), n * (jmax_col))
    for i = 1:n_coeff
        idx,idy = infovec[:,sorted_indices[i]]
        dmatrix_rec2D[idx,idy] = A_flat[sorted_indices[i]]
    end
    dmatrix_rec2D = sparse(dmatrix_rec2D);
    return dmatrix_rec2D
end

function Walsh_bicgstab_solver(dmatrix_rec2D, b, GProws,GPcols; tol=1e-8, max_iter=1000)
    n,jmax_col = size(GPcols.tag)
    x = spzeros(n)  # Initial guess (zero vector)
    Ax = Walsh_Multiplication(x, dmatrix_rec2D, GProws,GPcols)
    r = b - Ax   # Initial residual
    r_hat = copy(r) # Shadow residual (fixed)
    p = copy(r)
    v = zeros(n)
    s = zeros(n)
    t = zeros(n)
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    for iter in 1:max_iter
        rho_new = dot(r_hat, r)
        if abs(rho_new) < eps(Float64)
            break  # Breakdown condition
        end
        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        v = Walsh_Multiplication(p, dmatrix_rec2D, GProws, GPcols)
        alpha = rho_new / dot(r_hat, v)
        s = r - alpha * v

        if norm(s) < tol
            x += alpha * p
            println("Converged in $iter iterations.")
            return x
        end

        t = Walsh_Multiplication(s, dmatrix_rec2D, GProws,GPcols)
        omega = dot(t, s) / dot(t, t)
        x += alpha * p + omega * s
        r = s - omega * t

        rho_old = rho_new

        if norm(r) < tol
            println("Converged in $iter iterations.")
            return x
        end
    end

    println("Warning: BiCGSTAB did not converge within $max_iter iterations.")
    println(norm(r))
    return x
end

function bicgstab_solver(A, b::Vector; tol=1e-8, max_iter=1000)
    n = size(A, 2)
    x = spzeros(n)  # Initial guess (zero vector)
    r = b - A * x   # Initial residual
    r_hat = copy(r) # Shadow residual (fixed)
    p = copy(r)
    v = zeros(n)
    s = zeros(n)
    t = zeros(n)

    rho_old = 1.0
    alpha = 1.0
    omega = 1.0

    for iter in 1:max_iter
        rho_new = dot(r_hat, r)
        if abs(rho_new) < eps(Float64)
            break  # Breakdown condition
        end

        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)

        v = A * p
        alpha = rho_new / dot(r_hat, v)
        s = r - alpha * v

        if norm(s) < tol
            x += alpha * p
            println("Converged in $iter iterations.")
            return x
        end

        t = A * s
        omega = dot(t, s) / dot(t, t)
        x += alpha * p + omega * s
        r = s - omega * t

        rho_old = rho_new

        if norm(r) < tol
            println("Converged in $iter iterations.")
            return x
        end
    end

    println("Warning: BiCGSTAB did not converge within $max_iter iterations.")
    println(norm(r))
    return x
end

function Legendre_polynomial_uniform(N,M)
    # Order of expansion    
    # Function for computing beta coefficients
    #x0 = range(-1,1,length = N)  # Gauss-Legendre nodes
    #w0 = 2 * V[1, :].^2  # Gauss-Legendre weights
    x0 = sort(2 .* rand(M) .- 1)
    # Initialize Legendre polynomial matrix
    P = zeros(M, N)
    P[:, 1] .= 1  # P_0(x) = 1
    
    # Compute first-order Legendre polynomial
    for i in 1:M
        P[i, 2] = x0[i] * P[i, 1]
    end
    
    # Recurrence relation for higher-order Legendre polynomials
    for j in 1:N-2
        jj = j + 2
        for i in 1:M
            P[i, jj] = ((2*j + 1) * x0[i] * P[i, jj-1] - j * P[i, jj-2]) / (j + 1)
        end
    end

    # Plot first 10 Legendre polynomials at Gaussian nodes
    #plot(x0, P[:, 1:10], seriestype=:scatter, markershape=:circle, linestyle=:dash, title="Legendre Polynomials at Gaussian Nodes")

    # Normalize Legendre polynomials
    P = P * Diagonal(sqrt.((2 .* (0:N-1) .+ 1) ./ 2))

    # Display message
    println("Legendre polynomial matrix computed and plotted successfully.")
    return P, x0
end

function Legendre_polynomial(N)
    # Order of expansion    
    # Function for computing beta coefficients
    beta = n -> 0.5 ./ sqrt.(1 .- (2 .* n) .^ (-2))
    
    # Construct Jacobi matrix of Legendre polynomials
    T = diagm(-1 => beta(1:(N-1))) + diagm(1 => beta(1:(N-1)))
    
    # Compute eigenvalues (nodes) and eigenvectors (weights)
    U, V = eigen(T)
    x0 = U  # Gauss-Legendre nodes
    w0 = 2 * V[1, :].^2  # Gauss-Legendre weights
    
    # Initialize Legendre polynomial matrix
    P = zeros(N, N)
    P[:, 1] .= 1  # P_0(x) = 1
    
    # Compute first-order Legendre polynomial
    for i in 1:N
        P[i, 2] = x0[i] * P[i, 1]
    end
    
    # Recurrence relation for higher-order Legendre polynomials
    for j in 1:N-2
        jj = j + 2
        for i in 1:N
            P[i, jj] = ((2*j + 1) * x0[i] * P[i, jj-1] - j * P[i, jj-2]) / (j + 1)
        end
    end

    # Plot first 10 Legendre polynomials at Gaussian nodes
    #plot(x0, P[:, 1:10], seriestype=:scatter, markershape=:circle, linestyle=:dash, title="Legendre Polynomials at Gaussian Nodes")

    # Normalize Legendre polynomials
    P = Diagonal(sqrt.(w0)) * P * Diagonal(sqrt.((2 .* (0:N-1) .+ 1) ./ 2))

    # Display message
    println("Legendre polynomial matrix computed and plotted successfully.")
    return P, x0, w0
end

function partition_tree_fiedler_new(G::GraphSignal.GraphSig; method::Symbol = :Lrw, swapRegion::Bool = true)
    #
    # 0. Preliminary stuff
    #
    # constanInt
    N = G.length
    jmax = max(3 * floor(Int, log2(N)), 4) # jmax >= 4 is guaranteed.
    # This jmax is the upper bound of the true jmax; the true jmax
    # is computed as the number of columns of the matrix `rs` in the end.

    # TRACKING VARIABLES

    # `ind` records the way in which the nodes are indexed on each level
    ind = Vector{Int}(1:N)

    # `inds` records the way in which the nodes are indexed on all levels
    inds = zeros(Int, N, jmax)
    inds[:, 1] = ind

    # `rs` stands for regionstarInt, meaning that the index in `ind` of the first
    # point in region number `i` is `rs[i]`
    rs = zeros(Int, N + 1, jmax)
    rs[1, :] .= 1
    rs[2, 1] = N + 1

    #
    # 1. Partition the graph to yield rs, ind, and jmax
    #
    j = 1                       # j=1 in julia <=> j=0 in theory
    regioncount = 0
    rs0 = 0                     # define here for the whole loops,
                                # which differs from MAIntAB.
    while regioncount < N
        #regioncount = countnz(rs[:,j]) - 1
        regioncount = count(!iszero,rs[:, j]) - 1 # the number of regions on level j
        if j == jmax  # add a column to rs and inds for level j+1, if necessary
            rs = hcat(rs, vcat(Int(1), zeros(Int,N)))
            inds = hcat(inds, zeros(Int, N))
            jmax = jmax + 1
        end
        # for tracking the child regions
        rr = 1
        for r = 1:regioncount   # cycle through the parent regions
            rs1 = rs[r, j]      # the start of the parent region
            rs2 = rs[r + 1, j] # 1 node after the end of the parent region
            n = rs2 - rs1   # the number of nodes in the parent region
            if n > 1            # regions with 2 or more nodes
                indrs = ind[rs1:(rs2 - 1)]
                # partition the current region
                (pm, ) = partition_fiedler(G.W[indrs,indrs], method = method)
                # determine the number of poinInt in child region 1
                n1 = sum(pm .> 0)
                # switch regions 1 and 2, if necessary, based on the sum of
                # edge weighInt to the previous child region (i.e., cousin)
                if r == 1 && swapRegion && regioncount > 1
                    rs3 = rs[r + 2, j]
                    if sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .> 0]]) >
                        sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .< 0]])
                        pm = -pm
                        n1 = n - n1
                    end
                end
                if r > 1 && swapRegion && regioncount<=2
                    # MAIntAB: sum(sum(...)) instead of sum(...)
                    rs3 = rs[r + 2, j]
                    if sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .> 0]]) < sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .< 0]]) 
                        pm = -pm
                        n1 = n - n1
                    end
                end
                if r > 1 && r<regioncount && swapRegion && regioncount > 2
                    # MAIntAB: sum(sum(...)) instead of sum(...)
                    rs3 = rs[r + 2, j]
                    w1 = sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .> 0]])/sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .< 0]]) 
                    w2 = sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .> 0]])/sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .< 0]])
                    if w2/w1>1
                        pm = -pm
                        n1 = n - n1
                    end
                elseif r > 1 && r == regioncount && swapRegion && regioncount > 2
                    rs3 = rs[r + 2, j]
                    w2 = sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .> 0]])/sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .< 0]])
                    if w2>1
                        pm = -pm
                        n1 = n - n1
                    end
                end
                # update the indexing
                ind[rs1:(rs1 + n1 - 1)] = indrs[pm .> 0]
                ind[(rs1 + n1):(rs2 - 1)] = indrs[pm .< 0]
                # update the region tracking
                rs[rr + 1, j + 1] = rs1 + n1
                rs[rr + 2, j + 1] = rs2
                rr = rr + 2
                rs0 = rs1 +n1
            elseif n == 1       # regions with 1 node
                rs[rr + 1, j + 1] = rs2
                rr = rr + 1
                rs0 = rs1
            end # of if n > 1 ... elseif n==1 construct
        end # of for r=1:regioncount
        j = j + 1
        inds[:, j] = ind
    end # of while regioncount < N statement

    #
    # 2. Postprocessing
    #
    # get rid of excess columns in rs
    rs = rs[:, 1:(j - 1)]         # in MAIntAB, it was rs(:,j:end) = [];
    # get rid of excess columns in inds
    inds = inds[:, 1:(j - 1)]
    # create a GraphPart object
    return GraphPart(ind, rs; inds = inds, method = method)
end # of function partition_tree_fiedler

function ghwt_2d_sparse_new(A::Matrix{Float64},threshold::Float64,GProws::GraphPart,GPcols::GraphPart)
    m,n = Base.size(A)
    dvec,infovec = eghwt_bestbasis_2d(A, GProws, GPcols)
    dvec = dvec'
    A_flat = vec(dvec)  # Ensure it's a 1D array
    sorted_indices = sortperm(abs.(A_flat), rev=true)
    E = cumsum(abs.(A_flat[sorted_indices]) .^ 2) / sum(abs.(A_flat) .^ 2)    
    # Find index where energy exceeds threshold
    n_coeff = findfirst(x -> x > threshold, E)
    dvec_rec = A_flat[sorted_indices[1:n_coeff]]
    infovec_rec = infovec[:,sorted_indices[1:n_coeff]]
    return dvec_rec,infovec_rec
end

function Walsh_Multiplication_fast(signal, dvec_rec,infovec_rec, GProws,GPcols)
    #dm_x,~,~ = Main.get_Walsh_coefficients1D(signal, GPcols, 1)
    n=length(signal)
    G = gpath(n)
    G.f = reshape(signal,length(signal),1)
    dm_x = ghwt_analysis!(G, GP = GPcols)
    dm_x = vec(dm_x)
    dm_x = dm_x[infovec_rec[2,:]]
    L = dm_x .* dvec_rec
    m,n = size(GProws.tag)
    dm_g = zeros(m*n,1)
    for i = 1: length(L)
        ind = infovec_rec[1,i]
        dm_g[ind] += L[i]
    end
    dm_g = reshape(dm_g,m,n);
    f = ghwt_synthesis_aftermultiplication(dm_g,GProws)
    return f
end

function Walsh_Multiplication_fast2D(matrix, dvec_rec,infovec_rec, GProws,GPcols)
    ~,n = size(matrix)
    m,~ = size(GProws.tag)
    results = zeros(m,n)
    @threads for i in 1:n
        results[:,i] = Walsh_Multiplication_fast(matrix[:,i], dvec_rec,infovec_rec, GProws,GPcols)
    end
    return results
end

function partition_tree_fiedler_new2(G::GraphSignal.GraphSig; method::Symbol = :Lrw, swapRegion::Bool = true)
    #
    # 0. Preliminary stuff
    #
    # constanInt
    N = G.length
    jmax = max(3 * floor(Int, log2(N)), 4) # jmax >= 4 is guaranteed.
    # This jmax is the upper bound of the true jmax; the true jmax
    # is computed as the number of columns of the matrix `rs` in the end.

    # TRACKING VARIABLES

    # `ind` records the way in which the nodes are indexed on each level
    ind = Vector{Int}(1:N)

    # `inds` records the way in which the nodes are indexed on all levels
    inds = zeros(Int, N, jmax)
    inds[:, 1] = ind

    # `rs` stands for regionstarInt, meaning that the index in `ind` of the first
    # point in region number `i` is `rs[i]`
    rs = zeros(Int, N + 1, jmax)
    rs[1, :] .= 1
    rs[2, 1] = N + 1

    #
    # 1. Partition the graph to yield rs, ind, and jmax
    #
    j = 1                       # j=1 in julia <=> j=0 in theory
    regioncount = 0
    rs0 = 0                     # define here for the whole loops,
                                # which differs from MAIntAB.
    while regioncount < N
        #regioncount = countnz(rs[:,j]) - 1
        regioncount = count(!iszero,rs[:, j]) - 1 # the number of regions on level j
        if j == jmax  # add a column to rs and inds for level j+1, if necessary
            rs = hcat(rs, vcat(Int(1), zeros(Int,N)))
            inds = hcat(inds, zeros(Int, N))
            jmax = jmax + 1
        end
        # for tracking the child regions
        rr = 1
        for r = 1:regioncount   # cycle through the parent regions
            rs1 = rs[r, j]      # the start of the parent region
            rs2 = rs[r + 1, j] # 1 node after the end of the parent region
            n = rs2 - rs1   # the number of nodes in the parent region
            if n > 1            # regions with 2 or more nodes
                indrs = ind[rs1:(rs2 - 1)]
                # partition the current region
                (pm, ) = partition_fiedler(G.W[indrs,indrs], method = method)
                # determine the number of poinInt in child region 1
                n1 = sum(pm .> 0)
                # switch regions 1 and 2, if necessary, based on the sum of
                # edge weighInt to the previous child region (i.e., cousin)
                if r == 1 && swapRegion && regioncount > 1
                    rs3 = rs[r + 2, j]
                    if sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .> 0]]) >
                        sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .< 0]])
                        pm = -pm
                        n1 = n - n1
                    end
                end
                if r > 1 && swapRegion && regioncount <=2
                    # MAIntAB: sum(sum(...)) instead of sum(...)
                    rs3 = rs[r + 2, j]
                    if sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .> 0]]) < sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .< 0]]) 
                        pm = -pm
                        n1 = n - n1
                    end
                end
                if r > 1 && swapRegion && regioncount > 2
                    # MAIntAB: sum(sum(...)) instead of sum(...)
                    rs3 = rs[r + 2, j]
                    w1 = sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .> 0]])/sum(G.W[ind[rs0:(rs1 - 1)], indrs[pm .< 0]]) 
                    w2 = sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .> 0]])/sum(G.W[ind[rs2:(rs3 - 1)], indrs[pm .< 0]])
                    if w2/w1>1
                        pm = -pm
                        n1 = n - n1
                    end
                end
                # update the indexing
                ind[rs1:(rs1 + n1 - 1)] = indrs[pm .> 0]
                ind[(rs1 + n1):(rs2 - 1)] = indrs[pm .< 0]
                # update the region tracking
                rs[rr + 1, j + 1] = rs1 + n1
                rs[rr + 2, j + 1] = rs2
                rr = rr + 2
                rs0 = rs1 +n1
            elseif n == 1       # regions with 1 node
                rs[rr + 1, j + 1] = rs2
                rr = rr + 1
                rs0 = rs1
            end # of if n > 1 ... elseif n==1 construct
        end # of for r=1:regioncount
        j = j + 1
        inds[:, j] = ind
    end # of while regioncount < N statement

    #
    # 2. Postprocessing
    #
    # get rid of excess columns in rs
    rs = rs[:, 1:(j - 1)]         # in MAIntAB, it was rs(:,j:end) = [];
    # get rid of excess columns in inds
    inds = inds[:, 1:(j - 1)]
    # create a GraphPart object
    return GraphPart(ind, rs; inds = inds, method = method)
end # of function partition_tree_fiedler


function dyadic_partition(S::Matrix{Float64}, C_max::Int)
    n_rows, n_cols = size(S)
    L = round(Int, log2(n_cols / C_max))  # level depth based on C_max columns

    dyadic_blocks = []
    dyadic_infos = []
    for ℓ in 0:L
        level_blocks = []
        level_infos = []
        num_k = 2^(L - ℓ)           # number of column partitions (dyadic frequency bins)
        num_j = 2^ℓ     # number of row partitions

        row_block_size = ceil(Int, n_rows / num_j)
        col_block_size = ceil(Int, n_cols / num_k)

        for j in 1:num_j
            for k in 1:num_k
                row_start = (j - 1) * row_block_size + 1
                row_end   = min(j * row_block_size, n_rows)

                col_start = (k - 1) * col_block_size + 1
                col_end   = min(k * col_block_size, n_cols)

                push!(level_blocks, (
                    level = ℓ,
                    j = j,
                    k = k,
                    row_range = row_start:row_end,
                    col_range = col_start:col_end,
                    block = @view S[row_start:row_end, col_start:col_end]
                ))
                push!(level_infos, (
                    level = ℓ,
                    j = j,
                    k = k,
                    row_range = row_start:row_end,
                    col_range = col_start:col_end,
                ))
            end
        end

        push!(dyadic_blocks, level_blocks)
        push!(dyadic_infos, level_infos)
    end

    return dyadic_blocks,dyadic_infos
end


function compress_dyadic_blocks(dyadic_blocks, acc)
    L = length(dyadic_blocks) - 1  # Max level
    B = Dict()
    BL = Dict()
    P = Dict()
    # === Step 1: Level 0 (initial blocks) ===
    for block_info in dyadic_blocks[1]  # level 0
        j, k = block_info.j, block_info.k
        A = block_info.block
        Js, P0, r  = ID_col(A, acc)
        U = A[:,Js]
        B[(0, j, k)] = U
        P[(0, j, k)] = P0
    end

    # === Step 2: Hierarchical compression for levels ℓ = 1 to L ===
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks[ℓ+1] has level-ℓ blocks
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent j index at level ℓ-1

            # Get previous level blocks
            A1 = B[(ℓ - 1, j₀, 2k - 1)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k - 1).block
            A2 = B[(ℓ - 1, j₀, 2k)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k).block

            # Determine top or bottom halves
            h1 = size(A1, 1) ÷ 2
            h2 = size(A2, 1) ÷ 2

            if isodd(j)
                A1_half = A1[1:h1, :]
                A2_half = A2[1:h2, :]
            else
                A1_half = A1[h1+1:end, :]
                A2_half = A2[h2+1:end, :]
            end

            A = hcat(A1_half, A2_half)

            # Interpolative Decomposition
            Js, Pk, r  = ID_col(A, acc)
            U = A[:,Js]
            if ℓ == L
                BL[(ℓ, j)] = U
                B[(ℓ, j)] = U
                P[(ℓ, j)] = Pk
            else
                B[(ℓ, j, k)] = U
                P[(ℓ, j, k)] = Pk
            end
        end
    end

    return BL, P
end


function find_block(dyadic_blocks, ℓ, j, k)
    for block in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 0-indexed in levels
        if block.j == j && block.k == k
            return block
        end
    end
    error("Block not found at level $ℓ, j=$j, k=$k")
end

function ID_row(A::AbstractMatrix{Float64}, acc::Float64)
    # This function computes a row interpolative decomposition (ID)
    # such that A ≈ Z * A[Js, :], where Z[Js, :] = I_k
    # Output:
    #   Js :: Vector{Int}         -- indices of selected rows
    #   X  :: Matrix{Float64}     -- interpolation matrix (size m × k)
    #   out:: Int (optional)      -- estimated rank

    Js, X_raw, k = ID_col(transpose(A), acc)
    X = transpose(X_raw)
    return Js, X, k
end
function ID_col(A::AbstractMatrix{Float64}, acc::Float64)
    normA = norm(A)
    m, n = size(A)
    dim = min(m, n)

    if dim <= 2000
        ss = svdvals(A)
        s = 1
        k = max(count(x -> x / ss[1] > 0.1 * acc * normA, ss), s)
        _, R, ind = qr(A, Val(true))  # thin QR with pivoting
        T = R[1:k, 1:k] \ R[1:k, k+1:end]
        Z = zeros(k, n)
        Z[:, ind] = hcat(Matrix(I, k, k), T)
        Js = ind[1:k]
        return Js, Z, k
    else
        println("randomized method is used.")
        Q, B, ss = random_qb(A, acc)
        s = 1
        k = max(count(x -> x / ss[1] > 0.1 * acc * normA, ss), s)
        _, R, ind = qr(B, Val(true))  # thin QR with pivoting
        T = R[1:k, 1:k] \ R[1:k, k+1:end]
        Z = zeros(k, n)
        Z[:, ind] = hcat(Matrix(I, k, k), T)
        Js = ind[1:k]
        return Js, Z, k
    end
end

function random_qb(A::AbstractMatrix{Float64}, acc::Float64)
    normA = norm(A)
    m, n = size(A)
    s = min(m, n)
    b = min(450, s)
    tol = 0.1 * acc * normA * sqrt(s)

    Q = zeros(m, 0)
    B = zeros(0, n)

    for j in 1:s
        G = randn(n, b)
        U, _ = qr(A * G)
        if j > 1
            U, _ = qr(U - Q * (Q' * U))  # orthogonalize
        end
        newB = U' * A
        B = vcat(B, newB)
        Q = hcat(Q, U)
        A -= U * newB
        if norm(A) < tol
            ss = svdvals(B)
            k = round(Int, 0.97 * length(ss))
            if ss[k] > 0.1 * normA * acc
                tol /= 2
            else
                return Q, B, ss
            end
        end
    end
    ss = svdvals(B)
    return Q, B, ss
end


function apply_compressed_operator(B::Dict, P::Dict, alpha::Vector{Float64}, dyadic_blocks)
    L = maximum(k[1] for k in keys(P))  # determine max level
    beta = Dict{Any, Vector{Float64}}()  # holds intermediate β

    # Step 1: Apply first-level interpolation (ℓ = 0)
    for block_info in dyadic_blocks[1]  # level 0 blocks
        j, k = block_info.j, block_info.k
        col_start, col_end = block_info.col_range.start, block_info.col_range.stop
        α_block = alpha[col_start:col_end]
        beta[(0, j, k)] = P[(0, j, k)] * α_block
    end

    # Step 2: Recursive interpolation from ℓ = 1 to L
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 1-indexed
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent index in row direction

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β_half = vcat(β1, β2)

            key = ℓ == L ? (ℓ, j) : (ℓ, j, k)
            beta[key] = P[key] * β_half
        end
    end
     # Step 2 to L−1: ℓ = 2:(L-1)
    for ℓ in 2:(L - 1)
        for block_info in dyadic_blocks[ℓ + 1]
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β = vcat(β1, β2)

            beta[(ℓ, j, k)] = P[(ℓ, j, k)] * β
        end
    end

    # Step 3: Final matrix application at level L
    f_blocks = Dict{Int, Vector{Float64}}()
    for j in 1:2^L
        β_L_j = beta[(L, j)]
        B_L_j = B[(L, j)]

        @assert size(B_L_j, 2) == length(β_L_j) "Mismatch at j=$j: B cols = $(size(B_L_j, 2)), β length = $(length(β_L_j))"
        f_blocks[j] = B_L_j * β_L_j
    end

    # Step 4: Concatenate all output blocks
    f = vcat([f_blocks[j] for j in 1:2^L]...)
    return f
end

function compress_P(P, acc)
    dvec_dict = Dict()
    #infovec_dict = Dict()
    GProws_dict = Dict()
    GPcols_dict = Dict()
    for key in keys(P)
        block = P[key]
        GProws,GPcols = get_Walsh_partition(block)
        #dvec_rec,infovec_rec = ghwt_2d_sparse_new(block,1-acc^2,GProws,GPcols)
        dvec_rec = ghwt_2d_sparse(block,1-acc^2,GProws,GPcols)
        dvec_dict[key] = dvec_rec
        #infovec_dict[key] = infovec_rec
        GProws_dict[key] = GProws
        GPcols_dict[key] = GPcols
    end
    return dvec_dict, GProws_dict, GPcols_dict
end

function apply_compressed_operator_Walsh(B::Dict, alpha::Vector{Float64}, dyadic_blocks, dvec_dict, GProws_dict, GPcols_dict)
    L = maximum(k[1] for k in keys(dvec_dict))  # determine max level
    beta = Dict{Any, Vector{Float64}}()  # holds intermediate β

    # Step 1: Apply first-level interpolation (ℓ = 0)
    for block_info in dyadic_blocks[1]  # level 0 blocks
        j, k = block_info.j, block_info.k
        col_start, col_end = block_info.col_range.start, block_info.col_range.stop
        α_block = alpha[col_start:col_end]
        dvec_rec = dvec_dict[(0, j, k)]
        #infovec_rec = infovec_dict[(0, j, k)]
        GProws = GProws_dict[(0, j, k)]
        GPcols = GPcols_dict[(0, j, k)]
        #beta[(0, j, k)] = Walsh_Multiplication_fast(α_block, dvec_rec,infovec_rec, GProws,GPcols)
        beta[(0, j, k)] = Walsh_Multiplication(α_block, dvec_rec, GProws,GPcols)
    end

    # Step 2: Recursive interpolation from ℓ = 1 to L
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 1-indexed
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent index in row direction

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β_half = vcat(β1, β2)

            key = ℓ == L ? (ℓ, j) : (ℓ, j, k)
            dvec_rec = dvec_dict[key]
            #infovec_rec = infovec_dict[key]
            GProws = GProws_dict[key]
            GPcols = GPcols_dict[key]
            #beta[key] = Walsh_Multiplication_fast(β_half, dvec_rec,infovec_rec, GProws,GPcols)
            beta[key] = Walsh_Multiplication(β_half, dvec_rec, GProws,GPcols)
        end
    end
     # Step 2 to L−1: ℓ = 2:(L-1)
    for ℓ in 2:(L - 1)
        for block_info in dyadic_blocks[ℓ + 1]
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β = vcat(β1, β2)
            dvec_rec = dvec_dict[(ℓ, j, k)]
            #infovec_rec = infovec_dict[(ℓ, j, k)]
            GProws = GProws_dict[(ℓ, j, k)]
            GPcols = GPcols_dict[(ℓ, j, k)]
            #beta[(ℓ, j, k)] = Walsh_Multiplication_fast(β, dvec_rec,infovec_rec, GProws,GPcols)
            beta[(ℓ, j, k)] = Walsh_Multiplication(β, dvec_rec, GProws,GPcols)
        end
    end

    # Step 3: Final matrix application at level L
    f_blocks = Dict{Int, Vector{Float64}}()
    for j in 1:2^L
        β_L_j = beta[(L, j)]
        B_L_j = B[(L, j)]

        @assert size(B_L_j, 2) == length(β_L_j) "Mismatch at j=$j: B cols = $(size(B_L_j, 2)), β length = $(length(β_L_j))"
        f_blocks[j] = B_L_j * β_L_j
    end

    # Step 4: Concatenate all output blocks
    f = vcat([f_blocks[j] for j in 1:2^L]...)
    return f
end

function compress_P2(P, acc)
    dvec_dict = Dict()
    #infovec_dict = Dict()
    GProws_dict = Dict()
    GPcols_dict = Dict()
    for key in keys(P)
        #println(key)
        block = P[key]
        data_afterquest,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations_new(block,false,0)
        G = GraphSig(sparse(qrun.col_aff));
        GPcols = partition_tree_fiedler_new2(G; swapRegion = true);
        G = GraphSig(sparse(qrun.row_aff));
        GProws = partition_tree_fiedler_new2(G; swapRegion = true);
        #dvec_rec,infovec_rec = ghwt_2d_sparse_new(block,1-acc^2,GProws,GPcols)
        dvec_rec = ghwt_2d_sparse(block,1-acc^2,GProws,GPcols)
        dvec_dict[key] = dvec_rec
        #infovec_dict[key] = infovec_rec
        GProws_dict[key] = GProws
        GPcols_dict[key] = GPcols
    end
    return dvec_dict, GProws_dict, GPcols_dict
end

function compress_P3(P, acc)
    dvec_dict = Dict()
    #infovec_dict = Dict()
    GProws_dict = Dict()
    GPcols_dict = Dict()
    for key in keys(P)
        println(key)
        block = P[key]
        data_afterquest,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations_new(block,false,0)
        GProws,GPcols = get_Walsh_partition(data_afterquest)
        #dvec_rec,infovec_rec = ghwt_2d_sparse_new(block,1-acc^2,GProws,GPcols)
        dvec_rec = ghwt_2d_sparse(data_afterquest,1-acc^2,GProws,GPcols)
        dvec_dict[key] = dvec_rec
        #infovec_dict[key] = infovec_rec
        GProws_dict[key] = coifman_row_order
        GPcols_dict[key] = coifman_col_order
        GC.gc()
    end
    return dvec_dict, GProws_dict, GPcols_dict
end

function compress_dyadic_blocks_rank(dyadic_blocks, rr)
    L = length(dyadic_blocks) - 1  # Max level
    B = Dict()
    BL = Dict()
    P = Dict()
    # === Step 1: Level 0 (initial blocks) ===
    for block_info in dyadic_blocks[1]  # level 0
        j, k = block_info.j, block_info.k
        A = block_info.block
        Js, P0, r  = ID_col_rank(A, rr)
        U = A[:,Js]
        B[(0, j, k)] = U
        P[(0, j, k)] = P0
    end

    # === Step 2: Hierarchical compression for levels ℓ = 1 to L ===
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks[ℓ+1] has level-ℓ blocks
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent j index at level ℓ-1

            # Get previous level blocks
            A1 = B[(ℓ - 1, j₀, 2k - 1)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k - 1).block
            A2 = B[(ℓ - 1, j₀, 2k)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k).block

            # Determine top or bottom halves
            h1 = size(A1, 1) ÷ 2
            h2 = size(A2, 1) ÷ 2

            if isodd(j)
                A1_half = A1[1:h1, :]
                A2_half = A2[1:h2, :]
            else
                A1_half = A1[h1+1:end, :]
                A2_half = A2[h2+1:end, :]
            end

            A = hcat(A1_half, A2_half)

            # Interpolative Decomposition
            Js, Pk, r  = ID_col_rank(A, rr)
            U = A[:,Js]
            if ℓ == L
                BL[(ℓ, j)] = U
                B[(ℓ, j)] = U
                P[(ℓ, j)] = Pk
            else
                B[(ℓ, j, k)] = U
                P[(ℓ, j, k)] = Pk
            end
        end
    end

    return BL, P
end

function ID_col_rank(A::AbstractMatrix{Float64}, r)
    normA = norm(A)
    m, n = size(A)
    dim = min(m, n)

    if dim <= 2000
        ss = svdvals(A)
        k = min(r,length(ss))
        _, R, ind = qr(A, Val(true))  # thin QR with pivoting
        T = R[1:k, 1:k] \ R[1:k, k+1:end]
        Z = zeros(k, n)
        Z[:, ind] = hcat(Matrix(I, k, k), T)
        Js = ind[1:k]
        return Js, Z, k
    else
        println("randomized method is used.")
        Q, B, ss = random_qb(A, acc)
        k = min(r,length(ss))
        _, R, ind = qr(B, Val(true))  # thin QR with pivoting
        T = R[1:k, 1:k] \ R[1:k, k+1:end]
        Z = zeros(k, n)
        Z[:, ind] = hcat(Matrix(I, k, k), T)
        Js = ind[1:k]
        return Js, Z, k
    end
end

function ID_row_rank(A::AbstractMatrix{Float64}, r)
    # This function computes a row interpolative decomposition (ID)
    # such that A ≈ Z * A[Js, :], where Z[Js, :] = I_k
    # Output:
    #   Js :: Vector{Int}         -- indices of selected rows
    #   X  :: Matrix{Float64}     -- interpolation matrix (size m × k)
    #   out:: Int (optional)      -- estimated rank

    Js, X_raw, k = ID_col(transpose(A), r)
    X = transpose(X_raw)
    return Js, X, k
end

function apply_compressed_operator_Walsh2(B::Dict, alpha::Vector{Float64}, dyadic_blocks, dvec_dict, GProws_dict, GPcols_dict, GProws_dict2, GPcols_dict2)
    L = maximum(k[1] for k in keys(dvec_dict))  # determine max level
    beta = Dict{Any, Vector{Float64}}()  # holds intermediate β

    # Step 1: Apply first-level interpolation (ℓ = 0)
    for block_info in dyadic_blocks[1]  # level 0 blocks
        j, k = block_info.j, block_info.k
        col_start, col_end = block_info.col_range.start, block_info.col_range.stop
        α_block = alpha[col_start:col_end]
        dvec_rec = dvec_dict[(0, j, k)]
        #infovec_rec = infovec_dict[(0, j, k)]
        row_order = GProws_dict[(0, j, k)]
        col_order = GPcols_dict[(0, j, k)]
        GProws = GProws_dict2[(0, j, k)]
        GPcols = GPcols_dict2[(0, j, k)]
        #beta[(0, j, k)] = Walsh_Multiplication_fast(α_block, dvec_rec,infovec_rec, GProws,GPcols)
        f = Walsh_Multiplication(α_block[col_order], dvec_rec, GProws,GPcols)
        invID = invperm(row_order)
        beta[(0, j, k)] = f[invID]
    end

    # Step 2: Recursive interpolation from ℓ = 1 to L
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 1-indexed
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent index in row direction

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β_half = vcat(β1, β2)

            key = ℓ == L ? (ℓ, j) : (ℓ, j, k)
            dvec_rec = dvec_dict[key]
            row_order = GProws_dict[key]
            col_order = GPcols_dict[key]
            GProws = GProws_dict2[key]
            GPcols = GPcols_dict2[key]
            f = Walsh_Multiplication(β_half[col_order], dvec_rec, GProws,GPcols)
            invID = invperm(row_order)
            beta[key] = f[invID]
        end
    end
     # Step 2 to L−1: ℓ = 2:(L-1)
    for ℓ in 2:(L - 1)
        for block_info in dyadic_blocks[ℓ + 1]
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β = vcat(β1, β2)
            dvec_rec = dvec_dict[(ℓ, j, k)]
            #infovec_rec = infovec_dict[(ℓ, j, k)]
            row_order = GProws_dict[(ℓ, j, k)]
            col_order = GPcols_dict[(ℓ, j, k)]
            GProws = GProws_dict2[(ℓ, j, k)]
            GPcols = GPcols_dict2[(ℓ, j, k)]
            f = Walsh_Multiplication(β[col_order], dvec_rec, GProws,GPcols)
            invID = invperm(row_order)
            beta[(ℓ, j, k)] = f[invID]
        end
    end

    # Step 3: Final matrix application at level L
    f_blocks = Dict{Int, Vector{Float64}}()
    for j in 1:2^L
        β_L_j = beta[(L, j)]
        B_L_j = B[(L, j)]

        @assert size(B_L_j, 2) == length(β_L_j) "Mismatch at j=$j: B cols = $(size(B_L_j, 2)), β length = $(length(β_L_j))"
        f_blocks[j] = B_L_j * β_L_j
    end

    # Step 4: Concatenate all output blocks
    f = vcat([f_blocks[j] for j in 1:2^L]...)
    return f
end

function apply_compressed_operator_Walsh3(B::Dict, alpha::Vector{Float64}, dyadic_blocks, dvec_dict, GProws_dict, GPcols_dict)
    L = maximum(k[1] for k in keys(dvec_dict))  # determine max level
    beta = Dict{Any, Vector{Float64}}()  # holds intermediate β

    # Step 1: Apply first-level interpolation (ℓ = 0)
    for block_info in dyadic_blocks[1]  # level 0 blocks
        j, k = block_info.j, block_info.k
        col_start, col_end = block_info.col_range.start, block_info.col_range.stop
        α_block = alpha[col_start:col_end]
        dvec_rec = dvec_dict[(0, j, k)]
        #infovec_rec = infovec_dict[(0, j, k)]
        row_order = GProws_dict[(0, j, k)]
        col_order = GPcols_dict[(0, j, k)]
        Gm = gpath(length(row_order));
        GProws = partition_tree_fiedler(Gm; swapRegion = false);
        ghwt_analysis!(Gm, GP = GProws)
        Gn = gpath(length(col_order));
        GPcols = partition_tree_fiedler(Gn; swapRegion = false);
        ghwt_analysis!(Gn, GP = GPcols)
        #beta[(0, j, k)] = Walsh_Multiplication_fast(α_block, dvec_rec,infovec_rec, GProws,GPcols)
        f = Walsh_Multiplication(α_block[col_order], dvec_rec, GProws,GPcols)
        invID = invperm(row_order)
        beta[(0, j, k)] = f[invID]
    end

    # Step 2: Recursive interpolation from ℓ = 1 to L
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 1-indexed
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent index in row direction

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β_half = vcat(β1, β2)

            key = ℓ == L ? (ℓ, j) : (ℓ, j, k)
            dvec_rec = dvec_dict[key]
            #infovec_rec = infovec_dict[key]
            #GProws = GProws_dict[key]
            #GPcols = GPcols_dict[key]
            row_order = GProws_dict[key]
            col_order = GPcols_dict[key]
            Gm = gpath(length(row_order));
            GProws = partition_tree_fiedler(Gm; swapRegion = false);
            ghwt_analysis!(Gm, GP = GProws)
            Gn = gpath(length(col_order));
            GPcols = partition_tree_fiedler(Gn; swapRegion = false);
            ghwt_analysis!(Gn, GP = GPcols)
            #beta[key] = Walsh_Multiplication_fast(β_half, dvec_rec,infovec_rec, GProws,GPcols)
            #beta[key] = Walsh_Multiplication(β_half, dvec_rec, GProws,GPcols)
            f = Walsh_Multiplication(β_half[col_order], dvec_rec, GProws,GPcols)
            invID = invperm(row_order)
            beta[key] = f[invID]
        end
    end
     # Step 2 to L−1: ℓ = 2:(L-1)
    for ℓ in 2:(L - 1)
        for block_info in dyadic_blocks[ℓ + 1]
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β = vcat(β1, β2)
            dvec_rec = dvec_dict[(ℓ, j, k)]
            row_order = GProws_dict[(ℓ, j, k)]
            col_order = GPcols_dict[(ℓ, j, k)]
            Gm = gpath(length(row_order));
            GProws = partition_tree_fiedler(Gm; swapRegion = false);
            ghwt_analysis!(Gm, GP = GProws)
            Gn = gpath(length(col_order));
            GPcols = partition_tree_fiedler(Gn; swapRegion = false);
            ghwt_analysis!(Gn, GP = GPcols)
            #beta[(ℓ, j, k)] = Walsh_Multiplication_fast(β, dvec_rec,infovec_rec, GProws,GPcols)
            #beta[(ℓ, j, k)] = Walsh_Multiplication(β, dvec_rec, GProws,GPcols)
            f = Walsh_Multiplication(β[col_order], dvec_rec, GProws,GPcols)
            invID = invperm(row_order)
            beta[(ℓ, j, k)] = f[invID]
        end
    end

    # Step 3: Final matrix application at level L
    f_blocks = Dict{Int, Vector{Float64}}()
    for j in 1:2^L
        β_L_j = beta[(L, j)]
        B_L_j = B[(L, j)]

        @assert size(B_L_j, 2) == length(β_L_j) "Mismatch at j=$j: B cols = $(size(B_L_j, 2)), β length = $(length(β_L_j))"
        f_blocks[j] = B_L_j * β_L_j
    end

    # Step 4: Concatenate all output blocks
    f = vcat([f_blocks[j] for j in 1:2^L]...)
    return f
end

function compress_P_ratio(P, ratio)
    dvec_dict = Dict()
    #infovec_dict = Dict()
    GProws_dict = Dict()
    GPcols_dict = Dict()
    for key in keys(P)
        println(key)
        block = P[key]
        data_afterquest,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations_new(block,false,0)
        GProws,GPcols = get_Walsh_partition(data_afterquest)
        #dvec_rec,infovec_rec = ghwt_2d_sparse_new(block,1-acc^2,GProws,GPcols)
        dvec_rec = ghwt_2d_sparse_ratio(data_afterquest,ratio,GProws,GPcols)
        dvec_dict[key] = dvec_rec
        #infovec_dict[key] = infovec_rec
        GProws_dict[key] = coifman_row_order
        GPcols_dict[key] = coifman_col_order
        GC.gc()
    end
    return dvec_dict, GProws_dict, GPcols_dict
end

function ghwt_2d_sparse_ratio(A::Matrix{Float64},ratio,GProws::GraphPart,GPcols::GraphPart)
    m,n = Base.size(A)
    jmax_row = Base.size(GProws.rs,2)
    jmax_col = Base.size(GPcols.rs,2)
    dvec,infovec = eghwt_bestbasis_2d(A, GProws, GPcols)
    dvec = dvec'
    A_flat = vec(dvec)  # Ensure it's a 1D array
    dvec_rec = zeros(m*n)
    sorted_indices = sortperm(abs.(A_flat), rev=true)
    E = cumsum(abs.(A_flat[sorted_indices]) .^ 2) / sum(abs.(A_flat) .^ 2)    
    # Find index where energy exceeds threshold
    n_coeff = max(Int(floor(m*n*ratio)),1)
    dvec_rec[sorted_indices[1:n_coeff]] = A_flat[sorted_indices[1:n_coeff]]
    dmatrix_rec2D = zeros(m *(jmax_row), n * (jmax_col))
    for i = 1:n_coeff
        idx,idy = infovec[:,sorted_indices[i]]
        dmatrix_rec2D[idx,idy] = A_flat[sorted_indices[i]]
    end
    dmatrix_rec2D = sparse(dmatrix_rec2D);
    return dmatrix_rec2D
end

function compress_dyadic_blocks_Walsh(dyadic_blocks, acc,acc_Walsh)
    L = length(dyadic_blocks) - 1  # Max level
    B = Dict()
    BL = Dict()
    P = Dict()
    dvec_dict = Dict()
    row_dict = Dict()
    col_dict = Dict()
    ind_dict = Dict()
    # === Step 1: Level 0 (initial blocks) ===
    for block_info in dyadic_blocks[1]  # level 0
        j, k = block_info.j, block_info.k
        println([0,j,k])
        A = block_info.block
        #Js, P0, r  = ID_col(A, acc)
        Js,P0,r = ID_col(A, acc)
        U = A[:,Js]
        B[(0, j, k)] = U
        P[(0, j, k)] = P0
        dvec_dict[(0, j,k)] = sparse(P0)
        GC.gc()
    end
    
    # === Step 2: Hierarchical compression for levels ℓ = 1 to L ===
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks[ℓ+1] has level-ℓ blocks
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent j index at level ℓ-1

            # Get previous level blocks
            A1 = B[(ℓ - 1, j₀, 2k - 1)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k - 1).block
            A2 = B[(ℓ - 1, j₀, 2k)]#find_block(dyadic_blocks, ℓ - 1, j₀, 2k).block

            # Determine top or bottom halves
            h1 = size(A1, 1) ÷ 2
            h2 = size(A2, 1) ÷ 2

            if isodd(j)
                A1_half = A1[1:h1, :]
                A2_half = A2[1:h2, :]
            else
                A1_half = A1[h1+1:end, :]
                A2_half = A2[h2+1:end, :]
            end
            
            A = hcat(A1_half, A2_half)

            # Interpolative Decomposition
            Js,Pk,r,ind, dvec_rec,coifman_col_order,coifman_row_order = ID_col_Walsh(A, acc,acc_Walsh,0)
            row_order = invperm(coifman_row_order)
            U = A[:,Js]
            if ℓ == L
                println([ℓ,j])
                BL[(ℓ, j)] = U
                B[(ℓ, j)] = U
                P[(ℓ, j)] = Pk
                dvec_dict[(ℓ, j)] = dvec_rec
                row_dict[(ℓ, j)] = row_order
                col_dict[(ℓ, j)] = coifman_col_order
                ind_dict[(ℓ, j)] = ind
            else
                println([ℓ,j,k])
                B[(ℓ, j, k)] = U
                P[(ℓ, j, k)] = Pk
                dvec_dict[(ℓ, j,k)] = dvec_rec
                row_dict[(ℓ, j,k)] = row_order
                col_dict[(ℓ, j,k)] = coifman_col_order
                ind_dict[(ℓ, j,k)] = ind
            end
            GC.gc()
        end
    end

    return BL, P, dvec_dict,row_dict,col_dict, ind_dict
end



function ID_col_Walsh(A::AbstractMatrix{Float64}, acc::Float64,acc_Walsh::Float64,ifcos)
    normA = norm(A)
    m, n = size(A)
    dim = min(m, n)
    ss = svdvals(A)
    s = 1
    k = max(count(x -> x / ss[1] > 0.1 * acc * normA, ss), s)
    _, R, ind = qr(A, Val(true))  # thin QR with pivoting
    T = R[1:k, 1:k] \ R[1:k, k+1:end]
    Z = zeros(k, n)
    Z[:, ind] = hcat(Matrix(I, k, k), T)
    Js = ind[1:k]
    if isempty(T)
        dvec_rec = []
        coifman_col_order = []
        coifman_row_order = []
    else
        T_afterquest,coifman_col_order,coifman_row_order,qrun = build_kernel_permutations_new(T,false,ifcos)
        GProws,GPcols = get_Walsh_partition(T_afterquest)
        dvec_rec = ghwt_2d_sparse(T_afterquest,1-acc_Walsh.^2,GProws,GPcols)
    end
    return Js,Z,k,ind, dvec_rec,coifman_col_order,coifman_row_order
    
end

function apply_compressed_operator_Walsh4(B::Dict, alpha::Vector{Float64}, dyadic_blocks,dvec_dict,row_dict,col_dict, ind_dict)
    L = maximum(k[1] for k in keys(dvec_dict))  # determine max level
    beta = Dict{Any, Vector{Float64}}()  # holds intermediate β

    # Step 1: Apply first-level interpolation (ℓ = 0)
    for block_info in dyadic_blocks[1]  # level 0 blocks
        j, k = block_info.j, block_info.k
        col_start, col_end = block_info.col_range.start, block_info.col_range.stop
        α_block = alpha[col_start:col_end]
        dvec_rec = dvec_dict[(0, j, k)]
        row_order = row_dict[(0, j, k)]
        col_order = col_dict[(0, j, k)]
        ind = ind_dict[(0, j, k)]
        r = length(row_order)
        Gm = gpath(length(row_order));
        GProws = partition_tree_fiedler(Gm; swapRegion = false);
        ghwt_analysis!(Gm, GP = GProws)
        Gn = gpath(length(col_order));
        GPcols = partition_tree_fiedler(Gn; swapRegion = false);
        ghwt_analysis!(Gn, GP = GPcols)
        α_block2 = α_block[ind]
        sig =  α_block2[(r+1):end]
        f = Walsh_Multiplication(sig[col_order], dvec_rec, GProws,GPcols)
        f = α_block2[1:r] .+ f[row_order]
        beta[(0, j, k)] = f
    end

    # Step 2: Recursive interpolation from ℓ = 1 to L
    for ℓ in 1:L
        for block_info in dyadic_blocks[ℓ + 1]  # dyadic_blocks is 1-indexed
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)  # parent index in row direction

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β_half = vcat(β1, β2)

            key = ℓ == L ? (ℓ, j) : (ℓ, j, k)

            dvec_rec = dvec_dict[key]
            row_order = row_dict[key]
            col_order = col_dict[key]
            ind = ind_dict[key]
            r = length(row_order)
            Gm = gpath(length(row_order));
            GProws = partition_tree_fiedler(Gm; swapRegion = false);
            ghwt_analysis!(Gm, GP = GProws)
            Gn = gpath(length(col_order));
            GPcols = partition_tree_fiedler(Gn; swapRegion = false);
            ghwt_analysis!(Gn, GP = GPcols)
            β_half = β_half[ind]
            sig =  β_half[(r+1):end]
            f = Walsh_Multiplication(sig[col_order], dvec_rec, GProws,GPcols)
            f = β_half[1:r] .+ f[row_order]
            beta[key] = f
        end
    end
     # Step 2 to L−1: ℓ = 2:(L-1)
    for ℓ in 2:(L - 1)
        for block_info in dyadic_blocks[ℓ + 1]
            j, k = block_info.j, block_info.k
            j₀ = fld(j + 1, 2)

            β1 = beta[(ℓ - 1, j₀, 2k - 1)]
            β2 = beta[(ℓ - 1, j₀, 2k)]
            β = vcat(β1, β2)

            dvec_rec = dvec_dict[(ℓ, j, k)]
            row_order = row_dict[(ℓ, j, k)]
            col_order = col_dict[(ℓ, j, k)]
            ind = ind_dict[(ℓ, j, k)]
            r = length(row_order)
            Gm = gpath(length(row_order));
            GProws = partition_tree_fiedler(Gm; swapRegion = false);
            ghwt_analysis!(Gm, GP = GProws)
            Gn = gpath(length(col_order));
            GPcols = partition_tree_fiedler(Gn; swapRegion = false);
            ghwt_analysis!(Gn, GP = GPcols)
            β = β[ind]
            sig =  β[(r+1):end]
            f = Walsh_Multiplication(sig[col_order], dvec_rec, GProws,GPcols)
            f = β[1:r] .+ f[row_order]
            beta[(ℓ, j, k)] = f
        end
    end

    # Step 3: Final matrix application at level L
    f_blocks = Dict{Int, Vector{Float64}}()
    for j in 1:2^L
        β_L_j = beta[(L, j)]
        B_L_j = B[(L, j)]

        @assert size(B_L_j, 2) == length(β_L_j) "Mismatch at j=$j: B cols = $(size(B_L_j, 2)), β length = $(length(β_L_j))"
        f_blocks[j] = B_L_j * β_L_j
    end

    # Step 4: Concatenate all output blocks
    f = vcat([f_blocks[j] for j in 1:2^L]...)
    return f
end


end