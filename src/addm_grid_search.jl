"""
#!/usr/bin/env julia
Copyright (C) 2023, California Institute of Technology

This file is part of addm_toolbox.

addm_toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

addm_toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with addm_toolbox. If not, see <http://www.gnu.org/licenses/>.

---

Module: addm_grid_search.jl
Author: Lynn Yang, lynnyang@caltech.edu

Testing functions in aDDM Toolbox.
"""

using Pkg
Pkg.activate("addm")

using LinearAlgebra
using ProgressMeter
using BenchmarkTools

include("addm.jl")
include("util.jl")


function aDDM_grid_search(addm::aDDM, addmTrials::Dict{String, Vector{aDDMTrial}}, dList::Vector{Float64}, σList::Vector{Float64},
                          θList::Vector{Float64}, n::Int64, subject::String)
    """
    """
    dMLEList = Vector{Float64}(undef, n)
    σMLEList = Vector{Float64}(undef, n)
    θMLEList = Vector{Float64}(undef, n)

    NNL_List = Vector{}(undef, n)

    # Create an array of tuples for all parameter combinations.
    param_combinations = [(d, σ, θ) for d in dList, σ in σList, θ in θList]
    
    @showprogress for i in 1:n
        
        # Vectorized calculation of negative log-likelihood for all parameter combinations
        neg_log_like_array = [aDDM_negative_log_likelihood_threads(addm, addmTrials[subject], d, σ, θ) for (d, σ, θ) in param_combinations]
        
        # Find the index of the minimum negative log-likelihood and obtain the MLE parameters
        minIdx = argmin(neg_log_like_array)
        dMin, σMin, θMin = param_combinations[minIdx]
        
        # Store results in the preallocated arrays
        dMLEList[i] = dMin
        σMLEList[i] = σMin
        θMLEList[i] = θMin

        #NNL: NNL_List[i] = neg_log_like_array
    end

    return dMLEList, σMLEList, θMLEList #NNL:, NNL_List
end
