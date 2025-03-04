"""
    grid_search(data, likelihood_fn, param_grid, return_grid_likelihoods = false; 
                likelihood_args =  (timeStep = 10.0, approxStateStep = 0.1), return_trial_likelihoods = false)

Compute the likelihood of either observed or simulated data for all parameter combinations in `param_grid`.

# Arguments

## Required 

- `data`: Data for which the sum of negative log likelihoods will be computed for each trial.
  Should be a vector of `ADDM.Trial` objects.
- `likelihood_fn`: Name of likelihood function to be used to compute likelihoods. 
  The toolbox has `ADDM.aDDM_get_trial_likelihood` and `ADDM.aDDM_get_trial_likelihood` defined.
  If comparing different generative processes then leave at default value of `nothing`
  and make sure to define a `likelihood_fn` in the `param_grid`.
- `param_grid`: Grid of parameter combinations for which the sum of nll's for the `data` is 
  computed.
- `fixed_params`: Default `Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)`.
  Parameters required by the `likelihood_fn` that are not specified to vary across likelihood 
  computations.

## Optional 

- `return_grid_nlls`: Default `true`. If true, will return the sum of nll's for 
  each parameter combination
  in the grid search.
- `likelihood_args`: Default `(timeStep = 10.0, approxStateStep = 0.1)`. Additional 
  arguments to be passed onto `likelihood_fn`. 
- `return_model_posteriors`: Default `false`. If true, will return the posterior probability 
  for each parameter combination in `param_grid`.
- `model_priors`: priors for each model probability if not assummed to be uniform. Should be
  specified as a `Dict` with values of probabilities matching the keys for the correct model
  specified in `param_grid`.

# Returns
- `best_part`: `Dict` containing the parameter combination with the lowest nll.
- `all_nll_df`: `DataFrame` containing sum of nll's for each parameter combination.
- `posteriors`: Likelihood for each trial for each parameter combination.

"""
function grid_search(data, param_grid, likelihood_fn = nothing, 
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    return_grid_nlls = true,
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_model_posteriors = false,
                    model_priors = nothing)

  n = length(param_grid) # number of parameter combinations specified in param_grid
  # all_nll = Vector{}(undef, n)
  # trial_likelihoods = Vector{}(undef, n)
  all_nll = Dict()
  trial_likelihoods = Dict()

  # Pass fixed parameters to the model
  # These don't need to be updated for each combination of the parameter grid
  # model = ADDM.aDDM()
  model = aDDM()
  for (k,v) in fixed_params setproperty!(model, k, v) end

  # What should the structure of param_grid be?
  # Currently it is a dictionary of NamedTuples 
  # Note that if all_nll is defined as a Vector above
  # then this loop relies on the keys of param_grid being numbers
  # e.g. 1 => (d = 0.003, sigma = 0.01, theta = 0.2)
  # because the keys index k is used to index the all_nll vector as well
  for (k, cur_grid_params) in param_grid
    
    # If likelihood_fn is not defined as argument to the function 
    # it should be defined in the param_grid
    # Extract that info and create variable that contains executable function
    if (:likelihood_fn in keys(cur_grid_params))
      likelihood_fn_str = cur_grid_params[:likelihood_fn]
      if (occursin(".", likelihood_fn_str))
        space, func = split(likelihood_fn_str, ".")
        if space == "ADDM"
          likelihood_fn = getfield(ADDM, Symbol(func))
        else
          error("Defining likelihood functions from modules outside of ADDM or Main is not currently supported.")
        end
      else
        likelihood_fn = getfield(Main, Symbol(likelihood_fn_str))
      end
    end

    # Update the model with the current parameter combination
    if !(cur_grid_params isa Dict)
      for (k,v) in pairs(cur_grid_params) setproperty!(model, k, v) end
    else
      for (k,v) in cur_grid_params setproperty!(model, k, v) end
    end
    
    # Make sure param names are converted to Greek symbols
    # ADDM.convert_param_symbols(model)
    convert_param_symbols(model)

    if return_model_posteriors
      # all_nll[k], trial_likelihoods[k] = ADDM.compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
      #                         return_trial_likelihoods = true)
      all_nll[k], trial_likelihoods[k] = compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                              return_trial_likelihoods = true)

    else
      all_nll[k] = compute_trials_nll(model, data, likelihood_fn, likelihood_args)
    end
  
  end

  # Wrangle likelihood data and extract best pars robustly before returning

  # Extract best pars
  # TODo: Convert param names to greek letters?
  minIdx = argmin(all_nll)
  best_fit_pars = Dict(pairs(param_grid[minIdx]))
  best_pars = merge(best_fit_pars, fixed_params)
  best_pars[:nll] = all_nll[minIdx]

  if return_grid_nlls
      # Add param info to all_nll
      all_nll_df = DataFrame()
      for (k, v) in param_grid
        row = DataFrame(Dict(pairs(v)))
        row.nll .= all_nll[k]
        # Appending won't work when different generative processes are part of param_grid
        # append!(all_nll_df, row)
        # vcat avoids that and can bind rows with different columns
        all_nll_df = vcat(all_nll_df, row, cols=:union)
      end
    
    # Why is this conditional on return_grid_nlls?
    if return_model_posteriors
        # TODo: Add param info to trial_likelihoods? 
        
        # Process trial likelihoods to compute model posteriors for each parameter combination
        nTrials = length(data)
        nModels = length(param_grid)

        # Define posteriors as a dictionary with the same keys as param_grid
        # Initialize the first posteriors as a float at 0 (value doesn't matter)
        posteriors = Dict(zip(keys(param_grid), repeat([[0.0]], outer = nModels)))

        if isnothing(model_priors)          
          model_priors = Dict(zip(keys(param_grid), repeat([1/nModels], outer = nModels)))
        end

        # Trialwise posterior updating
        for t in 1:nTrials

          # Reset denominator p(data) for each trial
          denom = 0

          # Update denominator summing
          for comb_key in keys(param_grid)
            if t == 1
              # Changed the initial posteriors so use model_priors for first trial
              denom += (model_priors[comb_key] * trial_likelihoods[comb_key][t])
            else
              denom += (posteriors[comb_key][t-1] * trial_likelihoods[comb_key][t])
            end
          end

          # Calculate the posteriors after this trial.
          for comb_key in keys(param_grid)
            if t == 1
              prior = model_priors[comb_key]
              # Assigning values to arrays in dictionaries is strange
              # This changes the first value that was initialized at 0.0 to the updated posterior
              # After trial 1
              posteriors[comb_key] = [(trial_likelihoods[comb_key][t] * prior / denom)]
            else
              prior = posteriors[comb_key][t-1]
              # After changing the first value we can now add the updated posteriors for the 
              # following trials
              append!(posteriors[comb_key],[(trial_likelihoods[comb_key][t] * prior / denom)])
            end
          end
        end

      return best_pars, all_nll_df, posteriors
    else
      return best_pars, all_nll_df
    end
  else
    return best_pars
  end

end

# Check each value in the posteriors adds up to 1 after each trial update
# for i in 1:length(posteriors)
#   println(sum([x[i] for x in values(posteriors)]))
# end

# # Get only the last value in posteriors
# [x[nTrials] for x in values(posteriors)]