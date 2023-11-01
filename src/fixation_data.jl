"""
    FixationData(probFixLeftFirst, latencies, transitions, fixations; 
                 fixDistType="fixation")
    
# Arguments:
- `probFixLeftFirst`: Float64 between 0 and 1, empirical probability that
    the left item will be fixated first.
- `latencies`: Vector corresponding to the empirical distribution of
    trial latencies (delay before first fixation) in milliseconds.
- `transitions`: Vector corresponding to the empirical distribution
    of transitions (delays between item fixations) in milliseconds.
- `fixations`: Dict whose indexing is defined according to parameter
    fixDistType. Each entry is an array corresponding to the
    empirical distribution of item fixation durations in
    milliseconds.
- `fixDistType`: String, one of {'simple', 'difficulty', 'fixation'},
    determines how the fixation distributions are indexed. If
    'simple', fixation distributions are indexed only by type (1st,
    2nd, etc). If 'difficulty', they are indexed by type and by trial
    difficulty, i.e., the absolute value for the trial's value
    difference. If 'fixation', they are indexed by type and by the
    value difference between the fixated and unfixated items.
"""
struct FixationData
    
    probFixLeftFirst::Float64
    latencies::Vector{Number}
    transitions::Vector{Number}
    fixations::Dict 
    fixDistType::String 

    function FixationData(probFixLeftFirst, latencies, transitions, fixations; fixDistType="fixation")
        availableDistTypes = ["simple", "difficulty", "fixation"]
        if !(fixDistType in availableDistTypes)
            throw(RuntimeError("Argument fixDistType must be one of {simple, difficulty, fixation}"))
        end
        new(probFixLeftFirst, latencies, transitions, fixations, fixDistType)
    end
end

"""

Create empirical distributions from the data to be used when generating
model simulations.

# Arguments
- `data`: a dict, indexed by subjectId, where each entry is a list of
    Trial objects.
- `timeStep`: integer, minimum duration of a fixation to be considered, in
    miliseconds.
- `maxFixTime`: integer, maximum duration of a fixation to be considered, in
    miliseconds.
- `numFixDists`: integer, number of fixation types to use in the fixation
    distributions. For instance, if numFixDists equals 3, then 3 separate
    fixation types will be used, corresponding to the 1st, 2nd and other
    (3rd and up) fixations in each trial.
- `fixDistType`: string, one of {'simple', 'difficulty', 'fixation'}, used to
    determine how the fixation distributions should be indexed. If
    'simple', then fixation distributions will be indexed only by type
    (1st, 2nd, etc). If 'difficulty', they will be indexed by type and by
    trial difficulty, i.e., the absolute value for the trial's value
    difference. If 'fixation', they will be indexed by type and by the
    value difference between the fixated and unfixated items.
- `valueDiffs`: list of integers. If fixDistType is 'difficulty' or
    'fixation', valueDiffs is a range correspoding to the item values to
    be used when indexing the fixation distributions.
- `subjectIds`: list of strings corresponding to the subjects whose data
    should be used. If not provided, all existing subjects will be used.

# Return
- A FixationData object.
"""
function get_empirical_distributions(data::Dict; timeStep::Number = 10, 
                                     maxFixTime::Number = 3000, 
                                     numFixDists::Int64 = 3, fixDistType::String = "fixation", 
                                     valueDiffs::Vector{Int64} = collect(-3:1:3), 
                                     subjectIds::Vector{String} = String[])
    
    fixDistType = String(fixDistType)
    availableTypes = ["simple", "difficulty", "fixation"]
    if (!(fixDistType in availableTypes))
        throw(error("fixDistType must be one of {simple, difficulty, fixation}"))
    end
    
    countLeftFirst = 0
    countTotalTrials = 0
    latenciesList = Number[]
    transitionsList = Number[]
    fixationsList = Dict()
    for fixNumber in 1:numFixDists
        if fixDistType == "simple"
            fixationsList[fixNumber] = Number[]
        else
            fixationsList[fixNumber] = Dict()
            for valueDiff in valueDiffs
                fixationsList[fixNumber][valueDiff] = Number[]
            end
        end
    end
    
    subjectIds = length(subjectIds) > 0 ? [String(subj) for subj in subjectIds] : collect(keys(data))

    for subjectId in subjectIds
        for trial in data[subjectId]
            
            # Discard trial if it has 1 or less item fixations.
            items = trial.fixItem
            
            if (!any(items!=nothing) || length(vcat(items[findall(x -> x == 1, items)], items[findall(x -> x == 2, items)])) <= 1)
                continue
            end
            
            fixUnfixValueDiffs = Dict(1 => trial.valueLeft - trial.valueRight, 
                                      2 => trial.valueRight - trial.valueLeft)
            
            # Find the last item fixation in this trial.
            # Use excludeCount as the stopping point for trail.fixItem in the calculations below.
            excludeCount = 0
            for i in length(trial.fixItem):-1:1
                excludeCount += 1
                if (trial.fixItem[i] == 1 || trial.fixItem[i] == 2)
                    break
                end
            end
            
            # Iterate over this trial's fixations (skip the last item fixation)
            latency = 0
            firstItemFixReached = false
            fixNumber = 1
            for i in 1:length(trial.fixItem) - excludeCount
                if trial.fixItem[i] != 1 && trial.fixItem[i] != 2
                    if !firstItemFixReached
                        latency += trial.fixTime[i]
                    elseif (trial.fixTime[i] >= timeStep && trial.fixTime[i] <= maxFixTime)
                        push!(transitionsList, trial.fixTime[i])
                    end
                else
                    if !firstItemFixReached
                        firstItemFixReached = true
                        push!(latenciesList, latency)
                    end
                    if fixNumber == 1
                        countTotalTrials += 1
                        if trial.fixItem[i] == 1 # First fixation was left.
                            countLeftFirst += 1
                        end
                    end
                    if (trial.fixTime[i] >= timeStep && trial.fixTime[i] <= maxFixTime)
                        if fixDistType == "simple"
                            push!(fixationsList[fixNumber], trial.fixTime[i])
                        elseif fixDistType == "difficulty"
                            valueDiff = abs(trial.valueLeft - trial.valueRight)
                            push!(fixationsList[fixNumber][valueDiff], trial.fixTime[i])
                        elseif fixDistType == "fixation"
                            valueDiff = fixUnfixValueDiffs[trial.fixItem[i]]
                            push!(fixationsList[fixNumber][valueDiff], trial.fixTime[i])
                        end
                    end
                    if fixNumber < numFixDists
                        fixNumber += 1
                    end
                end
            end
        end
    end
    probFixLeftFirst = countLeftFirst / countTotalTrials
    latencies = latenciesList
    transitions = transitionsList
    fixations = Dict()

    for fixNumber in 1:numFixDists
        if fixDistType == "simple"
            fixations[fixNumber] = [fixationsList[fixNumber]]
        else
            fixations[fixNumber] = Dict()
            for valueDiff in valueDiffs
                fixations[fixNumber][valueDiff] = [fixationsList[fixNumber][valueDiff]]
            end
        end
    end

    return FixationData(probFixLeftFirst, latencies, transitions, fixations, fixDistType=fixDistType)
end