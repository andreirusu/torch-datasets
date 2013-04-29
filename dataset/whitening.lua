require 'torch'
require 'unsup'
require 'dataset'


-- ZCA-Whitening
function dataset.zca_whiten(data, means, P)
    local invP
    local auxdata = data:clone()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
        auxdata:resize(nsamples, n_dimensions)
    end
    if not means and not P then
        -- compute mean vector if not provided 
        means = torch.mean(auxdata, 1):squeeze()
        -- compute transformation matrix P if not provided
        local ce, cv = unsup.pcacov(auxdata)
        local ce = ce:add(1e-5):sqrt()
        local invce = ce:clone():pow(-1)
        local invdiag = torch.diag(invce)
        P = torch.mm(cv, invdiag)
        P = torch.mm(P, cv:t())

        -- compute inverse of the transformation
        local diag = torch.diag(ce)
        invP = torch.mm(cv:t(), diag)
        invP = torch.mm(invP, cv)
    end
    -- remove the means
    auxdata:add(torch.ger(torch.ones(nsamples), means):mul(-1))
    -- rotate in ZCA space
    auxdata = torch.mm(auxdata, P)

    data:copy(auxdata:resizeAs(data))
    return means, P, invP
end

