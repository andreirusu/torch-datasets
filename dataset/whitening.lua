require 'torch'
require 'unsup'
require 'dataset'


-- ZCA-Whitening
function dataset.zca_whiten(data, means, P)
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
        ce:add(1e-5):sqrt():pow(-1)
        local diag = torch.diag(ce)
	    P = torch.mm(cv, diag)
	    P = torch.mm(P, cv:t())
    end
    -- remove the means
	auxdata:add(torch.ger(torch.ones(nsamples), means):mul(-1))
    -- rotate in ZCA space
	auxdata = torch.mm(auxdata, P)
	
    data:copy(auxdata:resizeAs(data))
	return means, P
end

