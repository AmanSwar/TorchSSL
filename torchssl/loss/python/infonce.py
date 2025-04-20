import torch
import torch.nn as nn


def infonce_loss(
        q,
        k,
        queue,
        temperature=0.2
):
    
    l_pos = torch.einsum('nc , nc->n' , [q , k]).unsqueeze(-1)
    l_neg = torch.mm(q , queue.T)

    logits = torch.cat([l_pos , l_neg] , dim=1)
    logits /= temperature


    labels = torch.zeros(logits.shape[0] , dtype=torch.long , device=logits.device)
    loss = nn.CrossEntropyLoss()(logits , labels)

    return loss
    