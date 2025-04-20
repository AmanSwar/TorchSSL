import torch
import torch.nn as nn

class NTXentLoss(nn.Module):

    def __init__(self , temp=0.5 , device="cuda"):
        super().__init__()

        self.temp = temp
        self.device = device
        self.crit = nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self, batch_size):

        N = 2 * batch_size
        mask = torch.ones((N , N) , dtype=bool).to(self.device)
        mask.fill_diagonal_(False)

        for i in range(batch_size):
            mask[i , batch_size + i] = False
            mask[batch_size + i , i] = False

        return mask
    
    def forward(self , z_i , z_j):

        batch_size = z_i.shape[0]

        z = torch.cat([z_i , z_j], dim=0)
        z = nn.functional.normalize(z , dim=1)

        similarity_mat = torch.matmul(z , z.T)


        mask = self._get_correlated_mask(batch_size)

        sim_ij = torch.diag(similarity_mat , batch_size)
        sim_ji = torch.diag(similarity_mat , -batch_size)
        positives = torch.cat([sim_ij , sim_ji] ,dim=0).unsqueeze(1)

        negatives = similarity_mat[mask].view(2 * batch_size , -1)
        logits = torch.cat([positives , negatives] , dim=1)
        logits /= self.temp

        labels = torch.zeros(2 * batch_size , dtype=torch.long).to(self.device)
        loss = self.crit(logits , labels)
        loss /= 2 * batch_size

        return loss



    
