import torch.nn.functional as F
import torch.nn as nn
import torch


class Classifier(nn.Module):

    def __init__(self, hidden_size, num_class):
        super().__init__()

        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.reset_parameters()

    def forward(self, x):
        logits = self.linear(x)
        prediction = torch.argmax(logits, dim=1)

        return logits, prediction

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


# class SNN_Classifier(nn.Module):
    
#     def __init__(self, sampler, args):
#         super().__init__()

#         self.sampler = sampler
#         self.tau = args.tau

#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, reps, ood=[]):
        
#         label_matrix, idx, batch_size = self.sampler.sample()

#         embs = F.normalize(reps, dim=1)
#         support_reps = embs[idx]

#         similarity = self.softmax(embs @ support_reps.T / self.tau)

#         if len(ood) !=0:
#             num_class = label_matrix.size(1)
#             dim = num_class * batch_size
#             similarity = self.fill_sim_matrix(ood, similarity, num_class, dim, batch_size)

#         probs = similarity @ label_matrix
        
#         preds = torch.argmax(probs, 1)

#         return probs, preds

#     def fill_sim_matrix(self, ood, similarity, num_class, dim, batch_size):
#         sim_mat = torch.empty((similarity.size(0), dim), dtype=torch.float32, device=similarity.device)
#         cnt = 0
#         for i in range(num_class):
#             if i in ood:
#                 sim_mat[:, i*batch_size : (i+1)*batch_size] = 0
#             else:
#                 sim_mat[:, i*batch_size : (i+1)*batch_size] = similarity[:, cnt*batch_size : (cnt+1)*batch_size]
#                 cnt += 1

#         return sim_mat