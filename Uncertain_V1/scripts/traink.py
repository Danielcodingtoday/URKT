import torch.nn as nn
import torch
 
 ##训练参数k和b
class ModifiedModel(nn.Module):
    def __init__(self, certain_model,uncertain_model):
        super(ModifiedModel, self).__init__()
        self.certain_model = certain_model
        self.uncertain_model = uncertain_model
        # 初始化k和b参数
        self.k = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, q, s, pid, N):
        with torch.no_grad():  # 确保不会更新原始模型的参数
            y_c,logits_mean_c, *_  = self.certain_model.predict(q, s, pid, N)
            y_uc,logits_mean_uc, *_  = self.uncertain_model.predict(q, s, pid, N)
            _,_,_,weight = self.certain_model.get_loss(q, s, pid)
        
        return torch.sigmoid(self.k*weight+self.b)*torch.sigmoid(logits_mean_c)+(1-torch.sigmoid(self.k*weight+self.b))*torch.sigmoid(logits_mean_uc)