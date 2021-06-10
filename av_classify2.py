import torch
import torch.nn as nn
import torch.nn.functional as F

def string_to_num(string):
    strings = {
        "level1" : 64,
        "level2" : 128,
        "level3" : 256,
        "level4" : 512
    }

    return strings.get(string, None)

class AV_Model(nn.Module):
    def __init__(self, visual_net,a_l='level4',v_l='level4',data='MUSIC'):
        super(AV_Model, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        # 11个类
        if(data=='vgg'):
            self.class_layer = nn.Linear(string_to_num(v_l), 309)
        elif(data=='MUSIC'):
            self.class_layer = nn.Linear(string_to_num(v_l), 11)
        elif (data == 'Kinetic'):
            self.class_layer = nn.Linear(string_to_num(v_l), 39)
        elif (data == 'ActivityNet'):
            self.class_layer = nn.Linear(string_to_num(v_l), 200)
        elif (data == 'UCF101'):
            self.class_layer = nn.Linear(string_to_num(v_l), 101)
        elif (data == 'hmdb'):
            self.class_layer = nn.Linear(string_to_num(v_l), 51)
        elif (data == 'AVE'):
            self.class_layer = nn.Linear(string_to_num(v_l), 28)



    def forward(self, v_input):
        B, C, D, H, W = v_input.size()
        v_input = v_input.view(B*D, C, H, W)
        v_fea = self.visual_net(v_input)
        v_fea = F.relu(v_fea)
        v_fea = v_fea.view(B, D, -1)
        v_fea = torch.mean(v_fea,dim=1)
        #v_fea=self.dropout(v_fea)
        v_logits = self.class_layer(v_fea)
        #v_logits = self.dropout(v_logits)
        out = self.softmax(v_logits)
        return out



