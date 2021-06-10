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


class External_attention(nn.Module):

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.k = 32
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)

    def forward(self, x):
        idn = x
        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n
        x = x + idn
        x = F.relu(x)
        return x


# class EANet(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         n_classes = args.n_classes
#         self.backbone = resnet18(args)
#
#         self.linu = External_attention(3)
#         self.fc = nn.Linear(512, n_classes)
#
#     def forward(self, img, lbl=None, size=None):
#         x = self.backbone(img)
#         x = self.linu(x)
#
#         (_, _, C, H, W) = x.size()
#         x = x.permute(0, 2, 1, 3, 4)
#         x = F.adaptive_avg_pool3d(x, 1)
#         x = x.squeeze(2).squeeze(2).squeeze(2)
#         x = self.fc(x)
#
#         return x


class AV_Model(nn.Module):
    def __init__(self, visual_net,a_l='level4',v_l='level4',data='MUSIC'):
        super(AV_Model, self).__init__()

        # backbone net
        self.visual_net = visual_net
        self.linu = External_attention(6)
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
        v_fea = self.linu(v_fea)
        v_fea = torch.mean(v_fea,dim=1)
        #v_fea=self.dropout(v_fea)
        v_logits = self.class_layer(v_fea)
        #v_logits = self.dropout(v_logits)
        out = self.softmax(v_logits)
        return out



