import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOHeadVT(nn.Module):
    def __init__(self, dtype, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)

        in_dim_vis = in_dim

        if nlayers == 1:
            self.mlp_vis = nn.Linear(in_dim_vis, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim_vis, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp_vis = nn.Sequential(*layers)

        if nlayers == 1:
            self.mlp_text = nn.Linear(in_dim_vis, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim_vis, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp_text = nn.Sequential(*layers)

        if nlayers == 1:
            self.mlp_vt = nn.Linear(in_dim_vis, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim_vis, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp_vt = nn.Sequential(*layers)

        if nlayers == 1:
            self.mlp_vt_f = nn.Linear(in_dim, 512)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, 512))
            self.mlp_vt_f = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(512, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.dtype = dtype
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x , V_T):
        if V_T == "V":
            x_proj = self.mlp_vis(x.type(torch.float32))
            #x_proj = x.type(torch.float32)
        if V_T == "T":
            x_proj = self.mlp_text(x.type(torch.float32))
            #x_proj = x.type(torch.float32)
        if V_T == "V_T":
            #x = self.mlp_vt_f(x.type(torch.float32))
            x_proj = self.mlp_vt(x.type(torch.float32))
            #x = self.mlp_vt_f(x.type(torch.float32))
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x.type(torch.float32))
        return x_proj, logits

class DINOHead(nn.Module):
    def __init__(self, dtype, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        nlayers = 3
        if nlayers == 1:
            self.mlp_vt_f = nn.Linear(1024, 1024)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, 1024))
            self.mlp_vt_f = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.dtype = dtype
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = self.mlp_vt_f(x)
        x_proj = self.mlp(x.type(torch.float32))
        #x_proj = F.normalize(x_proj, dim=-1)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x.type(torch.float32))
        return x_proj, logits

class DINOHeadCLIP(nn.Module):
    def __init__(self, dtype, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.dtype = dtype
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True) #CLIP norm
        x_proj = self.mlp(x.type(torch.float32))
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x.type(torch.float32))
        return x_proj, logits



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # TODO inital to make sure it converge
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionCA(nn.Module):
    def __init__(self, dimEmb, dimHeads, numHeads, dimFF, dropout):
        super(MultiHeadAttentionCA, self).__init__()
        self.numHeads = numHeads
        self.dimEmb = dimEmb
        self.dimHeads = dimHeads
        self.scalarQK = (dimHeads * numHeads) ** -0.5

        self.W_Q = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads , bias=False)
        self.W_K = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)
        self.W_V = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)

        self.FC = nn.Linear(self.numHeads * self.dimHeads,
                            dimFF)
        self.DP = nn.Dropout(dropout)

        self._init_weights()

    def forward(self, q, k, v):

        batchSize = q.size(0)
        lenQ = q.size(1)
        lenK = k.size(1)
        lenV = v.size(1)

        Q = self.W_Q(q).reshape(batchSize, lenQ, self.numHeads , self.dimHeads).transpose(1, 2)
        K = self.W_K(k).reshape(batchSize, lenK, self.numHeads , self.dimHeads).transpose(1, 2)
        V = self.W_V(v).reshape(batchSize, lenV, self.numHeads, self.dimHeads).transpose(1, 2)
        scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K) * self.scalarQK

        # attn = nn.functional.softmax(scores, dim=-1)
        attn = nn.functional.sigmoid(scores)
        attn = F.normalize(attn, p=1, dim=-1)

        # [batchSize, numheads, lenQ, lenK] -> [batchSize, numheads, lenQ, dimHeads]
        context = torch.einsum('b h i j, b h j d -> b h i d', attn, V)
        # -> [batchSize, lenQ, numHeads, dim_kqv] -> [batchSize, lenQ, numHeads * dim_kqv]
        # context = rearrange(context, 'b h n d -> b n (h d)')
        context = context.permute(0, 2, 1, 3).contiguous().\
            view(batchSize, lenQ, self.numHeads * self.dimHeads)
        output = self.FC(context)
        output = self.DP(output)
        return output

    def _init_weights(self):
        nn.init.normal_(self.W_Q.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_K.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_V.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        # TODO initial FC weights to converge
        nn.init.xavier_normal_(self.FC.weight)

class MultiHeadAttentionCA_ML(nn.Module):
    def __init__(self, dimEmb, dimHeads, numHeads, dimFF, dropout):
        super(MultiHeadAttentionCA_ML, self).__init__()
        self.numHeads = numHeads
        self.dimEmb = dimEmb
        self.dimHeads = dimHeads
        self.scalarQK = (dimHeads * numHeads * 4) ** -0.5

        self.W_Q = nn.Linear(dimEmb * 2, self.dimHeads *
                             self.numHeads * 4 , bias=False)
        self.W_K = nn.Linear(dimEmb * 2, self.dimHeads *
                             self.numHeads * 4, bias=False)
        self.W_V = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)

        self.FC = nn.Linear(self.numHeads * self.dimHeads,
                            dimFF)
        self.DP = nn.Dropout(dropout)

        self._init_weights()

    def forward(self, q, k, v):

        batchSize = q.size(0)
        lenQ = q.size(1)
        lenK = k.size(1)
        lenV = v.size(1)

        Q = self.W_Q(q).reshape(batchSize, lenQ, self.numHeads , self.dimHeads * 4 ).transpose(1, 2)
        K = self.W_K(k).reshape(batchSize, lenK, self.numHeads , self.dimHeads * 4 ).transpose(1, 2)
        V = self.W_V(v).reshape(batchSize, lenV, self.numHeads, self.dimHeads).transpose(1, 2)
        scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K) * self.scalarQK

        # attn = nn.functional.softmax(scores, dim=-1)
        attn = nn.functional.sigmoid(scores)
        attn = F.normalize(attn, p=1, dim=-1)

        # [batchSize, numheads, lenQ, lenK] -> [batchSize, numheads, lenQ, dimHeads]
        context = torch.einsum('b h i j, b h j d -> b h i d', attn, V)
        # -> [batchSize, lenQ, numHeads, dim_kqv] -> [batchSize, lenQ, numHeads * dim_kqv]
        # context = rearrange(context, 'b h n d -> b n (h d)')
        context = context.permute(0, 2, 1, 3).contiguous().\
            view(batchSize, lenQ, self.numHeads * self.dimHeads)
        output = self.FC(context)
        output = self.DP(output)
        return output

    def _init_weights(self):
        nn.init.normal_(self.W_Q.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_K.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_V.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        # TODO initial FC weights to converge
        nn.init.xavier_normal_(self.FC.weight)


class MultiHeadAttentionSA(nn.Module):
    def __init__(self, dimEmb, dimHeads, numHeads, dimFF, dropout):
        super(MultiHeadAttentionSA, self).__init__()
        self.numHeads = numHeads
        self.dimEmb = dimEmb
        self.dimHeads = dimHeads
        self.scalar = (dimHeads * numHeads) ** -0.5

        self.W_Q = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)
        self.W_K = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)
        self.W_V = nn.Linear(dimEmb, self.dimHeads *
                             self.numHeads, bias=False)

        self.FC = nn.Linear(self.numHeads * self.dimHeads,
                            dimFF)
        self.DP = nn.Dropout(dropout)

        self._init_weights()

    def forward(self, q, k, v):
        """computting similarity by softmax or sigmoid?
        Args
        ----
            lenQ = len_k = len_v = len_sequence
            input:
                -> input_Q: [batchSize, lenQ, dimEmbedding]
                -> input_K: [batchSize, len_k, dimEmbedding]
                -> input_V: [batchSize, len_v, dimEmbedding]

        Returns
        -------
            output: [batchSize, len_sequence, dimEmb]
            attention: [batchSize, numHeads, lenQ, len_k]
        """
        batchSize = q.size(0)
        lenQ = q.size(1)
        lenK = k.size(1)
        lenV = v.size(1)

        Q = self.W_Q(q).reshape(batchSize, lenQ, self.numHeads,self.dimHeads).transpose(1, 2)
        K = self.W_K(k).reshape(batchSize, lenK, self.numHeads,self.dimHeads).transpose(1, 2)
        V = self.W_V(v).reshape(batchSize, lenV, self.numHeads,self.dimHeads).transpose(1, 2)
        scores = torch.einsum('b h i d, b h j d -> b h i j', Q, K) * self.scalar

        # attn = nn.functional.softmax(scores, dim=-1)
        attn = nn.functional.sigmoid(scores)
        attn = F.normalize(attn, p=1, dim=-1)

        # [batchSize, numheads, lenQ, lenK] -> [batchSize, numheads, lenQ, dimHeads]
        context = torch.einsum('b h i j, b h j d -> b h i d', attn, V)
        # -> [batchSize, lenQ, numHeads, dim_kqv] -> [batchSize, lenQ, numHeads * dim_kqv]
        # context = rearrange(context, 'b h n d -> b n (h d)')
        context = context.permute(0, 2, 1, 3).contiguous().\
            view(batchSize, lenQ, self.numHeads * self.dimHeads)
        output = self.FC(context)
        output = self.DP(output)
        return output

    def _init_weights(self):
        nn.init.normal_(self.W_Q.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_K.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        nn.init.normal_(self.W_V.weight, mean=0,
                        std=np.sqrt(2.0 / (self.dimEmb + self.dimHeads)))
        # TODO initial FC weights to converge
        nn.init.xavier_normal_(self.FC.weight)

class AL_Layer(nn.Module):
    def __init__(self, dimEmb, dimHeads, numHeads, dimFF, dropout):
        super().__init__()
        #self.norm1 = nn.LayerNorm(dimEmb)
        self.norm21 = nn.LayerNorm(dimEmb)
        self.norm22 = nn.LayerNorm(dimEmb)
        self.norm23 = nn.LayerNorm(dimEmb)
        self.norm3 = nn.LayerNorm(dimEmb)

        #self.SA = MultiHeadAttentionSA(dimEmb, dimHeads=dimHeads,
        #                             numHeads=numHeads, dimFF=dimFF,
        #                             dropout=dropout)
        self.CA = MultiHeadAttentionCA(dimEmb, dimHeads=dimHeads,
                                     numHeads=numHeads, dimFF=dimFF,
                                     dropout=dropout)
        self.ff = FeedForward(dimEmb, dimFF, dropout=dropout)

    def forward(self,  queryCA, embeddingsK, embeddingsV, pos=None, queryPos=None):
        'self attention'
        #queryNorm = querySA
        #queryAtten = self.SA(( queryNorm + DecoderEmb0), ( queryNorm + DecoderEmb0), DecoderEmb0) + self.norm1(DecoderEmb0)
        'cross attention'
        #querycat = torch.cat((queryAtten, queryCA), dim=2)
        queryAttenNorm = self.norm21(queryCA)
        embNormK = self.norm22(embeddingsK)
        embNormV = self.norm23(embeddingsV)

        output = self.CA(queryAttenNorm, embNormK, embNormV)
        'forward'
        output = self.norm3(output)
        output = self.ff(output) + output
        return output

    def addPosEmb(self, tensor, posEmb):
        return tensor if posEmb is None else tensor + posEmb

class AL_ML_Layer(nn.Module):
    def __init__(self, dimEmb, dimHeads, numHeads, dimFF, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimEmb)
        self.norm21 = nn.LayerNorm(dimEmb * 2)
        self.norm22 = nn.LayerNorm(dimEmb * 2)
        self.norm23 = nn.LayerNorm(dimEmb)
        self.norm3 = nn.LayerNorm(dimEmb)

        self.SA = MultiHeadAttentionSA(dimEmb, dimHeads=dimHeads,
                                     numHeads=numHeads, dimFF=dimFF,
                                     dropout=dropout)
        self.CA = MultiHeadAttentionCA_ML(dimEmb, dimHeads=dimHeads,
                                     numHeads=numHeads, dimFF=dimFF,
                                     dropout=dropout)
        self.ff = FeedForward(dimEmb, dimFF, dropout=dropout)

    def forward(self, image_new, image_old, text_old, DecoderEmb0, pos=None, queryPos=None):
        'self attention'
        input_norm = self.norm1(DecoderEmb0)
        SA_out = self.SA(DecoderEmb0, DecoderEmb0, DecoderEmb0) + input_norm
        'cross attention'
        Q = torch.cat((image_new, SA_out), dim=2)
        K = torch.cat((image_old, text_old), dim=2)
        V = text_old
        embNormQ = self.norm21(Q)
        embNormK = self.norm22(K)
        embNormV = self.norm23(V)

        output = self.CA(embNormQ, embNormK, embNormV)
        'forward'
        output = self.norm3(output)
        output = self.ff(output) + output
        return output

    def addPosEmb(self, tensor, posEmb):
        return tensor if posEmb is None else tensor + posEmb




class ALNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder0 = AL_Layer(512, 128, 4, 512, 0.1)


    def forward(self, proto_list_new_f, proto_list_base_f, text_feature_base_f ,args):

        Q = proto_list_new_f.unsqueeze(0).type(torch.float32)
        K = proto_list_base_f.unsqueeze(0).type(torch.float32)
        V = text_feature_base_f.unsqueeze(0).type(torch.float32)
        text_feature_new = torch.squeeze(self.decoder0(Q, K, V), dim = 0)
        return text_feature_new



class ALNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder0 = AL_Layer(768, 192, 4, 768, 0.1)
        self.decoder1 = AL_ML_Layer(768, 192, 4, 768, 0.1)
        self.decoder2 = AL_ML_Layer(768, 192, 4, 768, 0.1)
        self.decoder3 = AL_ML_Layer(768, 192, 4, 768, 0.1)
        self.decoder4 = AL_ML_Layer(768, 192, 4, 768, 0.1)
        self.decoder5 = AL_ML_Layer(768, 192, 4, 768, 0.1)


    def forward(self, proto_list_new_f, proto_list_base_f, text_feature_base_f ,args):
        image_new = proto_list_new_f.unsqueeze(0).type(torch.float32)
        image_old = proto_list_base_f.unsqueeze(0).type(torch.float32)
        text_old = text_feature_base_f.unsqueeze(0).type(torch.float32)

        Q = proto_list_new_f.unsqueeze(0).type(torch.float32)
        K = proto_list_base_f.unsqueeze(0).type(torch.float32)
        V = text_feature_base_f.unsqueeze(0).type(torch.float32)
        text_feature_new = self.decoder0(Q, K, V)
        if args.layers_AL == 1 : return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder1(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 2: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder2(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 3: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder3(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 4: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder4(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 5: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder5(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 6: return torch.squeeze(text_feature_new, dim=0)

class ALNetCLIP(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder0 = AL_Layer(512, 128, 4, 512, 0.1)
        self.decoder1 = AL_ML_Layer(512, 128, 4, 512, 0.1)
        self.decoder2 = AL_ML_Layer(512, 128, 4, 512, 0.1)
        self.decoder3 = AL_ML_Layer(512, 128, 4, 512, 0.1)
        self.decoder4 = AL_ML_Layer(512, 128, 4, 512, 0.1)
        self.decoder5 = AL_ML_Layer(512, 128, 4, 512, 0.1)

    def forward(self, proto_list_new_f, proto_list_base_f, text_feature_base_f ,args):
        image_new = proto_list_new_f.unsqueeze(0).type(torch.float32)
        image_old = proto_list_base_f.unsqueeze(0).type(torch.float32)
        text_old = text_feature_base_f.unsqueeze(0).type(torch.float32)

        Q = proto_list_new_f.unsqueeze(0).type(torch.float32)
        K = proto_list_base_f.unsqueeze(0).type(torch.float32)
        V = text_feature_base_f.unsqueeze(0).type(torch.float32)
        text_feature_new = self.decoder0(Q, K, V)
        if args.layers_AL == 1 : return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder1(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 2: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder2(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 3: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder3(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 4: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder4(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 5: return torch.squeeze(text_feature_new, dim=0)

        DecoderEmb = text_feature_new
        text_feature_new = self.decoder5(image_new, image_old, text_old, DecoderEmb)
        if args.layers_AL == 6: return torch.squeeze(text_feature_new, dim=0)

