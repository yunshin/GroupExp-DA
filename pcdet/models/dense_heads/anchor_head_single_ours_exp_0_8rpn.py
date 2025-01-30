import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template_ours import AnchorHeadTemplate_Ours
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import pdb
import time


class CrossAttentionLayer(nn.Module):
    def __init__(self, feature_dim, spatial_dim):
        super(CrossAttentionLayer, self).__init__()
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.norm_layer = nn.LayerNorm([feature_dim, spatial_dim, spatial_dim])
        self.conv_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.drop = nn.Dropout(0.1)  # Dropout for regularization
       

    def forward(self, mu, spatial_feature):

        batch_size, feature_dim, height, width = spatial_feature.size()

        
        
        # Step 1: Query Projection
        query = self.query_proj(mu)  # [1, 1, feature_dim]

        # Step 2: Flatten Spatial Feature
        spatial_feature_flat = spatial_feature.view(batch_size, feature_dim, height * width)

        # Step 3: Key and Value Projections
        key = self.key_proj(spatial_feature_flat.permute(0, 2, 1))  # [batch, height*width, feature_dim]
        value = self.value_proj(spatial_feature_flat.permute(0, 2, 1))

        # Step 4: Compute Attention Scores
        query = query.expand(batch_size, -1, -1)  # [batch, 1, feature_dim]
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch, 1, height*width]
        d_k = key.size(-1)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, 1, height*width]
        attention_weights = self.drop(attention_weights)

        # Step 5: Weighted Sum of Value Vectors
        context = torch.bmm(attention_weights, value)  # [batch, 1, feature_dim]

        # Step 6: Broadcast Context
        context_broadcast = context.view(context.size(0), context.size(2), 1, 1)  # [batch, feature_dim, 1, 1]
        context_broadcast = context_broadcast.expand(-1, -1, height, width)

        # Step 7: Combine with Spatial Feature
        combined_feature = spatial_feature + context_broadcast  # [batch, feature_dim, height, width]
        normalized_feature = self.norm_layer(combined_feature)

        # Step 8: Apply Convolution and Activation
        output_feature = self.conv_proj(normalized_feature)
        output_feature = F.relu(output_feature)
        
        return output_feature 
    
class MultiLayerCrossAttention(nn.Module):
    def __init__(self, num_layers, feature_dim, spatial_dim):
        super(MultiLayerCrossAttention, self).__init__()
        self.layers = nn.ModuleList([CrossAttentionLayer(feature_dim, spatial_dim) for _ in range(num_layers)])
        self.query_update_proj = nn.Linear(feature_dim, feature_dim)  # Optional query update mechanism

    def forward(self, mu, spatial_feature):
        for layer in self.layers:
            # Apply the cross-attention layer
            
            spatial_feature = layer(mu, spatial_feature)
            # Update the query using a projection (optional)
            mu = F.relu(self.query_update_proj(mu))
        return spatial_feature
    
class AnchorHeadSingle_Ours_EXP_0_8RPN(AnchorHeadTemplate_Ours):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        
        self.models = nn.ModuleDict()
        self.load_pseudo = False
        self.is_lt = model_cfg.get('IS_LT', False)
        self.coff = 0.2
        
        
        '''
        self.conv_proj = nn.Conv2d(
            input_channels, 512,
            kernel_size=1
        )
        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        
        self.models['cls0'] = self.conv_cls
        self.models['box0'] = self.conv_box

        self.models['dir0'] = self.conv_dir_cls
        '''
        self.models['cls0'] = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
        )
        self.models['box0'] = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        self.models['dir0'] = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
            kernel_size=1
        )
        self.num_group = model_cfg['NUM_GROUP']
        
        #self.mu = torch.nn.parameter.Parameter(torch.randn(self.num_group,1,512))
        #self.mu = nn.Parameter(torch.zeros(self.num_group, 512))
        #self.mu = nn.Parameter(torch.zeros(self.num_group, 512))
        #self.cov = nn.Parameter(torch.zeros(self.num_group, 512))
        #self.weight = nn.Parameter(torch.zeros(self.num_group))
        self.register_buffer('mu', torch.zeros(self.num_group, 512))
        self.register_buffer('cov', torch.zeros(self.num_group, 512))
        self.register_buffer('weight', torch.zeros(self.num_group))
        self.group_init = False
        if model_cfg.get('ONLY_OPTIM_GROUP', None):
            self.only_optim_group = model_cfg['ONLY_OPTIM_GROUP']
        else:
            self.only_optim_group = False

        if model_cfg.get('ONLY_OPTIM_ORI', None):
            self.only_optim_ori = model_cfg['ONLY_OPTIM_ORI']
        else:
            self.only_optim_ori = False

        if model_cfg.get('USE_GROUP_UNTIL_EPOCH', None):
            self.group_epoch = model_cfg['USE_GROUP_UNTIL_EPOCH']
        else:
            self.group_epoch = 10000

        if self.training == False:
            self.only_optim_group = False

        if model_cfg.get('FIX_GROUP', None):

            self.group_fix = True
        else:
            self.group_fix = False

        if model_cfg.get('MULTI_RPN', None):
            self.multi_rpn = True
        else:
            self.multi_rpn = False

       
       

        self.group_models = nn.ModuleDict()
        
        
        input_dim = 3
        output_dim = 512
        self.group_models['attn'] =  MultiLayerCrossAttention(2, 512, 188)#CrossAttentionLayer(512, 188)
        self.group_models['obj_mlp1'] = nn.Sequential(
                            nn.Conv1d(input_dim, 64, 1),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            nn.Conv1d(64, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Conv1d(128, 256, 1),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            nn.Conv1d(256, 512, 1),
                            nn.BatchNorm1d(512),
                            nn.ReLU()
                        )
        
        self.group_models['obj_mlp2'] = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(512, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(512, output_dim)# Output layer
                        )
        '''
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Fully connected layers for final output
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)# Output layer
        )
        '''
       
        self.init_weights()
        
        self.chk_mu = self.mu.detach().cpu().numpy()

    def init_weights(self):
        pi = 0.01

        for key in self.models:
            if 'cls' in key:
                #nn.init.constant_(self.models[key].bias, -np.log((1 - pi) / pi))
                nn.init.xavier_uniform_(self.models[key].weight)
                #nn.init.kaiming_uniform_(self.models[key].weight, nonlinearity='relu')
                nn.init.zeros_(self.models[key].bias)  # Initialize biases to zero
            elif 'box' in key:
                #nn.init.normal_(self.models[key].weight, mean=0, std=0.001)
                nn.init.normal_(self.models[key].weight, mean=0.0, std=0.001)
                nn.init.zeros_(self.models[key].bias)
        
        if self.multi_rpn:
            for key in self.group_models:
                if 'cls' in key:
                    #nn.init.constant_(self.models[key].bias, -np.log((1 - pi) / pi))
                    nn.init.xavier_uniform_(self.group_models[key].weight)
                    #nn.init.kaiming_uniform_(self.models[key].weight, nonlinearity='relu')
                    nn.init.constant_(self.group_models[key].bias, -np.log((1 - pi) / pi))  # Initialize biases to zero
                elif 'box' in key:
                    #nn.init.normal_(self.models[key].weight, mean=0, std=0.001)
                    nn.init.normal_(self.group_models[key].weight, mean=0.0, std=0.001)
                    nn.init.zeros_(self.group_models[key].bias)
        self.group_models['obj_mlp1'].apply(self.initialize_weights)
        self.group_models['obj_mlp2'].apply(self.initialize_weights)
        
        '''
        self.query_proj.apply(self.initialize_weights)
        self.key_proj.apply(self.initialize_weights)
        self.value_proj.apply(self.initialize_weights)
        self.norm_layer.apply(self.initialize_weights)
        '''
        #nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        #nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):  # Apply to Linear layers
            nn.init.xavier_uniform_(module.weight)  # Xavier initialization
            if module.bias is not None:  # Initialize bias if present
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):  # Example for Conv2d
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias) 
    def orthogonality_regularization(self, embeddings):
        """
        Regularization to ensure embeddings are orthogonal.
        
        Args:
            embeddings (torch.Tensor): Tensor of shape (N, D).
        
        Returns:
            torch.Tensor: Scalar regularization loss value.
        """
        similarity_matrix = torch.mm(embeddings, embeddings.T)  # Shape: (N, N)
        mask = torch.eye(similarity_matrix.size(0), device=embeddings.device).bool()
        off_diagonal = similarity_matrix.masked_fill(mask, 0)  # Exclude diagonal
        loss = off_diagonal.pow(2).mean()  # Penalize non-zero off-diagonal values
        return loss
    def pairwise_dissimilarity_loss(self, embeddings, margin=1.0):
        """
        Compute pairwise dissimilarity loss for a batch of embeddings.
        
        Args:
            embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of embeddings, D is the embedding dimension.
            margin (float): The margin to enforce dissimilarity.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        '''
        N = embeddings.size(0)  # Number of embeddings
        loss = 0.0
        
        similarity_matrix = torch.mm(embeddings, embeddings.T)  # Shape: (N, N)

        #embeddings = F.normalize(embeddings, p=2, dim=1)
        sims = []
        for i in range(N):
            for j in range(i + 1, N):  # Only compute for unique pairs (i, j)
                sim = F.cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])  # Cosine similarity
                sims.append(sim.item())
                loss += torch.clamp(margin - sim, min=0)  # Enforce dissimilarity
        '''
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings, embeddings.T)  # Cosine similarities
        N = embeddings.size(0)
        mask = torch.eye(N, device=embeddings.device).bool()  # Mask diagonal (self-similarity)
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)  # Remove self-similarity
        reg_loss = self.orthogonality_regularization(embeddings)
        loss = similarity_matrix.mean()


        
        if loss.item() < 0:
            use_loss = False
        else:
            use_loss = True
        #print('push away mat sim : {0} mean: {1} use_loss: {2}'.format(similarity_matrix, loss, use_loss))
        return loss + reg_loss *0.1, use_loss  # Average over all pairs
    def pairwise_similarity_loss(self, embeddings):
        """
        Compute pairwise similarity loss for a batch of embeddings.
        
        Args:
            embeddings (torch.Tensor): Tensor of shape (N, D), where N is the number of embeddings and D is the embedding dimension.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        N = embeddings.size(0)
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.T)  # Shape: (N, N)

        # Exclude self-similarity (diagonal elements)
        mask = torch.eye(similarity_matrix.size(0), device=embeddings.device).bool()
        off_diagonal = similarity_matrix.masked_fill(mask, 0)  # Exclude diagonal
        loss = off_diagonal.pow(2).mean()  # Penalize non-zero off-diagonal values
        # Loss is the negative of the mean similarity
        l1_reg = embeddings.abs().mean()

        avg_sim = similarity_matrix.mean()
        
        if avg_sim.item() > 0.4: 
            use_loss = False
        else:
            use_loss = True
        #print('closer mat sim : {0} use_loss: {1}'.format(avg_sim, use_loss))
        sim_loss = -off_diagonal.mean()
        return sim_loss + loss + l1_reg*0.01, use_loss
    def get_parwise_loss(self, descriptors, group_label):
        pos_loss = torch.zeros(1).float().cuda()[0]
        for group_idx in range(self.num_group):
            group_mask = group_label == group_idx 
            
            group_descriptors = descriptors[group_mask]
            
            pos_loss_, use_loss = self.pairwise_similarity_loss(group_descriptors)
            if use_loss:
                pos_loss += pos_loss_
        return pos_loss /self.num_group
    def get_distinctive_samples(self, data, K):
        # Step 1: Compute pairwise Euclidean distance matrix
        # Expand dimensions to compute pairwise distances
        #data_expanded = data.unsqueeze(1)  # Shape: (N, 1, D)
        distances = torch.cdist(data, data)  # Shape: (N, N)

        # Step 2: Calculate distinctiveness scores for each sample
        # Sum distances for each sample to get distinctiveness scores
        distinctiveness_scores = distances.sum(dim=1)  # Shape: (N,)

        # Step 3: Select the top K most distinctive samples
        # Sort by distinctiveness_scores in descending order
        
        _, top_samples_indices = torch.topk(distinctiveness_scores, K)
        top_samples = [i.item() for i in top_samples_indices]
        
        return top_samples_indices


    def masked_max_pooling(self, x, mask, loop=False):

        if loop == False:
            x = x.masked_fill(~mask.unsqueeze(1), float('-inf'))  # Mask out invalid points
        else:
            with torch.no_grad():
                x_list = []
                for idx in range(len(mask)):

                    x_ = x[idx].unsqueeze(0)
                    mask_ = mask[idx].unsqueeze(0)
                    x_ = x_.masked_fill(~mask_.unsqueeze(1), float('-inf'))  # Mask out invalid points
                
                    x_list.append(x_.cpu())
                x = torch.cat(x_list)
            
        return torch.max(x, dim=2).values

    def get_obj_descriptor(self, obj_points, obj_points_masks, gt_boxes, assigned_groups=None):

        batch_size = obj_points.shape[0]
        init = False
        
        obj_points_list = []
        mask_list = []
        batch_num_list = []
        valid_obj_mask = []

        for batch_idx in range(batch_size):
            # x shape: [B, N, input_dim]
            
            x = obj_points[batch_idx]
            mask = obj_points_masks[batch_idx]
            mask = mask == 1

            obj_num_mask = mask.sum(1) > 0
            gt_boxes_ = gt_boxes[batch_idx]
            gt_boxes_[obj_num_mask == False] = 0
            gt_boxes[batch_idx] = gt_boxes_
            


            x = x[obj_num_mask]
            
            if len(x) == 0:
                valid_obj_mask.append(False)
                continue
            else:
                valid_obj_mask.append(True)
            mask = mask[obj_num_mask]
            
            obj_points_list.append(x)
            mask_list.append(mask)

            
            batch_nums = torch.zeros(x.shape[0]).long().cuda()
            batch_nums[:] = batch_idx
            batch_num_list.append(batch_nums)

        try:
            obj_points_list = torch.cat(obj_points_list)
        except:
            pdb.set_trace()
        mask_list = torch.cat(mask_list)
        batch_num_list = torch.cat(batch_num_list)
        
        

        x = obj_points_list.permute(0, 2, 1)  # Change to [B, D, N] for Conv1d compatibility
        
        
        
        if x.shape[0] == 1:
            is_single = True
            x = x.repeat(2,1,1)
            mask_list = mask_list.repeat(2,1)
        else:
            is_single = False
        
        
        x = self.group_models['obj_mlp1'](x)  # Output shape: [B, 1024, N]
        

            
        x = self.masked_max_pooling(x, mask_list)  # Result shape: [B, 1024]
        descriptor = self.group_models['obj_mlp2'](x)  # Result shape: [B, output_dim]
        

        
        
        if is_single:
            mask_list = mask_list[0:1]
            x = x[0:1]
            descriptor = descriptor[0:1]
            
       
        valid_obj_mask = torch.tensor(np.array(valid_obj_mask)).cuda()
        
        
        return descriptor, batch_num_list, gt_boxes, valid_obj_mask


    def make_form_from_indices(self, obj_points, obj_points_masks, gt_boxes, indices, batch_num_list, obj_num_list):


        batch_size = gt_boxes.shape[0]

        max_box_num = 0
        for batch_idx in range(batch_size):

            #batch_mask = batch_num_list == batch_idx
            #num_box_batch = batch_mask.sum().item()
            
            batch_nums_selected = batch_num_list[indices]
            obj_nums_selected = obj_num_list[indices]
           
            
            batch_mask = batch_nums_selected == batch_idx
            obj_nums_selected = obj_nums_selected[batch_mask]

            num_box_batch = len(obj_nums_selected)
            if num_box_batch > max_box_num:
                max_box_num = num_box_batch

        new_boxes = torch.zeros(batch_size, max_box_num, 8)
        new_obj_points = torch.zeros(batch_size, max_box_num, obj_points.shape[2],3)
        new_obj_points_masks = torch.zeros(batch_size, max_box_num, obj_points.shape[2])

        for batch_idx in range(batch_size):

            batch_nums_selected = batch_num_list[indices]
            obj_nums_selected = obj_num_list[indices]
           
            
            batch_mask = batch_nums_selected == batch_idx
            obj_nums_selected = obj_nums_selected[batch_mask]
            

            batch_boxes = gt_boxes[batch_idx][obj_nums_selected]
            batch_obj_points = obj_points[batch_idx][obj_nums_selected]
            batch_obj_masks = obj_points_masks[batch_idx][obj_nums_selected]

            num_box_batch = len(batch_boxes)
            new_boxes[batch_idx,:num_box_batch] = batch_boxes
            new_obj_points[batch_idx,:num_box_batch] = batch_obj_points
            new_obj_points_masks[batch_idx,:num_box_batch] = batch_obj_masks
        

        return new_boxes.cuda(), new_obj_points.cuda(), new_obj_points_masks.cuda()

    def get_obj_descriptor_subsample(self, obj_points, obj_points_masks, gt_boxes, max_obj_num):

        batch_size = obj_points.shape[0]
        init = False
        
        obj_points_list = []
        mask_list = []
        batch_num_list = []
        obj_num_list = []
        valid_obj_mask = []

        for batch_idx in range(batch_size):
            # x shape: [B, N, input_dim]
            
            x = obj_points[batch_idx]
            mask = obj_points_masks[batch_idx]
            mask = mask == 1

            obj_num_mask = mask.sum(1) > 0
            gt_boxes_ = gt_boxes[batch_idx]
            gt_boxes_[obj_num_mask == False] = 0
            gt_boxes[batch_idx] = gt_boxes_
           
            box_nums = torch.where(obj_num_mask)
            

            x = x[obj_num_mask]
            
            if len(x) == 0:
                valid_obj_mask.append(False)
                continue
            else:
                valid_obj_mask.append(True)
            mask = mask[obj_num_mask]
            
            obj_points_list.append(x)
            mask_list.append(mask)

            
            batch_nums = torch.zeros(x.shape[0]).long().cuda()
            batch_nums[:] = batch_idx
            batch_num_list.append(batch_nums)
            obj_num_list.append(box_nums[0])

        try:
            obj_points_list = torch.cat(obj_points_list)
        except:
            pdb.set_trace()
        mask_list = torch.cat(mask_list)
        batch_num_list = torch.cat(batch_num_list)
        obj_num_list = torch.cat(obj_num_list)
        

        x = obj_points_list.permute(0, 2, 1)  # Change to [B, D, N] for Conv1d compatibility
        
        
        
        if x.shape[0] == 1:
            is_single = True
            x = x.repeat(2,1,1)
            mask_list = mask_list.repeat(2,1)
        else:
            is_single = False
        
        
        x = self.shared_mlp(x)  # Output shape: [B, 1024, N]
        
        
        # Number of distinctive features we want to select
        
        # Masked Global Pooling (choose max or average pooling)
        
        
        # Fully connected layers for classification
        
         
        x_ = self.masked_max_pooling(x, mask_list, loop=True)  # Result shape: [B, 1024]
        
        
        top_features_indicees = self.get_distinctive_samples(x_, max_obj_num)
        
        
        return top_features_indicees, batch_num_list, obj_num_list

    def apply_attn(self, mu, spatial_feature):
        
        query = self.query_proj(mu)  # [1, 1, 512]

        # Reshape spatial_feature to [batch_size, height * width, feature_dim]
        batch_size, feature_dim, height, width = spatial_feature.size()
        spatial_feature_flat = spatial_feature.view(batch_size, feature_dim, height * width)  # [6, 512, 188*188]

        # Project spatial_feature to key and value
        key = self.key_proj(spatial_feature_flat.permute(0, 2, 1))  # [6, 188*188, 512]
        value = self.value_proj(spatial_feature_flat.permute(0, 2, 1))  # [6, 188*188, 512]

        # Step 2: Compute attention scores (query-key dot product)
        # Broadcast query to match the batch size
        query = query.expand(batch_size, -1, -1)  # [6, 1, 512]
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [6, 1, 188*188]

        # Scale the attention scores by sqrt(d_k)
        d_k = key.size(-1)  # 512
        attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [6, 1, 188*188]
        attention_weights = self.drop(attention_weights)
        # Step 3: Compute the weighted sum of value vectors (context vector)
        context = torch.bmm(attention_weights, value)  # [6, 1, 512]

        # Step 4: Broadcast the context across the spatial dimensions (188x188)
        context_broadcast = context.view(context.size(0), context.size(2), 1, 1)  # [6, 512, 1, 1]
        context_broadcast = context_broadcast.expand(-1, -1, 188, 188)  # [6, 512, 188, 188]

        # Step 5: Combine the context with the spatial_feature (element-wise addition)
        combined_feature_add = spatial_feature + context_broadcast  # [6, 512, 188, 188]

        normalized_feature = self.norm_layer(combined_feature_add)
        
        output_feature = self.conv_proj(normalized_feature)
        # Step 7: Apply non-linear activation (ReLU or GELU)
        #output_feature = F.relu(output_feature)
        
        return output_feature
    
    def init_GMM(self, data_np):
        
        gmm = GaussianMixture(n_components=self.num_group, covariance_type='full', random_state=0)
        gmm.means_init = self.mu.cpu().numpy()
        gmm.fit(data_np)

        # Predict the cluster for each data point
        labels = gmm.predict(data_np)  # Array of shape (20,)

        # Optionally, retrieve the cluster probabilities for each point
        probs = gmm.predict_proba(data_np)  # Array of shape (20, n_components)
        
        #cov = gmm.covariances_
        #weight = gmm.weights_
        
        
        self.update_GMM(data_np, labels)
        return labels
    def update_GMM(self, data, labels, init=False):
        
        mu = torch.zeros(self.mu.shape).cuda()
        cov = torch.zeros(self.cov.shape).cuda()
        weights = torch.zeros(self.weight.shape).cuda()

        for idx_group in range(self.num_group):
            
            data_group = data[labels==idx_group]

            if len(data_group) == 0:
                mu[idx_group] = self.mu[idx_group]
                cov[idx_group] = self.cov[idx_group]
                weights[idx_group] = 0
                continue
            mu_ = data_group.mean(dim=0, keepdim=True)

            data_group_centered = data_group - mu_  # Centered data (15, 512)
            N = data_group.size(0)  # Number of samples, which is 15
            #covariance_matrix_ = (data_group_centered.T @ data_group_centered) / (N - 1)  # Shape: (512, 512)
            
            
            #covariance_matrix_ = torch.diag(variances)  # Shape: (512, 512)
            weight_ = len(data_group) / len(data)
 
            mu[idx_group] = mu_.squeeze()
            if len(data_group) == 1:
                cov[idx_group] = self.cov[idx_group]
            else:
                variances = torch.var(data_group_centered, dim=0, unbiased=True)  # Shape: (512,)
                cov[idx_group] = variances

            weights[idx_group] = weight_
        if torch.isnan(mu).any():
            pdb.set_trace()
        if init:
            
            self.mu += mu
            self.cov += cov
            self.weight += weights
        else:
           
            
            self.mu = (self.mu * (1-self.coff)) + (mu.detach() * self.coff) 
            self.cov = (self.cov * (1-self.coff)) + (cov.detach() * self.coff)
            self.weight = (self.weight * (1-self.coff)) + (weights.detach() * self.coff)
            '''
            self.mu = (self.mu * 0.8) + (mu.detach() * 0.2) 
            self.cov = (self.cov * 0.8) + (cov.detach() * 0.2)
            self.weight = (self.weight * 0.8) + (weights.detach() * 0.2)
            '''

    def determine_group(self, obj_descriptors):
    
        gmm = GaussianMixture(n_components=self.num_group, covariance_type='diag', reg_covar=1)
        

        new_means_init = self.mu.cpu().numpy()
        new_covariances_init = self.cov.cpu().numpy()
        new_weights_init = self.weight.cpu().numpy()

        # Reinitialize the GMM with updated parameters as initialization
        gmm.means_init = new_means_init
        #gmm.weights_init = new_weights_init# / (new_weights_init.sum() + 1e-6)
        
        gmm.precisions_init = 1 / (new_covariances_init + 1e-6)
        try:
            gmm.fit(obj_descriptors)
            labels = gmm.predict(obj_descriptors)  # Array of shape (20,)
            probs = gmm.predict_proba(obj_descriptors)  # Array of shape (20, n_components)
        except Exception as e:
            print('gmm fit error. Error: {0}'.format(e))
            dist = torch.cdist(torch.tensor(obj_descriptors).float().cuda(), self.mu)
            labels = torch.argmin(dist,1)
            labels = labels.cpu().numpy()
            probs = np.zeros((obj_descriptors.shape[0], self.num_group))
            probs[:, labels] = 1
            #labels = self.gmm.predict(obj_descriptors)
            #probs = self.gmm.predict_proba(obj_descriptors)
        
       
        
        labels = self.ensure_non_empty_clusters(obj_descriptors, labels, self.num_group)
        labels = torch.tensor(labels).long().cuda()
        
        if len(labels) != len(obj_descriptors):
                pdb.set_trace()
                print('group and obj num mismatch')

        for debug_idx in range(self.num_group):

            if (labels == debug_idx).sum() == 0:
                
                print('group {0} has 0 sample '.format(debug_idx))
                
        self.gmm = gmm
        return labels, probs
    
    def ensure_non_empty_clusters(self, data, labels, n_clusters):
        # Reassign samples if a cluster is empty
        data_tensor = torch.tensor(data).float().cuda()
        for cluster in range(n_clusters):
            if np.sum(labels == cluster) == 0:  # Check if cluster is empty
                # Find a nearby point to reassign
                non_empty_clusters = [i for i in range(n_clusters) if np.sum(labels == i) > 1]
                

                dist_to_each = torch.cdist(self.mu, data_tensor)

                for non_empty_idx in range(len(non_empty_clusters)):
                    source_mask = labels == non_empty_clusters[non_empty_idx]
                    source_nums = torch.arange(len(source_mask)).long().cuda()

                    dist_cluster = dist_to_each[cluster][source_mask]
                    source_nums = source_nums[source_mask]

                    min_idx = torch.argmin(dist_cluster)
                    obj_num = source_nums[min_idx].cpu().numpy()
                    labels[obj_num] = cluster
                '''
                source_cluster = non_empty_clusters[0]
                sample_idx = np.where(labels == source_cluster)[0][0]
                labels[sample_idx] = cluster
                '''
        return labels

    def get_group(self, obj_descriptors, assigned_groups=None):
        

        if self.group_init == False:

            obj_desc_numpy = obj_descriptors.detach().cpu().numpy()
            
            K = self.num_group

            # Initialize KMeans with desired parameters
            kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=42)
            # Fit the model to the data
            kmeans.fit(obj_desc_numpy)
            # Retrieve the cluster centers and labels
            centroids = kmeans.cluster_centers_  # Shape: (K, 512)
            kmeans_labels = kmeans.labels_              # Shape: (80,)

            for idx_group in range(self.num_group):
                
                mu = obj_descriptors[kmeans_labels==idx_group]
                mu = torch.mean(mu,0)

                if idx_group == 0:

                    mus = mu[None,:]
                else:
                    mus = torch.cat([mus, mu[None,:]],0)
            
            mu_for_optim = (self.mu.clone()* (1-self.coff)) + (mus * self.coff) 
            
        
            self.update_GMM(obj_descriptors.detach(), torch.tensor(kmeans_labels).long().cuda(), init=True)
            #label = self.init_GMM(obj_desc_numpy)
            
            label = torch.tensor(kmeans_labels).long().cuda()
            self.group_init = True

            print('\n\n Group Initialized! \n\n')
        else:
            if self.group_fix == False:
                label, probs = self.determine_group(obj_descriptors.detach().cpu().numpy())
            else:
                
                label = []
                for idx_group in range(len(assigned_groups)):
                
                    assigned_group_ = assigned_groups[idx_group]
                    valid_mask = assigned_group_ > -1
                    assigned_group_ = assigned_group_[valid_mask]
                    label_ = assigned_group_.long()
                    label.append(label_)
                
                label = torch.cat(label)
                if len(label) != len(obj_descriptors):
                    pdb.set_trace()
                    print('group and obj num mismatch')

            for idx_group in range(self.num_group):
                
                mu = obj_descriptors[label==idx_group]
                if len(mu) == 0:
                    mu = self.mu[idx_group].clone().detach()
                else:
                    mu = torch.mean(mu,0)

                if idx_group == 0:
                    mus = mu[None,:]
                else:
                    mus = torch.cat([mus, mu[None,:]],0)

            mu_for_optim = (self.mu.clone()* (1-self.coff)) + (mus * self.coff) 
            #if self.st == False or self.is_target:
            self.update_GMM(obj_descriptors.detach(), label, init=False)

        if torch.isnan(mu_for_optim).any():
            pdb.set_trace()
        


        return mu_for_optim, label

    def assign_gt(self, group_label, gt_boxes, batch_nums, data_dict):
        
        
        batch_size = gt_boxes.shape[0]
        group_boxes_list = []
        for group_idx in range(self.num_group):

            num_box_group = 0

            for batch_idx in range(batch_size):
                batch_mask = batch_nums == batch_idx
                try:
                    num_box_group_ = (group_label[batch_mask] == group_idx).sum()
                    
                except:
                    #it happens when the batch_size is 1, at the end of the epoch
                    pdb.set_trace()
                if num_box_group_ > num_box_group:
                    num_box_group = num_box_group_
            
            gt_boxes_group = torch.zeros((batch_size, num_box_group, 8)).float().cuda()

            for batch_idx in range(batch_size):

                batch_mask = batch_nums == batch_idx
                group_mask = (group_label[batch_mask] ==group_idx)
                
                boxes_batch = gt_boxes[batch_idx]
                boxes_batch = boxes_batch[boxes_batch[:,3]>0]
                try:
                    boxes_to_group = boxes_batch[group_mask]
                except:
                    pdb.set_trace()
                gt_boxes_group[batch_idx,:len(boxes_to_group)] = boxes_to_group
            group_boxes_list.append(gt_boxes_group)

        for idx_group in range(len(group_boxes_list)):

            data_dict['gt_boxes{0}'.format(idx_group+1)] = group_boxes_list[idx_group]
        return data_dict
    
    def forward_(self, mus, spatial_features_2d, idx, data_dict, is_group=False):

      
        if is_group:
            
            spatial_features_2d = self.group_models['attn'](mus, spatial_features_2d)
            #spatial_features_2d = self.apply_attn(mus, spatial_features_2d)
            
            data_dict['spatial_features_2d{0}'.format(idx)] = spatial_features_2d

            if self.multi_rpn:
                pdb.set_trace()
                cls_preds = self.group_models['cls'](spatial_features_2d)
                box_preds = self.group_models['box'](spatial_features_2d)
                #cls_preds = self.models['cls0'](spatial_features_2d)
                #box_preds = self.models['box0'](spatial_features_2d)
            else:
                cls_preds = self.models['cls0'](spatial_features_2d)
                box_preds = self.models['box0'](spatial_features_2d)
        else:
            cls_preds = self.models['cls0'](spatial_features_2d)
            box_preds = self.models['box0'](spatial_features_2d)
            
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if is_group: #group

            if self.multi_rpn:
                self.forward_ret_dict['cls_preds{0}'.format(idx)] = cls_preds
                self.forward_ret_dict['box_preds{0}'.format(idx)] = box_preds
                pdb.set_trace()
                dir_cls_preds = self.group_models['dir'](spatial_features_2d)
                #dir_cls_preds = self.models['dir0'](spatial_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                self.forward_ret_dict['dir_cls_preds{0}'.format(idx)] = dir_cls_preds
            else:

                self.forward_ret_dict['cls_preds{0}'.format(idx)] = cls_preds
                self.forward_ret_dict['box_preds{0}'.format(idx)] = box_preds

                dir_cls_preds = self.models['dir0'](spatial_features_2d)
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
                self.forward_ret_dict['dir_cls_preds{0}'.format(idx)] = dir_cls_preds
        else:
            self.forward_ret_dict['cls_preds'] = cls_preds
            self.forward_ret_dict['box_preds'] = box_preds

            dir_cls_preds = self.models['dir0'](spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
       
        if self.training:
            if is_group == False:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes']
                )
                self.forward_ret_dict.update(targets_dict)
            else:
                targets_dict = self.assign_targets(
                    gt_boxes=data_dict['gt_boxes{0}'.format(idx)]
                )
                new_targets_dict = {
                    'box_cls_labels{0}'.format(idx) : targets_dict['box_cls_labels'],
                    'box_reg_targets{0}'.format(idx) : targets_dict['box_reg_targets'],
                    'reg_weights{0}'.format(idx) : targets_dict['reg_weights'],
                    }
                self.forward_ret_dict.update(new_targets_dict)
        
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            
            
            
            if is_group: #group
                data_dict['batch_cls_preds{0}'.format(idx)] = batch_cls_preds
                data_dict['batch_box_preds{0}'.format(idx)] = batch_box_preds
                
                if False and self.training:
                    gt_boxes_group = data_dict['gt_boxes{0}'.format(idx)]

                    for idx_test in range(len(gt_boxes_group)):
                        gt_boxes_group_ = gt_boxes_group[idx_test]
                        gt_boxes_group_ = gt_boxes_group_[gt_boxes_group_[:,3]>0]

                        if len(gt_boxes_group_) > 0:
                            
                            cls_scores_ = torch.clone(batch_cls_preds[idx_test])
                            ious_boxes_ = boxes_iou3d_gpu(gt_boxes_group_[:,:7], torch.clone(batch_box_preds[idx_test]))
                            ious_max, max_indices = torch.max(ious_boxes_,1)
                            corr_scores = cls_scores_[max_indices]
                           
                            print('Sample {0} Group: {1}  max ious: {2} corr scores: {3}'.format(idx_test, idx, ious_max.detach().cpu().numpy(), corr_scores.detach().cpu().numpy()))

            else:

                data_dict['batch_cls_preds'] = batch_cls_preds
                data_dict['batch_box_preds'] = batch_box_preds
                data_dict['cls_preds_normalized'] = False
        
                #self.tmp_cls = torch.clone(batch_cls_preds.detach())
                #self.tmp_box = torch.clone(batch_box_preds.detach())
                '''
                gt_boxes_group = data_dict['gt_boxes']
                self.max_ori = []
                self.scores_ori = []
                for idx_test in range(len(gt_boxes_group)):
                        gt_boxes_group_ = gt_boxes_group[idx_test]
                        gt_boxes_group_ = gt_boxes_group_[gt_boxes_group_[:,3]>0]

                        if len(gt_boxes_group_) > 0:
                            
                            cls_scores_ = torch.clone(batch_cls_preds[idx_test])
                            ious_boxes_ = boxes_iou3d_gpu(gt_boxes_group_[:,:7], torch.clone(batch_box_preds[idx_test]))
                            ious_max, max_indices = torch.max(ious_boxes_,1)
                            corr_scores = cls_scores_[max_indices]
                            
                            self.max_ori.append(ious_max)
                            self.scores_ori.append(corr_scores)
                            print('Sample {0} Group: {1}  max ious: {2} corr scores: {3}'.format(idx_test, idx, ious_max.detach().cpu().numpy(), corr_scores.detach().cpu().numpy()))
                '''        
                
                

        return data_dict
    
    def sub_sample(self, obj_points, obj_points_masks, gt_boxes, max_obj_num=30):

        with torch.no_grad():
            sample_indices, batch_num_list, obj_num_list = self.get_obj_descriptor_subsample(obj_points, obj_points_masks, gt_boxes, max_obj_num)
            #batch_num_list : batch index of all boxes
            #obj_num_list: index of boxes in each batch

            new_boxes, new_obj_points, new_obj_points_masks = self.make_form_from_indices(obj_points, obj_points_masks, gt_boxes, 
            sample_indices, batch_num_list, obj_num_list)
        
        return new_boxes, new_obj_points, new_obj_points_masks


    def forward(self, data_dict):
        
        
        #cases to consider: #1 pseudo label collection #2 source training #3 st training #4 testing
        if 'pseudo_collection' in data_dict:
            
            if data_dict['cur_epoch'] <= self.group_epoch:
                self.only_optim_ori = False
            else:
                self.only_optim_ori = True #We do not use group after certain epoch for pseudo label collection
        else:# In case it's training or testing
            if 'st' in data_dict: #self training
                if data_dict['cur_epoch'] <= self.group_epoch and data_dict['source'] == False: 
                    self.only_optim_ori = False
                else:
                    self.only_optim_ori = True
                
            else: #source training or testing
                if self.training: 
                    self.only_optim_ori = False
                else:
                    self.only_optim_ori = True 

        if self.group_init == False:
            if (np.allclose(self.mu.cpu().numpy(), self.chk_mu) == False):
                self.group_init = True

        spatial_features_2d = data_dict['spatial_features_2d']
        
        if 'target' in data_dict:
            pdb.set_trace()
            self.is_target = data_dict['target']
            self.st = True
        else:
            self.st = False
        
        if self.training:
            num_sample = data_dict['gt_boxes'].shape[1]
            #if num_sample > 35:
            if False and num_sample > 100 and self.group_fix == False:
                torch.cuda.empty_cache()
                print('subsampled.. boxes: {0} points: {1}'.format(data_dict['gt_boxes'].shape, data_dict['obj_points'].shape))
                new_boxes, new_obj_points, new_obj_points_masks = self.sub_sample(data_dict['obj_points'], data_dict['obj_points_masks'],
                data_dict['gt_boxes'], max_obj_num=20)
              
                data_dict['obj_points'] = new_obj_points
                data_dict['obj_points_masks'] = new_obj_points_masks
                data_dict['gt_boxes'] = new_boxes
                
                
        
        if self.only_optim_ori == False and self.training and len(data_dict['gt_boxes']) > 0:

            if self.group_fix:
                obj_descriptors, batch_nums, gt_boxes, valid_obj_mask, assigned_groups = self.get_obj_descriptor(data_dict['obj_points'], data_dict['obj_points_masks'], data_dict['gt_boxes'], data_dict['assigned_groups'])
                data_dict['gt_boxes'] = gt_boxes
                mus, group_label = self.get_group(obj_descriptors, assigned_groups)
            else:
                
                
                obj_descriptors, batch_nums, gt_boxes, valid_obj_mask = self.get_obj_descriptor(data_dict['obj_points'], data_dict['obj_points_masks'], data_dict['gt_boxes'])
                
                data_dict['gt_boxes'] = gt_boxes
                mus, group_label = self.get_group(obj_descriptors)
            

            if self.group_fix == False:
                ctrast_loss, use_loss = self.pairwise_dissimilarity_loss(mus)
                if use_loss:
                    self.forward_ret_dict['ctrast_loss'] = ctrast_loss
                else:
                    self.forward_ret_dict['ctrast_loss'] = torch.zeros(1).float().cuda()[0]
                pos_loss = self.get_parwise_loss(obj_descriptors, group_label)
                self.forward_ret_dict['pos_loss'] = pos_loss
            
            del data_dict['obj_points']
            del data_dict['obj_points_masks']
            torch.cuda.empty_cache()
            
            data_dict = self.assign_gt(group_label, data_dict['gt_boxes'], batch_nums, data_dict)
          
        else:
            mus = self.mu
        
        if self.training == False or self.only_optim_group == False:
            data_dict = self.forward_(None, spatial_features_2d, 0, data_dict, is_group=False)
        
        if self.only_optim_ori == False:
            for group_idx in range(len(mus)):
                
                data_dict = self.forward_(mus[group_idx][None,:], spatial_features_2d, group_idx+1, data_dict, is_group=True)

        if self.only_optim_ori :
            self.forward_ret_dict['group'] = 0

            data_dict['num_group'] = 0
        else:
            self.forward_ret_dict['group'] = self.num_group

            data_dict['num_group'] = self.num_group
        
       
    
        return data_dict
