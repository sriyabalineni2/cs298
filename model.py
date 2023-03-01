
import json

# read the JSON file
with open('feature_video_mapping.json', 'r') as f:
    feature_video_mapping = json.load(f)

import numpy as np
import pandas as pd

# Load the numpy array from file
datas = np.load("mod_datas.npy", allow_pickle=True)


# Convert to pandas dataframe
df = pd.DataFrame(datas, columns=["user_id", "video_id", "hashtags"])

# Use factorize() method to convert to numeric values
df["user"] = pd.factorize(df["user_id"])[0]
df["video"] = pd.factorize(df["video_id"])[0]
df["hashtag"] = pd.factorize(df["hashtags"])[0]

# Save the resulting dataframe to numpy array
new_datas = np.array(df[["user", "video", "hashtag"]])

user_map = df[["user", "user_id"]].drop_duplicates().sort_values("user")
video_map = df[["video", "video_id"]].drop_duplicates().sort_values("video")

hashtag_map = df[["hashtag", "hashtags"]].drop_duplicates().sort_values("hashtags")

# Save the numpy array to file
np.save("new_datas.npy", new_datas)
datas = np.load("new_datas.npy", allow_pickle=True)

#saving the string to int representations in dicts
user_map_dict = user_map.set_index('user')['user_id'].to_dict()
video_map_dict =  video_map.set_index('video')['video_id'].to_dict()
hashtag_map_dict =  hashtag_map.set_index('hashtag')['hashtags'].to_dict()




import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import uniform

from torch_geometric.nn import SAGEConv, GATConv


class Net(torch.nn.Module):
    def __init__(self, features, uh_edge_index, v_uh_edge_index, batch_size, num_user, num_hashtag, num_video, dim_latent, feature_video_mapping,aggr='mean'):

          print(f"dim_latent: {dim_latent}, type: {type(dim_latent)}")
          print(f"features.shape[1]: {features.shape[1]}, type: {type(features.shape[1])}")
          super(Net, self).__init__()
       
          self.batch_size = batch_size
          self.num_user = num_user
          self.num_hashtag = num_hashtag
          self.num_video = num_video
          self.dim_feat = features.shape[1]
          self.dim_latent = dim_latent
          self.aggr = aggr
          self.uh_edge_index = uh_edge_index
          self.v_uh_edge_index = v_uh_edge_index
          self.feature_video_mapping=feature_video_mapping
          self.u_h_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_hashtag, self.dim_latent), requires_grad=True))
          self.video_features = torch.tensor(features, dtype=torch.float)

          print(self.video_features.shape, 'shape video feature')

          print(self.dim_feat, type(self.dim_feat), 'dim_feat')
          print(self.dim_latent, type(self.dim_latent), 'dim_latent')

          self.trans_video_layer = nn.Linear(self.dim_feat, self.dim_latent)
          #self.trans_video_layer = nn.Linear(2070, 64)

         # self.trans_video_layer = nn.Linear(int(2070), 64)


          self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_hashtag, self.dim_latent)))

          self.first_conv = GATConv(self.dim_latent, self.dim_latent, self.aggr)

          nn.init.xavier_normal_(self.first_conv.weight)      
          self.second_conv = SAGEConv(self.dim_latent, self.dim_latent, self.aggr)
          nn.init.xavier_normal_(self.second_conv.weight)
          self.third_conv = GATConv(self.dim_latent, self.dim_latent, self.aggr)
          nn.init.xavier_normal_(self.third_conv.weight)
          self.forth_conv = SAGEConv(self.dim_latent, self.dim_latent, self.aggr)
          nn.init.xavier_normal_(self.forth_conv.weight)

          self.user_video_layer = nn.Linear(2*self.dim_latent, self.dim_latent)
          self.user_hashtag_layer = nn.Linear(2*self.dim_latent, self.dim_latent)
     
      
          
    @profile
    def forward(self, item):     

        user_tensor = item[:,[0]]
        video_tensor = item[:,[1]]
        pos_hashtag_tensor = item[:,[2]]
       # neg_hashtag_tensor = item[:,[3]]

        x = F.leaky_relu(self.trans_video_layer(self.video_features))
        x = torch.cat((self.u_h_embedding, x), dim=0)
        x =  F.normalize(x)

        x = F.leaky_relu(self.first_conv(x, self.v_uh_edge_index))
        x = F.leaky_relu(self.second_conv(x, self.uh_edge_index))
        x = F.leaky_relu(self.third_conv(x, self.v_uh_edge_index))
        x = F.leaky_relu(self.forth_conv(x, self.uh_edge_index))
    
        
        self.result_embed = x[torch.arange(self.num_user+self.num_hashtag)]

        user_tensor = self.result_embed[user_tensor].squeeze(1)
        pos_hashtags_tensor = self.result_embed[pos_hashtag_tensor].squeeze(1)
        #neg_hashtags_tensor = self.result_embed[neg_hashtag_tensor].squeeze(1)
        
        video_tensor = self.video_features[self.feature_video_mapping[video_tensor]].squeeze(1)
        video_tensor = F.leaky_relu(self.trans_video_layer(video_tensor))
        user_specific_video = F.leaky_relu(self.user_video_layer(torch.cat((video_tensor, user_tensor), dim=1)))
        user_specific_pos_h = F.leaky_relu(  self.user_hashtag_layer(torch.cat((pos_hashtags_tensor, user_tensor), dim=1)))
        #user_specific_neg_h = F.leaky_relu(self.user_hashtag_layer(torch.cat((neg_hashtags_tensor, user_tensor), dim=1)))

        pos_scores = torch.sum(user_specific_video*user_specific_pos_h, dim=1)
        #neg_scores = torch.sum(user_specific_video*user_specific_neg_h, dim=1)
        print(pos_scores)
        return pos_scores, neg_scores


    def loss(self, data):
        pos_scores, neg_scores = self.forward(data)
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores-neg_scores)))
        return loss_value


def get_heter_edges(datas):
  uh_edges_set = set()
  v_uh_edges_set = set()
  for index, data in enumerate(datas):
      user,video, hashtag = data
      v_uh_edges_set.add((video, user))
      v_uh_edges_set.add((video, hashtag))
      uh_edges_set.add((user, hashtag))
      uh_edges_set.add((hashtag, user))
  return uh_edges_set, v_uh_edges_set

from torch.utils.data import DataLoader
from tqdm import tqdm
batch_size=32
dim_latent= 64
num_heads=1
aggr_mode='mean'

train_dataset  = np.load('train.npy')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
num_user= len(user_map_dict)
num_video=len(video_map_dict)
num_hashtag =len(hashtag_map)
features = np.load('combined_features.npy', allow_pickle=True)

test_dataset = np.load('test.npy', allow_pickle=True)
print('Data has been loaded.')
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
uh_edges_set, v_uh_edges_set = get_heter_edges(train_dataset)
#uh_edges_index = torch.tensor(list(uh_edges_set),dtype=torch.long).contiguous().t() 
#v_uh_edges_index = torch.tensor(list(v_uh_edges_set),dtype=torch.long).contiguous().t() 

uh_edges_index = torch.tensor(list(uh_edges_set)).t().to(device)
v_uh_edges_index = torch.tensor(list(v_uh_edges_set)).t().to(device)

learning_rate=1e-3
weight_decay=1e-3



dim_latent = int(dim_latent)
model = Net(features, uh_edges_index, v_uh_edges_index, 32, num_user, num_hashtag, num_video, dim_latent,feature_video_mapping, aggr_mode)
print(model.parameters(), 'model parameters')
optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}], weight_decay=weight_decay)
max_recall = 0.0
# step = 0
num_epoch= 50
for epoch in range(num_epoch):
    model.train()
    print('Now, training start ...')
    pbar = tqdm(total=len(train_dataset))
    sum_loss = 0.0
    for data in train_dataloader:
        print(data, data.shape, 'DATA IS')
        optimizer.zero_grad()
        loss = model.loss(data)
        loss.backward()
        optimizer.step()
        pbar.update(batch_size)
        sum_loss += loss
    print(sum_loss/batch_size)
    pbar.close()

    print('Validation start...')
    


