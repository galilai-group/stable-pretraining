"""Frozen upstream DAE-Net network classes for offline numerical parity.

Source and license: THIRD_PARTY_NOTICES.md (DAE-Net).

The five class definitions are unchanged; only the header and imports differ.
Formatting is disabled to preserve the original definitions. Source SHA256:
CRLF: 46bbe3d9071c09c9a201f6e5bd9b5ab970be07581c0da99ad1e75adb77081064
LF:   65bf2fe52cfc9fa0cdf65afdb4ba328bce977aca716fbe3d74db3c587c118a1e

The reference runs offline without ShapeNet data, pretrained weights or CUDA.
Its encoder forces float32, so end-to-end parity uses FP32; the decoder is also
compared in float64. Tests independently transcribe the released training loss
and include hand-calculated value and gradient checks.
"""

# ruff: noqa
# fmt: off
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_G(nn.Module):
	def __init__(self, input_dim, gf_dim, branch_num):
		super(MLP_G, self).__init__()
		self.input_dim = input_dim
		self.df_dim = gf_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.linear_d1 = nn.Conv1d(self.input_dim*self.branch_num, self.df_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_d2 = nn.Conv1d(self.df_dim*self.branch_num,                 self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_d3 = nn.Conv1d(self.df_dim*self.branch_num,                 self.input_dim*self.branch_num, 1, groups=self.branch_num, bias=True)

		self.linear_1 = nn.Conv1d(self.input_dim*self.branch_num, self.gf_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_2 = nn.Conv1d(self.gf_dim*self.branch_num,    self.gf_dim*self.branch_num, 1, groups=self.branch_num, bias=True)
		self.linear_3 = nn.Conv1d(self.gf_dim*self.branch_num,    self.branch_num,             1, groups=self.branch_num, bias=True)
		#nn.Conv1d(A*B, C*B, 1, groups=B)
		#weight [C*B, A, 1]
		#bias [C*B]

	def init(self, branch_id):
		nn.init.xavier_uniform_(self.linear_d1.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_d1.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_d2.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_d2.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_d3.weight[branch_id*self.input_dim:(branch_id+1)*self.input_dim])
		nn.init.constant_(self.linear_d3.bias[branch_id*self.input_dim:(branch_id+1)*self.input_dim],0)

		nn.init.normal_(self.linear_1.weight[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], mean=0.0, std= (2/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_1.bias[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], 0)
		nn.init.normal_(self.linear_2.weight[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], mean=0.0, std= (2/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_2.bias[branch_id*self.gf_dim:(branch_id+1)*self.gf_dim], 0)
		nn.init.constant_(self.linear_3.weight[branch_id:(branch_id+1)], -(3.14*4/self.gf_dim)**0.5 )
		nn.init.constant_(self.linear_3.bias[branch_id:(branch_id+1)], 0)

	def forward(self, points):
		#points [N, 3*B, P]

		out = self.linear_d1(points)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_d2(out)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_d3(out) + points

		out = self.linear_1(out)
		out = F.relu(out)

		out = self.linear_2(out)
		out = F.relu(out)

		out = self.linear_3(out)
		out = torch.sigmoid(out)

		return out

class MLP_D(nn.Module):
	def __init__(self, input_dim, z_dim, df_dim, branch_num):
		super(MLP_D, self).__init__()
		self.input_dim = input_dim
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.branch_num = branch_num

		self.linear_1 = nn.Conv1d((self.input_dim+self.z_dim)*self.branch_num, self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_2 = nn.Conv1d(self.df_dim*self.branch_num,                 self.df_dim*self.branch_num,    1, groups=self.branch_num, bias=True)
		self.linear_3 = nn.Conv1d(self.df_dim*self.branch_num,                 self.input_dim*self.branch_num, 1, groups=self.branch_num, bias=True)

	def init(self, branch_id):
		nn.init.xavier_uniform_(self.linear_1.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_1.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_2.weight[branch_id*self.df_dim:(branch_id+1)*self.df_dim])
		nn.init.constant_(self.linear_2.bias[branch_id*self.df_dim:(branch_id+1)*self.df_dim],0)
		nn.init.xavier_uniform_(self.linear_3.weight[branch_id*self.input_dim:(branch_id+1)*self.input_dim])
		nn.init.constant_(self.linear_3.bias[branch_id*self.input_dim:(branch_id+1)*self.input_dim],0)

	def forward(self, zs, points):
		#points [N, P, 3]
		affine_mat = zs[:,:,self.z_dim:self.z_dim+12]-0.5 #[N, B, 12]
		zs = zs[:,:,:self.z_dim] #[N, B, Z]

		N,P,_ = points.size()
		points = points.permute(0,2,1) #[N, 3, P]
		pointsx = torch.cat([points,torch.ones(N,1,P,device=points.device)],1) #[N, 4, P]
		ps = []
		pzs = []
		for i in range(self.branch_num):
			ps.append(affine_mat[:,i].view(N,3,4)@pointsx+points) #[N, 3, P]
			pzs.append(ps[i]) #[N, 3, P]
			pzs.append(zs[:,i].view(N,self.z_dim,1).repeat(1,1,P)) #[N, Z, P]
		ps = torch.cat(ps,1) #[N, 3*B, P]
		pzs = torch.cat(pzs,1) #[N, (3+Z)*B, P]

		out = self.linear_1(pzs)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		out = self.linear_2(out)
		out = F.leaky_relu(out, negative_slope=0.02, inplace=True)

		offsets = self.linear_3(out) #[N, 3*B, P]

		return affine_mat, ps, offsets

class generator(nn.Module):
	def __init__(self, z_dim, df_dim, gf_dim, branch_num):
		super(generator, self).__init__()
		self.point_dim = 3
		self.z_dim = z_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.deformers = MLP_D(self.point_dim,self.z_dim,self.df_dim,self.branch_num)
		self.generators = MLP_G(self.point_dim,self.gf_dim,self.branch_num)

	def init(self, branch_id):
		self.deformers.init(branch_id)
		self.generators.init(branch_id)

	def forward(self, points, zs, matrices, out_sum=False, out_branch=False, affine_only=False):
		if out_branch and not out_sum:
			#points [N,P,3]
			out = self.generators(points.permute(0,2,1).repeat(1,self.branch_num,1)) #[N,B,P]
			out = out.permute(0,2,1) #[N,P,B]
			return out

		#points [N,P,3]
		#zs [N,B,Z]
		#matrices [N,B,1]
		N,P,_ = points.size()
		affine_mat, affine_points, deformed_offsets = self.deformers(zs,points) #[N, 3*B, P]
		if affine_only:
			deformed_points = affine_points
		else:
			deformed_points = affine_points + deformed_offsets
		out = self.generators(deformed_points) #[N,B,P]
		deformed_offsets = deformed_offsets.view(N, self.branch_num, 3, P) * out.detach().view(N, self.branch_num, 1, P) #mask out empty space
		out = out.permute(0,2,1) #[N,P,B]

		if out_branch:
			return out

		matrices = matrices.view(N,1,self.branch_num).repeat(1,P,1) #[N,P,B]
		out = out*matrices
		out_sum = torch.sum(out,-1,keepdim=True)
		out_max = torch.max(out,-1,keepdim=True)[0]

		return out_sum, out_max, affine_mat, deformed_offsets

class encoder(nn.Module):
	def __init__(self, z_dim, ef_dim, branch_num):
		super(encoder, self).__init__()
		self.z_dim = z_dim + 12 # + affine transform matrix
		self.ef_dim = ef_dim
		self.branch_num = branch_num

		self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.norm_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.norm_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.norm_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.norm_4 = nn.InstanceNorm3d(self.ef_dim*8)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.ef_dim*16, 4, stride=1, padding=0, bias=True)

		self.linear_1 = nn.Linear(self.ef_dim*16, self.branch_num, bias=True)
		self.linear_2 = nn.Linear(self.ef_dim*16, self.branch_num*self.z_dim, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)

	def init(self, branch_id):
		nn.init.normal_(self.linear_1.weight[branch_id:(branch_id+1)], mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias[branch_id:(branch_id+1)], 0)
		nn.init.normal_(self.linear_2.weight[branch_id*self.z_dim:(branch_id+1)*self.z_dim], mean=0.0, std=0.02)
		nn.init.normal_(self.linear_2.bias[branch_id*self.z_dim:(branch_id+1)*self.z_dim], mean=0.0, std=0.1)
		#only scale and translate when init
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+1] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+2] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+4] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+6] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+8] = 0
		self.linear_2.bias.data[branch_id*self.z_dim:branch_id*self.z_dim+9] = 0

	def forward(self, inputs):
		out = inputs.float()
		out = F.leaky_relu(self.norm_1(self.conv_1(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_2(self.conv_2(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_3(self.conv_3(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.norm_4(self.conv_4(out)), negative_slope=0.02, inplace=True)
		out = F.leaky_relu(self.conv_5(out), negative_slope=0.02, inplace=True)
		out = out.view(-1, self.ef_dim*16)

		out1 = self.linear_1(out)
		out1 = out1.view(-1, self.branch_num, 1)
		out1 = torch.sigmoid(out1-8)

		out2 = self.linear_2(out)
		out2 = out2.view(-1, self.branch_num, self.z_dim)
		out2 = torch.sigmoid(out2)

		return out1, out2

class dae_network(nn.Module):
	def __init__(self, z_dim, ef_dim, df_dim, gf_dim, branch_num):
		super(dae_network, self).__init__()
		self.z_dim = z_dim
		self.ef_dim = ef_dim
		self.df_dim = df_dim
		self.gf_dim = gf_dim
		self.branch_num = branch_num

		self.encoder = encoder(self.z_dim, self.ef_dim, self.branch_num)
		self.generator = generator(self.z_dim, self.df_dim, self.gf_dim, self.branch_num)
