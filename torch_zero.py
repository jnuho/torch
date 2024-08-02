
# %%
import torch


torch.__version__


print(torch.cuda.device_count())
# %%

scalar = torch.tensor(7)
print("-----1.scalar----------------------------")
print("shape = ", scalar.shape)
print("ndim = ", scalar.ndim)
print("scalar.item() = ", scalar.item())

vector = torch.tensor([7, 7, 8])
print("-----2.vector----------------------------")
print("shape = ", vector.shape)
print("ndim = ", vector.ndim)

# 3. MATRIX
MATRIX = torch.tensor([[1, 2, 3],
                      [2, 4, 5]])
print(MATRIX)
print("-----3.MATRIX----------------------------")
print("shape = ", MATRIX.shape)
print("ndim = ", MATRIX.ndim)

# 4. TENSOR
TENSOR = torch.tensor([[
    [1, 2, 3, 4],
    [3, 6, 9, 12],
    [2, 4, 5, 9] ],
    [[10, 20, 30, 40],
    [30, 60, 90, 120],
    [20, 40, 50, 90]]])
print("-----4.TENSOR----------------------------")
print(TENSOR)
print(TENSOR.shape)
print(TENSOR.ndim) # == number of brackets


# %%
print("-----RANDOM----------------------------")
# random_tensor = torch.rand(size=(3, 64, 64))
random_tensor = torch.rand(3, 4, 2)
print(random_tensor.dtype)
print(random_tensor.ndim)
print(random_tensor.shape)
print("random_tensor = ", random_tensor)

# %%
print("-----ZEROS AND ONES----------------------------")
zeros = torch.zeros(size=(3, 4))
print(zeros)

ones = torch.ones(3,4)
print(ones)


# %%
print("------Creating a range and tensors like-------")
# zero_to_ten_deprecated = torch.range(0, 10)
# print(zero_to_ten_deprecated)

zeros_to_ten = torch.arange(start=0, end=10, step=1)
print(zeros_to_ten)

ten_zeros = torch.zeros_like(input=zeros_to_ten)
print(ten_zeros)
ten_ones = torch.ones_like(input=zeros_to_ten)
print(ten_ones)


# %%
print("-----Tensor datatypes-----------------")


# For precision in computing!
# The higher the precision value (8, 16, 32), the more detail and hence data used to express a number.
# `torch.float32` or `torch.float`

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device

# %%
print("-----Common issues: mismatch in tensor `shape`, `datatype` and `device`----")

print("For example, one of tensors is torch.float32 and the other is torch.float16 (PyTorch often likes tensors to be the same format).")

print("Or one of your tensors is on the CPU and the other is on the GPU (PyTorch likes calculations between tensors to be on the same device).")



# %%
print("---Getting information from tensors---")
print("shape - what shape is the tensor? (some operations require specific shape rules)")
print("dtype - what datatype are the elements within the tensor stored in?")
print("device - what device is the tensor stored on? (usually GPU or CPU)\n")

# Create a tensor
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Dimension (ndim) of tensor: {some_tensor.ndim}")
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}") # will default to CPU

# %%
import torch

print("---Tensor operations-----")
# Addition
# Substraction
print("----Multiplication (element-wise)----")
tensor = torch.tensor([1, 2, 3])
print(tensor * tensor)
print(torch.mul(tensor, tensor))
print(torch.mul(tensor, tensor))
# Division
print("----Matrix multiplication----")
tensor = torch.tensor([1, 2, 3])
print(torch.matmul(tensor, tensor))
print(tensor @ tensor)

# %%

%%time

import torch

# tensor = torch.tensor([1, 2, 3])
tensor = torch.arange(start=0, end=1000000, step=1)

# Matrix multiplication by hand 
# (avoid doing operations with for loops at all cost, they are computationally expensive)
value = 0
print(len(tensor))
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value
# %%

%%time
torch.matmul(tensor, tensor)

# %%
print("COMMON ERRROR!!")
# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

torch.matmul(tensor_A, tensor_B) # (this will error)

# %%
print(torch.matmul(tensor_A, tensor_B.T))
print(torch.mm(tensor_A, tensor_B.T))
# %%
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
# torch.manual_seed(42)
# This uses matrix multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")

# %%

print("Finding the min, max, mean, sum, etc (aggregation)")
x = torch.arange(0, 100, 10)

print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")


torch.min(x), torch.max(x), torch.mean(x.type(torch.float32)), torch.sum(x)
# %%
import torch

# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")
# %%

import torch

tensor = torch.arange(10, 100, 10)
print("Change tensor datatype")
# Create a tensor and check its datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
print(tensor)

# Create a float16 tensor
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16.dtype)
print(tensor_float16)

# Create a int8 tensor
tensor_int8 = tensor.type(torch.int8)
print(tensor_int8.dtype)
print(tensor_int8)

# %%
print("Reshaping, stacking, squeezing and unsqueezing")

import torch

x = torch.arange(0., 10.)
print(x)
print(x.shape)

x_reshaped = x.reshape(10)
print(x_reshaped)
x_reshaped = x.reshape(1, 10)
print(x_reshaped)


print("VIEW!!!")
print("A tensor view in PyTorch is a way to create a new tensor")
print("\tthat shares the same data as the original tensor")
print("\tbut has a different shape or strides.")

print("Views are used to reshape, transpose, or otherwise change the view")
print("\tof the data without copying the underlying data.")
print("\tThis makes operations more memory-efficient and faster.")
print(x)
z = x.view(2, 5)
print(z)

# Changing view (z) changes x
z[:, 0] = 5
z, x


print("-----Stack-----")
# Stack tensors on top of each other
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)

x_stacked = torch.stack([x, x, x, x], dim=1)

print(x_stacked)
# %%

print("-----Squeeze-----")
# Stack tensors on top of each other
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")

print("---Unsqueeze---")
print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")

# And to do the reverse of torch.squeeze() you can use torch.unsqueeze() to add a dimension value of 1 at a specific index.
## Add an extra dimension with unsqueeze
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")
# %%
print("-----permute-----")
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))

# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")