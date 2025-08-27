import torch.backends.mps

print(torch.backends.mps.is_available())

tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)


tensor_1 = tensor_1.to("mps:0")
tensor_2 = tensor_2.to("mps:0")
print(tensor_1 + tensor_2)