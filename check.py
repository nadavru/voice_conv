import pickle
import torch

print(torch.load("saved_models/iteration.pt"))

'''with open("pickle_check.pkl", 'rb') as f:
    data = pickle.load(f)
    print(data)'''