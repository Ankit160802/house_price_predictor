from django.shortcuts import render
from django.http import HttpResponse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


# model defining
class HousePrice(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Linear(in_features=5,out_features=10),
            nn.CELU(),
            nn.Linear(10,20),
            nn.CELU(),
            nn.Linear(20,30),
            nn.CELU(),
            nn.Linear(30,1)
        )
    def forward(self,x):
        return self.layer_stack(x)

# loading trained data of model
modelpath=Path('model')
model_name="housepredictonmodel"
modelsavepath=modelpath/model_name
model=HousePrice()
model.load_state_dict(torch.load(f=modelsavepath))

map_tra=['New Property', 'Resale']
map_fur=['Furnished', 'Semi-Furnished', 'Unfurnished']
map_loc=['agra',
 'ahmadnagar',
 'ahmedabad',
 'allahabad',
 'aurangabad',
 'badlapur',
 'bangalore',
 'belgaum',
 'bhiwadi',
 'bhiwandi',
 'bhopal',
 'bhubaneswar',
 'chandigarh',
 'chennai',
 'coimbatore',
 'dehradun',
 'durgapur',
 'ernakulam',
 'faridabad',
 'ghaziabad',
 'goa',
 'greater-noida',
 'guntur',
 'gurgaon',
 'guwahati',
 'gwalior',
 'haridwar',
 'hyderabad',
 'indore',
 'jabalpur',
 'jaipur',
 'jamshedpur',
 'jodhpur',
 'kalyan',
 'kanpur',
 'kochi',
 'kolkata',
 'kozhikode',
 'lucknow',
 'ludhiana',
 'madurai',
 'mangalore',
 'mohali',
 'mumbai',
 'mysore',
 'nagpur',
 'nashik',
 'navi-mumbai',
 'navsari',
 'nellore',
 'new-delhi',
 'noida',
 'palakkad',
 'palghar',
 'panchkula',
 'patna',
 'pondicherry',
 'pune',
 'raipur',
 'rajahmundry',
 'ranchi',
 'satara',
 'shimla',
 'siliguri',
 'solapur',
 'sonipat',
 'surat',
 'thane',
 'thrissur',
 'tirupati',
 'trichy',
 'trivandrum',
 'udaipur',
 'udupi',
 'vadodara',
 'vapi',
 'varanasi',
 'vijayawada',
 'visakhapatnam',
 'vrindavan',
 'zirakpur']



def predictingprice(location:int,Area:float,Transaction:int,Furnishing:int,Bathroom:int):
    list=[location,Area,Transaction,Furnishing,Bathroom]
    list=np.array(list)
    list=torch.from_numpy(list).type(torch.float32)
    model.eval()
    with torch.inference_mode():
        predicted=model(list).squeeze()
    predicted=predicted.cpu().detach().numpy()
    return predicted


def predict(request):
    return render(request,'index.html')


def result(request):
    location = request.GET.get('location')
    Area= float(request.GET.get('Area'))
    Transact= request.GET.get('Trans')
    furnish = request.GET.get('furnish')
    Bathroom= int(request.GET.get('Bathroom'))

    index_loc = map_loc.index(location)
    index_trans = map_tra.index(Transact)
    index_furnishing = map_fur.index(furnish)
    price=predictingprice(index_loc,Area,index_trans,index_furnishing,Bathroom)
    print("price: ")
    print(price)
    return render(request,'result.html',{'value':price})