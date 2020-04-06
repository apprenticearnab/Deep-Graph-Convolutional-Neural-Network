<<<<<<< HEAD
function knn(s,k) #calculates the distance(Eucleidian) between the points in and stores them in an adjacency matrix of order n*n n number of points
   inner=zeros(size(s,1),size(s,1),size(s,3))
   idx=zeros(size(s,1),k,size(s,3))
   y = PermutedDimsArray(s,(2,1,3))
   for j=1:size(s,3) 
       inner[:,:,j] = (-2).*(s[:,:,j]*y[:,:,j])
   end
   x1 = sum(s.^2 , dims=2)
   x2 = repeat(x1,outer=(1,size(s,1),1))
   dist = (-1).*x2 .- inner .- PermutedDimsArray(x2,(2,1,3))
   for i=1:size(s,1)
      for j=1:size(s,3)
          idx[i,:,j]=partialsortperm(dist[i,:,j], 1:k)
=======
using HDF5
using LinearAlgebra
using Flux
using Flux.Optimise: update!
using Zygote

fid = h5open("test_point_clouds.h5","r")
data = read(fid)

points = data["0"]["points"] 
#Here I have taken for simplicity only first data from the 3D MNIST dataset and its all point's coordinates(19000 points)
x = 3
a = reshape(points,x,19000)'
p = transpose(a)


function pairwise_distance(s) #calculates the distance(Eucleidian) between the points in and stores them in an adjacency matrix of order n*n n number of points 
   q = []
   for i in 1:190
      for j in 1:190   #as there is showing outofmemory error I have reduced the number of points further down to 190 
         push!(q,((s[i,1]-s[j,1]).^2).+((s[i,2]-s[j,2]).^2).+((s[i,3]-s[j,3]).^2))    
>>>>>>> cee2037dbd417de81ad60fb94095f22ebe4a5eff
      end
   end
   idx1 = convert(Array{Int64,3},idx)
   return idx1
end
@nograd knn
function edge(s,k)# calculates the k nearest neighbors for every points and forms a matrix with elements (xi,xi-xj) i!=j with dimentions k*6*n n = number of points
   idx = knn(s,k)
   z = zeros(size(idx,1),size(idx,2),size(s,2),size(idx,3))
   for i=1:size(idx,1)
      for j=1:size(idx,3)
         for t=1:size(idx,2)
            z[i,t,:,j]=s[idx[i,t,j],:,j]
         end
      end
   end 
   u = reshape(s,size(s,1),1,size(s,2),size(s,3))
   h = repeat(u,outer=(1,k,1,1))
   d = broadcast(-,u,z)
   feature = cat(d,h,dims=3)
   return feature
end
<<<<<<< HEAD
@nograd edge
#generating model as descripted in the paper , I have excluded the segmentation and transformation network 
struct DGCNN
     l1
     l2
     l3
     l4
     m
=======
function max(x,p)
    for i=1:p
      for j=1:20
       u[i,j] = maximum(x[i,:,j])
      end
    end
    return u
end    
#generating model as descripted in the paper , I have excluded the segmentation and transformation network , as the dimension of matrix (in our case 20*6) is already lower I have directly applied dense layer instead of applying further convolution to avoid loss of information we can reduce the output dimension we can avoid data overloading effect but it will be effective if we train the model in GPUs to avoid these kind of effects

layer1 = Chain(x->KNN(x,pairwise_distance(x),19),x->x[:,1:19,1:20],Dense(6,8),Dense(8,9),x->max(x,9))
layer2 = Chain(x->reshape(x,3,60)',x->KNN(x,pairwise_distance(x),19),x->x[:,1:19,1:20],Dense(6,8),Dense(8,9),x->max(x,9))
layer3 = Chain(x->reshape(x,3,60)',x->KNN(x,pairwise_distance(x),19),x->x[:,1:19,1:20],Dense(6,8),Dense(8,9),x->max(x,9))
layer4 = Chain(x->reshape(x,3,60)',x->KNN(x,pairwise_distance(x),19),x->x[:,1:19,1:20],Dense(6,10),Dense(10,18),x->max(x,18))
model = Chain(x->vcat(layer4(layer3(layer2(layer1(x)))),layer3(layer2(layer1(x))),layer2(layer1(x)),layer1(x)),x->reshape(x,3,300)',x->KNN(x,pairwise_distance(x),19),x->x[:,1:19,1:20],Dense(6,7),Dense(7,15),Dense(15,12),Dense(12,10),softmax)
    
opt = ADAM()
labellist = [1,0,0,0,0,0,0,0,0,0]   #As I am training the model on 3D MNIST dataset I have taken the labellist vector as onehot vector of length 10 and arbitarily taken that the first digit is 0 for training the model 
function loss(x,y)
     output = model(x)
     return crossentropy(output,y)
>>>>>>> cee2037dbd417de81ad60fb94095f22ebe4a5eff
end
function DGCNN(n_shapes::Int64,k::Int64) 
     layer1 = Chain(x->edge(x,k),Conv((1,1), 6=>8,relu),x->maximum(x,dims=2),x->reshape(x,size(x,1),size(x,3),size(x,4)))
     layer2 = Chain(x->edge(x,k),Conv((1,1), 16=>8,relu),x->maximum(x,dims=2),x->reshape(x,size(x,1),size(x,3),size(x,4)))
     layer3 = Chain(x->edge(x,k),Conv((1,1), 16=>16,relu),x->maximum(x,dims=2),x->reshape(x,size(x,1),size(x,3),size(x,4)))
     layer4 = Chain(x->edge(x,k),Conv((1,1), 32=>32,relu),x->maximum(x,dims=2),x->reshape(x,size(x,1),size(x,3),size(x,4)))
#model1 = Chain(layer1,layer2,layer3,layer4,x->reshape(x,(32*size(x,1)),size(x,3)))
"""function f(y) 
     z = cat(layer4(layer3(layer2(layer1(y)))),layer3(layer2(layer1(y))),layer2(layer1(y)),layer1(y),dims=2)
     return z
end"""
     model = Chain(x->cat(layer4(layer3(layer2(layer1(x)))),layer3(layer2(layer1(x))),layer2(layer1(x)),layer1(x),dims=2),x->reshape(x,(64*size(x,1)),size(x,3)),Dense(64*size(x,1),114),Dropout(0.5),Dense(114,118),Dropout(0.5),Dense(118,42),Dense(42,n_shapes),softmax)
     return DGCNN(layer1,layer2,layer3,layer4,model)
end
function (f::DGCNN)(x)
     return f.m(x)
end
@functor DGCNN

     

     
