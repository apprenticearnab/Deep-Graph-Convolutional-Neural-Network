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
      end
   end
   adj = reshape(q,190,190)'
   return adj
end
function KNN(s,adj_mat,k)# calculates the k nearest neighbors for every points and forms a matrix with elements (xi,xi-xj) i!=j with dimentions k*6*n n = number of points
   m = []
   g = zeros(36100)
   z = zeros(k,3,190)
   t = zeros(k,3,190)
   s_t = convert(Array,s)
   for i in 1:190
     for j in 1:190
        push!(m,(adj_mat[i,j]))
     end
   end
   d1 = reshape(m,190,190)'
   d = reshape(m,190,190)'
   for i in 1:190
      sort!(d[i,:])
   end
   for i in 1:190
      for j in 1:190
         for y in 1:190
            if d1[i,y] .== d[i,j]
               g[j.+(i-1).*190] = y
            end
         end
      end
   end
   g_t = reshape(g,190,190)'
   g_T = convert(Array,g_t)
   for i in 1:190
      for j in 2:k
        for b in 1:3
           z[j-1,b,i] = (s_t[Int64(g_T[i,1]),b]).-(s_t[Int64(g_T[i,j]),b])
           t[j-1,b,i] = (s_t[Int64(g_T[i,1]),b])
        end
       end
   end
   x = cat(z,t;dims=2)
   return x
end
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
end
function train(X,y_label)
     grads = gradient(() -> loss(X, y_label), Flux.params(model,layer1,layer2,layer3,layer4))    
     update!(opt, Flux.params(model,layer1,layer2,layer3,layer4) , grads)
     return loss(X,y_labels)
end
for epoch in 1:100
    println("-------Epoch : $epoch -------")
    current_loss = train(a ,labellist)
end 

     
