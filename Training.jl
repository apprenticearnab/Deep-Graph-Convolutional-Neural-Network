using HDF5
using Glob
using Base.Filesystem
using Flux: onehotbatch,onecold
using Flux.Data
using Statistics: mean
using LinearAlgebra
using Flux
using Flux.Optimise: update!
using Zygote
using Flux:@functor
using Zygote:@nograd
include("basedir/ModelNet.jl")
include("basedir/DGCNN.jl")
d = ModelNet("partition","basedir/folder name where the datasets exists",num_points,num_shapes)
v = ModelNet("partition","basedir/folder name where the datasets exists",num_points,num_shapes)
 
x = d[1]     #random x

opt = ADAM(0.001)

f = DGCNN(num_shapes,k)
vx = v[1][:,:,1:num_data_want_to_take]       #validation data
vy = v[2][:,1:num_data_want_to_take]

accuracy(x, y) = mean(onecold(f(x), 1:num_shapes) .== onecold((y), 1:num_shapes))

function loss(x,y)
     output = f(x)
     return Flux.crossentropy(output,y)
end
function train(data,epochs,opt,loss)
   for epoch in 1:epochs
      h=1
      total_loss=0
      for i=1:45
         grads = gradient(() -> loss(data[1][:,:,h:h+13],data[2][:,h:h+13]), Flux.params(f))  
         current_loss = loss(data[1][:,:,h:h+13],data[2][:,h:h+13])
         total_loss+=current_loss 
         update!(opt, Flux.params(f) , grads)
         h+=14
      end
      print("----------EPOCH:$epoch----------")
      print(total_loss)
   end
end
train(d, 20, opt, loss)

@show(accuracy(vx, vy))

