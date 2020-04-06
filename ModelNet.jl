function load_data(partition,basedir)
    DATA_DIR = joinpath(basedir,"modelnet40_ply_hdf5_2048")
    all_data =[]
    all_label =[]
    for file in glob("ply_data_" * partition * ".h5" , DATA_DIR)
        fid = h5open(file,"r")
        f = read(fid)
        data = f["data"][:]
        label = f["label"][:]
        append!(all_data, data)
        append!(all_label, label)
     end
     all_data=hcat(all_data)
     all-label=hcat(all_label) 
     return all_data,all_label
end
function ModelNet(partition,basedir,num_pts,num_shapes)
     d,l = load_data(partition,basedir)
     a = collect(0:39)
     s = vec(l)
     label = onehotbatch(s,a)
     label1 = label[1:num_shapes,:]
     u = reshape(d,3,2048,2048)
     p = permutedims(u, [2, 1, 3])
     pointclouds = p[1:num_pts,:,:]
     return pointclouds,label1
end
   
     
