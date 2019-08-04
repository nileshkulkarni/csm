getd = @(p)path(p,path); 
getd('toolbox_signal/');
getd('toolbox_general/');
getd('toolbox_graph/');
getd('toolbox_wavelet_meshes/');
startup
globals
global output_map_file
global class_name
name = [class_name '.off'];
[vertex,faces] = read_mesh(name);
n = size(vertex,2);
m = size(faces,2);
clear options; options.name = name;

clf;
options.lighting = 1;
plot_mesh(vertex,faces,options);
shading faceted;

weight = 'conformal';
weight = 'combinatorial';
switch weight
    case 'conformal'
        W = make_sparse(n,n);
        for i=1:3
            i1 = mod(i-1,3)+1;
            i2 = mod(i  ,3)+1;
            i3 = mod(i+1,3)+1;
            pp = vertex(:,faces(i2,:)) - vertex(:,faces(i1,:));
            qq = vertex(:,faces(i3,:)) - vertex(:,faces(i1,:));
            % normalize the vectors
            pp = pp ./ repmat( sqrt(sum(pp.^2,1)), [3 1] );
            qq = qq ./ repmat( sqrt(sum(qq.^2,1)), [3 1] );
            % compute angles
            ang = acos(sum(pp.*qq,1));
            a = max(1 ./ tan(ang),1e-1); % this is *very* important
            W = W + make_sparse(faces(i2,:),faces(i3,:), a, n, n );
            W = W + make_sparse(faces(i3,:),faces(i2,:), a, n, n );
        end
    case 'combinatorial'
        E = [faces([1 2],:) faces([2 3],:) faces([3 1],:)];
        E = unique_rows([E E(2:-1:1,:)]')';
        W = make_sparse( E(1,:), E(2,:), ones(size(E,2),1) );
end


d = full( sum(W,1) );
D = spdiags(d(:), 0, n,n);
iD = spdiags(d(:).^(-1), 0, n,n);
tW = iD * W;


vertex1 = vertex;
vertex1 = vertex1 - repmat( mean(vertex1,2), [1 n] );
vertex1 = vertex1 ./ repmat( sqrt(sum(vertex1.^2,1)), [3 1] );


% normal to faces
[normal,normalf] = compute_normal(vertex1,faces);
% center of faces
C = squeeze(mean(reshape(vertex1(:,faces),[3 3 m]), 2));
% inner product
I = sum(C.*normalf);

disp(['Ratio of inverted triangles:' num2str(sum(I(:)<0)/m, 3) '%']);


options.name = 'none';
clf;
options.face_vertex_color = double(I(:)>0);
plot_mesh(vertex1,faces,options);
colormap gray(256); axis tight;
shading faceted;

clf;
exo1;
vshape = vertex;
vsphere = vertex1;
face = faces-1;
save(output_map_file,'vsphere', 'vshape', 'face') 
