1. Have a model of the category as an obj file.
2. Convert the model to tsdf field.
3. Run Marching cubes to get on model
4. Model clean + close holes
  a. Merge Close vertices
  b. Quadratic Simplication 
  c. Remove non-manifold edges
5. Mesh-deform to create a parametrized mapping
