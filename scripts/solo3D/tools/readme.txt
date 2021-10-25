heightMapGenerator : 

Generate height map from .stl file object.

- Add new_object.stl in a new folder : objects/new_folder
- Modify path 
- Select X,Y range and number of points for the heighMap

--> Generate new_folder/heightmap/heightMap.dat ( [x,y, [height,Surface associated index]]) 
             new_folder/heightmap/surfaces.dat (liste of surfaces) 


--> Generate new_folder/meshes/object_* : useful to import object properly into pybullet (one single .stl do not allow to load properly the collision shape with .stl)
