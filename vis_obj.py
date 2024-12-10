import trimesh
import os
import numpy as np

def visualize_obj_files(obj_paths, show_together=False):
    """
    Visualiza uno o múltiples archivos .obj
    
    Args:
        obj_paths: Puede ser una ruta a un archivo .obj o una lista de rutas
        show_together: Si es True, muestra todos los meshes juntos. Si es False, los muestra secuencialmente
    """

    if isinstance(obj_paths, str):
        obj_paths = [obj_paths]

    meshes = []

    for path in obj_paths:
        try:
            if not os.path.exists(path):
                print(f"Error: El archivo {path} no existe")
                continue

            mesh = trimesh.load(path, force='mesh')
      
            mesh.vertices -= mesh.vertices.mean(axis=0)

            scale = 1.0 / np.max(mesh.vertices)
            mesh.vertices *= scale
            
            meshes.append(mesh)
            print(f"Mesh cargado exitosamente: {path}")
            
        except Exception as e:
            print(f"Error al cargar {path}: {e}")
    
    if not meshes:
        print("No se pudo cargar ningún mesh")
        return

    if show_together and len(meshes) > 1:
 
        scene = trimesh.Scene()
        
        #Añadir cada mesh con un offset para que no se superpongan
        for i, mesh in enumerate(meshes):
           
            mesh_copy = mesh.copy()
            mesh_copy.vertices += np.array([i * 2, 0, 0])  # Offset en el eje X
            scene.add_geometry(mesh_copy)

        scene.show(smooth=True, flags={'cull': False})
    else:
        # Mostrar meshes secuencialmente
        for i, mesh in enumerate(meshes):
            print(f"\nMostrando mesh {i+1}/{len(meshes)}")
            mesh.show(smooth=True, flags={'cull': False})

def main():
   
    single_obj = "./output_stitching/143/obj_files/prediction_0000.obj"
   
    multiple_objs = [
        "./output_stitching/143/obj_files/prediction_0000.obj",
        "./output_stitching/143/obj_files/prediction_0001.obj",
        "./output_stitching/143/obj_files/prediction_0002.obj"
    ]
    
    # Descomentar la line a usar
    #visualize_obj_files(single_obj)
    visualize_obj_files(multiple_objs, show_together=True)
    # visualize_obj_files(multiple_objs, show_together=False)

if __name__ == "__main__":
    main()