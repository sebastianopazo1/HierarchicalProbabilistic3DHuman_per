import torch
import numpy as np
import trimesh
from models.smpl_official import SMPL
from utils.eval_utils import make_xz_ground_plane

def recursive_to(module, device):
    """
    Mueve recursivamente todos los parámetros y buffers al dispositivo especificado.
    """
    for child in module.children():
        recursive_to(child, device)
    for name, param in module._parameters.items():
        if param is not None:
            module._parameters[name] = param.to(device)
    for name, buffer in module._buffers.items():
        if buffer is not None:
            module._buffers[name] = buffer.to(device)

def visualize_smpl_mesh(betas=None, pose=None, device=None):
    """
    Visualiza el mesh SMPL en 3D usando trimesh
    Args:
        betas: Parámetros de forma SMPL (opcional)
        pose: Parámetros de pose SMPL (opcional) 
        device: Dispositivo para cálculos (cuda/cpu)
    """
    # Determinar dispositivo
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Inicializar modelo SMPL y moverlo al dispositivo correcto
    try:
        smpl = SMPL(model_path='model_files/smpl/SMPL_NEUTRAL.pkl',
                    gender='neutral',
                    batch_size=1,
                    device=device)
        smpl.to(device)  # Mover modelo al dispositivo
        recursive_to(smpl, device)  # Asegurar que todos los parámetros y buffers estén en el dispositivo
    except Exception as e:
        print(f"Error initializing SMPL model: {e}")
        raise

    # Crear parámetros por defecto si no se proporcionan
    if betas is None:
        betas = torch.zeros(1, 10, device=device)
    else:
        betas = betas.clone().detach().to(device)
        
    if pose is None:
        pose = torch.zeros(1, 72, device=device)
    else:
        pose = pose.clone().detach().to(device)

    print(f"betas device: {betas.device}")
    print(f"pose device: {pose.device}")

    # Obtener vértices y caras del modelo SMPL
    with torch.no_grad():
        try:
            smpl_output = smpl(betas=betas,
                              body_pose=pose[:, 3:],
                              global_orient=pose[:, :3])
        except Exception as e:
            print(f"Error in SMPL forward pass: {e}")
            raise

    # Mover resultados a CPU para visualización
    vertices = smpl_output.vertices.cpu().numpy()[0]
    faces = smpl.faces  # smpl.faces ya es un numpy.ndarray

    # Asegurar que el mesh esté en el plano XZ
    vertices = make_xz_ground_plane(vertices[None])[0]

    # Crear mesh con trimesh
    mesh = trimesh.Trimesh(vertices=vertices,
                          faces=faces,
                          process=False)

    # Visualizar
    mesh.show(smooth=True,
              flags={'cull': False})

    return mesh

def main():
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ejemplo de parámetros personalizados (opcional)
    betas = torch.zeros(1, 10, device=device)  # Parámetros de forma neutral
    pose = torch.zeros(1, 72, device=device)   # Pose en A-pose
    
    # Pose básica de ejemplo (brazos levantados)
    pose[0, 47] = -np.pi/3.0  # Brazo izquierdo
    pose[0, 50] = np.pi/3.0   # Brazo derecho

    try:
        # Visualizar mesh - pasando explícitamente el dispositivo
        mesh = visualize_smpl_mesh(betas=betas, 
                                 pose=pose,
                                 device=device)
        print("¡Mesh visualizado exitosamente!")
    except Exception as e:
        print(f"Error al visualizar mesh: {e}")
        raise

if __name__ == '__main__':
    main()
