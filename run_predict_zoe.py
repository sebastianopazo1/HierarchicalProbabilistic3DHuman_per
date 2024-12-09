import os
import torch
import numpy as np
import argparse
from PIL import Image
from models.smpl_official import SMPL
from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector
from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from predict.predict_poseMF_shapeGaussian_net import predict_poseMF_shapeGaussian_net
from visualize_mesh import visualize_smpl_mesh
from configs import paths

torch.backends.cudnn.benchmark = True  # Optimiza la velocidad de inferencia
torch.backends.cuda.matmul.allow_tf32 = True  # Permite TF32 en GPUs Ampere
def load_zoedepth_model(device):
    """Carga el modelo ZoeDepth desde torch.hub con un modelo MiDaS más estable"""
    try:
        # Trigger fresh download of MiDaS repo
        torch.hub.help("intel-isl/MiDaS", "DPT_Large", force_reload=True)  # Changed from DPT_BEiT_L_384
        
        # Load ZoeDepth with different configuration
        repo = "isl-org/ZoeDepth"
        conf = {
            "model_type": "zoedepth",
            "version": "v1",
            "midas_model_type": "DPT_Large"  # Changed from DPT_BEiT_L_384
        }
        
        model = torch.hub.load(repo, "ZoeD_N", pretrained=True, config=conf)
        model.to(device)
        model.eval()
        print("Modelo ZoeDepth cargado exitosamente")
        return model
    except Exception as e:
        print(f"Error cargando ZoeDepth: {str(e)}")
        return None

def setup_smpl_models(device, pose_shape_weights_path, pose2D_hrnet_weights_path):
    """Configura los modelos necesarios para SMPL"""
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()

    # HRNet para detección de joints 2D
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)

    # Detector de bordes
    edge_detect_model = CannyEdgeDetector(
        non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
        gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
        gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
        threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD
    ).to(device)

    # Modelo SMPL
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender='neutral').to(device)

    # Predictor de distribución de forma y pose 3D
    pose_shape_dist_model = PoseMFShapeGaussianNet(
        smpl_parents=smpl_model.parents.tolist(),
        config=pose_shape_cfg
    ).to(device)
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])

    return hrnet_model, edge_detect_model, smpl_model, pose_shape_dist_model, pose_shape_cfg

def process_image(image_path, zoe_model, device):
    """Procesa la imagen y retorna solo la profundidad estimada"""
    try:
        # Cargar y procesar imagen
        image = Image.open(image_path).convert("RGB")
        
        # Obtener profundidad con ZoeDepth
        with torch.no_grad():
            depth = zoe_model.infer_pil(image)
            h, w = depth.shape
            center_depth = np.mean(depth[h//3:2*h//3, w//3:2*w//3])
            
        return depth, center_depth, np.array(image)
        
    except Exception as e:
        print(f"Error procesando imagen: {str(e)}")
        return None, None, None

def run_predict(device,
                image_dir,
                save_dir,
                pose_shape_weights_path,
                pose2D_hrnet_weights_path):
    
    try:
        # Configurar ZoeDepth
        repo = "isl-org/ZoeDepth"
        conf = {
            "model_type": "zoedepth",
            "version": "v1",
            "midas_model_type": "DPT_Large",
            "memory_efficient": True,
            "force_keep_ar": True
        }
        
        # Cargar modelo ZoeDepth
        zoe_model = torch.hub.load(repo, "ZoeD_N", pretrained=True, config=conf)
        zoe_model.to(device)
        zoe_model.eval()
        
        # Procesar imágenes
        with torch.no_grad():
            for image_name in os.listdir(image_dir):
                if not image_name.endswith(('.jpg', '.png', '.jpeg')):
                    continue
                    
                print(f"Procesando {image_name}...")
                image_path = os.path.join(image_dir, image_name)
                
                # Obtener profundidad
                depth, center_depth, image = process_image(image_path, zoe_model, device)
                if depth is None:
                    continue
                
                # Generar y mostrar SMPL directamente
                base_name = os.path.splitext(image_name)[0]
                mesh_path = os.path.join(save_dir, f"mesh_{base_name}.obj")
                
                # Aquí va la llamada a tu función de generación de SMPL
                # usando la profundidad estimada (depth)
                # generate_smpl(depth, center_depth, mesh_path)
                
                print(f"Mesh guardado en: {mesh_path}")
                
                # Liberar memoria
                del depth, center_depth, image
                torch.cuda.empty_cache()
                
    except Exception as e:
        print(f"Error en run_predict: {str(e)}")
    finally:
        torch.cuda.empty_cache()
        
    hrnet_model, edge_detect_model, smpl_model, pose_shape_dist_model, pose_shape_cfg = setup_smpl_models(
        device, pose_shape_weights_path, pose2D_hrnet_weights_path)

    # Crear directorio de salida
    os.makedirs(save_dir, exist_ok=True)

    # Procesar cada imagen
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, img_name)
            print(f"\nProcesando {img_name}...")

            # Obtener profundidad con ZoeDepth
            depth, center_depth, image = process_image(image_path, zoe_model, device)

            # Obtener predicción SMPL
            predictions = predict_poseMF_shapeGaussian_net(
                pose_shape_model=pose_shape_dist_model,
                pose_shape_cfg=pose_shape_cfg,
                smpl_model=smpl_model,
                hrnet_model=hrnet_model,
                hrnet_cfg=get_pose2D_hrnet_cfg_defaults(),
                edge_detect_model=edge_detect_model,
                device=device,
                image_dir=os.path.dirname(image_path),
                save_dir=save_dir,
                object_detect_model=None,
                joints2Dvisib_threshold=0.75
            )

            if predictions and len(predictions) > 0:
                pred = predictions[0]
                if 'vertices' in pred and pred['vertices'] is not None:
                    vertices = pred['vertices'][0]
                    
                    # Ajustar la posición del modelo SMPL según la profundidad de ZoeDepth
                    vertices_centered = vertices - vertices.mean(dim=0, keepdim=True)
                    vertices_scaled = vertices_centered * center_depth
                    
                    # Visualizar el modelo ajustado
                    try:
                        import trimesh
                        mesh = trimesh.Trimesh(
                            vertices=vertices_scaled.cpu().numpy(),
                            faces=smpl_model.faces,
                            process=False
                        )
                        mesh.show()
                        
                        # Guardar el mesh
                        output_path = os.path.join(save_dir, f'mesh_{img_name[:-4]}.obj')
                        mesh.export(output_path)
                        print(f"Mesh guardado en: {output_path}")
                        
                    except Exception as e:
                        print(f"Error en visualización: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str, required=True,
                        help='Directorio con imágenes de entrada')
    parser.add_argument('--save_dir', '-S', type=str, required=True,
                        help='Directorio donde se guardarán los resultados')
    parser.add_argument('--pose_shape_weights', '-W3D', type=str,
                        default='./model_files/poseMF_shapeGaussian_net_weights.tar')
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str,
                        default='./model_files/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDispositivo: {}'.format(device))

    run_predict(device=device,
                image_dir=args.image_dir,
                save_dir=args.save_dir,
                pose_shape_weights_path=args.pose_shape_weights,
                pose2D_hrnet_weights_path=args.pose2D_hrnet_weights)