import os
import torch
import torchvision
import numpy as np
import argparse

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths

from predict.predict_poseMF_shapeGaussian_net import predict_poseMF_shapeGaussian_net
from visualize_mesh import visualize_smpl_mesh
from renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

def save_mesh_as_obj(out_path, vertices, faces):
    """
    Saves vertex mesh as obj file.
    """
    try:
        with open(out_path, 'w') as fp:
            for v in vertices:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            if faces is not None:
                for f in faces + 1:  # Add 1 as OBJ uses 1-based indexing
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
        print(f"Successfully saved mesh to {out_path}")
    except Exception as e:
        print(f"Error saving mesh to {out_path}: {str(e)}")

def run_predict(device,
                image_dir,
                save_dir,
                pose_shape_weights_path,
                pose2D_hrnet_weights_path,
                pose_shape_cfg_path=None,
                already_cropped_images=False,
                visualise_samples=False,
                visualise_uncropped=False,
                joints2Dvisib_threshold=0.75,
                gender='neutral',
                visualize_3d=True):  # New parameter

    # ------------------------- Models -------------------------
    # Configs
    pose2D_hrnet_cfg = get_pose2D_hrnet_cfg_defaults()
    pose_shape_cfg = get_poseMF_shapeGaussian_cfg_defaults()
    if pose_shape_cfg_path is not None:
        pose_shape_cfg.merge_from_file(pose_shape_cfg_path)
        print('\nLoaded Distribution Predictor config from', pose_shape_cfg_path)
    else:
        print('\nUsing default Distribution Predictor config.')

    # Bounding box / Object detection model
    if not already_cropped_images:
        object_detect_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    else:
        object_detect_model = None

    # HRNet model for 2D joint detection
    hrnet_model = PoseHighResolutionNet(pose2D_hrnet_cfg).to(device)
    hrnet_checkpoint = torch.load(pose2D_hrnet_weights_path, map_location=device)
    hrnet_model.load_state_dict(hrnet_checkpoint, strict=False)
    print('\nLoaded HRNet weights from', pose2D_hrnet_weights_path)

    # Edge detector
    edge_detect_model = CannyEdgeDetector(non_max_suppression=pose_shape_cfg.DATA.EDGE_NMS,
                                          gaussian_filter_std=pose_shape_cfg.DATA.EDGE_GAUSSIAN_STD,
                                          gaussian_filter_size=pose_shape_cfg.DATA.EDGE_GAUSSIAN_SIZE,
                                          threshold=pose_shape_cfg.DATA.EDGE_THRESHOLD).to(device)

    # SMPL model
    print('\nUsing {} SMPL model with {} shape parameters.'.format(gender, str(pose_shape_cfg.MODEL.NUM_SMPL_BETAS)))
    smpl_model = SMPL(paths.SMPL,
                      batch_size=1,
                      gender=gender,
                      num_betas=pose_shape_cfg.MODEL.NUM_SMPL_BETAS).to(device)
    smpl_immediate_parents = smpl_model.parents.tolist()

    # 3D shape and pose distribution predictor
    pose_shape_dist_model = PoseMFShapeGaussianNet(smpl_parents=smpl_immediate_parents,
                                                   config=pose_shape_cfg).to(device)
    checkpoint = torch.load(pose_shape_weights_path, map_location=device)
    pose_shape_dist_model.load_state_dict(checkpoint['best_model_state_dict'])
    print('\nLoaded Distribution Predictor weights from', pose_shape_weights_path)

    # Create directory for OBJ files
    obj_save_dir = os.path.join(save_dir, 'obj_files')
    os.makedirs(obj_save_dir, exist_ok=True)

    # ------------------------- Predict -------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    
    predictions = predict_poseMF_shapeGaussian_net(
        pose_shape_model=pose_shape_dist_model,
        pose_shape_cfg=pose_shape_cfg,
        smpl_model=smpl_model,
        hrnet_model=hrnet_model,
        hrnet_cfg=pose2D_hrnet_cfg,
        edge_detect_model=edge_detect_model,
        device=device,
        image_dir=image_dir,
        save_dir=save_dir,
        object_detect_model=object_detect_model,
        joints2Dvisib_threshold=joints2Dvisib_threshold,
        visualise_uncropped=visualise_uncropped,
        visualise_samples=visualise_samples
    )

    if predictions is not None:
        try:
            # Initialize PyTorch3D renderer
            renderer = TexturedIUVRenderer(
                device=device,
                batch_size=1,
                img_wh=256,
                projection_type='perspective',
                perspective_focal_length=500,
                render_rgb=True
            )

            for idx, pred in enumerate(predictions):
                if 'vertices' in pred and pred['vertices'] is not None:
                    vertices = pred['vertices'][0].cpu().numpy() if torch.is_tensor(pred['vertices']) else pred['vertices'][0]
                    faces = pred['faces'].cpu().numpy() if torch.is_tensor(pred['faces']) else pred['faces']
                    
                    # Save as OBJ
                    obj_filename = f'prediction_{idx:04d}.obj'
                    obj_path = os.path.join(obj_save_dir, obj_filename)
                    save_mesh_as_obj(obj_path, vertices, faces)

                    # Visualize with PyTorch3D if enabled
                    # Replace the visualization part with this corrected version:
                    if visualize_3d:
                        try:
                            print(f"\nVisualizando mesh {idx}...")
                            
                            # If we have vertices directly, create mesh from vertices
                            if 'vertices' in pred and pred['vertices'] is not None:
                                vertices = pred['vertices'][0] if torch.is_tensor(pred['vertices']) else torch.tensor(pred['vertices'][0])
                                faces = pred['faces'] if torch.is_tensor(pred['faces']) else torch.tensor(pred['faces'])
                                
                                # Convert vertices to numpy for trimesh
                                vertices_np = vertices.cpu().numpy() if torch.is_tensor(vertices) else vertices
                                faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else faces
                                
                                # Create and show mesh directly with trimesh
                                import trimesh
                                mesh = trimesh.Trimesh(vertices=vertices_np,
                                                    faces=faces_np,
                                                    process=False)
                                mesh.show(smooth=True, flags={'cull': False})
                                
                            else:
                                # If we have pose and shape parameters, use SMPL model
                                betas = pred.get('betas', torch.zeros(1, 10, device=device))
                                pose = pred.get('pose', torch.zeros(1, 72, device=device))
                                
                                # Call visualize_smpl_mesh with the parameters
                                mesh = visualize_smpl_mesh(betas=betas, 
                                                        pose=pose,
                                                        device=device)
                                                
                        except Exception as e:
                            print(f"Error in 3D visualization for mesh {idx}: {str(e)}")
                    else:
                        print(f"Warning: No vertices found in prediction {idx}")
        except Exception as e:
            print(f"Error processing predictions: {str(e)}")
    else:
        print("Warning: No predictions were generated")

    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', '-I', type=str, help='Path to directory of test images.')
    parser.add_argument('--save_dir', '-S', type=str, help='Path to directory where test outputs will be saved.')
    parser.add_argument('--pose_shape_weights', '-W3D', type=str, default='./model_files/poseMF_shapeGaussian_net_weights.tar')
    parser.add_argument('--pose_shape_cfg', type=str, default=None)
    parser.add_argument('--pose2D_hrnet_weights', '-W2D', type=str, default='./model_files/pose_hrnet_w48_384x288.pth')
    parser.add_argument('--cropped_images', '-C', action='store_true', help='Images already cropped and centred.')
    parser.add_argument('--visualise_samples', '-VS', action='store_true')
    parser.add_argument('--visualise_uncropped', '-VU', action='store_true')
    parser.add_argument('--joints2Dvisib_threshold', '-T', type=float, default=0.75)
    parser.add_argument('--gender', '-G', type=str, default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visualize_3d', action='store_true', help='Visualize 3D mesh using PyTorch3D')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: {}'.format(device))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    run_predict(device=device,
                image_dir=args.image_dir,
                save_dir=args.save_dir,
                pose_shape_weights_path=args.pose_shape_weights,
                pose_shape_cfg_path=args.pose_shape_cfg,
                pose2D_hrnet_weights_path=args.pose2D_hrnet_weights,
                already_cropped_images=args.cropped_images,
                visualise_samples=args.visualise_samples,
                visualise_uncropped=args.visualise_uncropped,
                joints2Dvisib_threshold=args.joints2Dvisib_threshold,
                gender=args.gender,
                visualize_3d=args.visualize_3d)