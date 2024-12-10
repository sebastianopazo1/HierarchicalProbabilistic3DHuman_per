import os
import torch
import torchvision
import numpy as np
import argparse
import json

from models.poseMF_shapeGaussian_net import PoseMFShapeGaussianNet
from models.smpl_official import SMPL
from models.pose2D_hrnet import PoseHighResolutionNet
from models.canny_edge_detector import CannyEdgeDetector

from configs.poseMF_shapeGaussian_net_config import get_poseMF_shapeGaussian_cfg_defaults
from configs.pose2D_hrnet_config import get_pose2D_hrnet_cfg_defaults
from configs import paths
from utils.joints2d_utils import undo_keypoint_normalisation
from predict.predict_poseMF_shapeGaussian_net import predict_poseMF_shapeGaussian_net

def save_mesh_as_obj(out_path, vertices, faces):
    """
    Saves vertex mesh as obj file.
    :param out_path: path to save obj file.
    :param vertices: (num vertices, 3) numpy array of 3D vertices
    :param faces: (num faces, 3) numpy array of vertex indices
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
                gender='neutral'):

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
    json_save_dir = os.path.join(save_dir, 'joint_data')
    obj_save_dir = os.path.join(save_dir, 'obj_files')
    os.makedirs(obj_save_dir, exist_ok=True)
    os.makedirs(json_save_dir, exist_ok=True)

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
        obj_save_dir = os.path.join(save_dir, 'obj_files')
        json_save_dir = os.path.join(save_dir, 'joint_data')
        os.makedirs(obj_save_dir, exist_ok=True)
        os.makedirs(json_save_dir, exist_ok=True)

        pelvis_positions = {}
        
        # Imprimir información de debug
        print("Number of predictions:", len(predictions))
        if len(predictions) > 0:
            print("Keys in first prediction:", predictions[0].keys())

        try:
            for idx, pred in enumerate(predictions):
                # Obtener dimensiones originales de la imagen
                orig_height = pred.get('orig_height', None)
                orig_width = pred.get('orig_width', None)
                uncropped_bb = pred.get('uncropped_bb', None)

                # Debug info
                print(f"\nProcessing prediction {idx}")
                print(f"Original dimensions: {orig_width}x{orig_height}")
                print(f"Uncropped BB: {uncropped_bb}")

                if 'joints2D' in pred and pred['joints2D'] is not None:
                    joints2D = pred['joints2D']
                    if torch.is_tensor(joints2D):
                        joints2D = joints2D.cpu().numpy()

                    # Extraer coordenadas de la pelvis
                    if len(joints2D.shape) == 3:
                        pelvis_2d = joints2D[0, 0].tolist()
                    else:
                        pelvis_2d = joints2D[0].tolist()

                    # Convertir coordenadas si es necesario
                    if uncropped_bb is not None:
                        # Convertir de coordenadas normalizadas a coordenadas de imagen original
                        x_scale = (uncropped_bb[2] - uncropped_bb[0]) / pose_shape_cfg.DATA.PROXY_REP_SIZE
                        y_scale = (uncropped_bb[3] - uncropped_bb[1]) / pose_shape_cfg.DATA.PROXY_REP_SIZE
                        
                        orig_x = uncropped_bb[0] + pelvis_2d[0] * x_scale
                        orig_y = uncropped_bb[1] + pelvis_2d[1] * y_scale
                    else:
                        orig_x = pelvis_2d[0]
                        orig_y = pelvis_2d[1]

                    # Guardar en el diccionario
                    image_name = pred.get('image_name', f'image_{idx:04d}')
                    pelvis_positions[image_name] = {
                        'x': float(orig_x),
                        'y': float(orig_y),
                        'image_size': [
                            int(orig_width) if orig_width is not None else -1,
                            int(orig_height) if orig_height is not None else -1
                        ],
                        'bbox': [
                            float(uncropped_bb[0]) if uncropped_bb is not None else -1,
                            float(uncropped_bb[1]) if uncropped_bb is not None else -1,
                            float(uncropped_bb[2]) if uncropped_bb is not None else -1,
                            float(uncropped_bb[3]) if uncropped_bb is not None else -1
                        ]
                    }

            # Guardar el JSON
            json_path = os.path.join(json_save_dir, 'pelvis_positions.json')
            with open(json_path, 'w') as f:
                json.dump(pelvis_positions, f, indent=4)
            
            print(f"\nSaved pelvis positions to {json_path}")
            print("Sample of saved data:", next(iter(pelvis_positions.items())))

        except Exception as e:
            print(f"Error processing predictions: {str(e)}")
            import traceback
            traceback.print_exc()

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
                gender=args.gender)