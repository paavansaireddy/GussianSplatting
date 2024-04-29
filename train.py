
import os
import torch
import torch.nn.functional as tf
from random import randint
from utils.loss_utils import create_window
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
l_for_l = 0.0

def l1_loss(ntw, gt):
    return torch.abs((ntw - gt)).mean()

def chan_for_ssim(ig1, ig2, ws=11, svg=True):
    path = ig1.size(-3)
    screen = create_window(ws, path)

    if ig1.is_cuda:
        screen = screen.cuda(ig1.get_device())
    screen = screen.type_as(ig2)

    return chan_for_ssim2(ig1, ig2, screen, ws, path, svg)

def chan_for_ssim2(ig1, ig2, screen, ws, path, svg=True):
    
    x1 = tf.conv2d(ig1, screen, padding=ws // 2, groups=path)
    x1s= x1*x1
    x2 = tf.conv2d(ig2, screen, padding=ws // 2, groups=path)
    x2s =x2*x2
    x1x2 = x1 * x2
    ss1q = tf.conv2d(ig1 * ig1, screen, padding=ws // 2, groups=path) - x1s
    ss2q = tf.conv2d(ig2 * ig2, screen, padding=ws // 2, groups=path) - x2s
    s12 = tf.conv2d(ig1 * ig2, screen, padding=ws // 2, groups=path) - x1x2

    j1 = 0.01 ** 2
    j2 = 0.03 ** 2
    s_otp = ((2 * x1x2 + j1) * (2 * s12 + j2)) / ((x1s + x2s + j1) * (ss1q + ss2q + j2))

    if svg:
        return s_otp.mean()
    else:
        return s_otp.mean(1).mean(1).mean(1)


def training(dataset, opt, pipe, tst_itr, svg_itr, chpt_itr, chpt, debug_from):
    global l_for_l
    # Initialize training parameters and load checkpoint if available
    tnsr_wrtr = otp_prepare(dataset)
    gsns = GaussianModel(dataset.sh_degree)
    scn = Scene(dataset, gsns)
    gsns.training_setup(opt)
    itr1 = 0
    if chpt:
        model_params, itr1 = torch.load(chpt)
        gsns.restore(model_params, opt)
    bckgrd = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    # Initialize time tracking for performance metrics
    startitr = torch.cuda.Event(enable_timing=True)
    enditr = torch.cuda.Event(enable_timing=True)
    flag2= opt.iterations
    pgbr = tqdm(range(itr1, flag2), desc="Training progress")

    vptn_stk = None

    # Main training loop
    scd_itr=itr1+1
    scd_param=opt.iterations + 1
    for itr in range(scd_itr, scd_param):
        handle_network_gui(itr, gsns, pipe, bckgrd, dataset, network_gui)

        startitr.record()
        if itr % 1000 == 0:
            gsns.oneupSHdegree()

        viewpoint_cam = select_camera(vptn_stk, scn)

        if itr == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gsns, pipe, bckgrd)
        loss = compute_loss(render_pkg, viewpoint_cam, opt)

        loss.backward()
        enditr.record()
        # Handle logging, checkpointing, and saving
        manage_training_cycle(itr, gsns, scn, loss, l_for_l, tnsr_wrtr, svg_itr, chpt_itr)
        l_for_l=cal_loss(loss)
        # Update progress bar display
        update_progress_bar(pgbr, itr, l_for_l, opt.iterations)

    # Final clean-up after training loop
    pgbr.close()
    print("\nTraining complete.")
def update_progress_bar(pgbr, itr, l_for_l, total_iterations):
    """ Update the progress bar with the current iteration and estimated moving average loss. """
    pgbr.set_description(f"Training progress (iteration {itr}/{total_iterations})")
    pgbr.set_postfix({"Loss": f"{l_for_l:.6f}"})
    pgbr.update(1)

def handle_network_gui(itr, gsns, pipe, bckgrd, dataset, network_gui):
    """ Manage GUI network interactions if connected. """
    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn:
        try:
            imgn_bts = handle_custom_camera_view(gsns, pipe, bckgrd, dataset, network_gui)
            network_gui.send(imgn_bts, dataset.source_path)
            if not keep_training_alive(itr, opt.iterations, network_gui):
                break
        except Exception as e:
            network_gui.conn = None

def handle_custom_camera_view(gsns, pipe, bckgrd, dataset, network_gui):
    """ Process custom camera view from network GUI. """
    cam_cust, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
    if cam_cust:
        imgn = render(cam_cust, gsns, pipe, bckgrd, scaling_modifier)["render"]
        imgn_bts = memoryview((torch.clamp(imgn, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        return imgn_bts
    return None

def keep_training_alive(itr, total_iterations, network_gui):
    """ Determine whether to continue training based on GUI input and iteration count. """
    return network_gui.receive_training_signal() and (itr < total_iterations or network_gui.keep_alive())

def select_camera(vptn_stk, scn):
    """ Select a random camera from the training set for rendering. """
    if not vptn_stk:
        vptn_stk = scn.getTrainCameras().copy()
    return vptn_stk.pop(randint(0, len(vptn_stk)-1))

def compute_loss(render_pkg, viewpoint_cam, opt):
    """ Compute the combined loss for the current iteration. """
    image, gt_image = render_pkg["render"], viewpoint_cam.original_image.cuda()
    flag1 = l1_loss(image, gt_image)
    return (1.0 - opt.lambda_dssim) * flag1 + opt.lambda_dssim * (1.0 - chan_for_ssim(image, gt_image))

def manage_training_cycle(itr, gsns, scn, loss, l_for_l, tnsr_wrtr, svg_itr, chpt_itr):
    """ Handle logging, checkpointing, and saving within the training loop. """
    if itr in svg_itr:
        print("\n[ITER {}] gaussian check".format(itr))
        scn.save(itr)
    if itr in chpt_itr:
        print("\n[ITER {}] checkpoint check".format(itr))
        torch.save

def cal_loss(loss):
    global l_for_l
    l_for_l = 0.4 * loss.item() + 0.6 * l_for_l
    return l_for_l



def otp_prepare(args):
    if not args.model_path:
        uniq="Trained_output12"
        args.model_path = os.path.join("./output/",uniq)
        
    # Set up output folder
    print("Output is in : {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tnsr_wrtr = None
    if TENSORBOARD_FOUND:
        tnsr_wrtr = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tnsr_wrtr

def training_report(tnsr_wrtr, itr, l1_loss_value, total_loss, loss_function, elapsed, tst_itr, scn, render_func, render_args):
    log_training_metrics(tnsr_wrtr, itr, l1_loss_value, total_loss, elapsed)

    if itr in tst_itr:
        evaluate_models(tnsr_wrtr, itr, scn, render_func, render_args, tst_itr)

def log_training_metrics(tnsr_wrtr, itr, l1_loss_value, total_loss, elapsed):
    """ Log training metrics to TensorBoard. """
    if tnsr_wrtr:
        tnsr_wrtr.add_scalar('train_loss_patches/l1_loss', l1_loss_value.item(), itr)
        tnsr_wrtr.add_scalar('train_loss_patches/total_loss', total_loss.item(), itr)
        tnsr_wrtr.add_scalar('iter_time', elapsed, itr)

def evaluate_models(tnsr_wrtr, itr, scn, render_func, render_args, tst_itr):
    """ Evaluate the model on test and training datasets, and log the results. """
    torch.cuda.empty_cache()
    validation_configs = [
        {'name': 'test', 'cameras': scn.getTestCameras()},
        {'name': 'train', 'cameras': [scn.getTrainCameras()[idx % len(scn.getTrainCameras())] for idx in range(5, 30, 5)]}
    ]

    for config in validation_configs:
        evaluate_config(tnsr_wrtr, itr, scn, render_func, render_args, config, tst_itr)

    if tnsr_wrtr:
        tnsr_wrtr.add_histogram("scene/opacity_histogram", scn.gsns.get_opacity(), itr)
        tnsr_wrtr.add_scalar('total_points', len(scn.gsns.get_xyz()), itr)

def evaluate_config(tnsr_wrtr, itr, scn, render_func, render_args, config, tst_itr):
    """ Evaluate a specific configuration and log results. """
    if not config['cameras']:
        return

    l1_test, psnr_test = 0.0, 0.0
    idx = 0
    num_cameras = len(config['cameras'])

    # Use a while loop to iterate over the cameras
    while idx < num_cameras:
        viewpoint = config['cameras'][idx]
        image, gt_image = render_and_clamp_images(viewpoint, scn, render_func, render_args)

        # Log iamages to TensorBoard for the first few cameras
        if tnsr_wrtr and idx < 5:
            log_images(tnsr_wrtr, itr, config['name'], viewpoint.image_name, image, gt_image, tst_itr)

        # Accumulate L1 loss and PSNR values
        l1_test += l1_loss(image, gt_image).mean().item()
        psnr_test += psnr(image, gt_image).mean().item()

        # Increment the index
        idx += 1

    log_evaluation_metrics(tnsr_wrtr, itr, config['name'], l1_test, psnr_test, len(config['cameras']))

def render_and_clamp_images(viewpoint, scn, render_func, render_args):
    """ Render and clamp images for evaluation. """
    rendered_image = render_func(viewpoint, scn.gsns, *render_args)["render"]
    image = torch.clamp(rendered_image, 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    return image, gt_image

def log_images(tnsr_wrtr, itr, config_name, viewpoint_name, image, gt_image, tst_itr):
    """ Log rendered and ground truth images to TensorBoard. """
    image_log_tag = f"{config_name}_view_{viewpoint_name}/render"

    # Ensure the image tensor is in the correct format (add a batch dimension if needed)
    formatted_image = image[None]  # Add a batch dimension to the image tensor

    # Log the image to TensorBoard
    tnsr_wrtr.add_images(image_log_tag, formatted_image, global_step=itr)
    if itr == tst_itr[0]:
        gt_image_log_tag = f"{config_name}_view_{viewpoint_name}/ground_truth"
# Ensure the ground truth image tensor is in the correct format (add a batch dimension if needed)
        formatted_gt_image = gt_image[None]  # Add a batch dimension to the ground truth image tensor
        # Log the ground truth image to TensorBoard
        tnsr_wrtr.add_images(gt_image_log_tag, formatted_gt_image, global_step=itr)

def log_evaluation_metrics(tnsr_wrtr, itr, config_name, l1_test, psnr_test, num_cameras):
    """ Log evaluation metrics for a specific test configuration. """
    l1_test /= num_cameras
    psnr_test /= num_cameras
    print(f"\n[ITER {itr}] Evaluating {config_name}: L1 {l1_test:.6f} PSNR {psnr_test:.6f}")
    if tnsr_wrtr:
        # Calculate average L1 loss and PSNR over all cameras
        average_l1_loss = l1_test / num_cameras
        average_psnr = psnr_test / num_cameras

        # Log the metrics to TensorBoard using the calculated averages
        l1_loss_tag = f'{config_name}/loss_viewpoint - l1_loss'
        psnr_tag = f'{config_name}/loss_viewpoint - psnr'

        tnsr_wrtr.add_scalar(l1_loss_tag, average_l1_loss, itr)
        tnsr_wrtr.add_scalar(psnr_tag, average_psnr, itr)


def parse_arguments():
    psr1 = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(psr1)
    opt_params = OptimizationParams(psr1)
    pipe_params = PipelineParams(psr1)
    psr1.add_argument('--ip', type=str, default="127.0.0.1")
    psr1.add_argument('--port', type=int, default=6009)
    psr1.add_argument('--debug_from', type=int, default=-1)
    psr1.add_argument('--detect_anomaly', action='store_true', default=False)
    psr1.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    psr1.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    psr1.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    psr1.add_argument("--start_checkpoint", type=str, default=None)
    psr1.add_argument("--quiet", action="store_true")

    args = psr1.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    return args, model_params, opt_params, pipe_params

if __name__ == "__main__":
    args, model_params, opt_params, pipe_params = parse_arguments()
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_params = model_params.extract(args)
    opt_params = opt_params.extract(args)
    pipe_params = pipe_params.extract(args)
    training(model_params, opt_params, pipe_params, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
