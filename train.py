
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, chpt, debug_from):
    # Initialize training parameters and load checkpoint if available
    tnsr_wrtr = prepare_output_and_logger(dataset)
    gsns = GaussianModel(dataset.sh_degree)
    scn = Scene(dataset, gsns)
    gsns.training_setup(opt)
    itr1 = 0
    if chpt:
        model_params, itr1 = torch.load(chpt)
        gsns.restore(model_params, opt)
    bckgrd = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    # Initialize time tracking for performance metrics
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    progress_bar = tqdm(range(itr1, opt.iterations), desc="Training progress")

    ema_loss_for_log = 0.0
    viewpoint_stack = None

    # Main training loop
    scd_itr=itr1+1
    scd_param=opt.iterations + 1
    for iteration in range(scd_itr, scd_param):
        handle_network_gui(iteration, gsns, pipe, bckgrd, dataset, network_gui)

        iter_start.record()
        if iteration % 1000 == 0:
            gsns.oneupSHdegree()

        viewpoint_cam = select_camera(viewpoint_stack, scn)

        if iteration == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gsns, pipe, bckgrd)
        loss = compute_loss(render_pkg, viewpoint_cam, opt)

        loss.backward()
        iter_end.record()

        # Handle logging, checkpointing, and saving
        manage_training_cycle(iteration, gsns, scn, loss, ema_loss_for_log, tnsr_wrtr, saving_iterations, checkpoint_iterations)

        # Update progress bar display
        update_progress_bar(progress_bar, iteration, ema_loss_for_log, opt.iterations)

    # Final clean-up after training loop
    progress_bar.close()
    print("\nTraining complete.")
def update_progress_bar(progress_bar, iteration, ema_loss_for_log, total_iterations):
    """ Update the progress bar with the current iteration and estimated moving average loss. """
    progress_bar.set_description(f"Training progress (iteration {iteration}/{total_iterations})")
    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.6f}"})
    progress_bar.update(1)

def handle_network_gui(iteration, gsns, pipe, bckgrd, dataset, network_gui):
    """ Manage GUI network interactions if connected. """
    if network_gui.conn is None:
        network_gui.try_connect()
    while network_gui.conn:
        try:
            net_image_bytes = handle_custom_camera_view(gsns, pipe, bckgrd, dataset, network_gui)
            network_gui.send(net_image_bytes, dataset.source_path)
            if not keep_training_alive(iteration, opt.iterations, network_gui):
                break
        except Exception as e:
            network_gui.conn = None

def handle_custom_camera_view(gsns, pipe, bckgrd, dataset, network_gui):
    """ Process custom camera view from network GUI. """
    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
    if custom_cam:
        net_image = render(custom_cam, gsns, pipe, bckgrd, scaling_modifier)["render"]
        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        return net_image_bytes
    return None

def keep_training_alive(iteration, total_iterations, network_gui):
    """ Determine whether to continue training based on GUI input and iteration count. """
    return network_gui.receive_training_signal() and (iteration < total_iterations or network_gui.keep_alive())

def select_camera(viewpoint_stack, scn):
    """ Select a random camera from the training set for rendering. """
    if not viewpoint_stack:
        viewpoint_stack = scn.getTrainCameras().copy()
    return viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

def compute_loss(render_pkg, viewpoint_cam, opt):
    """ Compute the combined loss for the current iteration. """
    image, gt_image = render_pkg["render"], viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

def manage_training_cycle(iteration, gsns, scn, loss, ema_loss_for_log, tnsr_wrtr, saving_iterations, checkpoint_iterations):
    """ Handle logging, checkpointing, and saving within the training loop. """
    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
    if iteration in saving_iterations:
        print("\n[ITER {}] Saving gsns".format(iteration))
        scn.save(iteration)
    if iteration in checkpoint_iterations:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save


def prepare_output_and_logger(args):
    if not args.model_path:
        uniq="Trained_output"
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

def training_report(tnsr_wrtr, iteration, l1_loss_value, total_loss, loss_function, elapsed, testing_iterations, scn, render_func, render_args):
    log_training_metrics(tnsr_wrtr, iteration, l1_loss_value, total_loss, elapsed)

    if iteration in testing_iterations:
        evaluate_models(tnsr_wrtr, iteration, scn, render_func, render_args, testing_iterations)

def log_training_metrics(tnsr_wrtr, iteration, l1_loss_value, total_loss, elapsed):
    """ Log training metrics to TensorBoard. """
    if tnsr_wrtr:
        tnsr_wrtr.add_scalar('train_loss_patches/l1_loss', l1_loss_value.item(), iteration)
        tnsr_wrtr.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tnsr_wrtr.add_scalar('iter_time', elapsed, iteration)

def evaluate_models(tnsr_wrtr, iteration, scn, render_func, render_args, testing_iterations):
    """ Evaluate the model on test and training datasets, and log the results. """
    torch.cuda.empty_cache()
    validation_configs = [
        {'name': 'test', 'cameras': scn.getTestCameras()},
        {'name': 'train', 'cameras': [scn.getTrainCameras()[idx % len(scn.getTrainCameras())] for idx in range(5, 30, 5)]}
    ]

    for config in validation_configs:
        evaluate_config(tnsr_wrtr, iteration, scn, render_func, render_args, config, testing_iterations)

    if tnsr_wrtr:
        tnsr_wrtr.add_histogram("scene/opacity_histogram", scn.gsns.get_opacity(), iteration)
        tnsr_wrtr.add_scalar('total_points', len(scn.gsns.get_xyz()), iteration)

def evaluate_config(tnsr_wrtr, iteration, scn, render_func, render_args, config, testing_iterations):
    """ Evaluate a specific configuration and log results. """
    if not config['cameras']:
        return

    l1_test, psnr_test = 0.0, 0.0
    for idx, viewpoint in enumerate(config['cameras']):
        image, gt_image = render_and_clamp_images(viewpoint, scn, render_func, render_args)
        if tnsr_wrtr and idx < 5:
            log_images(tnsr_wrtr, iteration, config['name'], viewpoint.image_name, image, gt_image, testing_iterations)
        l1_test += l1_loss(image, gt_image).mean().item()
        psnr_test += psnr(image, gt_image).mean().item()

    log_evaluation_metrics(tnsr_wrtr, iteration, config['name'], l1_test, psnr_test, len(config['cameras']))

def render_and_clamp_images(viewpoint, scn, render_func, render_args):
    """ Render and clamp images for evaluation. """
    rendered_image = render_func(viewpoint, scn.gsns, *render_args)["render"]
    image = torch.clamp(rendered_image, 0.0, 1.0)
    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
    return image, gt_image

def log_images(tnsr_wrtr, iteration, config_name, viewpoint_name, image, gt_image, testing_iterations):
    """ Log rendered and ground truth images to TensorBoard. """
    tnsr_wrtr.add_images(f"{config_name}_view_{viewpoint_name}/render", image[None], global_step=iteration)
    if iteration == testing_iterations[0]:
        tnsr_wrtr.add_images(f"{config_name}_view_{viewpoint_name}/ground_truth", gt_image[None], global_step=iteration)

def log_evaluation_metrics(tnsr_wrtr, iteration, config_name, l1_test, psnr_test, num_cameras):
    """ Log evaluation metrics for a specific test configuration. """
    l1_test /= num_cameras
    psnr_test /= num_cameras
    print(f"\n[ITER {iteration}] Evaluating {config_name}: L1 {l1_test:.6f} PSNR {psnr_test:.6f}")
    if tnsr_wrtr:
        tnsr_wrtr.add_scalar(f'{config_name}/loss_viewpoint - l1_loss', l1_test, iteration)
        tnsr_wrtr.add_scalar(f'{config_name}/loss_viewpoint - psnr', psnr_test, iteration)


def parse_arguments():
    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipe_params = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
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
