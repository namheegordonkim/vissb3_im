from argparse import ArgumentParser
from dataclasses import dataclass
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Transitions
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from mujoco import mjx
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from utils.env_containers import EnvContainer
from utils.ppo import MyPPO
from utils.tree_utils import tree_stack
from utils.vecenv import MyVecEnv
from viz.visual_data import XMLVisualDataContainer
import jax
import jax.numpy as jp
import matplotlib
import numpy as np
import os
import pyvista as pv
import torch

# Disable XLA's default memory preallocation behavior to avoid memory issues.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@dataclass
class VisualArray:
    """
    Represents an array of meshes and actors for rendering visual components.

    Attributes:
        meshes (np.ndarray): Array containing mesh data for rendering.
        actors (np.ndarray): Array containing actor objects used for rendering the scene.
    """

    meshes: np.ndarray
    actors: np.ndarray


class AppState:
    """
    Maintains the state of the application, including the visual and environment components,
    as well as various settings for PPO, GAIL, and behavior cloning.

    Attributes:
        scene_meshes (np.ndarray): Scene's meshes used for visual rendering.
        scene_actors (np.ndarray): Scene's actors used for rendering the simulation.
        ghost_meshes (np.ndarray): Meshes for ghost visualizations used to display previous states.
        ghost_actors (np.ndarray): Actors for ghost visualizations.
        train_vecenv (MyVecEnv): Vectorized environment used for training.
        eval_vecenv (MyVecEnv): Vectorized environment used for evaluation.
        ppo (MyPPO): Proximal Policy Optimization agent used for training.
        dataset (dict): Collected dataset used for behavior cloning.
        bc (BC): Behavior cloning algorithm instance.
        gail (GAIL): Generative Adversarial Imitation Learning instance for adversarial training.
        Various other attributes for controlling the GUI state, training parameters, etc.
    """

    def __init__(
        self,
        scene_visuals: VisualArray,
        ghost_visuals: VisualArray,
        trail_visuals: VisualArray,
        train_vecenv: MyVecEnv,
        eval_vecenv: MyVecEnv,
        ppo: MyPPO,
        dataset: dict,
        bc: BC,
        gail: GAIL,
    ):
        self.scene_meshes = scene_visuals.meshes
        self.scene_actors = scene_visuals.actors
        self.ghost_meshes = ghost_visuals.meshes
        self.ghost_actors = ghost_visuals.actors
        self.trail_meshes = trail_visuals.meshes
        self.trail_actors = trail_visuals.actors
        self.train_vecenv = train_vecenv
        self.eval_vecenv = eval_vecenv
        self.ppo = ppo
        _, self.ppo_callback = self.ppo._setup_learn(int(1e6), None)  # Setup PPO learning callback.
        self.dataset = dataset
        self.bc = bc
        self.gail = gail

        # Initialize GUI state parameters
        self.bc_n_epochs = 1
        self.color_code = 0  # Color coding for different visual elements.
        self.data_playing = False
        self.dataset_frame = 0
        self.deterministic = False
        self.eval_obs = None
        self.eval_rewards = np.zeros((1, 1), dtype=float)
        self.first_time = True
        self.iter_i = 0
        self.iterating = False
        self.n_epochs = 1
        self.n_iters = 10
        self.play_mode = False
        self.pose_idx = 0
        self.reward_ratio = 0  # Ratio of GAIL reward to task reward.
        self.rollout_length = self.ppo.n_steps  # Number of steps in each rollout.
        self.show_axes = False
        self.show_ghost = True
        self.show_guides = True
        self.show_trails = True
        self.traj_frame = 0
        self.traj_idx = 0
        self.trajectory_t = 0
        self.trajectory_x = None


def setup_and_run_gui(pl: ImguiPlotter, app_state: AppState):
    """
    This function sets up and runs a graphical user interface (GUI) using imgui for controlling
    and visualizing reinforcement learning training and evaluation processes. It integrates both
    an imgui-based interface and a PyVista-based visualization panel for rendering 3D data.

    Args:
        pl (ImguiPlotter): A PyVista plotter used for rendering 3D visualizations within the GUI.
        app_state (AppState): The current application state holding training and evaluation environments,
                              neural networks, visual meshes, and various parameters for controlling the
                              reinforcement learning loop.

    """

    # Create imgui runner parameters for the GUI window
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "Viewer"  # Title of the GUI window
    runner_params.app_window_params.window_geometry.size = (1280, 720)  # Window size

    def gui():
        """
        This function defines the layout and interaction logic for the imgui interface. It controls
        the user inputs for resetting the environment, starting training, running evaluation tests,
        setting parameters for BC (Behavior Cloning), GAIL, PPO, and more. It also manages visualization
        elements for displaying the environment and training states.
        """

        # Apply a dark theme to the GUI
        hello_imgui.apply_theme(hello_imgui.ImGuiTheme_.imgui_colors_dark)

        # Get the current size of the imgui viewport (entire window area)
        viewport_size = imgui.get_window_viewport().size

        # ---------- PyVista portion ----------
        # Set the window size for the PyVista plotter and position it in the viewport
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))  # Half window size for PyVista
        imgui.set_next_window_pos(imgui.ImVec2(viewport_size.x // 2, 0))  # Position at the right half of the window
        imgui.set_next_window_bg_alpha(1.0)  # Set window transparency (fully opaque)

        # Create a new window for rendering the PyVista plotter content
        imgui.begin(
            "ImguiPlotter",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )
        # Render the PyVista plotter (3D environment visualization)
        pl.render_imgui()
        imgui.end()

        # ---------- GUI portion ----------
        # Set the window size for the control panel and position it in the viewport
        imgui.set_next_window_size(imgui.ImVec2(viewport_size.x // 2, viewport_size.y))  # Half window size for controls
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))  # Position at the left half of the window
        imgui.set_next_window_bg_alpha(1.0)  # Set window transparency (fully opaque)

        # Create a new window for the GUI controls
        imgui.begin(
            "Controls",
            flags=imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move,
        )

        # Section for controlling the evaluation environment
        imgui.text("Evaluation Environment Control")
        reset_clicked = imgui.button("Reset")  # Reset environment button

        # If this is the first time loading or the reset button is clicked, reset evaluation environment
        if app_state.first_time or reset_clicked:
            app_state.eval_obs = app_state.eval_vecenv.reset()

        # Checkbox for Play Mode and Deterministic settings
        imgui.same_line()
        changed, app_state.play_mode = imgui.checkbox("Play Mode", app_state.play_mode)
        imgui.same_line()
        changed, app_state.deterministic = imgui.checkbox("Deterministic", app_state.deterministic)  # If checked, the policy uses deterministic actions for evaluation

        imgui.separator()  # Horizontal line separator

        # Section for controlling the training environment
        imgui.text("Training Environment Control")
        imgui.text(f"# Parallel Environments: {app_state.train_vecenv.num_envs}")  # Display number of environments
        imgui.text(f"Rollout Length: {app_state.rollout_length}")  # Display the rollout length
        imgui.text(f"Eval Reward Stat: {app_state.eval_rewards.sum(-1).mean():.2f} +/- {app_state.eval_rewards.sum(-1).std():.2f}")  # Display reward stats

        # Slider for adjusting the ratio between GAIL reward and task reward
        changed, app_state.reward_ratio = imgui.slider_float("GAIL / Task Reward Ratio", app_state.reward_ratio, 0, 1)

        imgui.separator()

        # Buttons for triggering the rollout, learning, and testing processes
        rollout_clicked = imgui.button("Rollout")
        imgui.same_line()
        learn_clicked = imgui.button("Learn")
        imgui.same_line()
        test_clicked = imgui.button("Test")

        # Automatically trigger rollout, learn, and test when iterating
        if app_state.iterating:
            rollout_clicked = True
            learn_clicked = True
            test_clicked = True

        # Rollout process: collect data for PPO and GAIL training
        if rollout_clicked:
            app_state.ppo.env.venv.venv.trajectory = []
            app_state.ppo.collect_rollouts(
                app_state.ppo.env,
                app_state.ppo_callback,
                app_state.ppo.rollout_buffer,
                n_rollout_steps=app_state.rollout_length,
            )
            app_state.trajectory_x = tree_stack(app_state.ppo.env.venv.venv.trajectory, 1)

            # Combine GAIL's discriminator reward with the task's reward
            orig_reward = np.concatenate([app_state.ppo.ep_info_buffer], 0)
            disc_reward = app_state.ppo.rollout_buffer.rewards
            rewards = (1 - app_state.reward_ratio) * orig_reward + app_state.reward_ratio * disc_reward
            app_state.ppo.rollout_buffer.rewards = rewards
            app_state.ppo.rollout_buffer.compute_returns_and_advantage(app_state.ppo.values, app_state.ppo.dones)

        # Learn process: trigger PPO and GAIL training
        if learn_clicked:
            app_state.ppo.train()

            # GAIL discriminator training
            gen_trajs, ep_lens = app_state.gail.venv_buffering.pop_trajectories()
            app_state.gail._check_fixed_horizon(ep_lens)
            gen_samples = rollout.flatten_trajectories_with_rew(gen_trajs)
            app_state.gail._gen_replay_buffer.store(gen_samples)

            # Train the discriminator multiple times per iteration
            for _ in range(10):
                app_state.gail.train_disc()

        # Test process: evaluate the trained policy on the evaluation environment
        if test_clicked:
            app_state.eval_obs = app_state.eval_vecenv.reset()
            rewards = []
            for _ in range(app_state.rollout_length):
                action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=app_state.deterministic)[0]
                app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)
                rewards.append(reward)
            app_state.eval_rewards = np.stack(rewards, -1)

        # Slider to adjust the number of iterations for training
        changed, app_state.n_iters = imgui.slider_int("# Iterations", app_state.n_iters, 1, 100)

        # If iterating, keep track of the iteration count
        if app_state.iterating:
            app_state.iter_i += 1
            if app_state.iter_i >= app_state.n_iters:
                app_state.iterating = False

        # Button to start iterating
        iterate_clicked = imgui.button("Iterate")
        if iterate_clicked and not app_state.iterating:
            app_state.iterating = True
            app_state.iter_i = 0

        # Display a progress bar for the iteration process
        imgui.same_line()
        imgui.progress_bar(app_state.iter_i / app_state.n_iters, imgui.ImVec2(0, 0), f"{app_state.iter_i}/{app_state.n_iters}")

        # Checkbox to control visibility of the trails in the visualization
        changed, app_state.show_trails = imgui.checkbox("Show Trails", app_state.show_trails)

        # Radio buttons for selecting how to color the trails
        imgui.text("Trail Color Code")
        cc_radio_clicked1 = imgui.radio_button("Body", app_state.color_code == 0)
        if cc_radio_clicked1:
            app_state.color_code = 0
        imgui.same_line()
        cc_radio_clicked2 = imgui.radio_button("Step Reward", app_state.color_code == 1)
        if cc_radio_clicked2:
            app_state.color_code = 1
        imgui.same_line()
        cc_radio_clicked3 = imgui.radio_button("Cumulative Reward", app_state.color_code == 2)
        if cc_radio_clicked3:
            app_state.color_code = 2
        imgui.same_line()
        cc_radio_clicked4 = imgui.radio_button("Estimated Value", app_state.color_code == 3)
        if cc_radio_clicked4:
            app_state.color_code = 3
        imgui.same_line()
        cc_radio_clicked5 = imgui.radio_button("Advantage", app_state.color_code == 4)
        if cc_radio_clicked5:
            app_state.color_code = 4

        cc_radio_clicked = np.any([cc_radio_clicked1, cc_radio_clicked2, cc_radio_clicked3, cc_radio_clicked4, cc_radio_clicked5])

        imgui.separator()

        imgui.text("Trajectory Browser")
        changed, app_state.traj_idx = imgui.slider_int("Trajectory Index", app_state.traj_idx, 0, app_state.train_vecenv.num_envs)
        changed, app_state.traj_frame = imgui.slider_int("Frame", app_state.traj_frame, 0, app_state.rollout_length - 1)

        imgui.separator()

        imgui.text("Dataset Browser")
        changed, app_state.show_ghost = imgui.checkbox("Show Ghost", app_state.show_ghost)
        changed, app_state.dataset_frame = imgui.slider_int("Dataset Frame", app_state.dataset_frame, 0, app_state.dataset["infos/qpos"].shape[0])
        changed, app_state.data_playing = imgui.checkbox("Play", app_state.data_playing)

        imgui.separator()

        imgui.text("Behaviour Cloning Control")
        changed, app_state.bc_n_epochs = imgui.slider_int("# Epochs", app_state.bc_n_epochs, 1, 10)
        bc_clicked = imgui.button("Behaviour Clone")
        if bc_clicked:
            app_state.bc.train(n_epochs=app_state.bc_n_epochs)

        imgui.end()

        if app_state.data_playing:
            app_state.dataset_frame += 1
            if app_state.dataset_frame >= app_state.dataset["infos/qpos"].shape[0]:
                app_state.dataset_frame = 0

        rb_size = app_state.ppo.rollout_buffer.size()
        eval_state = app_state.eval_vecenv.state
        if app_state.play_mode:
            action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=app_state.deterministic)[0]
            app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)

        else:
            # Trails
            if app_state.show_trails and rb_size > 0:
                if rollout_clicked:
                    pos = np.array(app_state.trajectory_x.pos[app_state.traj_idx, :, 0])
                    quat = np.array(app_state.trajectory_x.rot[app_state.traj_idx, :, 0])
                    quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]
                    app_state.trail_meshes[0, 0].points = pos
                    app_state.trail_meshes[0, 0].lines = pv.MultipleLines(points=pos).lines

                app_state.trail_actors[0, 0].SetVisibility(True)
            else:
                app_state.trail_actors[0, 0].SetVisibility(False)

        # Animating
        for i in range(len(app_state.scene_actors)):
            if app_state.play_mode or app_state.trajectory_x is None:
                pos = np.array(eval_state.pipeline_state.x.pos[0, i])
                quat = np.array(eval_state.pipeline_state.x.rot[0, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m
            else:
                pos = np.array(app_state.trajectory_x.pos[app_state.traj_idx, app_state.traj_frame, i])
                quat = np.array(app_state.trajectory_x.rot[app_state.traj_idx, app_state.traj_frame, i])
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.scene_actors[i].user_matrix = m

        if app_state.show_ghost:
            qpos = jp.array(app_state.dataset["infos/qpos"][app_state.dataset_frame])
            qvel = jp.array(app_state.dataset["infos/qvel"][app_state.dataset_frame])
            data = mjx.make_data(app_state.eval_vecenv.env_container.env.sys)
            data = data.replace(qpos=qpos, qvel=qvel)
            data = jax.jit(mjx.forward)(app_state.eval_vecenv.env_container.env.sys, data)
            poses = np.array(data.xpos[1:])
            quats = np.array(data.xquat[1:])
            for i in range(len(app_state.ghost_actors)):
                pos = poses[i]
                quat = quats[i]
                quat[..., [0, 1, 2, 3]] = quat[..., [1, 2, 3, 0]]

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat).as_matrix()
                app_state.ghost_actors[i].user_matrix = m
                app_state.ghost_actors[i].SetVisibility(True)
        else:
            for i in range(len(app_state.ghost_actors)):
                app_state.ghost_actors[i].SetVisibility(False)

        # Coloring
        if app_state.show_trails and (app_state.iterating or cc_radio_clicked or rollout_clicked):
            colors = np.array([[0.5, 0.5, 0.5]]).repeat(app_state.trail_meshes[0, 0].n_points, 0)
            if app_state.color_code == 1:
                colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.rewards[..., app_state.traj_idx])[..., :3]
            elif app_state.color_code == 2:
                colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.returns.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
            elif app_state.color_code == 3:
                colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.values.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
            elif app_state.color_code == 4:
                colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.advantages.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
            app_state.trail_meshes[0, 0].point_data["color"] = colors

        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main(args):
    env_name = "halfcheetah"
    mjcf_path = "brax/envs/assets/half_cheetah.xml"

    backend = "mjx"
    batch_size = 1024
    episode_length = 256
    train_env_container = EnvContainer(env_name, backend, batch_size, True, episode_length)
    eval_env_container = EnvContainer(env_name, backend, 16, False, episode_length)
    train_vecenv = MyVecEnv(train_env_container, seed=0)
    eval_vecenv = MyVecEnv(eval_env_container, seed=0)

    if args.policy_path is not None:
        ppo = MyPPO.load(args.policy_path, train_vecenv)
    else:
        ppo = MyPPO(
            "MlpPolicy",
            train_vecenv,
            policy_kwargs={"log_std_init": -1, "net_arch": [64, 64]},
            learning_rate=3e-4,
            max_grad_norm=0.1,
            batch_size=16384,
            n_epochs=10,
            n_steps=episode_length,
            stats_window_size=episode_length,
        )

    dataset = torch.load("data/dataset.pth")
    transitions = Transitions(
        obs=dataset["observations"][:-1],
        acts=dataset["actions"][:-1],
        infos=dataset["infos/qpos"][:-1],
        next_obs=dataset["observations"][1:],
        dones=dataset["terminals"][:-1] | dataset["timeouts"][:-1],
    )

    bc = BC(
        policy=ppo.policy,
        observation_space=train_vecenv.observation_space,
        action_space=train_vecenv.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        batch_size=16384,
        optimizer_kwargs={"lr": 3e-4},
    )
    reward_net = BasicRewardNet(
        observation_space=train_vecenv.observation_space,
        action_space=train_vecenv.action_space,
        normalize_input_layer=RunningNorm,
        use_state=True,
        use_action=False,
        use_next_state=True,
        use_done=False,
    )
    gail = GAIL(
        demonstrations=transitions,
        reward_net=reward_net,
        venv=train_vecenv,
        gen_algo=ppo,
        demo_batch_size=16384,
    )

    pl = ImguiPlotter()
    plane_height = 0.0
    if env_name == "inverted_pendulum":
        plane_height = -0.5
    plane = pv.Plane(center=(0, 0, plane_height), direction=(0, 0, 1), i_size=100, j_size=10)

    pl.add_mesh(plane, show_edges=True)
    pl.add_axes()
    pl.camera.position = (0, -10, 0.1)
    pl.camera.focus = (0, 0, 0)
    pl.camera.up = (0, 0, 1)
    # pl.enable_shadows()
    visual = XMLVisualDataContainer(mjcf_path)
    n = len(visual.meshes)
    visuals = []
    for i in range(2):
        meshes = np.empty((n,), dtype=object)
        actors = np.empty((n,), dtype=object)
        for j, mesh in enumerate(visual.meshes):
            mesh = mesh.copy()  # clone so we don't mutate the original
            if i == 1:
                color = [1.0, 0.0, 0.0]
            else:
                color = [1.0, 1.0, 1.0]
            mesh.cell_data["color"] = np.array([color]).repeat(mesh.n_cells, 0)
            actor = pl.add_mesh(mesh, scalars="color", rgb=True, show_scalar_bar=False)
            meshes[j] = mesh
            actors[j] = actor
        visuals.append(VisualArray(meshes, actors))
    scene_visuals, ghost_visuals = visuals

    trail_meshes = np.empty((1, 1), dtype=object)
    trail_actors = np.empty((1, 1), dtype=object)
    color = [1.0, 1.0, 1.0]
    trail_mesh = pv.MultipleLines(points=np.zeros((2, 3)))
    trail_mesh.point_data["color"] = np.array([color]).repeat(trail_mesh.n_points, 0) * 1
    trail_actor = pl.add_mesh(trail_mesh, rgb=True, scalars="color", show_scalar_bar=False)
    trail_meshes[0, 0] = trail_mesh
    trail_actors[0, 0] = trail_actor
    trail_actor.SetVisibility(False)

    trail_visuals = VisualArray(trail_meshes, trail_actors)

    # Run the GUI
    app_state = AppState(scene_visuals, ghost_visuals, trail_visuals, train_vecenv, eval_vecenv, ppo, dataset, bc, gail)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--policy_path", type=str)
    args = parser.parse_args()

    main(args)
