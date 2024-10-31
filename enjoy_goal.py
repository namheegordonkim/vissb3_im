from argparse import ArgumentParser
from dataclasses import dataclass
from imgui_bundle import immapp
from imgui_bundle._imgui_bundle import imgui, hello_imgui
from pyvista_imgui import ImguiPlotter
from scipy.spatial.transform import Rotation
from utils.env_containers import EnvContainer
from utils.ppo import MyPPO
from utils.tree_utils import tree_stack
from utils.vecenv import MyVecEnv
from viz.visual_data import XMLVisualDataContainer
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
        trail_visuals: VisualArray,
        guide_visuals: VisualArray,
        train_vecenv: MyVecEnv,
        eval_vecenv: MyVecEnv,
        ppo: MyPPO,
    ):
        self.scene_meshes = scene_visuals.meshes
        self.scene_actors = scene_visuals.actors
        self.trail_meshes = trail_visuals.meshes
        self.trail_actors = trail_visuals.actors
        self.guide_meshes = guide_visuals.meshes
        self.guide_actors = guide_visuals.actors
        self.train_vecenv = train_vecenv
        self.eval_vecenv = eval_vecenv
        self.ppo = ppo
        _, self.ppo_callback = self.ppo._setup_learn(int(1e6), None)  # Setup PPO learning callback.

        # Initialize GUI state parameters
        self.color_code = 0  # Color coding for different visual elements.
        self.curriculum_stage = 1
        self.data_playing = False
        self.dataset_frame = 0
        self.deterministic = False
        self.eval_obs = None
        self.eval_rewards = np.zeros((1, 1), dtype=float)
        self.eval_trajectory_x = None
        self.first_time = True
        self.iter_i = 0
        self.iterating = False
        self.n_epochs = 1
        self.n_iters = 20
        self.play_mode = False
        self.policy_log_std = -1 * np.ones(2)
        self.pose_idx = 0
        self.reward_ratio = 0  # Ratio of GAIL reward to task reward.
        self.rollout_length = self.ppo.n_steps  # Number of steps in each rollout.
        self.show_axes = False
        self.show_ghost = True
        self.show_guides = True
        self.show_trails = True
        self.trail_source = 0
        self.train_trajectory_x = None
        self.train_trajectory_x = None
        self.traj_frame = 0
        self.traj_idx = 0
        self.trajectory_t = 0


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

        # Refactored stuff to reuse over and over
        center_offset = np.array([[(10 - app_state.curriculum_stage) * 0.02, 0, 0]])
        old_radius = 0.01
        new_radius = 0.01 + app_state.curriculum_stage * 0.02
        radius_scale = new_radius / old_radius

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
            app_state.eval_vecenv.reset()
            q = app_state.eval_vecenv.state.pipeline_state.q
            qd = app_state.eval_vecenv.state.pipeline_state.qd
            q = q.at[..., 2:].set(q[..., 2:] * radius_scale + center_offset[..., :2])
            new_first_pipeline_state = app_state.eval_vecenv.env_container.jit_env_pipeline_init(q, qd)
            new_first_obs = app_state.eval_vecenv.env_container.jit_env_get_obs(new_first_pipeline_state)
            app_state.eval_obs = new_first_obs
            app_state.eval_vecenv.state = app_state.eval_vecenv.state.replace(pipeline_state=new_first_pipeline_state, obs=new_first_obs)
            app_state.eval_vecenv.trajectory = [app_state.eval_vecenv.state.pipeline_state.x]

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
        # changed, app_state.reward_ratio = imgui.slider_float("GAIL / Task Reward Ratio", app_state.reward_ratio, 0, 1)

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

        if rollout_clicked:
            # Perform a reset based on curriculum
            app_state.train_vecenv.reset()
            q = app_state.train_vecenv.state.pipeline_state.q
            qd = app_state.train_vecenv.state.pipeline_state.qd
            q = q.at[..., 2:].set(q[..., 2:] * radius_scale + center_offset[..., :2])

            new_first_pipeline_state = app_state.train_vecenv.env_container.jit_env_pipeline_init(q, qd)
            new_first_obs = app_state.train_vecenv.env_container.jit_env_get_obs(new_first_pipeline_state)

            app_state.train_vecenv.state = app_state.train_vecenv.state.replace(pipeline_state=new_first_pipeline_state, obs=new_first_obs)
            app_state.train_vecenv.trajectory = [app_state.train_vecenv.state.pipeline_state.x]
            app_state.train_vecenv.state.info["first_pipeline_state"] = new_first_pipeline_state
            app_state.train_vecenv.state.info["first_obs"] = new_first_obs
            app_state.ppo._last_obs = np.array(new_first_obs)

            app_state.ppo.policy.log_std.data[:] = torch.as_tensor(app_state.policy_log_std, dtype=torch.float)
            app_state.ppo.collect_rollouts(
                app_state.train_vecenv,
                app_state.ppo_callback,
                app_state.ppo.rollout_buffer,
                n_rollout_steps=app_state.rollout_length,
            )
            app_state.train_trajectory_x = tree_stack(app_state.train_vecenv.trajectory, 1)

        if learn_clicked:
            app_state.ppo.train()
            app_state.policy_log_std = app_state.ppo.policy.log_std.data.detach().cpu().numpy()

        if test_clicked:
            app_state.eval_vecenv.reset()
            q = app_state.eval_vecenv.state.pipeline_state.q
            qd = app_state.eval_vecenv.state.pipeline_state.qd
            q = q.at[..., 2:].set(q[..., 2:] * radius_scale + center_offset[..., :2])
            new_first_pipeline_state = app_state.eval_vecenv.env_container.jit_env_pipeline_init(q, qd)
            new_first_obs = app_state.eval_vecenv.env_container.jit_env_get_obs(new_first_pipeline_state)
            app_state.eval_obs = new_first_obs
            app_state.eval_vecenv.state = app_state.eval_vecenv.state.replace(pipeline_state=new_first_pipeline_state, obs=new_first_obs)
            app_state.eval_vecenv.trajectory = [app_state.eval_vecenv.state.pipeline_state.x]

            rewards = []
            for _ in range(app_state.rollout_length):
                action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=True)[0]
                app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)
                rewards.append(reward)
            app_state.eval_rewards = np.stack(rewards, -1)
            app_state.eval_trajectory_x = tree_stack(app_state.eval_vecenv.trajectory, 1)

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

        imgui.separator()

        # Checkbox to control visibility of the trails in the visualization
        st_checked, app_state.show_trails = imgui.checkbox("Show Trails", app_state.show_trails)

        # Radio buttons for selecting how to color the trails
        imgui.text("Trail Color Code")
        cc_radio_clicked0, app_state.color_code = imgui.radio_button("Body", app_state.color_code, 0)
        imgui.same_line()
        cc_radio_clicked1, app_state.color_code = imgui.radio_button("Step Reward", app_state.color_code, 1)
        imgui.same_line()
        cc_radio_clicked2, app_state.color_code = imgui.radio_button("Cumulative Reward", app_state.color_code, 2)
        imgui.same_line()
        cc_radio_clicked3, app_state.color_code = imgui.radio_button("Estimated Value", app_state.color_code, 3)
        imgui.same_line()
        cc_radio_clicked4, app_state.color_code = imgui.radio_button("Advantage", app_state.color_code, 4)
        cc_radio_clicked = np.any([cc_radio_clicked0, cc_radio_clicked1, cc_radio_clicked2, cc_radio_clicked3, cc_radio_clicked4])

        imgui.separator()

        imgui.text("Trail Source")
        ts_radio_clicked1, app_state.trail_source = imgui.radio_button("Train", app_state.trail_source, 0)
        imgui.same_line()
        ts_radio_clicked2, app_state.trail_source = imgui.radio_button("Eval", app_state.trail_source, 1)
        ts_radio_clicked = ts_radio_clicked1 or ts_radio_clicked2

        imgui.text("Trajectory Browser")
        traj_idx_ceiling = app_state.train_vecenv.num_envs - 1 if app_state.trail_source == 0 else app_state.eval_vecenv.num_envs - 1
        traj_idx_changed, app_state.traj_idx = imgui.slider_int("Trajectory Index", app_state.traj_idx, 0, traj_idx_ceiling)
        changed, app_state.traj_frame = imgui.slider_int("Frame", app_state.traj_frame, 0, app_state.rollout_length)

        imgui.separator()

        imgui.text("Curriculum Control")
        imgui.checkbox("Show Curriculum Guide", True)
        changed, app_state.curriculum_stage = imgui.slider_int("Stage", app_state.curriculum_stage, 0, 10)
        for i in range(app_state.policy_log_std.shape[0]):
            changed, app_state.policy_log_std[i] = imgui.slider_float(f"Policy Logstd {i}", app_state.policy_log_std[i].item(), -5, 0)

        imgui.end()

        rb_size = app_state.ppo.rollout_buffer.size()
        eval_state = app_state.eval_vecenv.state
        if app_state.play_mode:
            action = app_state.ppo.policy.predict(app_state.eval_obs, deterministic=app_state.deterministic)[0]
            app_state.eval_obs, reward, done, _ = app_state.eval_vecenv.step(action)

        # Trails
        for i in range(1):
            if app_state.show_trails:
                offset = np.array([[0.11, 0, 0]])
                if rollout_clicked or test_clicked or traj_idx_changed or ts_radio_clicked or st_checked:
                    if app_state.trail_source == 0 and app_state.train_trajectory_x is not None:
                        pos = np.array(app_state.train_trajectory_x.pos[app_state.traj_idx, 1:, -2])
                        quat = np.array(app_state.train_trajectory_x.rot[app_state.traj_idx, 1:, -2])
                        pos = Rotation.from_quat(quat, scalar_first=True).apply(offset) + pos
                        app_state.trail_meshes[i, 0].points = pos
                        app_state.trail_meshes[i, 0].lines = pv.MultipleLines(points=pos).lines
                        app_state.trail_actors[i, 0].SetVisibility(True)

                    elif app_state.trail_source == 1 and app_state.eval_trajectory_x is not None:
                        pos = np.array(app_state.eval_trajectory_x.pos[app_state.traj_idx, 1:, -2])
                        quat = np.array(app_state.eval_trajectory_x.rot[app_state.traj_idx, 1:, -2])
                        pos = Rotation.from_quat(quat, scalar_first=True).apply(offset) + pos
                        app_state.trail_meshes[i, 0].points = pos
                        app_state.trail_meshes[i, 0].lines = pv.MultipleLines(points=pos).lines
                        app_state.trail_actors[i, 0].SetVisibility(True)
                    else:
                        app_state.trail_actors[i, 0].SetVisibility(False)
            else:
                app_state.trail_actors[i, 0].SetVisibility(False)

        # Animating
        for i in range(len(app_state.scene_actors)):
            if app_state.play_mode or (app_state.train_trajectory_x is None and app_state.eval_trajectory_x is None):
                pos = np.array(eval_state.pipeline_state.x.pos[0, i])
                quat = np.array(eval_state.pipeline_state.x.rot[0, i])

                m = np.eye(4)
                m[:3, 3] = pos
                m[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
                app_state.scene_actors[i].user_matrix = m
            else:
                if app_state.trail_source == 0 and app_state.train_trajectory_x is not None:
                    pos = np.array(app_state.train_trajectory_x.pos[app_state.traj_idx, app_state.traj_frame, i])
                    quat = np.array(app_state.train_trajectory_x.rot[app_state.traj_idx, app_state.traj_frame, i])

                    m = np.eye(4)
                    m[:3, 3] = pos
                    m[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
                    app_state.scene_actors[i].user_matrix = m

                elif app_state.trail_source == 1 and app_state.eval_trajectory_x is not None:
                    pos = np.array(app_state.eval_trajectory_x.pos[app_state.traj_idx, app_state.traj_frame, i])
                    quat = np.array(app_state.eval_trajectory_x.rot[app_state.traj_idx, app_state.traj_frame, i])

                    m = np.eye(4)
                    m[:3, 3] = pos
                    m[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
                    app_state.scene_actors[i].user_matrix = m

        # Coloring
        for i in range(1):
            if app_state.show_trails and (app_state.iterating or cc_radio_clicked or rollout_clicked or test_clicked or traj_idx_changed or ts_radio_clicked):
                color = [1.0, 1.0, 1.0]
                colors = np.array([color]).repeat(app_state.trail_meshes[i, 0].n_points, 0)
                if app_state.trail_source == 0:
                    if app_state.color_code == 1:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.rewards[..., app_state.traj_idx])[..., :3]
                    elif app_state.color_code == 2:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.returns.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
                    elif app_state.color_code == 3:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.values.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
                    elif app_state.color_code == 4:
                        colors = matplotlib.colormaps["viridis"](app_state.ppo.rollout_buffer.advantages.reshape(app_state.rollout_length, -1)[..., app_state.traj_idx])[..., :3]
                app_state.trail_meshes[i, 0].point_data["color"] = colors

        # Curriculum Guide
        for i in range(1):
            if app_state.show_guides:
                app_state.guide_actors[i, 0].SetVisibility(True)
                app_state.guide_meshes[i, 0].points = pv.Cylinder(radius=1, direction=(0.0, 0.0, 0.1), height=0.01).points
                app_state.guide_meshes[i, 0].points[..., :2] *= new_radius
                app_state.guide_meshes[i, 0].points += center_offset
            else:
                app_state.guide_actors[i, 0].SetVisibility(False)

        app_state.first_time = False

    runner_params.callbacks.show_gui = gui
    runner_params.imgui_window_params.default_imgui_window_type = hello_imgui.DefaultImGuiWindowType.no_default_window
    immapp.run(runner_params=runner_params)


def main(args):
    env_name = "reacher"
    mjcf_path = "brax/envs/assets/reacher.xml"

    backend = "mjx"
    batch_size = 1024
    episode_length = 256
    train_reset_every = 256

    # train_env_container = EnvContainer(env_name, backend, batch_size, True, episode_length)
    train_env_container = EnvContainer(env_name, backend, batch_size, True, train_reset_every)
    eval_env_container = EnvContainer(env_name, backend, 16, False, episode_length)
    train_vecenv = MyVecEnv(train_env_container, seed=0)
    eval_vecenv = MyVecEnv(eval_env_container, seed=0)

    if args.policy_path is not None:
        ppo = MyPPO.load(args.policy_path, train_vecenv)
    else:
        ppo = MyPPO(
            "MlpPolicy",
            train_vecenv,
            policy_kwargs={"log_std_init": -1, "net_arch": [256, 256], "activation_fn": torch.nn.ReLU},
            learning_rate=3e-4,
            max_grad_norm=0.1,
            batch_size=16384,
            n_epochs=10,
            n_steps=episode_length,
            stats_window_size=episode_length,
        )

    pl = ImguiPlotter()
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=1, j_size=1)

    pl.add_mesh(plane, show_edges=True)
    pl.add_axes()
    pl.camera.position = (0, 0, 1.0)
    pl.camera.focus = (0, 0, 0)
    pl.camera.up = (1, 0, 0)
    # pl.enable_shadows()
    visual = XMLVisualDataContainer(mjcf_path)
    n = len(visual.meshes)
    meshes = np.empty((n,), dtype=object)
    actors = np.empty((n,), dtype=object)
    for j, mesh in enumerate(visual.meshes):
        mesh = mesh.copy()  # clone so we don't mutate the original
        if j == len(visual.meshes) - 1:  # target in reacher
            color = [1.0, 0.0, 0.0]
        else:
            color = [1.0, 1.0, 1.0]
        mesh.cell_data["color"] = np.array([color]).repeat(mesh.n_cells, 0)
        actor = pl.add_mesh(mesh, scalars="color", rgb=True, show_scalar_bar=False)
        meshes[j] = mesh
        actors[j] = actor
    scene_visuals = VisualArray(meshes, actors)

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

    guide_meshes = np.empty((1, 1), dtype=object)
    guide_actors = np.empty((1, 1), dtype=object)
    color = [0.0, 1.0, 0.0]
    guide_mesh = pv.Cylinder(radius=1, direction=(0.0, 0.0, 0.1), height=0.01)
    guide_mesh.cell_data["color"] = np.array([color]).repeat(guide_mesh.n_cells, 0)
    guide_actor = pl.add_mesh(guide_mesh, scalars="color", rgb=True, show_scalar_bar=False, opacity=0.5)
    guide_meshes[0, 0] = guide_mesh
    guide_actors[0, 0] = guide_actor
    guide_visuals = VisualArray(guide_meshes, guide_actors)

    # Run the GUI
    app_state = AppState(scene_visuals, trail_visuals, guide_visuals, train_vecenv, eval_vecenv, ppo)
    setup_and_run_gui(pl, app_state)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--policy_path", type=str)
    args = parser.parse_args()

    main(args)
