import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st

import config_io
import runner

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDED_DATA_DIR = os.path.join(REPO_ROOT, "recorded_data")
REF_TRAJ_DIR = os.path.join(REPO_ROOT, "lti_system", "reference_trajectories")
DIST_SEQ_DIR = os.path.join(REPO_ROOT, "lti_system", "disturbance_sequences")

st.set_page_config(page_title="DDSMPC Control Panel", layout="wide")


def list_subdirs(path):
    return sorted(
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    )


def matrix_editor(label, matrix, key):
    df = pd.DataFrame(matrix)
    edited = st.data_editor(df, key=key, num_rows="fixed")
    return edited.to_numpy().tolist()


if "config" not in st.session_state:
    st.session_state.config = config_io.load_config()

cfg = st.session_state.config

st.title("Data-Driven Stochastic MPC — Control Panel")

tab_config, tab_run = st.tabs(["Configuration", "Run"])

with tab_config:
    st.subheader("Data storage")
    col1, col2 = st.columns(2)
    with col1:
        cfg["data_storage"]["record_data"] = st.checkbox(
            "Record data", value=cfg["data_storage"]["record_data"])
    with col2:
        cfg["data_storage"]["folder_name"] = st.text_input(
            "Folder name", value=cfg["data_storage"]["folder_name"])

    if cfg["data_storage"]["record_data"]:
        target = os.path.join(RECORDED_DATA_DIR, cfg["data_storage"]["folder_name"])
        if os.path.isdir(target):
            st.warning(
                f"`recorded_data/{cfg['data_storage']['folder_name']}` already "
                "exists. Running now would crash (the underlying code refuses "
                "to overwrite existing runs)."
            )
            suggestion = (
                f"{cfg['data_storage']['folder_name']}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            if st.button(f"Use suggested name: {suggestion}"):
                cfg["data_storage"]["folder_name"] = suggestion
                st.rerun()

    st.subheader("LTI system")
    st.caption("A (system) matrix")
    cfg["lti_system"]["a_system_matrix"] = matrix_editor(
        "A", cfg["lti_system"]["a_system_matrix"], "a_matrix")
    st.caption("B (input) matrix")
    cfg["lti_system"]["b_input_matrix"] = matrix_editor(
        "B", cfg["lti_system"]["b_input_matrix"], "b_matrix")
    st.caption("Initial state x0")
    cfg["lti_system"]["x_initial_state"] = matrix_editor(
        "x0", cfg["lti_system"]["x_initial_state"], "x0_matrix")

    st.caption("Disturbance type per state")
    dist_options = ["gaussian", "uniform", "triangular", "lognormal"]
    new_types = []
    cols = st.columns(len(cfg["lti_system"]["types_of_disturbances"]))
    for i, (col, current) in enumerate(
        zip(cols, cfg["lti_system"]["types_of_disturbances"])
    ):
        with col:
            new_types.append(
                st.selectbox(
                    f"State {i + 1}", dist_options,
                    index=dist_options.index(current), key=f"dist_type_{i}",
                )
            )
    cfg["lti_system"]["types_of_disturbances"] = new_types

    st.subheader("Simulation parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        cfg["risk_param"] = st.number_input(
            "Risk parameter", value=float(cfg["risk_param"]),
            min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
        cfg["number_of_measurements"] = st.number_input(
            "Number of measurements", value=int(cfg["number_of_measurements"]),
            min_value=1, step=1)
    with col2:
        ref_traj_options = list_subdirs(REF_TRAJ_DIR)
        cfg["reference_trajectory"] = st.selectbox(
            "Reference trajectory", ref_traj_options,
            index=ref_traj_options.index(cfg["reference_trajectory"])
            if cfg["reference_trajectory"] in ref_traj_options else 0)
        dist_seq_options = list_subdirs(DIST_SEQ_DIR)
        cfg["disturbance_sequence"] = st.selectbox(
            "Disturbance sequence", dist_seq_options,
            index=dist_seq_options.index(cfg["disturbance_sequence"])
            if cfg["disturbance_sequence"] in dist_seq_options else 0)
    with col3:
        cfg["add_measurement_noise"] = st.checkbox(
            "Add measurement noise", value=cfg["add_measurement_noise"])
        cfg["number_of_inputs"] = st.number_input(
            "Number of inputs (trajectory length)",
            value=int(cfg["number_of_inputs"]), min_value=1, step=1)
        cfg["prediction_horizon"] = st.number_input(
            "Prediction horizon", value=int(cfg["prediction_horizon"]),
            min_value=1, step=1)

    st.subheader("Disturbance estimation")
    estimator_options = ["discounted_kde", "gaussian_process"]
    cfg["disturbance_estimation"] = st.selectbox(
        "Estimator", estimator_options,
        index=estimator_options.index(cfg["disturbance_estimation"])
        if cfg["disturbance_estimation"] in estimator_options else 0)

    if cfg["disturbance_estimation"] == "discounted_kde":
        dk = cfg["discounted_kde"]
        col1, col2, col3 = st.columns(3)
        with col1:
            dk["base_of_exponential_weights"] = st.number_input(
                "Base of exponential weights",
                value=float(dk["base_of_exponential_weights"]), format="%.3f")
        with col2:
            dk["density_number_of_past_samples_considered"] = st.number_input(
                "Past samples considered",
                value=int(dk["density_number_of_past_samples_considered"]),
                min_value=1, step=1)
            dk["number_eval_points"] = st.number_input(
                "Evaluation points", value=int(dk["number_eval_points"]),
                min_value=1, step=1)
        with col3:
            interv_max = st.number_input(
                "Evaluation interval (symmetric, +/-)",
                value=float(dk["interv_max"]), min_value=0.0, step=0.1)
            dk["interv_max"] = interv_max
            dk["interv_min"] = -interv_max
    else:
        gp = cfg["gaussian_process"]
        gp["density_number_of_past_samples_considered"] = st.number_input(
            "Past samples considered",
            value=int(gp["density_number_of_past_samples_considered"]),
            min_value=1, step=1)

    st.subheader("Cost matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("R (input cost)")
        cfg["mpc_cost_matrices"]["R"] = matrix_editor(
            "R", cfg["mpc_cost_matrices"]["R"], "r_matrix")
    with col2:
        st.caption("Q (state/tracking cost)")
        cfg["mpc_cost_matrices"]["Q"] = matrix_editor(
            "Q", cfg["mpc_cost_matrices"]["Q"], "q_matrix")

    st.subheader("Constraints")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("G_u")
        cfg["constraints"]["G_u"] = matrix_editor(
            "G_u", cfg["constraints"]["G_u"], "gu_matrix")
        st.caption("g_u")
        cfg["constraints"]["g_u"] = matrix_editor(
            "g_u", cfg["constraints"]["g_u"], "gu_bound")
    with col2:
        st.caption("G_x")
        cfg["constraints"]["G_x"] = matrix_editor(
            "G_x", cfg["constraints"]["G_x"], "gx_matrix")
        st.caption("g_x")
        cfg["constraints"]["g_x"] = matrix_editor(
            "g_x", cfg["constraints"]["g_x"], "gx_bound")

    if st.button("Save configuration", type="primary"):
        config_io.save_config(cfg)
        st.success("Saved to config/config.yaml")

with tab_run:
    st.subheader("Run simulation")
    st.caption(
        "Runs the current on-disk `config/config.yaml`. Save your changes "
        "in the Configuration tab first."
    )

    folder_name = cfg["data_storage"]["folder_name"]
    record_data = cfg["data_storage"]["record_data"]
    target = os.path.join(RECORDED_DATA_DIR, folder_name)
    blocked = record_data and os.path.isdir(target)

    if blocked:
        st.error(
            f"`recorded_data/{folder_name}` already exists. Pick a new "
            "folder name in the Configuration tab before running."
        )

    if st.button("Run", disabled=blocked, type="primary"):
        save_dir = target if record_data else None
        log_area = st.empty()
        log_lines = []
        try:
            for line in runner.run_simulation(save_dir=save_dir):
                log_lines.append(line)
                log_area.code("\n".join(log_lines[-200:]))
            st.success("Simulation finished.")

            if save_dir is not None:
                animation_path = os.path.join(save_dir, "animation.mp4")
                distribution_path = os.path.join(save_dir, "distribution.png")
                if os.path.exists(animation_path):
                    st.video(animation_path)
                if os.path.exists(distribution_path):
                    st.image(distribution_path)
        except RuntimeError as exc:
            st.error(str(exc))
