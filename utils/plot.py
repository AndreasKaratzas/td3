
import sys
sys.path.append('../')

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def compile_plots(log_f_name: str, plot_dir: str):
    df = build_log_dataframe(log_f_name)
    generate_avg_length_plot(df, plot_dir)
    generate_avg_reward_plot(df, plot_dir)
    generate_avg_q_val_plot(df, plot_dir)
    generate_loss_actor_plot(df, plot_dir)
    generate_loss_critic_plot(df, plot_dir)


def build_log_dataframe(log_f_name):
    df = pd.read_csv(log_f_name, delim_whitespace=True, engine='python', skiprows=3, header=0)
    df = df.reset_index()
    prev_cols = list(df.columns)
    df = df.drop(columns=[i for i in prev_cols[:11]])
    prev_cols = list(df.columns)  
    return df.rename(
        columns={
            prev_cols[0]: 'avg_length', 
            prev_cols[1]: 'avg_reward', 
            prev_cols[2]: 'avg_q_val', 
            prev_cols[3]: 'loss_actor', 
            prev_cols[4]: 'loss_critic'
        }
    )


def generate_avg_length_plot(df: pd.DataFrame, plot_dir: str):
    fig, axs = plt.subplots(figsize=(12, 12))
    df.avg_length.plot(ax=axs)
    fig.patch.set_facecolor('white')
    axs.set_facecolor('xkcd:white')
    axs.set_title('Agent mean achieved test episode length', fontsize=22)
    axs.set_xlabel("Epoch", fontsize=16)
    axs.set_ylabel("Average episode length", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / Path("avg_length.png"))


def generate_avg_reward_plot(df: pd.DataFrame, plot_dir: str):
    fig, axs = plt.subplots(figsize=(12, 12))
    df.avg_reward.plot(ax=axs)
    fig.patch.set_facecolor('white')
    axs.set_facecolor('xkcd:white')
    axs.set_title('Agent mean accumulated test episode reward', fontsize=22)
    axs.set_xlabel("Epoch", fontsize=16)
    axs.set_ylabel("Average accumulated reward", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / Path("avg_reward.png"))


def generate_avg_q_val_plot(df: pd.DataFrame, plot_dir: str):
    fig, axs = plt.subplots(figsize=(12, 12))
    df.avg_q_val.plot(ax=axs)
    fig.patch.set_facecolor('white')
    axs.set_facecolor('xkcd:white')
    axs.set_title('Agent accumulated Q value', fontsize=22)
    axs.set_xlabel("Epoch", fontsize=16)
    axs.set_ylabel("Accumulated Q value", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / Path("avg_q_val.png"))


def generate_loss_actor_plot(df: pd.DataFrame, plot_dir: str):
    fig, axs = plt.subplots(figsize=(12, 12))
    df.loss_actor.plot(ax=axs)
    fig.patch.set_facecolor('white')
    axs.set_facecolor('xkcd:white')
    axs.set_title('Agent actor module loss', fontsize=22)
    axs.set_xlabel("Epoch", fontsize=16)
    axs.set_ylabel("Actor criterion value", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / Path("loss_actor.png"))


def generate_loss_critic_plot(df: pd.DataFrame, plot_dir: str):
    fig, axs = plt.subplots(figsize=(12, 12))
    df.loss_critic.plot(ax=axs)
    fig.patch.set_facecolor('white')
    axs.set_facecolor('xkcd:white')
    axs.set_title('Agent critic module loss', fontsize=22)
    axs.set_xlabel("Epoch", fontsize=16)
    axs.set_ylabel("Critic criterion value", fontsize=16)
    fig.tight_layout()
    fig.savefig(plot_dir / Path("loss_critic.png"))
