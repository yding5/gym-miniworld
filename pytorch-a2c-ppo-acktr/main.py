import copy
import glob
import os
import time
import types
from collections import deque

import gym
import gym_miniworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy, AE, RNN, Detector, VAER, SimGAN
from storage import RolloutStorage
#from visualize import visdom_plot
from utils import make_var

import datetime

args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)
        
def Detector_to_symbolic(x):
    reshaped_x = x.view(-1,6,5)
    #print(reshaped_x[:,:,:3])
    idx = torch.max(reshaped_x[:,:,:3], dim=2, keepdim=True)[1].float()
    #print(idx)
    res = torch.cat([idx,reshaped_x[:,:,3:]], dim=2)
    res = res.view(-1,18)
    #print(res)
    return res
        

def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if args.cuda else "cpu")
    
    ##
    UID = 'exp_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    step_log = []
    reward_log = []
    
    ## To be used to selec environment
    mode = 'normal'
    
    # encoder type
    encoder = 'sym_VAE'
    if encoder == 'symbolic':
        embedding_size = (18,)
    elif encoder == 'AE':
        embedding_size = (200,)
    elif encoder == 'VAE':
        embedding_size = (100,)
    elif encoder == 'sym_VAE':
        embedding_size = (118,)
    else:
        raise NotImplementedError('fff')
    
    
    # load pre-trained AE
    #AE = VAEU([128,128])
    #model_path = '/hdd_c/data/miniWorld/trained_models/VAE/dataset_4/VAEU.pth'
    #AE = torch.load(model_path)
    #AE.eval()
    
    # load pre-trained VAE
    VAE = VAER([128,128])
    model_path = '/hdd_c/data/miniWorld/trained_models/VAE/dataset_5/VAER.pth'
    VAE = torch.load(model_path).to(device)
    VAE.eval()
    
    # load pre-trained detector
    Detector_model = Detector
    model_path = '/hdd_c/data/miniWorld/trained_models/Detector/dataset_5/Detector_resnet18_e14.pth'
    Detector_model = torch.load(model_path).to(device)
    
    # load pre-trained RNN
    RNN_model = RNN(200, 128)
    model_path = '/hdd_c/data/miniWorld/trained_models/RNN/RNN1.pth'
    RNN_model = torch.load(model_path).to(device)
    RNN_model.eval()
    
    
    """
    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None
    """


    
    
    
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)
    
    print(envs.observation_space.shape)

    #actor_critic = Policy(envs.observation_space.shape, envs.action_space,
    #    base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic = Policy(embedding_size, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)


    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    #rollouts = RolloutStorage(args.num_steps, args.num_processes,
    #                    envs.observation_space.shape, envs.action_space,
    #                    actor_critic.recurrent_hidden_state_size)
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        embedding_size, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)


    obs = envs.reset()
    #print(obs.size())
    #obs = make_var(obs)
    print(obs.size())
    with torch.no_grad():
        if encoder == 'symbolic':
            
            z = Detector_model(obs)
            print(z.size())
            z = Detector_to_symbolic(z)
            rollouts.obs[0].copy_(z)
        elif encoder == 'AE':
            z = AE.encode(obs)
            rollouts.obs[0].copy_(z)
        elif encoder == 'VAE':
            z = VAE.encode(obs)[0]
            rollouts.obs[0].copy_(z)
        elif encoder == 'sym_VAE':
            z_vae = VAE.encode(obs)[0]
            z_sym = Detector_model(obs)
            z_sym = Detector_to_symbolic(z_sym)
            z = torch.cat((z_vae,z_sym),dim=1)
            rollouts.obs[0].copy_(z)
        else:
            raise NotImplementedError('fff')

    
        
    #rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    for j in range(num_updates):
        #print(j)
        for step in range(args.num_steps):
            # Sample actions
            #print(step)
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            #print(action)
            with torch.no_grad():
                obs, reward, done, infos = envs.step(action)
                if encoder == 'symbolic':
                    #print(obs.size())
                    np.save('/hdd_c/data/miniWorld/training_obs_{}.npy'.format(step),obs.detach().cpu().numpy())
                    z = Detector_model(obs/255.0)
                    z = Detector_to_symbolic(z)
                    #print(z)
                    np.save('/hdd_c/data/miniWorld/training_z_{}.npy'.format(step),z.detach().cpu().numpy())
                elif encoder == 'AE':
                    z = AE.encode(obs)
                elif encoder == 'VAE':
                    z = VAE.encode(obs)[0]
                elif encoder == 'sym_VAE':
                    z_vae = VAE.encode(obs)[0]
                    z_sym = Detector_model(obs)
                    z_sym = Detector_to_symbolic(z_sym)
                    z = torch.cat((z_vae,z_sym),dim=1)
                else:
                    raise NotImplementedError('fff')
                #obs = make_var(obs)
                

            """
            for info in infos:
                if 'episode' in info.keys():
                    print(reward)
                    episode_rewards.append(info['episode']['r'])
            """

#             # FIXME: works only for environments with sparse rewards
#             for idx, eps_done in enumerate(done):
#                 if eps_done:
#                     episode_rewards.append(reward[idx])
                    
            # FIXME: works only for environments with sparse rewards
            for idx, eps_done in enumerate(done):
                if eps_done:
                    #print('done')
                    episode_rewards.append(infos[idx]['accumulated_reward'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            #rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)
            rollouts.insert(z, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            print('Saving model')
            print()

            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model, hasattr(envs.venv, 'ob_rms') and envs.venv.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        #print(len(episode_rewards))
        
        step_log.append(total_num_steps)
        reward_log.append(np.mean(episode_rewards))
        step_log_np = np.asarray(step_log)
        reward_log_np = np.asarray(reward_log)
        np.savez_compressed('/hdd_c/data/miniWorld/log/{}.npz'.format(UID),step=step_log_np, reward=reward_log_np)
        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}, success rate {:.2f}\n".
                format(
                    j, total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    np.count_nonzero(np.greater(episode_rewards, 0)) / len(episode_rewards)
                )
            )

        if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
            eval_envs = make_vec_envs(args.env_name, args.seed + args.num_processes, args.num_processes,
                                args.gamma, eval_log_dir, args.add_timestep, device, True)

            if eval_envs.venv.__class__.__name__ == "VecNormalize":
                eval_envs.venv.ob_rms = envs.venv.ob_rms

                # An ugly hack to remove updates
                def _obfilt(self, obs):
                    if self.ob_rms:
                        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                        return obs
                    else:
                        return obs

                eval_envs.venv._obfilt = types.MethodType(_obfilt, envs.venv)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
                len(eval_episode_rewards),
                np.mean(eval_episode_rewards)
            ))

        """
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
        """
    envs.close()
if __name__ == "__main__":
    main()
