from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np

""" 
    정책을 사용해서 게임 한 판을 진행하고, 그 과정을 기록 
            게임 초기화
            ↓
            현재 관측·state·가능한 행동을 기록
            ↓
            mac에 행동 선택 요청
            ↓
            선택한 행동을 게임에 실행
            ↓
            보상·종료 여부 기록
            ↓
            게임이 끝날 때까지 반복
"""
class EpisodeRunner:

    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if args.env == "sc2":
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        elif args.env == "gymma":
            self.env = env_REGISTRY[self.args.env](
                **self.args.env_args,
                common_reward=self.args.common_reward,
                reward_scalarisation=self.args.reward_scalarisation,
            )
        self.episode_limit = self.env.episode_limit
        self.t = 0 # 현재 게임 안에서 몇 스텝 진행했는가

        self.t_env = 0 # 현재 정책이 어느 학습 시점의 정책인지 

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    """ 기록 양식과 공유 mac 연결 """
    def setup(self, scheme, groups, preprocess, mac):
        # partial : 인자를 미리 채운 함수를 만드는 도구 / 게임 기록용 빈 배치를 만드는 함수
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, nolog=False, pretrain=False):
        self.reset()

        terminated = False
        episode_return = 0
        # 이전 게임에서 이어진 hidden state 초기화
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task) 

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()], # 팀 전체의 global state
                "avail_actions": [self.env.get_avail_actions()], # 각 에이전트가 현재 실행할 수 있는 행동 표시
                "obs": [self.env.get_obs()], # 각 에이전트의 local obs
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # [B, N]
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                task=self.task,
                test_mode=test_mode,
            )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # 게임 종료 후의 마지막 관측도 저장
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch,
            t_ep=self.t,
            t_env=self.t_env,
            task=self.task,
            test_mode=test_mode,
        )

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = f"{'pretrain/' if pretrain else ''}{self.task}/{'test_' if test_mode else ''}"
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        if not nolog:
            if test_mode and len(self.test_returns) == self.args.test_nepisode:
                self._log(cur_returns, cur_stats, log_prefix)
            elif (
                not test_mode
                and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval
            ):
                self._log(cur_returns, cur_stats, log_prefix)
                if "offline" not in self.args.run_file:
                    if hasattr(self.mac.action_selector, "epsilon"):
                        self.logger.log_stat(
                            f"{self.task}/epsilon",
                            self.mac.action_selector.epsilon,
                            self.t_env,
                        )
                self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def evaluate(self):
        self.reset()
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        render_images = [self.env.render()]

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size, task=self.task)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                task=self.task,
                test_mode=True,
            )

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

            render_images.append(self.env.render())

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, task=self.task, test_mode=True
        )

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats
        cur_returns = self.test_returns
        log_prefix = f"{self.task}/eval_"
        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + env_info.get(k, 0)
                for k in set(cur_stats) | set(env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        if len(self.test_returns) == self.args.test_nepisode:
            self._log(cur_returns, cur_stats, log_prefix)

        return render_images
