import numpy as np
import time
from games import Soccer, SoccerPLUS


class Policy:
    def __init__(self, game:Soccer, player_num):
        self.game = game
        self.player_num = player_num
        self.action_map = self.game.action_map
        self.me = self.game.agent if player_num else self.game.opponent
        self.opp = self.game.opponent if player_num else self.game.agent
        self.my_goal = self.game.getGoalLocation(player_num)
        self.opp_goal = self.game.getGoalLocation(not player_num)
        self.has_ball = self.game.agent_keep_ball if player_num else not self.game.agent_keep_ball
        self.policy_type = [1,2,3,4]
        # policy 1: if not has ball, get ball. if has ball get to goal
        # policy 3: if not has ball, get ball. if has ball get to goal and keep away
        # policy 2: if not has ball, defence the goal if distance larger than 2, if less than 2 get closer. if has ball get to goal
        # policy 4: if not has ball, defence the goal if distance larger than 2. if less than 2 get closer. if has ball get to goal and keep away

    def moving(self, closer: bool,player_number:bool, locations: np.ndarray, actions: np.ndarray):
        assert locations.ndim == 2 and actions.ndim == 2
        current_location = self.me if player_number else self.opp
        current_dis = np.sum(abs(locations - current_location), axis=1)
        next_locations = current_location + actions
        min_dis, max_dis, mean_dis = [], [], []
        for next in next_locations:
            distances = locations - next
            distances = np.sum(abs(distances), axis=1)
            min_dis.append(np.min(distances))
            max_dis.append(np.max(distances))
            mean_dis.append(np.mean(distances))
        min_dis = np.array(min_dis) - np.min(current_dis)
        max_dis = np.array(max_dis) - np.max(current_dis)
        mean_dis = np.array(mean_dis) - np.mean(current_dis)
        scores = (min_dis + max_dis + mean_dis) * (1 if not closer else -1)
        scores[np.argwhere(scores < 0)] = 0
        return scores

    def validActionAll(self, player_number):
        all_actions = []
        current_location = self.me if player_number else self.opp
        for i in range(len(self.action_map)):
            all_actions.append(self.action_map[i])
        all_actions = np.array(all_actions)
        new_locations = current_location + all_actions
        valid_actions = self.game.validLocationAll(new_locations)
        valid_new_locations = new_locations[valid_actions]
        valid_actions_index = np.arange(len(new_locations))[valid_actions]
        return valid_new_locations, valid_actions_index

    def grab_ball(self):
        assert not self.has_ball
        assert sum(abs(self.me - self.opp)) <= 1

    def keep_ball(self):
        assert self.has_ball
        assert sum(abs(self.me - self.opp)) <= 1

    def policy1(self, valid_actions):
        if self.has_ball:
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal,actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp, actions=valid_actions)
        actions_prob = actions_score / sum(actions_score)
        return np.random.choice(valid_actions, p=actions_prob)

    def policy2(self, valid_actions):
        if self.has_ball:
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
            actions_score += self.moving(closer=False, player_number=True, locations=self.opp, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp, actions=valid_actions)
        actions_prob = actions_score / sum(actions_score)
        return np.random.choice(valid_actions, p=actions_prob)

    def policy3(self, valid_actions):
        if self.has_ball:
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
            actions_score += self.moving(closer=False, player_number=True, locations=self.opp, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp_goal, actions=valid_actions)
        actions_prob = actions_score / sum(actions_score)
        return np.random.choice(valid_actions, p=actions_prob)

    def policy4(self, valid_actions):
        if self.has_ball:
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp_goal, actions=valid_actions)
        actions_prob = actions_score / sum(actions_score)
        return np.random.choice(valid_actions, p=actions_prob)

    def get_actions(self, policy_num):
        valid_new_locations, valid_actions_index = self.validActionAll(player_number=self.player_num)


if __name__ == "__main__":
    policy_types = list(range(5))
    env = Soccer()
    env.reset()
    my_policy = Policy(game=env, player_num=True)
    while True:
        env.render()
        s_, reward, done, _ = env.step(my_policy.get_actions(1), np.random.randint(env.n_actions))
        if done:
            env.reset()
        time.sleep(1)

