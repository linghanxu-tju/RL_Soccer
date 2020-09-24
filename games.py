# coding=utf-8
import numpy as np
import imageio
from gym import spaces
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot  as plt
import time

CELL, BLOCK, AGENT_GOAL, OPPONENT_GOAL, AGENT, OPPONENT = range(6)
WIN, LOSE, TIE = 5, -5, 0
UP, RIGHT, DOWN, LEFT, HOLD, DASH_UP_LEFT, DASH_UP_RIGHT, DASH_DOWN_LEFT, DASH_DOWN_RIGHT, DASH_UP, DASH_RIGHT, DASH_DOWN, DASH_LEFT, = range(13)
UNIT = 40
VISUAL = True

class Player:
    energy_cost = {
        UP: 0.5,
        RIGHT: 0.5,
        DOWN: 0.5,
        LEFT: 0.5,
        HOLD: 1,
        DASH_UP: -1,
        DASH_RIGHT: -1,
        DASH_DOWN: -1,
        DASH_LEFT: -1,
        DASH_UP_LEFT: -1,
        DASH_UP_RIGHT: -1,
        DASH_DOWN_LEFT: -1,
        DASH_DOWN_RIGHT: -1,
    }

    def __init__(self, player_num, max_energy):
        self.player_num = player_num
        self.location = np.array([0, 0])
        self.max_energy = max_energy
        self.energy = self.max_energy

    def set_location(self, location):
        self.location = location

    def get_location(self):
        return self.location

    def get_energy(self):
        return self.energy

    def set_energy(self,energy_level):
        assert energy_level <= self.max_energy
        self.energy = energy_level

    def update_energy(self, action):
        if self.has_ball() and action in [UP, RIGHT, DOWN, LEFT, HOLD]:
            self.energy += self.energy_cost[action] - 0.5
        else:
            self.energy += self.energy_cost[action]
        self.energy = max(min(self.max_energy, self.energy), 0)

    def energy_enough(self, action):
        return False if self.energy + self.energy_cost[action] < 0 else True

    def has_ball(self):
        return self.keeping_ball

    def set_ball(self, has_ball):
        self.keeping_ball = has_ball

    def change_ball(self):
        self.keeping_ball = not self.keeping_ball


class Soccer(tk.Tk if VISUAL else object):
    playground = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
                  3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
                  3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]
    action_map = {
        UP: np.array([-1, 0]),
        RIGHT: np.array([0, 1]),
        DOWN: np.array([1, 0]),
        LEFT: np.array([0, -1]),
        HOLD: np.array([0, 0])}

    def __init__(self,visual=False):
        super(Soccer, self).__init__()
        self.size = int(np.sqrt(len(self.playground)))
        self.max_steps = len(self.playground) * 2
        self.agent = np.array([self.size // 2, 2])
        self.opponent = np.array([self.size // 2, self.size - 3])
        self.grids = np.array(self.playground).reshape(self.size, self.size)
        self.agent_keep_ball = bool(np.random.randint(0, 2))
        self.action_space = [UP, RIGHT, DOWN, LEFT, HOLD]
        self.n_actions = len(self.action_space)
        self.n_features = 5
        self.counter = 0
        # creat canvas
        self.visual =visual
        if self.visual:
            self.visualize()
            self.update_canvas()
            self.canvas.pack()

    def step(self, act_a, act_o):
        self.counter += 1
        new_pos_a = self.agent + self.action_map[act_a]
        new_pos_o = self.opponent + self.action_map[act_o]

        reward, done, s_ = 0, False, []
        if self.counter >= self.max_steps:
            reward = TIE
            done = True
        # opponent win
        if self.grids[tuple(new_pos_o)] == 3 and not self.agent_keep_ball:
            reward = LOSE
            done = True

        # agent win
        if self.grids[tuple(new_pos_a)] == 2 and self.agent_keep_ball:
            reward = WIN
            done = True

        # valid check for opponent and agent
        if self.grids[tuple(new_pos_a)] in (1, 2, 3):
            new_pos_a = self.agent

        if self.grids[tuple(new_pos_o)] in (1, 2, 3):
            new_pos_o = self.opponent

        # collision
        if np.array_equal(new_pos_a, new_pos_o) and self.grids[tuple(new_pos_a)] != 1:
            self.agent_keep_ball = not self.agent_keep_ball

        # print(self.canvas.coords(self.agent_rect))
        self.agent = new_pos_a
        self.opponent = new_pos_o

        # update canvas
        if self.visual:
            self.canvas.delete(self.agent_rect)
            self.canvas.delete(self.opp_rect)
            self.canvas.delete(self.ball_rect)
            self.update_canvas()
        s_ = [self.agent[0]/self.size, self.agent[1]/self.size, self.opponent[0]/self.size, self.opponent[1]/self.size]
        if self.agent_keep_ball:
            s_.append(0)
        else:
            s_.append(1)
        s_ = np.array(s_[:5])
        return s_, reward, done, {}

    # reset position and ball
    def reset(self):
        self.counter = 0
        self.agent = np.array([self.size // 2, 2])
        self.opponent = np.array([self.size // 2, self.size - 3])
        self.agent_keep_ball = bool(np.random.randint(0, 2))
        if self.visual:
            self.update()
        s_ = [self.agent[0]/self.size, self.agent[1]/self.size, self.opponent[0]/self.size, self.opponent[1]/self.size]
        if self.agent_keep_ball:
            s_.append(0)
        else:
            s_.append(1)
        s_ = np.array(s_[:5])
        return s_

    def update_canvas(self):
        self.agent_rect = self.canvas.create_rectangle(self.agent[1] * UNIT, self.agent[0] * UNIT,
                                                       (self.agent[1] + 1) * UNIT, (self.agent[0] + 1) * UNIT,
                                                       fill='red')
        self.opp_rect = self.canvas.create_rectangle(self.opponent[1] * UNIT, self.opponent[0] * UNIT,
                                                     (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT,
                                                     fill='blue')
        if self.agent_keep_ball:
            self.ball_rect = self.canvas.create_oval(
                (self.agent[1] * UNIT, self.agent[0] * UNIT, (self.agent[1] + 1) * UNIT, (self.agent[0] + 1) * UNIT),
                fill='white')
        else:
            self.ball_rect = self.canvas.create_oval(self.opponent[1] * UNIT, self.opponent[0] * UNIT,
                                                     (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT,
                                                     fill='white')

    # render array
    def render(self):
        m = np.copy(self.grids)
        m[tuple(self.agent)] = 4
        m[tuple(self.opponent)] = 5
        if self.agent_keep_ball:
            m[tuple(self.agent)] += 2
        else:
            m[tuple(self.opponent)] += 2
        # print(m, end='\n\n')
        if self.visual:
            self.update()
        return m.reshape(len(self.playground))

    # render img
    def visualize(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.size * UNIT,
                                width=self.size * UNIT)
        # create grids
        for c in range(0, self.size * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.size * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.size * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.size * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        m = np.copy(self.grids)
        m[tuple(self.agent)] = 4
        m[tuple(self.opponent)] = 5
        # print(m)
        for j in range(self.size):
            for i in range(self.size):
                if m[j, i] == 1:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='black')
                elif m[j, i] == 2 or m[j, i] == 3:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='white')
                elif m[j, i] == 0 or m[j, i] == 4 or m[j, i] == 5:
                    self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='green')

    def close(self):
        if self.visual:
            self.canvas.destroy()

    def is_valid_moving(self,location, action):
        if self.grids[tuple(location + action)] != 1:
            return True
        else:
            return False

    # The following method is mainly used to construct manual policy
    @staticmethod
    def getDistance(location1, location2):
        return abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])

    def getPlayerDistance(self):
        return abs(self.agent[0]-self.opponent[0]) + abs(self.agent[1]-self.opponent[1])

    def getGoalLocation(self, player_num):
        goal_value = 2 if player_num else 3
        locations = np.argwhere(self.grids == goal_value)
        return locations

    def getGoalDistance(self, goal_locations, target_location):
        return np.sum(abs(goal_locations - target_location), axis=1)

    def getNextLocation(self, now_location, action):
        return now_location + action

    def getNextLocationAll(self, now_location):
        action_offset = np.array([self.action_map[i] for i in range(self.n_actions)])
        return action_offset + now_location

    def validLocation(self, new_location):
        return not self.grids[tuple(new_location)] == 1

    def validLocationAll(self, new_locations):
        valid = []
        for location in new_locations:
            valid.append(not self.grids[tuple(location)] == 1)
        return np.array(valid)


    def reachGoal(self, new_location,player_num):
        goal_value = 2 if player_num else 3
        return False if self.grids[tuple(new_location)] == goal_value else True

    def reachGoalAll(self, new_locations,player_num):
        goal_value = 2 if player_num else 3
        return self.grids[tuple(new_locations)] == goal_value


class SoccerPLUS(Soccer):
    action_map = {
        UP: np.array([-1, 0]),
        RIGHT: np.array([0, 1]),
        DOWN: np.array([1, 0]),
        LEFT: np.array([0, -1]),
        HOLD: np.array([0, 0]),
        DASH_UP_LEFT: np.array([-1, -1]),
        DASH_UP_RIGHT: np.array([-1, 1]),
        DASH_DOWN_LEFT: np.array([1, -1]),
        DASH_DOWN_RIGHT: np.array([1, 1]),
        DASH_UP: np.array([-2, 0]),
        DASH_RIGHT: np.array([0, 2]),
        DASH_DOWN: np.array([2, 0]),
        DASH_LEFT: np.array([0, -2]),
    }

    def __init__(self,visual=False):
        super().__init__(visual=visual)
        self.player_agent = Player(player_num=True, max_energy=2)
        self.player_opponent = Player(player_num=False, max_energy=2)
        self.player_agent.set_location(self.agent)
        self.player_opponent.set_location(self.opponent)
        self.player_agent.set_ball(self.agent_keep_ball)
        self.player_opponent.set_ball(not self.agent_keep_ball)
        self.n_actions = len(self.action_map)
        self.n_features = 8

    def step(self, act_a, act_o):
        assert np.array_equal(self.player_agent.get_location(), self.agent) and np.array_equal(self.player_opponent.get_location(), self.opponent)
        assert self.agent_keep_ball == self.player_agent.has_ball()
        # print("=" * 20 + " DEBUG_INFO " + "=" * 20)
        # print("Agent Action:\t{},\tOpponent Action:\t{}".format(self.action_map[act_a], self.action_map[act_o]))
        self.counter += 1
        act_a, act_o = self.action_energy_check(act_a, act_o)
        # get intended new position
        new_pos_a = self.agent + self.action_map[act_a]
        new_pos_o = self.opponent + self.action_map[act_o]
        if (max(new_pos_a) >= self.size or max(new_pos_o)>=self.size or min(new_pos_a) < 0 or min(new_pos_o)) and self.done < 0:
            print("HERE")
        # round end check
        reward, done = self.round_end_check(new_pos_a, new_pos_o)
        new_pos_a, new_pos_o, act_a, act_o = self.moving_validation(new_pos_a, new_pos_o, act_a, act_o)
        # collision check
        new_pos_a, new_pos_o = self.collision_check(new_pos_a, new_pos_o)
        # used for update the canvas
        self.agent = new_pos_a
        self.opponent = new_pos_o
        # update the player object
        self.player_agent.set_location(new_pos_a)
        self.player_opponent.set_location(new_pos_o)
        # update energy
        self.player_agent.update_energy(act_a)
        self.player_opponent.update_energy(act_o)
        # update canvas
        if self.visual:
            self.canvas.delete(self.agent_rect)
            self.canvas.delete(self.opp_rect)
            self.canvas.delete(self.ball_rect)
            self.update_canvas()
        s_ = [self.agent[0]/self.size,
              self.agent[1]/self.size,
              self.player_agent.get_energy()/self.player_agent.max_energy,
              self.opponent[0]/self.size,
              self.opponent[1]/self.size,
              self.player_opponent.get_energy()/self.player_opponent.max_energy,
              int(self.agent_keep_ball),
              self.counter/self.max_steps,
              ]
        s_ = np.array(s_,dtype=np.float)
        if done:
            self.done = done
        # DEBUG INFO
        # print("Agent location:\t{},\tOppo location:\t{}".format(self.player_agent.get_location(), self.player_opponent.get_location()))
        # print("Agent energy:\t{},\tOppo energy:\t{}".format(self.player_agent.get_energy(), self.player_opponent.get_energy()))
        # print("Agent hasBall:\t{},\tOppo hasBall:\t{}".format(self.player_agent.has_ball(), self.player_opponent.has_ball()))
        # print("=" * 20 + " DEBUG_INFO " + "=" * 20)
        return s_, reward, done, {}

    def reset(self):
        super().reset()
        self.done = False
        self.player_agent.set_location(self.agent)
        self.player_opponent.set_location(self.opponent)
        self.player_agent.set_energy(2)
        self.player_opponent.set_energy(2)
        self.player_agent.set_ball(self.agent_keep_ball)
        self.player_opponent.set_ball(not self.agent_keep_ball)
        s_ = [self.agent[0]/self.size,
              self.agent[1]/self.size,
              self.player_agent.get_energy()/self.player_agent.max_energy,
              self.opponent[0]/self.size,
              self.opponent[1]/self.size,
              self.player_opponent.get_energy()/self.player_opponent.max_energy,
              int(self.agent_keep_ball),
              self.counter/self.max_steps,
              ]
        s_ = np.array(s_,dtype=np.float)
        return s_

    def collision_check(self, new_pos_a, new_pos_o):
        assert self.player_agent.has_ball() != self.player_opponent.has_ball()
        if np.array_equal(new_pos_a, new_pos_o) and self.grids[tuple(new_pos_a)] != 1:
            if self.player_agent.has_ball() or all(np.equal(new_pos_a, self.opponent)):
                new_pos_a = self.agent
            if self.player_opponent.has_ball() or all(np.equal(new_pos_o, self.agent)):
                new_pos_o = self.opponent
            self.player_agent.change_ball()
            self.player_opponent.change_ball()
            self.agent_keep_ball = not self.agent_keep_ball
        return new_pos_a, new_pos_o

    def round_end_check(self,new_pos_a, new_pos_o):
        reward, done = 0, False
        # exceed maximum steps
        if self.counter >= self.max_steps:
            reward = TIE
            done = True
        # opponent win
        if self.grids[tuple(new_pos_o)] == 3 and not self.agent_keep_ball:
            reward = LOSE
            done = True

        # agent win
        if self.grids[tuple(new_pos_a)] == 2 and self.agent_keep_ball:
            reward = WIN
            done = True
        return reward, done

    def moving_validation(self, new_pos_a, new_pos_o, act_a, act_o):
        if self.grids[tuple(new_pos_a)] in (1, 2, 3):
            new_pos_a = self.agent
            act_a = HOLD

        if self.grids[tuple(new_pos_o)] in (1, 2, 3):
            new_pos_o = self.opponent
            act_o = HOLD

        return new_pos_a, new_pos_o, act_a, act_o

    def action_energy_check(self, act_a, act_o):
        if not self.player_agent.energy_enough(act_a):
            if act_a == DASH_UP: act_a = UP
            elif act_a == DASH_LEFT: act_a = LEFT
            elif act_a == DASH_DOWN: act_a = DOWN
            elif act_a == DASH_RIGHT: act_a = RIGHT
            else: act_a = HOLD
        if not self.player_opponent.energy_enough(act_o):
            if act_o == DASH_UP: act_o = UP
            elif act_o == DASH_LEFT: act_o = LEFT
            elif act_o == DASH_DOWN: act_o = DOWN
            elif act_o == DASH_RIGHT: act_o = RIGHT
            else: act_o = HOLD
        return act_a, act_o

    def energy_update(self, act_a, act_o):
        self.player_agent.update_energy(act_a)
        self.player_opponent.update_energy(act_o)


if __name__ == '__main__':
    env = SoccerPLUS(VISUAL)
    env.reset()
    # agent strategy
    agent_actions = [RIGHT] * 6 + [RIGHT] * 7
    # opponent strategy, you can initialize it randomly
    opponent_actions = [LEFT] * 6 + [RIGHT] * 7

    for step, (a_a, a_o) in enumerate(zip(agent_actions, opponent_actions)):
        env.render()
        s_, reward, done, _ = env.step(a_a, a_o)
        print(step, s_)
        if done:
            break
        time.sleep(1)
        # env.after(100, run_maze)
        # env.mainloop()
        # env.render()
