'''
MIT License

Copyright (c) 2018-2020 Ekaterina Nikonova,
Research School of Computer Science, Australian National University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This version of the agent is based on the work presented in: https://arxiv.org/abs/1910.01806

'''

from __future__ import division

import numpy as np
import random
import tensorflow.compat.v1 as tf
import math
import socket
import json
import os
from PIL import ImageFile
from threading import Thread
from client.agent_client import AgentClient, GameState
from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from computer_vision.GroundTruthReader import GroundTruthReader,NotVaildStateError
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

TOTAL_STEPS = 1000000
MODEL_PATH = "./src/demo/RL/Models/ddqn_model"

tf.disable_eager_execution()


class StateMaker():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[480, 840, 3], dtype=tf.float32)
            self.output = tf.image.per_image_standardization(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 80, 20, 310, 770)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def make(self, sess, state):
        return sess.run(self.output, {self.input_state: state})


class ClientRLAgent(Thread):
    def __init__(self):
                #initialise colormap for the ground truth reader
        f = open('./ColorMap.json','r')
        result = json.load(f)

        self.look_up_matrix = np.zeros((len(result),256))
        self.look_up_obj_type = np.zeros(len(result)).astype(str)

        obj_number = 0
        for d in result:

            if 'effects_21' in d['type']:
                obj_name = 'Platform'

            elif 'effects_34' in d['type']:
                obj_name = 'TNT'

            elif 'ice' in d['type']:
                obj_name = 'Ice'

            elif 'wood' in d['type']:
                obj_name = 'Wood'

            elif 'stone' in d['type']:
                obj_name = 'Stone'

            else:
                obj_name = d['type'][:-2]

            obj_color_map = d['colormap']

            self.look_up_obj_type[obj_number] = obj_name
            for pair in obj_color_map:
                self.look_up_matrix[obj_number][int(pair['x'])] = pair['y']

            obj_number+=1

        #normalise the look_up_matrix
        self.look_up_matrix = self.look_up_matrix / np.sqrt((self.look_up_matrix**2).sum(1)).reshape(-1,1)
        # Wrapper of the communicating messages
        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.ar = AgentClient(**sc_json_config[0])
        try:
            self.ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))
        self.current_level = -1
        self.failed_counter = 0
        self.solved = []
        self.tp = SimpleTrajectoryPlanner()
        self.sling_center = None
        self.id = 28888
        self.first_shot = True
        self.prev_target = None

    def get_slingshot_center(self):
        ground_truth = self.ar.get_ground_truth_without_screenshot()
        ground_truth_reader = GroundTruthReader(ground_truth,self.look_up_matrix,self.look_up_obj_type)
        sling = ground_truth_reader.find_slingshot_mbr()[0]
        sling.width, sling.height = sling.height, sling.width
        self.sling_center = self.tp.get_reference_point(sling)

    def update_no_of_levels(self):
        levels = self.ar.get_number_of_levels()

        if levels > len(self.solved):
            for n in range(len(self.solved), levels):
                self.solved.append(0)

        if levels < len(self.solved):
            self.solved = self.solved[:levels]

        print('No of Levels: ' + str(levels))
        return levels

    def get_next_level(self):
        level = (self.current_level + 1) % len(self.solved)
        if level == 0:
            level = len(self.solved)
        return level

    def check_my_score(self):
        """
         * Run the Client (Naive Agent)
        *"""
        scores = self.ar.get_all_level_scores()
        print(" My score: ")
        level = 1
        for i in scores:
            print(" level ", level, "  ", i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1

    def run(self):
        win_count = 0
        while True:
            try:

                tf.reset_default_graph()
                init = tf.global_variables_initializer()
                state_maker = StateMaker()

                if not os.path.exists(MODEL_PATH):
                    os.makedirs(MODEL_PATH)

                with tf.Session() as sess:
                    sess.run(init)

                    saver = tf.train.import_meta_graph(MODEL_PATH + '/model-ddqn.ckpt.meta')
                    saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH + '/'))
                    graph = tf.get_default_graph()
                    online_in = graph.get_tensor_by_name('X_1:0')

                    # Run loop
                    info = self.ar.configure(self.id)
                    self.solved = [0 for x in range(self.ar.get_number_of_levels())]

                    max_scores = np.zeros([len(self.solved)])

                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)

                    if self.current_level >= 457:
                        print("Shutong: win_count: ", win_count)

                    s = 'None'
                    d = False
                    first_time_in_level_in_episode = True

                    for env_step in range(1, TOTAL_STEPS):
                        game_state = self.ar.get_game_state()
                        r = self.ar.get_current_score()

                        print ('current score ', r)
                        print('win count: ', win_count)

                        #if(game_state == GameState.UNSTABLE):
                        #    self.get_next_level()

                        # First check if we are in the won or lost state
                        # to adjust the reward and done flag if needed
                        if game_state == GameState.WON:
                            # save current state before reloading the level
                            s = self.ar.do_screenshot()
                            s = state_maker.make(sess, s)
                            n_levels = self.update_no_of_levels()
                            print("number of levels " , n_levels)
                            self.check_my_score()
                            self.current_level = self.get_next_level()
                            self.ar.load_level(self.current_level)

                            win_count += 1

                            # Update reward and done
                            d = 1
                            first_time_in_level_in_episode = True


                        elif game_state == GameState.LOST:
                            # save current state before reloading the level
                            s = self.ar.do_screenshot()
                            s = state_maker.make(sess, s)

                            # check for change of number of levels in the game
                            n_levels = self.update_no_of_levels()
                            print("number of levels " , n_levels)
                            self.check_my_score()
                            # If lost, then restart the level
                            self.failed_counter += 1
                            if self.failed_counter > 1:  # for testing , go directly to the next level

                                self.failed_counter = 0
                                self.current_level = self.get_next_level()
                                self.ar.load_level(self.current_level)
                            else:
                                print("restart")
                                self.ar.load_level(self.current_level)

                            # Update reward and done
                            d = 1
                            first_time_in_level_in_episode = True


                        if(game_state == GameState.PLAYING):
                            # Start of the episode
                            if (first_time_in_level_in_episode):
                                # If first time in level reset states
                                s = 'None'
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                self.ar.fully_zoom_out()
                                first_time_in_level_in_episode = False

                            self.get_slingshot_center()

                            if self.sling_center == None:
                                print ('sling ', self.sling_center)
                                continue
                            s = self.ar.do_screenshot()
                            s = state_maker.make(sess, s)

                            a = sess.run(graph.get_tensor_by_name('ArgMax_1:0'), feed_dict={online_in: [s]})

                            tap_time = int(random.randint(65, 100))
                            ax_pixels = -int(40 * math.cos(math.radians(a)))
                            ay_pixels = int(40 * math.sin(math.radians(a)))

                            print("Shoot: " + str(ax_pixels) + ", " + str(ay_pixels) + ", " + str(tap_time))
                            # Execute a in the environment
                            self.ar.shoot(int(self.sling_center.X), int(self.sling_center.Y), ax_pixels,
                                               ay_pixels, 0, tap_time, False)

                        elif game_state == GameState.LEVEL_SELECTION:
                            print("unexpected level selection page, go to the last current level : ", self.current_level)
                            self.ar.load_level(self.current_level)

                        elif game_state == GameState.MAIN_MENU:
                            print("unexpected main menu page, reload the level : ", self.current_level)
                            self.ar.load_level(self.current_level)

                        elif game_state == GameState.EPISODE_MENU:
                            print("unexpected episode menu page, reload the level: ", self.current_level)
                            self.ar.load_level(self.current_level)


            except Exception as e:
                print("Error: ", e)
            finally:
                time.sleep(10)
