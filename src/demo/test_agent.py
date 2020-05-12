import sys

sys.path.append('./src')
from datetime import datetime
import time
import numpy as np

sys.path.append('..')
from threading import Thread
import random
import json
import socket
from os import system, rename
from math import cos, sin, degrees, pi
from client.agent_client import AgentClient, GameState, RequestCodes
from trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner
from computer_vision.GroundTruthReader import GroundTruthReader, NotVaildStateError
from computer_vision.game_object import GameObjectType
from utils.point2D import Point2D
import logging


class ClientNaiveAgent(Thread):
    """Naive agent (server/client version)"""

    def __init__(self, name_offset, difficulty):
        # Wrapper of the communicating messages

        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)

        self.ar = AgentClient(**sc_json_config[0])

        with open('./src/client/server_observer_client_config.json', 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)

        # the observer agent can only execute 6 command: configure, screenshot
        # and the four groundtruth related ones
        self.observer_ar = AgentClient(**observer_sc_json_config[0])

        try:
            self.ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        try:
            self.observer_ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        self.current_level = -1
        self.failed_counter = 0
        self.solved = []
        self.tp = SimpleTrajectoryPlanner()
        self.id = 28888
        self.first_shot = True
        self.prev_target = None

        # initialise colormap for the ground truth reader
        f = open('./ColorMap.json', 'r')
        result = json.load(f)

        self.look_up_matrix = np.zeros((len(result), 256))
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

            obj_number += 1

        # normalise the look_up_matrix
        self.look_up_matrix = self.look_up_matrix / np.sqrt((self.look_up_matrix ** 2).sum(1)).reshape(-1, 1)
        self._logger = logging.getLogger("ClientNaiveAgent")

        # shutong: my parameters
        self.max_win = 5
        self.init_angle = 300
        self.last_angle = 1200
        self.step = 25 * difficulty
        self.level_name_offset = name_offset

        # shutong: variables for test
        self.angle = self.init_angle
        self.win_time = 0
        self.win_angle = []
        self.prev_bird = ''
        self.target_num_of_bird = 0  # 4 birds, 0: red, 1: blue, 2: yellow, 3: black
        self.current_num_of_bird = 0  # same as above

    def clean_parameter_next_bird(self):
        self.angle = self.init_angle
        self.win_time = 0
        self.win_angle = []
        self.target_num_of_bird += 1
        self.current_num_of_bird = 0

    def clean_parameter_next_level(self):
        self.angle = self.init_angle
        self.win_time = 0
        self.win_angle = []
        self.target_num_of_bird = 0  # 4 birds, 0: red, 1: blue, 2: yellow, 3: black
        self.current_num_of_bird = 0
        self.first_shot = True
        # at most 99 levels, more levels will cause counting problems
        if self.current_level == 99:
            sys.exit(0)

    def clean_parameter_restart(self):
        self.current_num_of_bird = 0

    def write_result(self):
        real_level = self.current_level + 100 * self.level_name_offset
        f = open("good_buildings/level_result-%s.txt" % real_level, "w")
        for i in self.win_angle:
            f.write(str(i[0]) + ' ' + str(i[1]) + '\n')
        f.close()
        # windows:
        system(
            "copy \"ScienceBirds\\sciencebirds_win\\Science Birds_Data\\StreamingAssets\\Levels\\novelty_level_0\\type1\\Levels\\level-" + str(
                self.current_level).zfill(2) + ".xml\" good_buildings")
        system(
            "copy \"ScienceBirds\\sciencebirds_win\\Science Birds_Data\\StreamingAssets\\Levels\\novelty_level_0\\type1\\Levels\\level-" + str(
                self.current_level).zfill(2) + ".txt\" good_buildings")
        # linux:
        # system(
        #     "cp ScienceBirds/sciencebirds_linux/sciencebirds_linux_Data/StreamingAssets/Levels/novelty_level_0/type1/Levels/level-" + str(
        #         self.current_level).zfill(2) + ".xml good_buildings")
        # system(
        #     "cp ScienceBirds/sciencebirds_linux/sciencebirds_linux_Data/StreamingAssets/Levels/novelty_level_0/type1/Levels/level-" + str(
        #         self.current_level).zfill(2) + ".txt good_buildings")
        # rename("good_buildings/level-" + str(self.current_level).zfill(2) + ".txt", "good_buildings/level-" + str(self.level_name_offset) + str(self.current_level).zfill(2) + ".txt")
        # rename("good_buildings/level-" + str(self.current_level).zfill(2) + ".xml", "good_buildings/level-" + str(self.level_name_offset) + str(self.current_level).zfill(2) + ".xml")


    def sample_state(self, request=RequestCodes.GetNoisyGroundTruthWithScreenshot, frequency=0.5):
        """
         sample a state from the observer agent
         this method allows to be run in a different thread
         NOTE: Setting the frequency too high, i.e. <0.01 may cause lag in science birds game
               due to the calculation of the groundtruth
        """
        while (True):
            vision = None
            if request == RequestCodes.GetGroundTruthWithScreenshot:
                image, ground_truth = self.observer_ar.get_ground_truth_with_screenshot()
                # set to true to ignore invalid state and return the vision object regardless
                # of #birds and #pigs
                vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)
                vision.set_screenshot(image)

            elif request == RequestCodes.GetGroundTruthWithoutScreenshot:
                ground_truth = self.observer_ar.get_ground_truth_without_screenshot()
                vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)

            elif request == RequestCodes.GetNoisyGroundTruthWithScreenshot:
                image, ground_truth = self.observer_ar.get_noisy_ground_truth_with_screenshot()
                vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)
                vision.set_screenshot(image)

            elif request == RequestCodes.GetNoisyGroundTruthWithoutScreenshot:
                ground_truth = self.observer_ar.get_noisy_ground_truth_without_screenshot()
                vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)
            time.sleep(frequency)

    def get_next_level(self):
        level = 0
        unsolved = False
        # all the level have been solved, then get the first unsolved level
        for i in range(len(self.solved)):
            if self.solved[i] == 0:
                unsolved = True
                level = i + 1
                if level <= self.current_level and self.current_level < len(self.solved):
                    continue
                else:
                    return level

        if unsolved:
            return level
        level = (self.current_level + 1) % len(self.solved)
        if level == 0:
            level = len(self.solved)
        return level

    def check_my_score(self):
        """
         * Run the Client (Naive Agent)
        *"""
        scores = self.ar.get_all_level_scores()
        # print(" My score: ")
        level = 1
        for i in scores:
            print(" level ", level, "  ", i)
            if i > 0:
                self.solved[level - 1] = 1
            level += 1
        return scores

    def update_no_of_levels(self):
        # check the number of levels in the game
        n_levels = self.ar.get_number_of_levels()

        # if number of levels has changed make adjustments to the solved array
        if n_levels > len(self.solved):
            for n in range(len(self.solved), n_levels):
                self.solved.append(0)

        if n_levels < len(self.solved):
            self.solved = self.solved[:n_levels]

        print('No of Levels: ' + str(n_levels))

        return n_levels

    def run(self):
        sim_speed = 50
        self.ar.configure(self.id)
        self.observer_ar.configure(self.id)
        self.ar.set_game_simulation_speed(sim_speed)
        n_levels = self.update_no_of_levels()

        self.solved = [0 for x in range(n_levels)]

        # load the initial level (default 1)
        # Check my score
        self.check_my_score()

        self.current_level = self.get_next_level()
        self.ar.load_level(self.current_level)

        '''
        Uncomment this section to run TEST for requesting groudtruth via different thread
        '''
        gt_thread = Thread(target=self.sample_state)
        gt_thread.start()
        '''
        END TEST
        '''

        # ar.load_level((byte)9)
        while True:
            # test for multi-thread groundtruth reading

            print('solving level: {}'.format(self.current_level))
            if self.first_shot:
                # shutong: to detect the unstable building
                self.first_shot = False
                state = self.solve(-450)
                if self.ar.get_current_score() > 0:
                    self.clean_parameter_next_level()
                    n_levels = self.update_no_of_levels()
                    self.check_my_score()
                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)
                    self.tp = SimpleTrajectoryPlanner()
                else:
                    self.clean_parameter_restart()
                    self.win_time = 0
                    self.win_angle = []
                    self.ar.load_level(self.current_level)
                continue
            while self.current_num_of_bird < self.target_num_of_bird:
                state = self.solve(-450)
                print("+1")
                self.current_num_of_bird += 1

            state = self.solve(self.angle)
            if self.ar.get_current_score() > 0:
                self.angle += self.step
            else:
                self.angle += self.step * 3

            # If the level is solved , go to the next level
            if state == GameState.WON:
                self.win_time += 1
                self.win_angle.append((self.target_num_of_bird, self.angle))

                if self.target_num_of_bird == 3:
                    print("good level, only black work")
                    self.write_result()
                    self.clean_parameter_next_level()

                    n_levels = self.update_no_of_levels()
                    # /System.out.println(" loading the level " + (self.current_level + 1) )
                    self.check_my_score()
                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)
                    self.tp = SimpleTrajectoryPlanner()
                elif self.win_time > self.max_win and self.target_num_of_bird == 0:
                    # even the weak one can knock it down many times, go to the next one
                    print("too easy, next level")
                    self.clean_parameter_next_level()

                    n_levels = self.update_no_of_levels()
                    # /System.out.println(" loading the level " + (self.current_level + 1) )
                    self.check_my_score()
                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)
                    self.tp = SimpleTrajectoryPlanner()
                elif self.angle >= self.last_angle and (self.win_time <= self.max_win or self.target_num_of_bird > 0):
                    # all birds now treated as black except red
                    print("good building")
                    self.write_result()
                    self.clean_parameter_next_level()

                    n_levels = self.update_no_of_levels()
                    # /System.out.println(" loading the level " + (self.current_level + 1) )
                    self.check_my_score()
                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)
                    self.tp = SimpleTrajectoryPlanner()
                else:
                    print("win restart")
                    self.clean_parameter_restart()

                    # print(self.target_num_of_bird)
                    self.ar.load_level(self.current_level)

            elif state == GameState.LOST:
                n_levels = self.update_no_of_levels()

                if self.angle >= self.last_angle:
                    print("even black bird fails, go next level")
                    self.clean_parameter_next_level()

                    n_levels = self.update_no_of_levels()
                    # /System.out.println(" loading the level " + (self.current_level + 1) )
                    self.check_my_score()
                    self.current_level = self.get_next_level()
                    self.ar.load_level(self.current_level)
                    self.tp = SimpleTrajectoryPlanner()
                else:
                    print("lose restart")
                    self.clean_parameter_restart()
                    self.ar.load_level(self.current_level)

            elif state == GameState.LEVEL_SELECTION:
                print("unexpected level selection page, go to the last current level : " \
                      , self.current_level)
                self.ar.load_level(self.current_level)

            elif state == GameState.MAIN_MENU:
                print("unexpected main menu page, reload the level : " \
                      , self.current_level)
                self.ar.load_level(self.current_level)

            elif state == GameState.EPISODE_MENU:
                print("unexpected episode menu page, reload the level: " \
                      , self.current_level)
                self.ar.load_level(self.current_level)

            else:
                # not winning or lossing
                # need to be clean up
                if self.angle >= self.last_angle:
                    if self.win_time > self.max_win and self.target_num_of_bird == 0:
                        print("too easy, go next level")
                        self.clean_parameter_next_level()

                        n_levels = self.update_no_of_levels()
                        # /System.out.println(" loading the level " + (self.current_level + 1) )
                        self.check_my_score()
                        self.current_level = self.get_next_level()
                        self.ar.load_level(self.current_level)
                        self.tp = SimpleTrajectoryPlanner()
                    elif self.win_time <= 0:
                        print("too hard, go next bird")
                        self.clean_parameter_next_bird()
                        self.ar.load_level(self.current_level)
                    else:
                        print("good buildings")
                        self.write_result()
                        self.clean_parameter_next_level()

                        n_levels = self.update_no_of_levels()
                        # /System.out.println(" loading the level " + (self.current_level + 1) )
                        self.check_my_score()
                        self.current_level = self.get_next_level()
                        self.ar.load_level(self.current_level)
                        self.tp = SimpleTrajectoryPlanner()
                else:
                    self.clean_parameter_restart()
                    self.ar.load_level(self.current_level)



    def _updateReader(self, dtype):
        '''
        update the ground truth reader with 4 different types of ground truth if the ground truth is vaild
        otherwise, return the state.

        str type : groundTruth_screenshot , groundTruth, NoisygroundTruth_screenshot,NoisygroundTruth
        '''

        self.showGroundTruth = False

        if dtype == 'groundTruth_screenshot':
            image, ground_truth = self.ar.get_ground_truth_with_screenshot()
            vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)
            vision.set_screenshot(image)
            self.showGroundTruth = True  # draw the ground truth with screenshot or not

        elif dtype == 'groundTruth':
            ground_truth = self.ar.get_ground_truth_without_screenshot()

            vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)

        elif dtype == 'NoisygroundTruth_screenshot':
            image, ground_truth = self.ar.get_noisy_ground_truth_with_screenshot()
            vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)
            vision.set_screenshot(image)
            self.showGroundTruth = True  # draw the ground truth with screenshot or not

        elif dtype == 'NoisygroundTruth':
            ground_truth = self.ar.get_noisy_ground_truth_without_screenshot()
            vision = GroundTruthReader(ground_truth, self.look_up_matrix, self.look_up_obj_type)

        return vision

    def solve(self, angle):
        """
        * Solve a particular level by shooting birds directly to pigs
        * @return GameState: the game state after shots.
        """

        ground_truth_type = 'groundTruth'

        vision = self._updateReader(ground_truth_type)
        if not vision.is_vaild():
            return self.ar.get_game_state()

        if self.showGroundTruth:
            vision.showResult()

        sling = vision.find_slingshot_mbr()[0]
        # TODO: look into the width and height issue of Traj planner
        sling.width, sling.height = sling.height, sling.width

        # get all the pigs
        pigs = vision.find_pigs_mbr()
        print("no of pigs: " + str(len(vision.find_pigs_mbr())))
        for pig in pigs:
            print("pig location: " + str(pig.get_centre_point()))
        state = self.ar.get_game_state()

        # if there is a sling, then play, otherwise skip.
        if sling != None:
            # If there are pigs, we pick up a pig randomly and shoot it.
            if pigs:
                release_point = None
                # random pick up a pig
                pig = pigs[random.randint(0, len(pigs) - 1)]
                temp_pt = pig.get_centre_point()

                # TODO change computer_vision.cv_utils.Rectangle
                # to be more intuitive
                _tpt = Point2D(temp_pt[1], temp_pt[0])

                # if the target is very close to before, randomly choose a
                # point near it
                if self.prev_target != None and self.prev_target.distance(_tpt) < 10:
                    _angle = random.uniform(0, 1) * pi * 2
                    _tpt.X = _tpt.X + int(cos(_angle)) * 10
                    _tpt.Y = _tpt.Y + int(sin(_angle)) * 10
                    print("Randomly changing to ", _tpt)

                self.prev_target = Point2D(_tpt.X, _tpt.Y)

                ################estimate the trajectory###################
                print('################estimate the trajectory###################')

                pts = self.tp.estimate_launch_point(sling, _tpt)
                pts = []

                if not pts:
                    # Add logic to deal with unreachable target
                    print("the target is not reachable directly with the birds")
                    print("no trajectory can be provided")
                    print("just shoot...")
                    release_point = Point2D(-100, angle)


                elif len(pts) == 1:
                    release_point = pts[0]
                elif len(pts) == 2:
                    # System.out.println("first shot " + first_shot)
                    # randomly choose between the trajectories, with a 1 in
                    # 6 chance of choosing the high one
                    if random.randint(0, 5) == 0:
                        release_point = pts[1]
                    else:
                        release_point = pts[0]

                ref_point = self.tp.get_reference_point(sling)

                # Get the release point from the trajectory prediction module
                tap_time = 0
                if release_point != None:
                    release_angle = self.tp.get_release_angle(sling, release_point)
                    print("Release Point: ", release_point)
                    print("Release Angle: ", degrees(release_angle))
                    tap_interval = 0

                    birds = vision.find_birds()
                    bird_on_sling = vision.find_bird_on_sling(birds, sling)
                    bird_type = bird_on_sling.type
                    print("bird_type ", bird_type)
                    if bird_type == GameObjectType.REDBIRD:
                        tap_interval = 0  # start of trajectory
                    elif bird_type == GameObjectType.YELLOWBIRD:
                        tap_interval = 65 + random.randint(0, 24)  # 65-90% of the way
                    elif bird_type == GameObjectType.WHITEBIRD:
                        tap_interval = 50 + random.randint(0, 19)  # 50-70% of the way
                    elif bird_type == GameObjectType.BLACKBIRD:
                        tap_interval = 0  # do not tap black bird
                    elif bird_type == GameObjectType.BLUEBIRD:
                        tap_interval = 65 + random.randint(0, 19)  # 65-85% of the way
                    else:
                        print("should not happen")
                        tap_interval = 60

                    tap_time = self.tp.get_tap_time(sling, release_point, _tpt, tap_interval)

                else:
                    print("No Release Point Found")
                    return self.ar.get_game_state()

                # check whether the slingshot is changed. the change of the slingshot indicates a change in the scale.
                self.ar.fully_zoom_out()

                vision = self._updateReader(ground_truth_type)
                if isinstance(vision, GameState):
                    return vision
                if self.showGroundTruth:
                    vision.showResult()

                _sling = vision.find_slingshot_mbr()[0]
                _sling.width, _sling.height = _sling.height, _sling.width

                if _sling != None:
                    scale_diff = (sling.width - _sling.width) ** 2 + (sling.height - _sling.height) ** 2
                    if scale_diff < 25:
                        dx = int(release_point.X - ref_point.X)
                        dy = int(release_point.Y - ref_point.Y)

                        if dx < 0:
                            print('ref point ', ref_point.X, ',', ref_point.Y)
                            self.ar.shoot(ref_point.X, ref_point.Y, dx, dy, 0, tap_time, False)
                            state = self.ar.get_game_state()
                            if state == GameState.PLAYING:
                                vision = self._updateReader(ground_truth_type)
                                if isinstance(vision, GameState):
                                    return vision
                                if self.showGroundTruth:
                                    vision.showResult()
                    else:
                        print("Scale is changed, can not execute the shot, will re-segement the image")
                else:
                    print("no sling detected, can not execute the shot, will re-segement the image")
        return state


if __name__ == "__main__":
    na = ClientNaiveAgent()
    na.run()
