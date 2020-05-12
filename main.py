import sys
from os import system

import src.demo.prove_agent as pr
import src.demo.single_building_prove_agent as sbpr
import src.demo.test_agent as test
import src.demo.ddqn_agent as ddq
import src.demo.naive_agent_groundtruth as na


def prove():
	print("sb prove running")
	agent = sbpr.ClientNaiveAgent()
	agent.run()


def main(*args):
	print("test running")
	agent = test.ClientNaiveAgent(name_offset=0, difficulty=1)
	agent.run()


# useless
def file_test():
	real_level = 1
	system(
		"copy \"ScienceBirds\\sciencebirds_win\\Science Birds_Data\\StreamingAssets\\Levels\\novelty_level_0\\type1\\Levels\\level-" + str(
			real_level).zfill(2) + ".xml\" good_buildings")
	system(
		"copy \"ScienceBirds\\sciencebirds_win\\Science Birds_Data\\StreamingAssets\\Levels\\novelty_level_0\\type1\\Levels\\level-" + str(
			real_level).zfill(2) + ".txt\" good_buildings")


if __name__ == "__main__":
	prove()
