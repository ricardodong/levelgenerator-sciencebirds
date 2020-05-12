import os
from combine_structures import read_level_property
from combine_structures import read_level_xml
from combine_structures import read_test_result
from random import uniform

bird_names = {'1':"BirdRed", '2':"BirdBlue", '3':"BirdYellow", '4':"BirdBlack", '5':"BirdWhite"}

good_buildings = "good buildings"
good_buildings_number = []
all_files = []
for parent, dirnames, filenames in os.walk(good_buildings):
    all_files = filenames
for filename in all_files:
    if ".xml" in filename:
        good_buildings_number.append(filename.split("-")[1].split(".")[0])

ground_count = 0
platform_count = 0
TNT_count = 0
for level_number in good_buildings_number:
    f_level = open("good buildings/level-" + level_number + ".xml", "r")
    f_property = open("good buildings/level-" + level_number + ".txt", "r")
    if level_number[0] == '0':
        f_bird = open("good buildings/level_result-" + level_number[1] + ".txt", "r")
    else:
        f_bird = open("good buildings/level_result-" + level_number + ".txt", "r")
    item_lines = read_level_xml(f_level)
    level_property = read_level_property(f_property)
    bird_type, shooting_angles = read_test_result(f_bird)
    # print(bird_type)
    if bird_type:
        for line in item_lines:
            if 'TNT' in line:
                TNT_count += 1
                break

print("TNT_count: ", TNT_count)
print("ground_count: ", ground_count)
print("platform_count: ", platform_count)