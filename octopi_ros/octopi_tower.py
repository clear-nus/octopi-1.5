#!/usr/bin/env python

import rospy
import random
import pickle
from std_msgs.msg import String, Float64
import numpy as np
from scipy.spatial.transform import Rotation as R
import requests
import ast

experiment_number = 0
experiment_subsection = 0
TIMER = 7

#### TODO replace the port numbers and IP address with your own.

class GSLogger():
    def __init__(self, sub_type="Single"):
        self.tactile_pub = rospy.Publisher('/gsmini_command', String, queue_size=10)
        self.verbose_pub = rospy.Publisher('/verbose_output', String, queue_size=10)
        rospy.Subscriber('/gsmini_command', String, queue_size=1, callback=self.targetCallback)
        self.vis_pub = rospy.Publisher("/vis", String, queue_size=1)

    def targetCallback(self, msg):
        print("receiving data: {}". format(msg.data))
        

        if msg.data == 'describe and rank':
            req = {'object_ids': '1,2,3'}
            suffix = 'describe_and_rank'
            url = 'http://127.0.0.1:8001/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            response = response_dict['response']
            verbose_msg = String()
            verbose_msg.data = response
            self.verbose_pub.publish(verbose_msg)
            self.hardness_rank = response_dict["hardness"]
            self.roughness_rank = response_dict["roughness"]

        elif msg.data == 'reset':
            req = {}
            suffix = msg.data
            url = 'http://127.0.0.1:8001/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            print(response_dict["status"])
            # req = {}
            # suffix = msg.data
            # url = 'http://127.0.0.1:8002/' + suffix ### NEW PORT
            # x = requests.post(url, params = req)
            # # rospy.sleep(TIMER)
            # response_dict = ast.literal_eval(x.content)
            # print(response_dict["status"])

        elif 'describe' in msg.data:
            req = {'object_ids': msg.data.split("describe ")[1]}
            print(req)
            suffix = 'describe'
            url = 'http://127.0.0.1:8002/' + suffix ### NEW PORT
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            print(response_dict)
            verbose_msg = String()
            verbose_msg.data = response_dict['response']
            self.verbose_pub.publish(verbose_msg)
            print(response_dict['response'])

        elif msg.data == 'rank':
            req = {'object_ids': '1,2,3'}
            suffix = msg.data
            url = 'http://127.0.0.1:8001/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            verbose_msg = String()
            verbose_msg.data = response_dict['response']
            self.verbose_pub.publish(verbose_msg)
            print(response_dict['response'])
            self.hardness_rank = response_dict["hardness"]
            self.roughness_rank = response_dict["roughness"]

        elif 'ask' in msg.data:
            req = {'query': msg.data[4:]}
            suffix = 'ask'
            url = 'http://127.0.0.1:8002/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            print(x.content)
            response_dict = ast.literal_eval(x.content)
            print(response_dict['response'])
            verbose_msg = String()
            verbose_msg.data = response_dict['response']
            self.verbose_pub.publish(verbose_msg)

        elif 'prompt' in msg.data:
            prompt = "Identify each object held in the cardboard holder with non-visual details necessary for tactile reasoning, from right to left. Format your answer as 'Object 1: details, object name.\nObject 2:...' with less than 5 words each."
            req = {'prompt': prompt}
            suffix = 'describe_rgb'
            url = 'http://127.0.0.1:8002/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            print(response_dict)
            verbose_msg = String()
            verbose_msg.data = response_dict['response']['generation']
            self.verbose_pub.publish(verbose_msg)
            pass

        elif 'guess from objects' in msg.data:
            req = {'object_candidates': msg.data.split('guess from objects ')[1]}
            suffix = 'guess_from_objects'
            url = 'http://127.0.0.1:8002/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            print(response_dict['response'])
            verbose_msg = String()
            verbose_msg.data = response_dict['response']
            self.verbose_pub.publish(verbose_msg)

        elif 'guess from touch' in msg.data:
            req = {'target_object': msg.data.split('guess from touch ')[1]}
            suffix = 'guess_from_touch'
            url = 'http://127.0.0.1:8002/' + suffix
            x = requests.post(url, params = req)
            # rospy.sleep(TIMER)
            response_dict = ast.literal_eval(x.content)
            print(response_dict['response'])
            verbose_msg = String()
            verbose_msg.data = response_dict['response']
            self.verbose_pub.publish(verbose_msg)

        elif 'publish sort' in msg.data:
            rank_msg = String()
            criteria = msg.data.split('publish sort ')[1]
            sort_criteria = self.hardness_rank if criteria == 'hardness' else self.roughness_rank
            for i in range(len(sort_criteria)):
                rank_msg.data = "movesort {} {}".format(sort_criteria[i], i)
                self.tactile_pub.publish(rank_msg)
                rospy.sleep(10)
            print('done')

        print("done")
        

def main():
    rospy.init_node('command', anonymous=True)
    GS_log = GSLogger()
    print("Listening for requests")
    rospy.spin()

if __name__ == '__main__':
    main()