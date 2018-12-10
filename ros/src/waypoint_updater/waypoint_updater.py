#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish
MAX_DECEL = 0.5 # max deceleration, acceleration should not exceed 10 m/s^2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb) # subscriber for /current_pose
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb) # subscriber for /base_waypoints
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb) # subscriber for /traffic_waypoint
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb) # subscriber for /obstacle_waypoint
        
        # publisher for /final_waypoints
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # member variables
        self.base_lane = None
        self.pose = None
        # self.base_waypoints = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None

        # make a loop to get the target frequenzy
        self.loop()

    ### loop to get the target frequency
    def loop(self):
        rate = rospy.Rate(5) # keep the rate of the loop by 50 Hz
        while not rospy.is_shutdown():
            if self.pose and self.base_lane:
                # Publish waypoints
                self.publish_waypoints()
            rate.sleep()

    ### find the next waypoint in front of the car
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x # current x position
        y = self.pose.pose.position.y # current y position
        closest_idx = self.waypoint_tree.query([x, y],1)[1] # index of nearest waypoint

        # check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx] # coordinates of nearest waypoints
        prev_coord = self.waypoints_2d[closest_idx - 1] # coordinates of the waypoint in front of the nearest waypoint

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord) # cloest waypoint
        prev_vect = np.array(prev_coord) # waypoint in front of the nearest waypoint
        pos_vect = np.array([x,y]) # current position

        # get the dot product of the following 2 vectors
        # 1. vector from nearest waypoint to the waypoint in front of it
        # 2. vector from current position to next waypoint
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect) 
        # if dot product is positive, the closest waypoint is behind the current position
        # so the next waypoint will be used
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        # returns the id of the closest waypoint in front of the car
        return closest_idx

    ### publishes the waypoints of the generated lane
    def publish_waypoints(self):
        # use generare_lane() to generate a lane of the next waypoints
        final_lane = self.generate_lane()
        # publish the next waypoints to final_waypoints
        self.final_waypoints_pub.publish(final_lane)

    ### generate a new lane
    def generate_lane(self):
        lane = Lane() # create a new lane
        # find the id of the closest waypoint in front of the car with get_closest_waypoint()
        closest_idx = self.get_closest_waypoint_idx()
        # calculate the id of the farthest waypoint
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        # base_waypoints = self.base_waypoints[closest_idx:farthest_idx]
        # get the waypoints from closest to farthest
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        # check if there is a stopline between closest to farthest waypoint
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints # use the waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx) # decelerate to stop the car
        return lane

    ### set waypoints to decelerate
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = [] # base waypoint should not be modified
        for i, wp in enumerate(waypoints):
            p = Waypoint() # create new waypoint
            p.pose = wp.pose # set the position of the current waypoint
            # get the id of the stop point
            # substact 2 to stop there with the nose of the car
            stop_idx = max(self.stopline_wp_idx - closest_idx -2,0)
            # calculate the distance to the stop index; will be 0, if stop point is behind
            dist = self.distance(waypoints, i, stop_idx)
            # calculate the velocity for deceleration; is dependent to the distance
            vel = math.sqrt(2 * MAX_DECEL * dist)
            # if the velocity is small enough, it will be set to 0
            if vel < 1.:
                vel = 0.
            # set the velocity for the waypoint
            # won't be greater than the speed limit for the waypoint
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            # append the waypoint with the (new) velocity to temp
            temp.append(p)
        return temp
    
    ### stores the pose and will be called frequently
    def pose_cb(self, msg):
        self.pose = msg

    ### callback function to store the waypoints for the subscriber
    # will be called once
    def waypoints_cb(self, waypoints):
        self.base_lane = waypoints
        if not self.waypoints_2d: # must be initialized first
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d) # to find the closest waypoint faster

    ### Callback for /traffic_waypoint message
    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    ### Callback for /obstacle_waypoint message
    def obstacle_cb(self, msg):
        self.stopline_wp_idx = msg.data

    ### gets the linear velocity (x-direction) for a single waypoint
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    ### sets the linear velocity (x-direction) for a single waypoint in a list of waypoints
    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    ### Computes the distance between two waypoints in a list along the piecewise linear arc
    # connecting all waypoints between the two
    # may be helpful in determining the velocities for a sequence of waypoints
    # leading up to a red light
    # (the velocities should gradually decrease to zero starting some distance from the light)
    # will return 0 if wp1 is bigger than wp2
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
