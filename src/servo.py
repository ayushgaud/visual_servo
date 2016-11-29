#!/usr/bin/env python

import math
import caffe
import numpy as np
import tf
import time
import roslib
import sys
import rospy
import cv2
import std_msgs.msg
from sensor_msgs.msg import Image
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from ftplib import FTP
import rospkg

class image_converter:

  def __init__(self):

    caffe.set_device(0)
    caffe.set_mode_gpu()
    rospack = rospkg.RosPack()

    model_def = rospack.get_path('visual_servo') + '/model/deployfnet.prototxt'
    model_weights = rospack.get_path('visual_servo') + '/model/flownet_step_e-4ssize30k_iter_75000.caffemodel'

    global net
    net = caffe.Net(model_def,      # defines the structure of the model
                  model_weights,  # contains the trained weights
                  caffe.TEST)     # use test mode (e.g., don't perform dropout)
    global transformer
    transformer = caffe.io.Transformer({'img0': net.blobs['img0'].data.shape,'img1': net.blobs['img1'].data.shape})
    transformer.set_transpose('img0', (2,0,1))
    transformer.set_transpose('img1', (2,0,1))
    print("Network Initialization Complete!")
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/bebop/image_raw",Image,self.callback)
    self.pose_pub = rospy.Publisher("/bebop/self.cmd_vel", geometry_msgs.msg.Twist, queue_size=1)
    #self.img_cap = rospy.Publisher("/bebop/snapshot", std_msgs.msg.Empty, queue_size=1)
    #self.img_cap.publish()
    rospy.Subscriber('/bebop/odom',Odometry,self.pose_callback)
    
    rospy.Timer(rospy.Duration(0.05), self.command)
    rospy.Timer(rospy.Duration(0.1), self.process)
    
    self.flag = 0
    self.img2=cv2.imread('f.png')  
    self.img2 = cv2.resize(self.img2,(512,384), interpolation = cv2.INTER_LINEAR)
    self.img1 = self.img2

  def callback(self,data):
    try:
       global cv_image
       self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
       self.img1 = cv2.resize(self.cv_image,(512,384), interpolation = cv2.INTER_LINEAR)
    except CvBridgeError, e:
      print e
  def command(self,event=None):
    try:
      self.error = abs(self.curr_pose.position.x - self.cmd.linear.x) + abs(self.curr_pose.position.y - self.cmd.linear.y) + abs(self.curr_pose.position.z - self.cmd.linear.z)
      thr = 5e-2
      if self.error >= thr:
        #self.pose_pub.publish(self.cmd)
        print("self.error: ", self.error)
    except Exception as e:
      print e.message, e.args
      self.error = 0

  def pose_callback(self,data):
    self.curr_pose = data.pose.pose
  
  def process(self,event=None):
    try:

      t1 = rospy.get_rostime()
      #cv2.imshow("Image_Filtered", self.img1)
      #cv2.waitKey(3)
      

      net.blobs['img0'].data[...] = transformer.preprocess('img0', self.img1)
      net.blobs['img1'].data[...] = transformer.preprocess('img1', self.img2)
      tic=time.time()
      output=net.forward()
      toc=time.time()
      posexyz = output['fc_pose_xyz'][0]
      poseq= output['fc_pose_wpqr'][0]
      #pose=np.zeros(7)
      #pose[0:3]=posexyz
      #pose[3:]=poseq
      #print pose

      lmda1 = 0.7
      lmda2 = 0.07
      thresh = 0.1
      # threshold for velocity values
      
      quaternion = (poseq[1], poseq[2], poseq[3], poseq[0])
      q_inv = tf.transformations.quaternion_inverse(quaternion)

      r_mat = tf.transformations.quaternion_matrix(q_inv)
      trans = -posexyz/1000

      angle, direction, point = tf.transformations.rotation_from_matrix(r_mat)
      tvel = -1 * lmda1 * np.dot(np.transpose(r_mat[0:3, 0:3]), trans)
      q_vel = tf.transformations.quaternion_about_axis(-1 * lmda2 * angle, direction)
      euler = tf.transformations.euler_from_quaternion(q_vel)
      
      # camera frame to ENU
      #y = -x
      #z = -y
      #x = z

      self.cmd = geometry_msgs.msg.Twist()
      self.cmd.linear.x =  np.sign(tvel[2])*min(abs(tvel[2]),thresh)
      self.cmd.linear.y = -np.sign(tvel[0])*min(abs(tvel[0]),thresh)
      self.cmd.linear.z = -np.sign(tvel[1])*min(abs(tvel[1]),thresh)

      #self.cmd.angular.x = euler[2]
      #self.cmd.angular.y = -euler[0]
      self.cmd.angular.z = -euler[1]

      r = rospy.Rate(30) # 30hz

      #print("Command: ", self.cmd)
      #print("Curr_pose: ",self.curr_pose)

    except Exception as e:
      print e.message, e.args
def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  try:
    rospy.spin()  
  except KeyboardInterrupt:
    print "Shutting down"
  cv2.destroyWindow("Image")
if __name__ == '__main__':
    main(sys.argv)
