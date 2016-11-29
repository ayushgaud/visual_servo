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
    #transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    #transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("/bebop/image_raw",Image,self.callback)
    self.pose_pub = rospy.Publisher("/bebop/cmd_vel", geometry_msgs.msg.Twist, queue_size=1)
    self.img_cap = rospy.Publisher("/bebop/snapshot", std_msgs.msg.Empty, queue_size=1)
    self.img_cap.publish()
    rospy.Timer(rospy.Duration(0.1), self.process)
    self.flag = 0

  def callback(self,data):
    try:
       global cv_image
       self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError, e:
      print e
  def process(self,event=None):
    try:

      t1 = rospy.get_rostime()
      #img1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
      ftp = FTP('192.168.42.1')
      ftp.login()
      ftp.cwd('/internal_000/Bebop_2/media')
      data = []
      ftp.dir(data.append)
      a = "\n".join(s for s in data if 'jpg' in s)
      filename = a[a.find('B',a.rfind('jpg') - 45, a.rfind('jpg')) :a.rfind('jpg') + 3]
      print filename
      try:
      	ftp.retrbinary("RETR " + filename ,open(filename, 'wb').write)
      	ftp.delete(filename)
      	ftp.quit()
      
      	img1 = cv2.imread(filename)
      except:
      	self.img_cap.publish()
      	self.process()

      #img = cv2.resize(img,(1536,864), interpolation = cv2.INTER_LINEAR)

      #img1 = img[864*0.5 - 184 - 100:864*0.5 + 184 - 100,1536*0.5 - 320:1536*0.5 + 320]

      #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

      #img1 = cv2.cvtColor(img1,36)
      #y,u,v = cv2.split(img1)

      #v = clahe.apply(v)

      #img1 = cv2.cvtColor(cv2.merge((y,u,v)),38)

      
      img1 = cv2.resize(img1,(512,384), interpolation = cv2.INTER_LINEAR)
      cv2.imshow("Image_Filtered", img1)
      cv2.waitKey(3)
      img2=cv2.imread('f.jpg')

      #img2 = cv2.cvtColor(img2,36)
      #y,u,v = cv2.split(img2)
      
      #v = clahe.apply(v)
      #img2 = cv2.cvtColor(cv2.merge((y,u,v)),38)

      img2 = cv2.resize(img2,(512,384), interpolation = cv2.INTER_LINEAR)
      net.blobs['img0'].data[...] = transformer.preprocess('img0', img1)
      net.blobs['img1'].data[...] = transformer.preprocess('img1', img2)
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

      cmd = geometry_msgs.msg.Twist()
      cmd.linear.x =  np.sign(tvel[2])*min(abs(tvel[2]),thresh)
      cmd.linear.y = -np.sign(tvel[0])*min(abs(tvel[0]),thresh)
      cmd.linear.z = -np.sign(tvel[1])*min(abs(tvel[1]),thresh)

      #cmd.angular.x = euler[2]
      #cmd.angular.y = -euler[0]
      cmd.angular.z = -euler[1]

      r = rospy.Rate(30) # 30hz
      prev = rospy.get_rostime()
      now = rospy.get_rostime()

      if self.flag > 0:
      	print cmd	
      	while now.secs - prev.secs < 4:
		    self.pose_pub.publish(cmd)
		    r.sleep()
		    now = rospy.get_rostime()
		    
      self.flag = 1
      cmd.linear.x = 0
      cmd.linear.y = 0
      cmd.linear.z = 0

      cmd.angular.x = 0
      cmd.angular.y = 0
      cmd.angular.z = 0
      prev = rospy.get_rostime()
      now = rospy.get_rostime()
      
      while now.secs - prev.secs < 0.1:
        self.pose_pub.publish(cmd)
        r.sleep()
        now = rospy.get_rostime()
      self.img_cap.publish() 
      #print now.secs - t1.secs
    except:
      pass
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(3)
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
