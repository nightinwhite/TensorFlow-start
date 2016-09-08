#coding: utf-8
import numpy as np
import cv2
import math

class MLS():
    def __init__(self, P, Q, alpha = 0.5):
        self.alpha = 0.5
        self.P = P
        self.Q = Q

    def get_newpos(self, v):
        #print "v：{0}".format(v)
        w = []
        P_s = [0, 0]  # p*
        Q_s = [0, 0]  # Q*
        w_sum = 0
        for i in range(len(self.P)):
            if v == self.P[i]:
                return self.Q[i]
            wi = 1./((v[0]-self.P[i][0]) ** 2 + (v[1]-self.P[i][1]) ** 2) ** self.alpha
            w.append(wi)
            P_s[0] += wi * self.P[i][0]
            P_s[1] += wi * self.P[i][1]
            Q_s[0] += wi * self.Q[i][0]
            Q_s[1] += wi * self.Q[i][1]
            w_sum += wi
        #print "w：{0}".format(w)

        P_s[0] /= w_sum
        P_s[1] /= w_sum
        Q_s[0] /= w_sum
        Q_s[1] /= w_sum
        #print "P_s：{0}".format(P_s)
        #print "Q_s：{0}".format(Q_s)

        P_u = []   # P^
        Q_u = []   # Q^
        for i in range(len(self.P)):
            P_u.append([self.P[i][0] - P_s[0], self.P[i][1] - P_s[1]])
            Q_u.append([self.Q[i][0] - Q_s[0], self.Q[i][1] - Q_s[1]])
        #print "P_u：{0}".format(P_u)
        #print "Q_u：{0}".format(Q_u)

        fr = np.asarray([0, 0],dtype=np.float64)    #Ai
        for i in range(len(self.P)):
            tmp_one = w[i]
            tmp_two = np.asarray([[P_u[i][0], P_u[i][1]], [P_u[i][1], -P_u[i][0]]])
            tmp_three = np.asarray([[v[0]-P_s[0], v[1]-P_s[1]],[v[1]-P_s[1], -(v[0]-P_s[0])]])
            Ai = (tmp_one*tmp_two.dot(tmp_three.transpose()))
            #print "Ai：{0}".format(Ai)
            fr += np.asarray(Q_u[i]).dot(Ai)
        #print "fr：{0}".format(fr)
        tmp_one = math.sqrt((v[0]-P_s[0]) ** 2 + (v[1]-P_s[1]) ** 2)
        tmp_two = math.sqrt(fr[0] ** 2+fr[1] ** 2)
        tmp_three = np.asarray(Q_s)
        if tmp_two == 0:
            return tmp_three
        return (tmp_one/tmp_two)*fr+tmp_three

    def transpose(self, image):
        image = image.transpose((1, 0, 2))
        image_x = image.shape[0]
        image_y = image.shape[1]
        image[50, :] = [0, 0, 0]
        image[:, 50] = [0, 0, 0]
        new_image = np.full((image_x+100, image_y+100, 3), 255, dtype=np.uint8)
        for i in range(image_x):
            for j in range(image_y):
                [m, n] = self.get_newpos([i, j])
                new_image[int(m), int(n)] = image[i, j]
        new_image = new_image.transpose((1, 0, 2))

        return new_image






        pass
