import numpy as np
import copy
import math


GREATER_THAN = 0
LESS_THAN = 1
INSIDE = 2

"""
This defines the sliding polygon
"""
class SliderSystem(object):

    """
    Sets up the params for the dynamical system
    """
    def __init__(self, polygon, mass=1.0, fmax=1.0, mu=0.1):
        self.polygon = polygon
        self.x,self.y = self.polygon.getCentroid()
        self.theta = 0
        self.mu = float(mu)
        self.mass = float(mass)
        self.fmax = float(fmax)
        self.trajectory = [ [self.x, self.y, self.theta] ]

        self.bounding_circle_radius = max([math.sqrt((v[0] - self.x) ** 2 + (v[1] - self.y) ** 2) for v in self.polygon.vertex_list])

    def parametrize_by_bounding_circle(self, contact_point, contact_velocity):
        """TODO: add doc"""
        a = (contact_velocity[0]**2 + contact_velocity[1]**2)
        b = (2 * contact_point[0] * contact_velocity[0] + 2 * contact_point[1] * contact_velocity[1])
        c = (contact_point[0] ** 2 + contact_point[1] ** 2 - self.bounding_circle_radius ** 2)
        if (b**2 - 4 * a * c) < 0:
            print("unable to parametrize by bounding circle: line of force does not touch bounding circle")
            return None
        else:
            t1 = (-b + math.sqrt(b**2 - 4 * a * c))/(2*a)
            t2 = (-b - math.sqrt(b**2 - 4 * a * c))/(2*a)
            return [(contact_point[0] + t2 * contact_velocity[0], contact_point[1] + t2 * contact_velocity[1]), \
                    (contact_point[0] + t1 * contact_velocity[0], contact_point[1] + t1 * contact_velocity[1])]


    """
    Calculates the tan of the motion cone boundaries 
    (you don't actually need the angles)
    """
    def _getMotionConeParams(self, contact_point):
        px = self.x - contact_point[0] 
        py = self.y - contact_point[1]
        c = self.fmax/self.mass
        ut = (self.mu * (c**2) - px*py + self.mu* (py**2))\
               / ((c**2) + (py**2) - px*py*self.mu)

        ub = (-self.mu * (c**2) - px*py - self.mu* (py**2))\
               / ((c**2) + (py**2) + px*py*self.mu)
        return (ut, ub)


    """
    Determines the contact mode >, <, or inside
    """
    def _getContactMode(self, contact_point, contact_velocity):

        vn, vt = self.polygon.getContactVel(contact_point, contact_velocity)
        ut,ub = self._getMotionConeParams(contact_point)

        if vt > ut*vn:
            return GREATER_THAN
        elif vt < ub*vn:
            return LESS_THAN
        else:
            return INSIDE

    """Translate local frame coordinate system to global coordinate"""
    def _local_to_global(self, point):
        c, s = np.cos(self.theta), np.sin(self.theta)
        translated = (c * point[0] - s * point[1] + self.x, s * point[0] + c * point[1] + self.y)
        return translated
        

    """
    Steps the model forward
    """
    def _step(self, contact_point, contact_velocity):

        #if no contact break
        if not self.polygon.edgePointContact(contact_point):
            self.trajectory[-1].append(contact_point)
            self.trajectory[-1].append(contact_velocity)
            self.trajectory += [ [self.x, self.y, self.theta] ]
            return

        print("contact_point")
        print(contact_point)
        print("contact_velocity")
        print(contact_velocity)
        print(self.theta)
        print(self.x, self.y)

        contact_point = self._local_to_global(contact_point)
        contact_velocity = self._local_to_global(contact_velocity)

        print("contact_point")
        print(contact_point)
        print("contact_velocity")
        print(contact_velocity)

        xd = self._motion_model(contact_point, contact_velocity)
        self.x = self.x + xd[0,0]
        self.y = self.y + xd[1,0]
        self.theta = self.theta + xd[2,0]
        self.polygon.translate(xd[0,0],xd[1,0])
        self.polygon.rotate(xd[2,0])

        #add the velocity to the trajectory
        self.trajectory[-1].append(contact_point)
        self.trajectory[-1].append(contact_velocity)

        self.trajectory += [ [self.x, self.y, self.theta] ]


    """
    Implements the swtiched linear dynamics model
    """
    def _motion_model(self, contact_point, contact_velocity):

        a , b = np.cos(self.theta), np.sin(self.theta)
        C = np.matrix('{} {}; {} {}'.format(a, b, -b, a))
        u = np.matrix('{}; {}'.format(contact_velocity[0], contact_velocity[1]))


        c = self.fmax/self.mass
        px = self.x - contact_point[0] 
        py = self.y - contact_point[1]
        contact_norm = c**2 + px**2 + py**2
        Q = np.matrix('{} {}; {} {}'.format(c**2 + px, 
                                                px*py, 
                                                px*py, 
                                                c**2 + py)) * 1.0/(contact_norm)

        
        ut,ub = self._getMotionConeParams(contact_point)
        mode = self._getContactMode(contact_point,contact_velocity)


        parametrized = {INSIDE: {'P': np.eye(2),
                                 'b': np.matrix('{} {}'.format(-py/(contact_norm), px)),
                                 'c': np.zeros((1,2))},

                        GREATER_THAN: {'P': np.matrix('{} {}; {} {}'.format(1, 0,ut,0)),
                                       'b': np.matrix('{} {}'.format((-py+ut*px)/(contact_norm), 0)),
                                       'c': np.matrix('{} {}'.format(-ut, 0))},

                        LESS_THAN: {'P': np.matrix('{} {}; {} {}'.format(1, 0,ub,0)),
                                       'b': np.matrix('{} {}'.format((-py+ub*px)/(contact_norm), 0)),
                                       'c': np.matrix('{} {}'.format(-ub, 0))} }

        A = np.vstack([C.T*Q*parametrized[mode]['P'], \
                        parametrized[mode]['b'], 
                        parametrized[mode]['c']])

        return A*u


    def __str__(self):
        return str({'x': self.x, 'y': self.y, 'theta': self.theta})



"""
Geometric structure that does some useful
coordinate transformations.
"""
class Polygon2D(object):

    """
    Give it a list of points [(x,y)] in clockwise order
    """
    def __init__(self, vertex_list):
        self.vertex_list = vertex_list
        self.N = len(vertex_list)
        self.centroid = self.getCentroid()

    """
    Primitive useful for calculating areas and centroids
    """
    def _vertexIterator(self):
        for i in range(self.N):
            j = (i + 1) % self.N

            xi = self.vertex_list[i][0]
            xj = self.vertex_list[j][0]
            yi = self.vertex_list[i][1]
            yj = self.vertex_list[j][1]

            yield {'cur':(xi,yi), \
                   'next':(xj, yj), \
                   'det': (xi*yj - xj*yi) }


    """
    Get the area of the polygon
    """
    def getArea(self, signed=False):
        if signed:
            return 0.5*sum([v['det'] for v in self._vertexIterator()])
        else:
            return np.abs(0.5*sum([v['det'] for v in self._vertexIterator()]))

    """
    Gets the com of the polygon
    """
    def getCentroid(self):
        z = 1.0/(6*self.getArea(signed=True))
        cx = z*sum([(v['cur'][0]+v['next'][0])*v['det'] for v in self._vertexIterator()])
        cy = z*sum([(v['cur'][1]+v['next'][1])*v['det'] for v in self._vertexIterator()])
        return (cx, cy)

    """
    String representation useful for debugging
    """
    def __str__(self):
        return str({'vertices': self.vertex_list, \
                    'area': self.getArea(), \
                    'centroid': self.getCentroid()})



    """
    rotates the polygon
    """
    def rotate(self,theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        
        for i, v in enumerate(self.vertex_list):
            rotated = R * np.matrix('{};{}'.format(self.vertex_list[i][0],\
                                                               self.vertex_list[i][1]))
            self.vertex_list[i] = (rotated[0,0], rotated[1,0])
        self.centroid = self.getCentroid()

    """
    translates the polygon
    """
    def translate(self,dx,dy):
        
        for i, v in enumerate(self.vertex_list):
            self.vertex_list[i] = (self.vertex_list[i][0] + dx, \
                                    self.vertex_list[i][1] + dy)
        self.centroid = self.getCentroid()


    def getContactVel(self, contact_point, contact_velocity):

        dist = [((contact_point[0] - v['cur'][0])**2 +  \
                 (contact_point[1] - v['cur'][1])**2 +  \
                 (contact_point[0] - v['next'][0])**2 +  \
                 (contact_point[1] - v['next'][1])**2, v) \
                    for v in self._vertexIterator()]
        print(dist)

        dist.sort(key=lambda x: x[0])

        edge = dist[0][1]
        contact_dest = (contact_point[0] + contact_velocity[0], \
                          contact_point[1] + contact_velocity[1])
        speed = np.sqrt(contact_velocity[0]**2 + contact_velocity[0]**2)

        ray1 = np.array((edge['next'][0] - edge['cur'][0], edge['next'][1] - edge['cur'][1]))
        ray2 = np.array((contact_dest[0] - contact_point[0], contact_dest[1] - contact_point[1]))

        cx = np.arccos(np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2)))
        cy = np.arcsin(np.dot(ray1, ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2)))

        return cx*speed, cy*speed


    def coordList(self):
        
        x = [v['cur'][0] for v in self._vertexIterator()]
        x += [x[0]]

        y = [v['cur'][1] for v in self._vertexIterator()]
        y += [y[0]]

        return x,y


    """
    Given a contact point returns whether the point is in contact with the edge
    """
    def edgePointContact(self, p1, distance=0.01):
        contacts = [p1 for v in self._vertexIterator() \
                        if self.pointToLineDistance(p1, v['cur'], v['next']) < distance] 
        return (len(contacts) > 0)


    """
    p2l distance
    """
    def pointToLineDistance(self, p1, e1, e2):
        numerator = np.abs((e2[1] - e1[1])*p1[0] - (e2[0] - e1[0])*p1[1] + e2[0]*e1[1] - e1[0]*e2[1])
        normalization =  np.sqrt((e2[1] - e1[1])**2 + (e2[0] - e1[0])**2)
        print(numerator/normalization)
        return numerator/normalization


"""
renders backwards
"""
def visualizeTrajectory(ss):
    import matplotlib.pyplot as plt
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    x,y = ss.polygon.coordList()
    plt.plot(x,y, c='k', linewidth=4)

    N = len(ss.trajectory)
    for i in range(1, N-1):
        
        t = ss.trajectory[N-i]
        tp = ss.trajectory[N-i-1]
        dx, dy, dtheta = (tp[0] - t[0], tp[1] - t[1], tp[2] - t[2])

        ss.polygon.rotate(dtheta)
        ss.polygon.translate(dx,dy)
        
        x,y = ss.polygon.coordList()
        plt.plot(x,y,c=(0,0,0,(N-i+0.1)/N), linewidth=4)
        print(i, t, tp)


    plt.show()







p1 = Polygon2D([(0,0),(0,1),(1,1),(1,0)])

s = SliderSystem(p1)

s._step((0,0.5), (1,0))

s._step((0,0.5), (0.2,0))

s._step((0,0.5), (0.2,0))

s._step((0,0.5), (0.2,0))

print(s.trajectory)

visualizeTrajectory(s)


