import numpy as np

####################
# Helper Functions #
####################
def get_centroid(vertices):
	area = 0
	c_x = 0
	c_y = 0
	n = len(vertices)
	for i in range(n):
		diff = vertices[i][0]*vertices[(i+1)%n][1] - vertices[(i+1)%n][0]*vertices[i][1]
		area += diff
		c_x += ((vertices[i][0] + vertices[(i+1)%n][0])*diff)
		c_y += ((vertices[i][1] + vertices[(i+1)%n][1])*diff)
	return c_x/(3*area), c_y/(3*area)

#################
# Polygon Class #
#################
class Polygon:
	def __init__(self, vertices):
		self.centroid_x, self.centroid_y = get_centroid(vertices)
		self.vertices = [(vertex[0] - self.centroid_x, vertex[1] - self.centroid_y) for vertex in vertices]
		